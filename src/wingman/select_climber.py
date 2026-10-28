"""Stage 2: choose the intended climber and follow only that person.

Step A - inspect (no choice made yet):
    python -m wingman.select_climber --run outputs/RUN --list --at 2.0
  prints every track ID with when/where it appears and saves
  <run>/preview_<t>s.jpg with IDs drawn, so a human can identify the climber.

Step B - select:
    python -m wingman.select_climber --run outputs/RUN --track-id 1
    python -m wingman.select_climber --run outputs/RUN --point 360,700 --at 2.0
  (--point picks the track whose box contains that pixel at that time.)

Identity rules: we follow exactly the chosen track ID. When it disappears the
frames are marked "lost" - we never jump to whichever person is nearest/largest.
If the tracker gives the climber a NEW id after an occlusion, the script prints
candidate ids; a human must confirm with --link NEW_ID@TIME, and that decision is
saved in selection.json.

Outputs: <run>/selected.csv (one row per processed frame, status + joints),
         <run>/selection.json, <run>/overlay_selected.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wingman.keypoints import KEYPOINT_NAMES, draw_skeleton, kp_columns
from wingman.video_io import open_writer, put_label

KP_COLS = kp_columns()


def load(run):
    meta = json.loads((run / "meta.json").read_text())
    det = pd.read_csv(run / "detections.csv")
    frames = pd.read_csv(run / "frames.csv")
    return meta, det, frames


def read_frame(meta, frame_idx):
    cap = cv2.VideoCapture(meta["video"])
    for _ in range(frame_idx):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def nearest_frame(frames, t):
    return int(frames.iloc[(frames.t_s - t).abs().argmin()].frame_idx)


def list_tracks(run, meta, det, frames, at):
    tracked = det[det.track_id >= 0]
    summary = tracked.groupby("track_id").agg(
        first_s=("t_s", "min"), last_s=("t_s", "max"), n_frames=("frame_idx", "count"),
        med_box_h=("y2", lambda s: float(np.median(s - tracked.loc[s.index, "y1"]))),
        med_cx=("x1", lambda s: float(np.median((s + tracked.loc[s.index, "x2"]) / 2))),
        med_cy=("y1", lambda s: float(np.median((s + tracked.loc[s.index, "y2"]) / 2))),
        med_conf=("box_conf", "median"),
    ).round(2)
    print(f"{len(frames)} frames processed; tracks found:\n{summary.to_string()}")
    print(f"Frames with no tracked person: {(frames.n_tracked == 0).sum()}")

    fi = nearest_frame(frames, at)
    img = read_frame(meta, fi)
    for _, r in det[det.frame_idx == fi].iterrows():
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
        put_label(img, f"ID {int(r.track_id)}", (p1[0], max(40, p1[1] - 12)), 1.2, (0, 255, 0))
    put_label(img, f"t={at:.2f}s frame {fi}", (15, 45), 1.0)
    path = run / f"preview_{at:.1f}s.jpg"
    cv2.imwrite(str(path), img)
    print(f"Preview with IDs saved to {path}")


def build_selection(det, frames, chain):
    """chain: list of (track_id, from_time_s) sorted by time. Returns per-frame table."""
    rows = []
    det_by_frame = {f: g for f, g in det.groupby("frame_idx")}
    for _, fr in frames.iterrows():
        active = [tid for tid, t0 in chain if fr.t_s >= t0 - 1e-6]
        tid = active[-1] if active else None
        g = det_by_frame.get(fr.frame_idx)
        hit = g[g.track_id == tid] if (g is not None and tid is not None) else None
        row = {"frame_idx": int(fr.frame_idx), "t_s": fr.t_s, "selected_track_id": tid}
        if tid is None:
            row["status"] = "before_selection"
        elif hit is not None and len(hit):
            r = hit.iloc[0]
            row.update({"status": "tracked", **{c: r[c] for c in ["x1", "y1", "x2", "y2", "box_conf"] + KP_COLS}})
        else:
            row["status"] = "lost"  # selected person not found in this frame; joints left empty
        rows.append(row)
    return pd.DataFrame(rows, columns=["frame_idx", "t_s", "status", "selected_track_id",
                                       "x1", "y1", "x2", "y2", "box_conf"] + KP_COLS)


def intervals(sel, status):
    """Contiguous [start_s, end_s] runs where sel.status == status."""
    out, start, prev = [], None, None
    for t, s in zip(sel.t_s, sel.status):
        if s == status and start is None:
            start = t
        if s != status and start is not None:
            out.append([round(start, 3), round(prev, 3)])
            start = None
        prev = t
    if start is not None:
        out.append([round(start, 3), round(prev, 3)])
    return out


def suggest_links(det, sel, chain_ids, window_s=3.0):
    """For each loss, list other track ids seen soon after it starts, with their overlap
    with the climber's last box. Includes ids that began *before* the loss (a duplicate
    box that takes over when the original id ends) - we saw this on real footage."""
    tips = []
    tracked_det = det[det.track_id >= 0]
    for a, b in intervals(sel, "lost"):
        last = sel[(sel.t_s < a) & (sel.status == "tracked")].tail(1)
        if last.empty:
            continue
        lb = last.iloc[0][["x1", "y1", "x2", "y2"]].astype(float).values
        # first appearance of each other id at or after the loss began
        after = tracked_det[tracked_det.t_s >= a - 1e-6].groupby("track_id").t_s.min()
        for tid, t0 in after.items():
            if tid in chain_ids or t0 > a + window_s:
                continue
            r = det[(det.track_id == tid) & (det.t_s == t0)].iloc[0]
            bb = np.array([r.x1, r.y1, r.x2, r.y2], dtype=float)
            ix = max(0, min(lb[2], bb[2]) - max(lb[0], bb[0]))
            iy = max(0, min(lb[3], bb[3]) - max(lb[1], bb[1]))
            inter = ix * iy
            iou = inter / ((lb[2] - lb[0]) * (lb[3] - lb[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter)
            tips.append({"lost_from_s": a, "candidate_id": int(tid), "appears_s": round(t0, 2),
                         "iou_with_last_box": round(float(iou), 2)})
    return tips


def render(run, meta, sel, chain, draw_min_conf):
    info = meta["video_info"]
    cap = cv2.VideoCapture(meta["video"])
    for _ in range(int(sel.frame_idx.iloc[0])):
        cap.grab()
    writer, _ = open_writer(run / "overlay_selected.mp4", info["fps"], info["width"], info["height"])
    for _, r in sel.iterrows():
        ok, img = cap.read()
        if not ok:
            break
        if r.status == "tracked":
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
            draw_skeleton(img, {n: (r[f"{n}_x"], r[f"{n}_y"], r[f"{n}_conf"]) for n in KEYPOINT_NAMES},
                          draw_min_conf)
            put_label(img, f"CLIMBER (track {int(r.selected_track_id)})", (15, 80), 0.8, (0, 255, 0))
        elif r.status == "lost":
            put_label(img, "SELECTED CLIMBER LOST - no measurement", (15, 80), 0.8, (0, 0, 255))
        put_label(img, f"t={r.t_s:6.2f}s  frame {int(r.frame_idx)}", (15, 40), 0.9)
        writer.write(img)
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--at", type=float, default=None, help="time (s) for preview / --point")
    ap.add_argument("--track-id", type=int)
    ap.add_argument("--point", help="x,y pixel inside the climber's box at time --at")
    ap.add_argument("--link", action="append", default=[],
                    help="human-confirmed re-identification NEW_ID@TIME_S (repeatable)")
    ap.add_argument("--selected-by", default="unspecified", help="who made the choice (pseudonym)")
    ap.add_argument("--draw-min-conf", type=float, default=0.5)
    args = ap.parse_args()

    run = Path(args.run)
    meta, det, frames = load(run)
    at = args.at if args.at is not None else float(frames.t_s.iloc[0])
    if args.list:
        list_tracks(run, meta, det, frames, at)
        return

    if args.point:
        px, py = map(float, args.point.split(","))
        fi = nearest_frame(frames, at)
        g = det[(det.frame_idx == fi) & (det.x1 <= px) & (det.x2 >= px) & (det.y1 <= py) & (det.y2 >= py)
                & (det.track_id >= 0)]
        if g.empty:
            raise SystemExit(f"No tracked box contains ({px},{py}) at frame {fi}. Run --list first.")
        g = g.assign(area=(g.x2 - g.x1) * (g.y2 - g.y1)).sort_values("area")
        tid, how = int(g.iloc[0].track_id), f"point ({px},{py}) at t={at}s, frame {fi}"
    elif args.track_id is not None:
        tid, how = args.track_id, f"track id {args.track_id} chosen after inspecting preview"
    else:
        raise SystemExit("Give --list, --track-id or --point")

    t_first = float(det[det.track_id == tid].t_s.min())
    chain = [(tid, t_first)]
    for spec in args.link:
        nid, t = spec.split("@")
        chain.append((int(nid), float(t)))
    chain.sort(key=lambda c: c[1])

    sel = build_selection(det, frames, chain)
    sel.to_csv(run / "selected.csv", index=False, float_format="%.4f")
    n = len(sel)
    counts = sel.status.value_counts().to_dict()
    report = {
        "selected_by": args.selected_by, "method": how,
        "chain": [{"track_id": c[0], "from_s": c[1]} for c in chain],
        "status_counts": counts,
        "tracked_fraction_after_selection": round(
            counts.get("tracked", 0) / max(1, n - counts.get("before_selection", 0)), 4),
        "lost_intervals_s": intervals(sel, "lost"),
        "suggested_links_unconfirmed": suggest_links(det, sel, [c[0] for c in chain]),
        "note": "Track ids are tracker guesses; inspect overlay_selected.mp4 for identity switches.",
    }
    (run / "selection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    render(run, meta, sel, chain, args.draw_min_conf)
    print(f"Wrote {run/'selected.csv'} and {run/'overlay_selected.mp4'}")


if __name__ == "__main__":
    main()
