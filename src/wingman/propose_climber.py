"""Propose the climber in many runs, let a human confirm, then apply the selection.

Step 1 - propose (no choice is applied):
    python -m wingman.propose_climber --name s960
  For every outputs/<name>/<video>/ it writes review.jpg (6 frames across the video,
  all boxes drawn, the proposed climber highlighted) and adds a row to
  labels/climber_selection_<name>.csv with confirmed = "" (blank).

  Proposal rule: start from the track with the highest (vertical travel x frames seen)
  - climbers climb and stay visible, bystanders stay on the floor - then join tracks that begin/end
  within 5 s and near it in place (a re-detection after the tracker lost the
  climber). This is only a suggestion - it can be wrong.

  To re-propose specific unconfirmed rows after changing the rule:
    python -m wingman.propose_climber --name s960 --redo p5_orange,p5_green

Step 2 - a human opens each review.jpg and edits the CSV:
  confirmed = yes   (proposal is right)
  or corrects track_id / links, then sets confirmed = yes
  reviewer = initials

Step 3 - apply (only rows with confirmed = yes):
    python -m wingman.propose_climber --name s960 --apply [--no-video]
  runs select_climber for each confirmed row. Unconfirmed rows are skipped and listed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wingman.video_io import put_label
from wingman.paths import dataset_file

PROJECT = Path(__file__).resolve().parents[2]
COLS = ["video_id", "track_id", "links", "confirmed", "reviewer", "notes"]


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def near(a, b):
    """Boxes overlap, or centres are within one box height of each other
    (the climber may have moved a little during a gap of up to a few seconds)."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return box_iou(a, b) > 0.1 or np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 1.0 * (a[3] - a[1])


def propose_chain(det, max_gap_s=5.0, overlap_s=2.0):
    """Return [(track_id, from_s), ...]: the main climber track, plus nearby
    continuations before and after it.

    Main track = highest (vertical travel of the box centre x frames seen): a person who
    climbs AND stays visible. Each half alone failed on real footage (p5, dev videos):
      - frames seen only -> a bystander standing at the floor the whole time
      - travel only      -> a person at the picture's edge moving fast for ~9 s"""
    det = det[det.track_id >= 0]
    if det.empty:
        return []
    cy = (det.y1 + det.y2) / 2
    spans = det.assign(cy=cy).groupby("track_id").agg(
        first=("t_s", "min"), last=("t_s", "max"), n=("t_s", "size"),
        travel=("cy", lambda s: s.max() - s.min()))
    main = int((spans.travel * spans.n).idxmax())
    box = lambda tid, t: det[(det.track_id == tid) & (det.t_s == t)][["x1", "y1", "x2", "y2"]].values[0]
    used, chain = {main}, [(main, float(spans.loc[main, "first"]))]

    cur = main  # extend forwards in time
    while True:
        end_t, end_box = spans.loc[cur, "last"], box(cur, spans.loc[cur, "last"])
        cands = [(tid, s) for tid, s in spans.iterrows() if tid not in used
                 and end_t - overlap_s <= s["first"] <= end_t + max_gap_s and s["last"] > end_t]
        best = None
        for tid, s in cands:
            seen = det[(det.track_id == tid) & (det.t_s > end_t)]
            if not seen.empty and near(end_box, seen.iloc[0][["x1", "y1", "x2", "y2"]].values):
                if best is None or s["n"] > spans.loc[best, "n"]:
                    best = tid
        if best is None:
            break
        nxt = det[(det.track_id == best) & (det.t_s > end_t)].t_s.min()
        chain.append((int(best), float(nxt)))
        used.add(best)
        cur = best

    cur = main  # extend backwards in time
    while True:
        start_t, start_box = spans.loc[cur, "first"], box(cur, spans.loc[cur, "first"])
        cands = [(tid, s) for tid, s in spans.iterrows() if tid not in used
                 and start_t - max_gap_s <= s["last"] <= start_t + overlap_s and s["first"] < start_t]
        best = None
        for tid, s in cands:
            seen = det[(det.track_id == tid) & (det.t_s < start_t)]
            if not seen.empty and near(start_box, seen.iloc[-1][["x1", "y1", "x2", "y2"]].values):
                if best is None or s["n"] > spans.loc[best, "n"]:
                    best = tid
        if best is None:
            break
        # earlier track is followed from its own start; `cur` takes over once it ends
        takeover = det[(det.track_id == cur) & (det.t_s > spans.loc[best, "last"])].t_s.min()
        if pd.notna(takeover):
            chain = [(tid, float(takeover)) if tid == cur else (tid, t) for tid, t in chain]
        chain = [(int(best), float(spans.loc[best, "first"]))] + chain
        used.add(best)
        cur = best
    return sorted(chain, key=lambda c: c[1])


def review_sheet(run, det, chain, path, n_tiles=6):
    """6 frames across the video: proposed climber green with its id, others grey."""
    meta_video = str(dataset_file(json.loads((run / "meta.json").read_text())["video"]))
    frames = pd.read_csv(run / "frames.csv")
    t_of = dict(zip(frames.frame_idx, frames.t_s))
    want = set(np.linspace(frames.frame_idx.min(), frames.frame_idx.max(), n_tiles).astype(int))
    chain_ids = {c[0] for c in chain}
    cap = cv2.VideoCapture(meta_video)
    tiles, fi = [], 0
    while len(tiles) < len(want):
        ok, img = cap.read()
        if not ok:
            break
        if fi in want:
            for r in det[det.frame_idx == fi].itertuples():
                mine = r.track_id in chain_ids
                color = (0, 255, 0) if mine else (160, 160, 160)
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 4 if mine else 2)
                put_label(img, f"ID {r.track_id}", (int(r.x1), max(40, int(r.y1) - 10)), 1.1, color)
            put_label(img, f"t={t_of[fi]:.1f}s", (15, 50), 1.3)
            tiles.append(cv2.resize(img, (360, 640)))
        fi += 1
    cap.release()
    sheet = np.hstack(tiles)
    banner = np.zeros((70, sheet.shape[1], 3), np.uint8)
    put_label(banner, f"{run.name}   proposed climber chain: "
              + "  ->  ".join(f"ID {t} from {s:.1f}s" for t, s in chain), (10, 45), 0.9, (0, 255, 0))
    cv2.imwrite(str(path), np.vstack([banner, sheet]))


def propose(name, redo=()):
    root = PROJECT / "outputs" / name
    csv_path = PROJECT / "labels" / f"climber_selection_{name}.csv"
    old = pd.read_csv(csv_path, dtype=str).fillna("") if csv_path.exists() else pd.DataFrame(columns=COLS)
    confirmed_redo = old[old.video_id.isin(redo) & (old.confirmed.str.lower() == "yes")].video_id.tolist()
    if confirmed_redo:
        raise SystemExit(f"Refusing to redo rows a human already confirmed: {confirmed_redo}")
    old = old[~old.video_id.isin(redo)]  # only the listed, unconfirmed rows are replaced
    rows = []
    for run in sorted(p for p in root.iterdir() if (p / "detections.csv").exists()):
        if run.name in set(old.video_id):
            continue  # never overwrite a row a human may already have edited
        det = pd.read_csv(run / "detections.csv")
        chain = propose_chain(det)
        review_sheet(run, det, chain, run / "review.jpg")
        rows.append({"video_id": run.name,
                     "track_id": chain[0][0] if chain else "",
                     "links": ";".join(f"{t}@{s:.2f}" for t, s in chain[1:]),
                     "confirmed": "", "reviewer": "", "notes": ""})
        print(f"{run.name}: proposed {chain}  -> {run / 'review.jpg'}")
    pd.concat([old, pd.DataFrame(rows, columns=COLS)]).to_csv(csv_path, index=False)
    print(f"Proposals in {csv_path}. Nothing is applied until 'confirmed' is set to yes.")


def apply(name, no_video):
    csv_path = PROJECT / "labels" / f"climber_selection_{name}.csv"
    sel = pd.read_csv(csv_path, dtype=str).fillna("")
    todo = sel[sel.confirmed.str.lower() == "yes"]
    skipped = sel[sel.confirmed.str.lower() != "yes"].video_id.tolist()
    for r in todo.itertuples():
        cmd = [sys.executable, "-m", "wingman.select_climber", "--run",
               str(PROJECT / "outputs" / name / r.video_id), "--track-id", r.track_id,
               "--selected-by", r.reviewer or "unspecified"]
        for link in filter(None, r.links.split(";")):
            cmd += ["--link", link]
        if no_video:
            cmd.append("--no-video")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{r.video_id}: {'ok' if res.returncode == 0 else 'FAILED ' + res.stderr[-500:]}")
    if skipped:
        print(f"Not applied (not confirmed): {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="setting name used by batch.py, e.g. s960")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--redo", default="", help="comma list of video ids to re-propose (unconfirmed rows only)")
    args = ap.parse_args()
    if args.apply:
        apply(args.name, args.no_video)
    else:
        propose(args.name, [v for v in args.redo.split(",") if v])


if __name__ == "__main__":
    main()
