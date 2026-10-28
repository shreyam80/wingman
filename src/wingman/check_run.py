"""Automatic sanity checks for one run folder. Prints PASS / WARN / FAIL per check.

These check that outputs are internally consistent and match the source video.
They do NOT check that joints are anatomically correct - that still needs a human
looking at the overlay.

Usage:
    python -m wingman.check_run --run outputs/p1_orange_16-36_s
Exit code is 1 if any check FAILs (so it can be used in scripts).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wingman.keypoints import KEYPOINT_NAMES

results = []


def report(status, name, detail):
    results.append(status)
    print(f"[{status:4}] {name}: {detail}")


def check(cond, name, ok_detail, bad_detail, fail_status="FAIL"):
    """Details may be lambdas so they are only built when that branch applies."""
    detail = ok_detail if cond else bad_detail
    report("PASS" if cond else fail_status, name, detail() if callable(detail) else detail)


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    info = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-jump", type=float, default=0.5,
                    help="torso-centre jump between consecutive frames, in box heights, "
                         "above which we warn about a possible identity switch")
    args = ap.parse_args()
    run = Path(args.run)

    # 1. Files present
    need = ["meta.json", "frames.csv", "detections.csv", "overlay_all.mp4"]
    missing = [f for f in need if not (run / f).exists()]
    check(not missing, "files present", "all stage-1 outputs exist", f"missing {missing}")
    if missing:
        sys.exit(1)
    meta = json.loads((run / "meta.json").read_text())
    vi = meta["video_info"]
    fps, W, H = vi["fps"], vi["width"], vi["height"]
    frames = pd.read_csv(run / "frames.csv")
    det = pd.read_csv(run / "detections.csv")

    # 2. Timestamps: one row per frame, increasing, spaced 1/fps apart
    n = meta["excerpt"]["frames_processed"]
    check(len(frames) == n, "frame count", f"{n} frames logged", f"{len(frames)} rows vs {n} processed")
    steps = frames.t_s.diff().dropna()
    check(frames.t_s.is_monotonic_increasing and steps.gt(0).all(), "timestamps increasing",
          f"{frames.t_s.iloc[0]:.2f}s to {frames.t_s.iloc[-1]:.2f}s", "timestamps go backwards or repeat")
    bad = (steps - 1 / fps).abs() > 0.5 / fps
    check(not bad.any(), "timestamp spacing", f"every step = {1 / fps:.3f}s (no dropped frames)",
          lambda: f"{bad.sum()} irregular gaps, e.g. at t={frames.t_s[bad.index[bad]].iloc[0]:.2f}s", "WARN")

    # 3. Detection values in valid ranges
    conf_cols = [f"{k}_conf" for k in KEYPOINT_NAMES]
    check(det[conf_cols + ["box_conf"]].stack().between(0, 1).all(), "confidences in 0-1",
          "all joint and box confidences valid", "confidence outside 0-1")
    slack = 0.05 * max(W, H)
    inside = det.x1.ge(-slack) & det.y1.ge(-slack) & det.x2.le(W + slack) & det.y2.le(H + slack)
    check(inside.all(), "boxes inside image", f"all {len(det)} boxes within {W}x{H}",
          f"{(~inside).sum()} boxes outside the frame")
    dup = det[det.track_id >= 0].duplicated(["frame_idx", "track_id"]).sum()
    check(dup == 0, "track ids unique per frame", "no id used twice in one frame", f"{dup} duplicates")

    # 4. Selected-person behaviour (only if selection has been made)
    if (run / "selected.csv").exists():
        sel = pd.read_csv(run / "selected.csv")
        chosen = json.loads((run / "selection.json").read_text())
        chain_ids = {c["track_id"] for c in chosen["chain"]}
        check(sel.frame_idx.tolist() == frames.frame_idx.tolist(), "selected rows align",
              "one selected.csv row per processed frame", "selected.csv frames differ from frames.csv")
        check(sel.status.isin(["tracked", "lost", "before_selection"]).all(), "status values",
              str(sel.status.value_counts().to_dict()), "unexpected status value")
        tr = sel[sel.status == "tracked"]
        check(tr.selected_track_id.isin(chain_ids).all(), "never another person",
              f"every tracked row uses chosen/confirmed id(s) {sorted(chain_ids)}",
              "a tracked row uses an id the human did not choose")
        lost = sel[sel.status != "tracked"]
        check(lost[[f"{k}_x" for k in KEYPOINT_NAMES]].isna().all().all(), "no invented joints",
              f"{len(lost)} lost/unselected rows have empty joints", "lost rows contain joint values")
        # Big jump of the torso centre between consecutive frames suggests an identity switch.
        cx = tr[["left_shoulder_x", "right_shoulder_x", "left_hip_x", "right_hip_x"]].mean(axis=1)
        cy = tr[["left_shoulder_y", "right_shoulder_y", "left_hip_y", "right_hip_y"]].mean(axis=1)
        consecutive = tr.frame_idx.diff() == 1
        jump = np.hypot(cx.diff(), cy.diff()) / (tr.y2 - tr.y1)
        big = jump[consecutive] > args.max_jump
        check(not big.any(), "no sudden torso jumps",
              lambda: f"max jump {jump[consecutive].max():.2f} box heights (limit {args.max_jump})",
              lambda: f"{big.sum()} jumps, first at t={tr.t_s[big.index[big]].iloc[0]:.2f}s - inspect for an ID switch",
              "WARN")
        print(f"       tracked fraction after selection: {chosen['tracked_fraction_after_selection']:.1%}; "
              f"lost intervals: {chosen['lost_intervals_s']}")
    else:
        report("WARN", "selection", "no selected.csv yet - run select_climber")

    # 5. Overlay videos match the input and play
    for name in ("overlay_all.mp4", "overlay_selected.mp4"):
        if not (run / name).exists():
            continue
        cnt, vfps, vw, vh = video_frames(run / name)
        check(cnt == n and abs(vfps - fps) < 0.01 and (vw, vh) == (W, H), f"{name} format",
              f"{cnt} frames, {vfps} FPS, {vw}x{vh} (same as input)",
              f"{cnt} frames, {vfps} FPS, {vw}x{vh} vs input {n}, {fps}, {W}x{H}")
    # Compare a few overlay frames with the original, in a band of rows the drawings
    # rarely cover. Small differences = video compression; large = wrong frame/offset.
    src = cv2.VideoCapture(meta["video"])
    ovl = cv2.VideoCapture(str(run / "overlay_all.mp4"))
    start = meta["excerpt"]["start_frame"]
    for _ in range(start):
        src.grab()
    probes, diffs = {0, n // 2, n - 1}, []
    for i in range(n):
        ok1, a = src.read()
        ok2, b = ovl.read()
        if not (ok1 and ok2):
            break
        if i in probes:
            diffs.append(float(np.abs(a[H - 200:H - 60].astype(int) - b[H - 200:H - 60].astype(int)).mean()))
    check(len(diffs) == 3 and max(diffs) < 10, "overlay matches source",
          f"mean pixel difference {[round(d, 1) for d in diffs]} (compression only)",
          f"pixel differences {diffs} - overlay may be offset from the source")

    fails = results.count("FAIL")
    print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
