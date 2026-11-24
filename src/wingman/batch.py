"""Run Stage 1 (run_pose) on many dataset videos with one model setting.

Videos come from data/manifest.csv. Results go to outputs/<name>/<video_id>/.
A video that already has meta.json is skipped, so an interrupted batch can simply be
started again.

Usage (from the project folder):
    python -m wingman.batch --name s960 --model yolo11s-pose.pt --imgsz 960 \
        --participants p1,p3,p5 --no-video
    python -m wingman.batch --name s960 --model yolo11s-pose.pt --imgsz 960   # all 22
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="setting name, used as the output folder, e.g. s960")
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--participants", help="comma list of participant groups, e.g. p1,p3,p5 (default: all)")
    ap.add_argument("--manifest", default=str(PROJECT / "data/manifest.csv"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    if args.participants:
        m = m[m.participant_group.isin(args.participants.split(","))]
    out_root = PROJECT / "outputs" / args.name
    print(f"{len(m)} videos, {m.duration_s.sum() / 60:.1f} min of footage -> {out_root}")

    for i, row in enumerate(m.itertuples(), 1):
        out = out_root / row.video_id
        if (out / "meta.json").exists():
            print(f"[{i}/{len(m)}] {row.video_id}: already done, skipping")
            continue
        cmd = [sys.executable, "-m", "wingman.run_pose", "--video", row.path, "--out", str(out),
               "--model", args.model, "--imgsz", str(args.imgsz),
               "--tracker", str(PROJECT / "configs/bytetrack.yaml")]
        if args.no_video:
            cmd.append("--no-video")
        t0 = time.perf_counter()
        # Run from models/ so the official weights are downloaded to / loaded from there.
        res = subprocess.run(cmd, cwd=PROJECT / "models", capture_output=True, text=True)
        status = "ok" if res.returncode == 0 else f"FAILED\n{res.stderr[-2000:]}"
        print(f"[{i}/{len(m)}] {row.video_id} ({row.duration_s:.0f}s): {status} "
              f"in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
