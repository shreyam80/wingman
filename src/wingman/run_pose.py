"""Stage 1: run a pretrained pose model + ByteTrack on a video excerpt.

Saves EVERY tracked person (not just the climber) so person selection can be
done/redone later without re-running the model:

    <out>/detections.csv  one row per tracked person per frame: track id, box,
                          17 joints (x, y in source pixels; conf 0-1)
    <out>/frames.csv      one row per processed frame (even if nobody detected):
                          timestamp, number of people, inference time
    <out>/meta.json       model file, package versions, tracker settings, video
                          properties, excerpt, device, total runtime
    <out>/overlay_all.mp4 inspection video: every track's box, ID and skeleton

Usage:
    python -m wingman.run_pose --video PATH --start 0 --end 20 --out outputs/p1_orange_0-20
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from wingman.keypoints import KEYPOINT_NAMES, draw_skeleton, kp_columns
from wingman.video_io import open_writer, put_label, video_info

TRACK_COLORS = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 128, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="excerpt start, seconds")
    ap.add_argument("--end", type=float, default=None, help="excerpt end, seconds (default: video end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt", help="official Ultralytics checkpoint")
    ap.add_argument("--tracker", default="configs/bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="longest side the model sees, pixels")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum person-box confidence")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--draw-min-conf", type=float, default=0.5,
                    help="joints below this confidence are not drawn in the overlay")
    args = ap.parse_args()

    video = Path(args.video).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    start_f = int(round(args.start * info["fps"]))
    end_f = info["frame_count"] if args.end is None else min(info["frame_count"], int(round(args.end * info["fps"])))

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video))
    for _ in range(start_f):  # sequential skip: exact frame positions, unlike seeking
        cap.grab()
    writer, codec = open_writer(out / "overlay_all.mp4", info["fps"], info["width"], info["height"])

    det_f = open(out / "detections.csv", "w", newline="")
    det_w = csv.writer(det_f)
    det_w.writerow(["frame_idx", "t_s", "track_id", "x1", "y1", "x2", "y2", "box_conf"] + kp_columns())
    frm_f = open(out / "frames.csv", "w", newline="")
    frm_w = csv.writer(frm_f)
    frm_w.writerow(["frame_idx", "t_s", "n_tracked", "infer_ms"])

    t0 = time.perf_counter()
    n_done = 0
    for frame_idx in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            print(f"Video ended early at frame {frame_idx}")
            break
        # Decoder timestamp of this frame (ms). Used instead of assuming frame/fps.
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        ti = time.perf_counter()
        r = model.track(frame, persist=True, tracker=args.tracker, imgsz=args.imgsz,
                        conf=args.conf, device=args.device, verbose=False)[0]
        infer_ms = (time.perf_counter() - ti) * 1000

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.id is not None
               else np.full(len(boxes), -1))  # -1: detected but not (yet) assigned a track
        kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))

        vis = frame.copy()
        for i in range(len(boxes)):
            det_w.writerow([frame_idx, f"{t_s:.4f}", ids[i]] + [f"{v:.1f}" for v in boxes[i]]
                           + [f"{confs[i]:.3f}"]
                           + [f"{v:.3f}" if k % 3 == 2 else f"{v:.1f}" for k, v in enumerate(kps[i].ravel())])
            color = TRACK_COLORS[ids[i] % len(TRACK_COLORS)] if ids[i] >= 0 else (128, 128, 128)
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            put_label(vis, f"ID {ids[i]}  {confs[i]:.2f}", (x1, max(30, y1 - 10)), 0.8, color)
            draw_skeleton(vis, {n: tuple(kps[i][j]) for j, n in enumerate(KEYPOINT_NAMES)}, args.draw_min_conf)
        frm_w.writerow([frame_idx, f"{t_s:.4f}", len(boxes), f"{infer_ms:.1f}"])
        put_label(vis, f"t={t_s:6.2f}s  frame {frame_idx}  tracked={len(boxes)}", (15, 40), 0.9)
        put_label(vis, f"{Path(args.model).stem} + ByteTrack | joints conf>={args.draw_min_conf}",
                  (15, info["height"] - 20), 0.6)
        writer.write(vis)
        n_done += 1
        if n_done % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {n_done}/{end_f - start_f} frames, {n_done / el:.1f} frames/s")

    total = time.perf_counter() - t0
    for h in (cap, writer):
        h.release()
    det_f.close()
    frm_f.close()

    meta = {
        "video": str(video), "video_info": info,
        "excerpt": {"start_s": args.start, "end_s": args.end, "start_frame": start_f,
                    "end_frame_exclusive": start_f + n_done, "frames_processed": n_done},
        "model": args.model, "imgsz": args.imgsz, "box_conf_min": args.conf, "device": args.device,
        "tracker_file": args.tracker, "tracker_settings": yaml.safe_load(open(args.tracker)),
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                     "opencv": cv2.__version__, "python": platform.python_version()},
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "keypoint_order": KEYPOINT_NAMES,
        "coordinates": "pixels in the source video frame; (0,0) top-left",
        "overlay_codec": codec,
        "runtime_s": round(total, 2),
        "processing_fps": round(n_done / total, 2) if total else None,
        "realtime_factor": round((n_done / info["fps"]) / total, 2) if total else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done: {n_done} frames in {total:.1f}s ({meta['processing_fps']} frames/s, "
          f"{meta['realtime_factor']}x real time). Output: {out}")


if __name__ == "__main__":
    main()
