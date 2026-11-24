"""Layer 1 simulation: fly a simulated drone in front of real dataset climbs.

Each video frame (25 per second):
  1. the simulated drone moves (SimDrone.step)
  2. the simulated camera renders what the drone sees (SimCamera.picture)
  3. at loop.rate_hz only: find people -> keep the lock -> controller -> speed command,
     which reaches the drone after loop.latency_s
  4. score against the truth: the human-confirmed climber box from Model 2 (m960).
     The truth is used ONLY for scoring, never for the drone's decisions.

Usage:
    python -m wingman.follow.run_follow --videos p4_orange,p8_orange
    python -m wingman.follow.run_follow --videos all --no-video --name sim_kp08
    python -m wingman.follow.run_follow --videos p4_orange --set detector.conf=0.15 --name conf015
Outputs (outputs/sim/<name>/):
    <video>/log.csv          one row per frame: drone height/speed, lock state, truth position
    <video>/events.json      lock, lost, relock, needs_pilot moments
    <video>/summary.json     scores for this video
    <video>/sim_view.mp4     what the drone saw (unless --no-video)
    summary.csv              one row per video;  config_used.yaml  exact settings used
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from wingman.follow.camera import SimCamera
from wingman.follow.config import PROJECT, load_config
from wingman.follow.controller import VerticalController
from wingman.follow.detector import DetectorFollower
from wingman.follow.drone import SimDrone
from wingman.follow.geometry import WallMap
from wingman.follow.hybrid import HybridFollower
from wingman.paths import dataset_file
from wingman.video_io import open_writer, put_label


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def visible_fraction(box, W, H):
    """Share of the box's area that lies inside the picture."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    ix = max(0.0, min(x2, W) - max(x1, 0))
    iy = max(0.0, min(y2, H) - max(y1, 0))
    return ix * iy / area if area > 0 else 0.0


def load_truth(video_id, wall_map):
    """frame_idx -> climber wall box (left, bottom, right, top), from confirmed m960 results."""
    sel = pd.read_csv(PROJECT / "outputs/m960" / video_id / "selected.csv")
    sel = sel[sel.status == "tracked"]
    return {int(r.frame_idx): wall_map.box_to_wall(r.x1, r.y1, r.x2, r.y2) for r in sel.itertuples()}


def simulate(video_id, cfg, out, device, make_video, max_seconds=None):
    m = pd.read_csv(PROJECT / "data/manifest.csv").set_index("video_id").loc[video_id]
    video = dataset_file(m.path)
    wall_map = WallMap(video.parent / f"{m.route}_bolt_holes.json")
    truth = load_truth(video_id, wall_map)
    first = min(truth)
    l, b, r, t_ = truth[first]
    drone = SimDrone(cfg, start_x=(l + r) / 2, start_y=(b + t_) / 2)  # pilot centred the climber
    cam = SimCamera(cfg, wall_map)
    # "tap": the user selects the climber; in simulation that is the true box at the start
    tap_box = cam.view(drone.x, drone.y).box_px(l, b, r, t_) if cfg.lock.start == "tap" else None
    if cfg.lock.mode == "hybrid":
        if tap_box is None:
            raise ValueError("lock.mode hybrid needs lock.start: tap")
        follower = HybridFollower(cfg, cam.W, cam.H, device, tap_box)
    else:
        follower = DetectorFollower(cfg, cam.W, cam.H, device, tap_box)
    ctrl = VerticalController(cfg, cam.view_h)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds:
        n = min(n, int(max_seconds * fps))
    for _ in range(first):
        cap.grab()
    out.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out / "sim_view.mp4", fps, cam.W, cam.H)[0] if make_video else None

    period = 1.0 / cfg.loop.rate_hz
    next_tick, people, box, cmd = first / fps, [], None, 0.0
    rows, events, infer_ms = [], [], []
    t_wall = time.perf_counter()
    for fi in range(first, n):
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if fi > first:
            drone.step(1.0 / fps)
        img = cam.picture(frame, drone.x, drone.y)

        tick = t + 1e-9 >= next_tick
        if tick:
            ti = time.perf_counter()
            box, event = follower.update(t, img)
            people = follower.people
            infer_ms.append((time.perf_counter() - ti) * 1000)
            v_frac = None if box is None else (box[1] + box[3]) / 2 / cam.H
            cmd = ctrl.update(v_frac, period)
            drone.send_up_speed(cmd)
            next_tick += period
            if event:
                events.append({"t_s": round(t, 2), "event": event, "track_id": follower.track_id,
                               "sharpness_ratio": round(float(cam.sharpness_ratio(drone.x, drone.y)), 3)})

        view = cam.view(drone.x, drone.y)
        row = {"frame_idx": fi, "t_s": round(t, 3), "drone_y_m": round(drone.y, 3),
               "drone_vy_mps": round(drone.vy, 3), "cmd_up_mps": round(cmd, 3),
               "lock_state": follower.state, "tick": tick,
               "sharpness_ratio": round(cam.sharpness_ratio(drone.x, drone.y), 3)}
        gt = truth.get(fi)
        gt_px = view.box_px(*gt) if gt else None
        if gt_px:
            row.update({"truth_visible_frac": round(visible_fraction(gt_px, cam.W, cam.H), 3),
                        "truth_centre_frac": round((gt_px[1] + gt_px[3]) / 2 / cam.H, 3)})
            if tick and box is not None:
                row["wrong_person"] = iou(box, gt_px) < cfg.sim.wrong_person_iou
        rows.append(row)

        if writer:
            for pb in people:
                cv2.rectangle(img, tuple(map(int, pb[:2])), tuple(map(int, pb[2:])), (160, 160, 160), 1)
            if gt_px:
                cv2.rectangle(img, tuple(map(int, gt_px[:2])), tuple(map(int, gt_px[2:])), (255, 255, 255), 1)
            if box is not None:
                cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 255, 0), 2)
            y_t = int(cfg.controller.target_point_frac * cam.H)
            cv2.line(img, (0, y_t), (cam.W, y_t), (0, 200, 255), 1)
            put_label(img, f"t={t:5.1f}s {follower.state}", (8, 24), 0.6,
                      (0, 255, 0) if follower.state == "LOCKED" else (0, 0, 255))
            put_label(img, f"cmd {cmd:+.2f} m/s  h={drone.y:.2f} m", (8, 48), 0.6)
            writer.write(img)
    cap.release()
    if writer:
        writer.release()

    log = pd.DataFrame(rows)
    log.to_csv(out / "log.csv", index=False)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    scored = log.dropna(subset=["truth_visible_frac"])
    sharp = cfg.sim.min_sharpness_ratio
    real = scored[scored.sharpness_ratio >= sharp]
    ticks = log[log.tick]
    accel = np.abs(np.diff(log.drone_vy_mps)) * fps
    # "Moving the wrong way": drone moving (> 0.05 m/s) while the climber is outside the
    # dead zone on the OTHER side of the target - i.e. overshooting / running away.
    off = cfg.controller.target_point_frac - scored.truth_centre_frac  # > 0: climber above target
    wrong_way = ((off.abs() > cfg.controller.deadband_frac) & (scored.drone_vy_mps.abs() > 0.05)
                 & (np.sign(off) != np.sign(scored.drone_vy_mps)))
    s = {
        "video_id": video_id,
        "seconds": round(len(log) / fps, 1),
        "frames_with_truth": len(scored),
        "climber_fully_in_frame": round((scored.truth_visible_frac >= 0.999).mean(), 4),
        "climber_mostly_in_frame": round((scored.truth_visible_frac >= 0.9).mean(), 4),
        "climber_out_of_frame": round((scored.truth_visible_frac < 0.5).mean(), 4),
        "median_offset_frac": round((scored.truth_centre_frac - cfg.controller.target_point_frac).abs().median(), 4),
        "p95_offset_frac": round((scored.truth_centre_frac - cfg.controller.target_point_frac).abs().quantile(0.95), 4),
        "locked_fraction_of_ticks": round((ticks.lock_state == "LOCKED").mean(), 4),
        "lost_events": sum(e["event"] == "lost" for e in events),
        "relocks": sum(e["event"] == "relock" for e in events),
        "needs_pilot_events": sum(e["event"] == "needs_pilot" for e in events),
        "needs_pilot_realistic": int(sum(e["event"] == "needs_pilot" and e["sharpness_ratio"] >= sharp for e in events)),
        "realistic_frames_frac": round(len(real) / len(scored), 4) if len(scored) else None,
        "fully_in_frame_realistic": round((real.truth_visible_frac >= 0.999).mean(), 4) if len(real) else None,
        "wrong_way_frac": round(wrong_way.mean(), 4),
        "wrong_person_ticks": int(log.get("wrong_person", pd.Series(dtype=bool)).fillna(False).sum()),
        "max_speed_mps": round(log.drone_vy_mps.abs().max(), 3),
        "max_accel_mps2": round(float(accel.max()) if len(accel) else 0.0, 3),
        "median_detect_ms": round(float(np.median(infer_ms)), 1) if infer_ms else None,
        "sim_x_real_time": round(len(log) / fps / (time.perf_counter() - t_wall), 2),
    }
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="comma list of video ids, or 'all'")
    ap.add_argument("--config", default=None, help="default: configs/follow.yaml")
    ap.add_argument("--name", default="sim", help="output folder under outputs/sim/")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None, help="only simulate the start of each video")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one setting, e.g. --set detector.conf=0.15 (repeatable)")
    args = ap.parse_args()

    cfg, raw = load_config(args.config, args.set)
    root = PROJECT / "outputs/sim" / args.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_used.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    vids = (pd.read_csv(PROJECT / "data/manifest.csv").video_id.tolist() if args.videos == "all"
            else args.videos.split(","))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    summaries = []
    for v in vids:
        s = simulate(v, cfg, root / v, device, not args.no_video, args.max_seconds)
        summaries.append(s)
        print(f"{v}: fully in frame {s['climber_fully_in_frame']:.1%}, out {s['climber_out_of_frame']:.1%}, "
              f"lost {s['lost_events']}, needs pilot {s['needs_pilot_events']} "
              f"({s['needs_pilot_realistic']} in sharp pictures), wrong person {s['wrong_person_ticks']}, "
              f"wrong way {s['wrong_way_frac']:.1%}, sharp {s['realistic_frames_frac']:.0%}, "
              f"{s['sim_x_real_time']}x real time")
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    print(f"-> {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
