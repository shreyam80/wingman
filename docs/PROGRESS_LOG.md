# Wingman progress log

Status words: **planned** (written down only) · **implemented** (code exists) ·
**run** (executed on real data) · **verified** (output checked by inspection/tests).

## 2026-10-28 – Milestone 1: environment and sample  ✅ verified

- Laptop: macOS 26.3.1, Apple M3 Pro, 18 GB RAM, **~30 GB free disk (94% full)**.
  Python 3.11.7 isolated env via uv in `.venv` (project moved from ~/wingman to ~/CIS Senior Design Docs/wingman; env rebuilt there, checks re-passed 15/15).
- Versions: ultralytics 8.4.191, torch 2.14.1 (MPS GPU available), opencv 5.0.0,
  numpy 2.4.6. Full pin: `requirements.lock.txt`.
- Dataset already extracted at `~/Downloads/wingman dataset` (20 GB, unmodified).
  README read. License **CC BY 4.0** (Zenodo API metadata; README itself has none).
  22 small videos / 10 participant groups (p2a, p2b = same person) – `data/manifest.csv`
  with SHA-256 hashes.
- README facts that matter: annotations refer to small 720×1280 25 FPS videos;
  holdUsage.csv = hold contact intervals per limb incl. occlusion (NOT joint or pause
  ground truth); bolt_holes.json = pixel→wall-cm homography (possible later use).
- Inspected p1_orange (88.5 s): footage is stabilized (shifting black borders);
  climber shrinks from ~2/3 frame height to ~10% near top; small bystanders at floor.

## 2026-10-28 – Milestone 2: real pose/tracking overlay  ✅ run + verified on one excerpt

Excerpt: p1_orange, 16–36 s (500 frames), ByteTrack defaults (`configs/bytetrack.yaml`).

| | yolo11n-pose | yolo11s-pose |
| --- | --- | --- |
| Frames with climber tracked | 484/500 | 498/500 |
| Frames with all 4 torso joints conf ≥ 0.5 | 484/500 | 498/500 |
| Median inference / frame (MPS) | 10.4 ms | 12.1 ms |
| Whole-run speed | 1.27× real time (incl. warm-up) | 2.15× real time |
| Extra tracks | ID 2 (1 frame, duplicate on climber) | ID 3 (7 frames, duplicate on climber) |

- MPS vs CPU joints identical on 3 test frames (0.00 px), so GPU output is trusted.
- Nano failure (inspected): climber clearly visible but undetected in wide-legged
  back-facing poses (30.5 s, 31.9–32.1 s, 35.6–36.0 s).
- **Decision: baseline = yolo11s-pose.pt** (sha256 1060bda4…). Nano kept for comparison.
- Selection: track 1, chosen by Claude after inspecting frames; **pending team
  confirmation**. 498/500 tracked, lost at 28.68 s and 35.64 s (single frames,
  marked lost, not filled). No identity switch observed in inspected frames.
- Checks: timestamps strictly increasing, 0.04 s steps; overlay 500 frames @ 25 FPS,
  H.264; matches source pixels (mean diff 2.55/255 in unannotated area).

What we can honestly say: "On a 20-second excerpt of one participant, pretrained
YOLO11s-pose with ByteTrack followed the selected climber in 498 of 500 frames at
about 2× real time on a laptop. The smaller nano model missed 16 frames where the
climber was visible."
Not yet supported: anything about other participants, the upper (small-climber) part
of the route, pause detection, or accuracy of joint positions (no joint ground truth).

## 2026-10-28 – Testing tools  ✅ verified
- `check_run.py`: 15 automatic checks. p1_orange_16-36_s: 15 pass. Nano run: 9 pass,
  1 warn (no selection made). Negative test: planted fake ID switch → correctly
  FAIL ("never another person") + WARN (torso jump at 28.00 s).
- `docs/WALKTHROUGH.md`: step-by-step guide for the team to reproduce and test.
- `docs/FILE_GUIDE.md`: plain-language explanation of every file and design decision.
- `docs/CODE_WALKTHROUGH.md`: line-by-line code walkthrough with real output rows.
- Project put under git (.gitignore excludes .venv, model weights, output videos/images) and pushed to private GitHub repo https://github.com/shreyam80/wingman.

## 2026-10-28 – Full p1 orange video  ✅ run + verified
- yolo11s, imgsz 640: climber received 5 track IDs (1→3→11→15→17) after >1.2 s misses near the top; 247 frames with no detection; top-section coverage 71.9%.
- yolo11s, imgsz 960 (18.6 ms/frame, 1.75× real time): climber keeps track 1 for the whole climb, 97.6% tracked, longest loss 0.7 s (77.8–78.5 s). Bystanders (IDs 3, 8, 13, bottom right) now detected and correctly not linked (overlap 0.0). 15/15 checks pass.
- **Decision: imgsz 960** becomes the default setting for our runs.
- Fixed suggest_links: it now finds a duplicate box that began before the climber's ID ended.
- Observed failure: left/right joint swap near the top (86 s).

## 2026-11-18/19 – Model comparison on 6 development videos  ✅ run + verified
New tools: `batch.py` (many videos), `propose_climber.py` (suggest climber, human confirms,
then apply), `eval_holds.py` (joints vs the dataset's hold-contact labels),
`compare.py` (one table), `configs/eval.yaml` (tolerance 0.5 torso lengths, joint
conf >= 0.5 – fixed 2026-11-18 before any result). Hold ids confirmed 0-based (6 routes;
median distance 0.0 vs 0.89 torso lengths if shifted by one).

Development split: p1, p3, p5 (6 videos, 8.6 min). Held out: p2, p4, p6, p7, p8, p9, p10.

| Median over 6 dev videos | s960 | s1280 | m960 |
| --- | --- | --- | --- |
| Speed vs real time | 1.92x | 1.31x | 0.90x |
| Climber tracked | 96.9% | 99.2% | 99.8% |
| IDs for the climber | 1.5 | 1.5 | 1 |
| Longest loss | 0.92 s | 0.34 s | 0.02 s |
| Hands at labelled hold | 92.4% | 96.1% | 99.4% |
| Feet at labelled hold | 83.4% | 83.8% | 89.5% |
| L/R swaps hands / feet (pooled) | 3.7% / 8.0% | 1.7% / 4.7% | 1.4% / 2.3% |

- Climber proposal rule: "seen in most frames" picked a bystander (p5_orange, s960 and
  s1280); "travels most" picked a person at the picture edge. Final rule: vertical travel
  x frames seen. All 18 picks confirmed by shreyamu after visual review.
- **Decision (shreyamu): m960 = yolo11m-pose.pt, imgsz 960** for after-flight analytics.
  Live drone following is a separate component (Model 1 on the Jetson), not this one.

## 2026-11-19 – m960 on all 22 videos; held-out evaluation  ✅ run + verified
- 22/22 videos processed; 16 new climber picks confirmed by shreyamu; overlay videos made
  (`outputs/m960/*/overlay_selected.mp4`, 391 MB, not in git). All 22 pass check_run.
- p2a_orange, p2a_green, p4_green ran 4-5x slower than the rest (laptop load) – their
  speed numbers are not representative.

| Held-out: 7 climbers, 16 videos, 17.1 min | m960 |
| --- | --- |
| Climber tracked (median) | 100% |
| All 4 torso joints usable (median) | 99.8% |
| Hands at labelled hold (median per video) | 97.8% (range 93.7-100%) |
| Feet at labelled hold (median per video) | 81.6% (range 60.7-97.3%) |
| L/R swaps, pooled | hands 1.0% (32/3267), feet 0.6% (65/11093) |
| Hand contact frames evaluable | 81.6% (rest: wrist low confidence / no estimate) |

What we can honestly say: "On 7 climbers not used to develop the system, the analytics
model tracked the selected climber in every frame and placed the hand at the dataset's
labelled hold 97.8% of the time (median per video); foot placement was weaker at 81.6%."
Not supported: exact joint accuracy (hit = within 0.5 torso lengths), other gyms, moving
(drone) camera, live speed on the Jetson. Feet: cause of low videos (p8_orange,
p9_green) not yet investigated.

## 2026-11-23 – Model 1 (live follower) Layer 1 simulation  ✅ run (simulation only)
Built `src/wingman/follow/` (settings in `configs/follow.yaml`), 13 passing tests, MAVSDK
4.2.1 installed (PX4Drone and LiveCamera written, NOT tested). Dataset bolt-hole files:
top reference points reversed in most 4-point files; `geometry.WallMap` refits row by row.

- Tuning (p4_orange, p5_orange only): yolo11n at conf 0.3 kept the climber fully in frame
  68-70%; yolo11s at conf 0.15 -> 100% / 95.7%. Defaults changed to yolo11s, conf 0.15.
- All 22 videos (`outputs/sim/all22_v1/`), the 20 NOT used for tuning (24.0 min):
  climber fully in frame median 100% (min 79.0%, p2b_green); out of frame max 7.6%;
  **wrong-person locks 0**; speed never above 0.45 m/s (limit 0.5).
  "Needs pilot" (lock lost > 2 s, drone holds) in 10 of 20 videos, every time with the
  climber fully visible near the centre: the detector missed them. Cause found at least
  in part in the simulator: at those heights the dataset video has 136-165 px per wall
  metre but the simulated view needs ~229, so the view is upscaled and soft. A real
  drone camera would not have this. Not yet known how the real camera behaves.
- What we can honestly say: "In simulation on 20 dataset climbs, the follow logic kept
  the climber fully in frame for the median climb and never locked onto a bystander;
  brief detector misses caused it to stop and request the pilot in half the climbs."
  Not supported: real-flight performance, Jetson speed, PX4 behaviour.

## Next (planned)
- Partner: analytics (torso pauses, limb movement) on `outputs/m960/*/selected.csv`;
  hand-label pauses on held-out videos BEFORE running the detector on them.
- Inspect low foot accuracy (p8_orange, p9_green).
- Model 1 (live follow): laptop "virtual drone camera" simulation, then Jetson
  (TensorRT, measured fps), PX4 SITL, bench tests with props off.
