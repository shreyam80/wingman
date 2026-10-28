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
- Project put under git (.gitignore excludes .venv, model weights, output videos/images).

## Next (planned)
- Run yolo11s on two more p1 excerpts: a pause section and the top of the route
  (climber small) – the plan's three-excerpt check.
- Confirm climber selection with the team.
- Milestone 3/4: torso-pause detector from `selected.csv` with configurable thresholds.
- Before looking at detector output: team labels pauses on held-out participants.
