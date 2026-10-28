# Hands-on walkthrough

Do these in order in a terminal. Each part says what you are looking at and why.
Everything here only reads the original videos; nothing in Downloads is changed.

## Part 0 – Get into the project

```bash
cd ~/wingman
source .venv/bin/activate
```
`source .venv/bin/activate` switches your terminal to this project's private Python,
which has the exact package versions in `requirements.lock.txt`. Your prompt will
show `(.venv)`. Type `deactivate` to leave.

## Part 1 – See what exists

```bash
ls src/wingman outputs outputs/p1_orange_16-36_s
```
- `src/wingman/` – the code (read it in this order: `keypoints.py`, `run_pose.py`,
  `select_climber.py`, `check_run.py`).
- `outputs/p1_orange_16-36_s/` – the run we keep (small model). `..._n` is the nano
  comparison run.

Watch the two videos:
```bash
open outputs/p1_orange_16-36_s/overlay_all.mp4
open outputs/p1_orange_16-36_s/overlay_selected.mp4
```
In `overlay_all` look for the brief extra box around 28 s (ID 3, a duplicate on the
climber). In `overlay_selected` that duplicate is gone because we follow one ID only.

## Part 2 – Look at the numbers behind the video

```bash
open -a "Numbers" outputs/p1_orange_16-36_s/selected.csv   # or Excel / Google Sheets
```
One row = one video frame (0.04 s). Find the row with `t_s` = 28.68: `status` is
`lost` and every joint column is empty. That is a deliberate rule: no measurement
is invented where the model saw nothing.

Pick any `tracked` row and compare `left_wrist_y` with `left_shoulder_y`. Pixel y
grows downward, so a wrist *above* the shoulder has a *smaller* y.

## Part 3 – Run the automatic checks

```bash
python -m wingman.check_run --run outputs/p1_orange_16-36_s
```
Expect 15 PASS. Each line names what is checked (timestamps, no invented joints,
never another person, overlay matches source …). See the docstring at the top of
`src/wingman/check_run.py` for what these checks do *not* prove.

## Part 4 – Run the pipeline yourself on new footage

Choose a different 20-second piece – the top of the route, where the climber is small:
```bash
cd models
python -m wingman.run_pose --video ~/Downloads/"wingman dataset/p1/orange.mp4" \
  --start 66 --end 86 --model yolo11s-pose.pt --tracker ../configs/bytetrack.yaml \
  --out ../outputs/p1_orange_66-86_s
cd ..
python -m wingman.select_climber --run outputs/p1_orange_66-86_s --list --at 70
open outputs/p1_orange_66-86_s/preview_70.0s.jpg
```
Decide which ID is the climber yourself, then:
```bash
python -m wingman.select_climber --run outputs/p1_orange_66-86_s --track-id <ID> --selected-by <your initials>
python -m wingman.check_run --run outputs/p1_orange_66-86_s
open outputs/p1_orange_66-86_s/overlay_selected.mp4
```
Write down: processing speed (printed at the end of run_pose), tracked fraction,
lost intervals, and whether the skeleton looks right when the climber is small.
This is real evidence you produced – it goes into the progress log.

## Part 5 – Experiment (optional)

Change one setting at a time and compare with the check script and your eyes:
- `--imgsz 960` – the model sees a larger image; may help small climbers, slower.
- `--draw-min-conf 0.3` – draw lower-confidence joints; see what the model is unsure about.
- `--model yolo11n-pose.pt` – the nano model, to see its misses yourself.

## Why we made each decision

| Decision | Why |
| --- | --- |
| Use pretrained weights, no training | Learned parameters already exist for "human body joints"; training needs thousands of labelled images and days. First measure whether existing models fail. |
| Small `*.mp4`, not `*_large.mp4` | Dataset labels are in small-video frame numbers; 4K files are 1–2.4 GB and slower. |
| Save every person in stage 1, select in stage 2 | Model runs once; you can change the selection without re-running it. |
| Follow one track ID, mark `lost` | Prevents silently measuring a bystander or a duplicate box. |
| ByteTrack defaults, copied into `configs/` | Change settings only after a measured problem; the copy records what was used. |
| Switched nano → small | Same excerpt: nano missed 16 visible frames, small missed 2, for ~2 ms/frame more. |
| GPU (MPS) vs CPU test | MPS has had pose bugs; identical output let us trust the faster GPU. |
| Decoder timestamps, not frame ÷ 25 | Durations of pauses must be in real seconds. |

## What you can present (only things verified so far)

Say: "We ran pretrained YOLO11s-pose (Ultralytics 8.4.191) with ByteTrack on a
20-second excerpt of one climber from the public The Way Up dataset (CC BY 4.0).
The selected climber was tracked in 498 of 500 frames at about 2× real time on a
MacBook M3 Pro. The smaller nano model missed 16 frames where the climber was
clearly visible, so we chose the small model. Gaps are marked as lost rather than
filled in."

Show: `overlay_selected.mp4`, the n vs s comparison table in `PROGRESS_LOG.md`, a
nano failure frame (`outputs/inspect/n_overlay_frames.jpg`, bottom-left), and the
check script output.

Do not say yet: anything about pauses, joint accuracy (we have no joint ground truth),
other climbers, or the drone.
