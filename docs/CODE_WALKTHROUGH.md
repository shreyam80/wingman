# Code walkthrough – what we did, and what every line is for

Written 2026-10-28. Line numbers match the code in this commit.
Open any file mentioned here in VS Code (or click it in the Claude app) and follow along.

---

## Part 1 – Every step taken so far, with evidence

| # | Step | What happened | Evidence you can open |
|---|------|---------------|-----------------------|
| 1 | Checked the laptop | M3 Pro, 18 GB RAM, 30 GB disk free, Apple GPU usable | `docs/PROGRESS_LOG.md` (top) |
| 2 | Checked the dataset | 22 videos, 10 people, license CC BY 4.0, labels are hold-contacts (not joints) | `data/manifest.csv` |
| 3 | Built the environment | Private Python 3.11 + exact package versions | `requirements.lock.txt` |
| 4 | Wrote the pipeline | 6 Python files, explained below | `src/wingman/` |
| 5 | First run, 20 s, nano model | Climber found in 484/500 frames; missed visible poses | `outputs/p1_orange_16-36_n/` |
| 6 | Same 20 s, small model | 498/500 frames → switched to small model | `outputs/p1_orange_16-36_s/` |
| 7 | GPU vs CPU test | Identical joints → GPU is safe to use | `PROGRESS_LOG.md` |
| 8 | Wrote the checker, tested it | 15 checks pass; planted fake mistake is caught | run `check_run.py` |
| 9 | Git + GitHub | Private repo shreyam80/wingman | github.com/shreyam80/wingman |
| 10 | Full 88 s video, small model, 640 px | Climber got **5 different IDs**; only 72% coverage at the top | `outputs/p1_orange_full_s/` |
| 11 | Full video, 960 px | **One ID the whole climb**, 97.6% coverage; bystanders now detected and correctly ignored | `outputs/p1_orange_full_s960/` |

**Current best setting:** `yolo11s-pose.pt`, `--imgsz 960`, ByteTrack defaults.
**Not done yet:** pause detector, manual labels, evaluation.

---

## Part 2 – The pipeline in one picture

```
 ~/Downloads/wingman dataset/p1/orange.mp4     (original, never modified)
            │
            ▼
 ┌──────────────────────┐   detections.csv  ← every person, every frame
 │  run_pose.py          │─► frames.csv      ← one line per frame (time, speed)
 │  (AI model + tracker) │   meta.json       ← all settings used
 └──────────────────────┘   overlay_all.mp4 ← video with everyone drawn
            │
            ▼
 ┌──────────────────────┐   selected.csv         ← only the climber, per frame
 │  select_climber.py    │─► selection.json       ← who/how chosen, lost periods
 │  (human picks person) │   overlay_selected.mp4 ← video with only the climber
 └──────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │  check_run.py         │─► PASS / WARN / FAIL printed on screen
 └──────────────────────┘

 Helpers used by all three: keypoints.py (joint names, drawing), video_io.py (video files)
 Separate: manifest.py (inventory of all 22 dataset videos)
```

---

## Part 3 – File by file, line by line

### `src/wingman/run_pose.py` (141 lines) – the most important file

**Lines 1–15 – description.** The text between `"""` marks is a note for humans; Python
ignores it. It says what the file makes and how to run it.

**Lines 17–32 – imports.** "Borrow tools from libraries":
- `cv2` = OpenCV, reads/writes video and draws on pictures
- `numpy` = fast number lists
- `torch`, `ultralytics`, `YOLO` = the AI model
- `yaml` = reads the tracker settings file
- `from wingman.keypoints import ...` = our own joint list and drawing function

**Line 34 – `TRACK_COLORS`.** Five colours; each track ID gets one so different people
look different in the overlay.

**Lines 37–50 – the settings you can type on the command line.** `argparse` turns
`--imgsz 960` into `args.imgsz = 960`. Each has a default:

| Option | Default | Meaning |
|---|---|---|
| `--video` | (required) | which video |
| `--start`, `--end` | 0, end | which seconds to process |
| `--out` | (required) | folder for results |
| `--model` | yolo11n-pose.pt | which weights file |
| `--tracker` | configs/bytetrack.yaml | tracker settings |
| `--imgsz` | 640 | size the model shrinks each picture to (we now use 960) |
| `--conf` | 0.25 | ignore person boxes the model is less than 25% sure about |
| `--device` | mps if available | Apple GPU, else normal processor |
| `--draw-min-conf` | 0.5 | only draw joints ≥ 50% confidence |

**Lines 52–57 – work out which frames to process.**
```python
start_f = int(round(args.start * info["fps"]))      # 16 s × 25 fps = frame 400
```

**Line 59 – `model = YOLO(args.model)`.** Loads the pretrained weights. If the file
isn't in the folder, Ultralytics downloads the official one (that's why we run from `models/`).

**Lines 60–62 – open the video and skip to the start.**
```python
for _ in range(start_f):  # sequential skip
    cap.grab()
```
`grab()` moves forward one frame without decoding the picture. We do it one at a time
because "jump to frame 400" in compressed video sometimes lands on the wrong frame.

**Line 63** – open the output video file (`overlay_all.mp4`).

**Lines 65–70 – create the two CSV files and write their header rows.**
`kp_columns()` (from keypoints.py) produces the 51 joint column names:
`nose_x, nose_y, nose_conf, left_eye_x, …, right_ankle_conf` (17 joints × 3).

**Line 74 – the main loop: `for frame_idx in range(start_f, end_f):`.**
Everything indented below runs once per frame – 2213 times for the full video.

**Lines 75–78** – read the next frame. `ok` is False if the video ended early.

**Line 80 – the timestamp.**
```python
t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
```
Asks the video file itself "what time is this frame?" in milliseconds, converts to seconds.

**Lines 82–85 – THE AI STEP.**
```python
r = model.track(frame, persist=True, tracker=args.tracker, imgsz=args.imgsz,
                conf=args.conf, device=args.device, verbose=False)[0]
```
- `model.track` = find people + joints, then match them to people from earlier frames.
- `persist=True` = "remember previous frames" – this is what makes IDs stay the same.
- Lines 82 and 85 time this call → `infer_ms` (milliseconds the model took).

**Lines 87–91 – unpack the answer into plain arrays.**
- `boxes` = one row per person: `x1, y1, x2, y2` (top-left and bottom-right corners)
- `confs` = how sure the model is that each box is a person
- `ids` = track IDs (`-1` if the tracker hasn't assigned one)
- `kps` = shape (people, 17, 3): for each person, 17 joints × (x, y, confidence)
- `.cpu().numpy()` moves numbers off the GPU into normal memory.

**Lines 93–102 – for each person: save a row, then draw them.**
Line 95 writes the CSV row. Line 97 formats numbers: confidences with 3 decimals,
pixel positions with 1 decimal. Lines 98–102 draw the box, "ID 1 0.87" label and skeleton.

**Line 103** – write one line to `frames.csv` (even if 0 people found – so gaps are visible).

**Lines 104–107** – write the time and model name onto the picture, add it to the video.

**Lines 109–111** – print progress every 100 frames.

**Lines 113–117** – close all files properly (otherwise the video can be unplayable).

**Lines 119–135 – `meta.json`.** A record of *everything* about this run: video
properties, frames processed, model, imgsz, device, tracker settings, package versions,
computer, runtime, frames per second, speed relative to real time.

**What one frame becomes – a real example** (frame 650, 26.0 s, full 960 px run):

`detections.csv` (some columns):
```
frame_idx  t_s  track_id  x1     y1     x2     y2     box_conf  left_wrist_x  left_wrist_y  left_wrist_conf
650        26.0 1         238.4  495.3  415.0  806.9  0.874     313.2         584.8         0.736
650        26.0 3         651.3  808.2  691.2  938.0  0.265     672.7         853.1         0.247
```
Two people: track 1 is the climber (box 177 px wide × 312 px tall, 87% sure). Track 3 is
a small, low-confidence box at bottom right – a bystander.

`frames.csv`: `650, 26.0, 2, 18.4` → frame 650, 26 s, 2 people, model took 18.4 ms.

---

### `src/wingman/select_climber.py` (229 lines) – follow one person

**Lines 36–41 `load`** – reads `meta.json`, `detections.csv`, `frames.csv` into tables.

**Lines 43–50 `read_frame`** – gets one picture from the original video (for the preview).

**Lines 52–53 `nearest_frame`** – "which processed frame is closest to time t?"

**Lines 56–77 `list_tracks`** (the `--list` mode) – groups detections by track ID and
prints a summary per ID: first/last second seen, number of frames, typical box height,
typical position, typical confidence. Then saves `preview_<t>s.jpg` with IDs drawn.
**Purpose:** a human can see which ID is the climber *before* choosing.

**Lines 80–99 `build_selection` – the core rule.** For every processed frame:
```python
active = [tid for tid, t0 in chain if fr.t_s >= t0 - 1e-6]   # line 85
tid = active[-1] if active else None                          # line 86
```
`chain` is the list of IDs that a human said are the climber, each with a start time
(normally just one: `[(1, 0.0)]`). Line 85–86 picks the right one for this moment.
```python
hit = g[g.track_id == tid]                                    # line 88
```
Looks in this frame's detections for **exactly that ID**. Not the biggest box, not the
nearest box – only that ID. Then (lines 90–96):
- no ID chosen yet → `before_selection`
- ID found → `tracked`, copy its box and 51 joint values
- ID not found → `lost`, joints left **empty**

Real example from the full run, near the top of the wall:
```
frame_idx  t_s    status   selected_track_id  left_hip_x
1944       77.76  tracked  1                  319.2
1945       77.80  lost     1                  (empty)
```

**Lines 102–114 `intervals`** – turns the per-frame list into time ranges, e.g. lost from
77.80 to 78.48 s.

**Lines 117–141 `suggest_links`** – when the climber is lost, lists other IDs that show
up around then, and how much their box overlaps the climber's last box (0 = no overlap,
1 = identical). **It only suggests – it never adds them.** (Fixed today: it now also
finds a duplicate box that began *before* the loss and takes over.)

**Lines 144–164 `render`** – re-reads the original video and draws only the climber
(green) or a red "SELECTED CLIMBER LOST" banner → `overlay_selected.mp4`.

**Lines 168–177 – command-line options:** `--list`, `--at`, `--track-id`, `--point`,
`--link NEW_ID@TIME` (human-confirmed ID change), `--selected-by`.

**Lines 187–195 – choose by clicking-style point.** Finds boxes that contain the pixel
(x, y) at that time; if several, picks the smallest (most specific).

**Lines 201–206 – build the chain** from the chosen ID plus any `--link`s, sorted by time.

**Lines 208–225 – save everything.** `selected.csv`, then `selection.json` containing:
who chose, how, the chain, status counts, tracked fraction, lost intervals,
unconfirmed suggestions. Then renders the video.

---

### `src/wingman/check_run.py` (150 lines) – the inspector

**Lines 25–34** – `report` prints one result line; `check` decides PASS or FAIL/WARN.
**Lines 36–41** – reads a video's frame count, speed, and size.
**Lines 44–148 `main`** – the checks, in order:

| Section | Checks | Why |
|---|---|---|
| 1 | needed files exist | can't check what isn't there |
| 2 | one row per frame; times always increase; steps exactly 0.04 s | pause durations depend on correct time |
| 3 | confidences 0–1; boxes inside picture; no ID twice in a frame | catches corrupted or misread output |
| 4 | selected rows align with frames; tracked rows use only chosen IDs; lost rows have no joints; no sudden torso jumps | the "never another person, never invented" rules |
| 5 | output videos have same frame count, fps, size as input; their pictures match the original | proves the overlay is the real video, frame-aligned |

Last lines: prints totals and exits with code 1 if anything FAILed.

---

### `src/wingman/keypoints.py` (61 lines)

- **Lines 9–14 `KEYPOINT_NAMES`** – the 17 joints in the model's order.
- **Lines 17–25 `SKELETON`** – which pairs to connect with a line.
- **Line 28** – colours: left orange, right blue, middle white.
- **Lines 39–44 `kp_columns`** – makes the 51 CSV column names.
- **Lines 47–61 `draw_skeleton`** – keeps only joints with confidence ≥ the cut-off,
  draws lines between connected pairs, then dots on joints.

### `src/wingman/video_io.py` (33 lines)

- **Lines 5–17 `video_info`** – width, height, fps, frame count, duration.
- **Lines 20–27 `open_writer`** – tries H.264 (`avc1`, plays on Mac), else `mp4v`.
- **Lines 30–33 `put_label`** – draws text twice: thick black, then thin colour (outline).

### `src/wingman/manifest.py` (65 lines)

- **Lines 17–18** – source and license text written into every row.
- **Lines 21–26 `sha256`** – reads a file in 1 MB pieces and computes its fingerprint.
- **Lines 29–61 `main`** – for each `p*` folder and each route: measure the small video,
  fingerprint it, record participant group (`p2a`/`p2b` → `p2`), write `data/manifest.csv`.

### `configs/bytetrack.yaml`
Tracker settings. Key ones: `track_high_thresh 0.25` (how sure a box must be to continue a
track), `track_buffer 30` (frames to remember a vanished person = 1.2 s),
`match_thresh 0.8` (how strict matching boxes between frames is).

---

## Part 4 – Technical progress

| Milestone | Planned | Implemented | Run | Verified |
|---|---|---|---|---|
| 1 Environment + sample | ✅ | ✅ | ✅ | ✅ manifest, license, video properties |
| 2 Pose + tracking overlay | ✅ | ✅ | ✅ 20 s + full video | ✅ checks + visual inspection |
| 3 Saved measurements | ✅ | ✅ | ✅ | ✅ `selected.csv` checked |
| 4 Pause bookmarks | ✅ | ❌ next | ❌ | ❌ |
| 5 Evaluation + labels | ✅ | ❌ | ❌ | ❌ |

**Known failures (honest list):**
1. At 640 px the climber was missed for > 1.2 s near the top, so the tracker issued new IDs (5 IDs in one climb). Fixed by 960 px.
2. Duplicate half-body boxes on the climber (e.g. 28 s). Harmless because we follow one ID.
3. Near the top, left/right can be swapped (frame 2150, 86 s: orange "left" joints drawn on the climber's right side). Matters later for "which hand" features.
4. Still 2.4% of frames lost, longest 0.7 s (77.8–78.5 s).

---

## Part 5 – How to see it yourself

```bash
cd ~/"CIS Senior Design Docs/wingman"
source .venv/bin/activate
```
1. **Watch the result:** `open outputs/p1_orange_full_s960/overlay_selected.mp4` – scrub
   to 77.8 s to see the red LOST banner.
2. **See bystanders being ignored:** `open outputs/p1_orange_full_s960/overlay_all.mp4` –
   bottom right, small boxes with other IDs.
3. **Read the numbers:** open `outputs/p1_orange_full_s960/selected.csv` in Numbers/Excel.
4. **Read the settings used:** open `outputs/p1_orange_full_s960/meta.json`.
5. **Run the checks:** `python -m wingman.check_run --run outputs/p1_orange_full_s960`
6. **Read the code with this document side by side** in VS Code: `code .`
7. **Make a change and watch the effect:** rerun `run_pose` with `--imgsz 640` into a
   new folder and compare `selection.json` – you'll reproduce failure 1 yourself.
