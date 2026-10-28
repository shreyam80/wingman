# File guide – what every file does, in plain words

## The big picture first

A video is just a stack of pictures (25 per second here). Our program:

1. **Looks at each picture** and asks a ready-made AI model: "where are the people,
   and where are their joints?" (`run_pose.py`)
2. **Picks one person** – the climber – and follows only them. (`select_climber.py`)
3. **Double-checks the results** for mistakes. (`check_run.py`)
4. *(next)* Finds moments where the climber's body stayed still – "pauses".

Each step saves its answers to files, and the next step reads those files. This is
on purpose: if step 2 needs changing, we don't redo the slow step 1.

```
video ──► run_pose ──► detections.csv ──► select_climber ──► selected.csv ──► (pause detector)
                                                                   │
                                                              check_run
```

### Three words you need

- **Model / weights** – a program that already learned what human bodies look like
  from many thousands of labelled photos. The "weights" file (`yolo11s-pose.pt`) is
  what it learned, saved as numbers. We download and *use* it; we never train it.
- **Keypoints / joints** – 17 body points the model finds: nose, eyes, ears,
  shoulders, elbows, wrists, hips, knees, ankles. Each comes with **x, y** (position
  in the picture, in pixels) and **confidence** (0 to 1: how sure the model is).
- **Track ID** – a number the tracker gives each person so the same person keeps the
  same number from picture to picture. It is a guess, not proof.

**Pixel coordinates:** (0, 0) is the *top-left* corner. x grows to the right, y grows
*downwards*. So a hand raised above the head has a **smaller** y than the head.

---

## `src/wingman/` – the code

### `keypoints.py` – the shared dictionary of joints
**What it does:** lists the 17 joint names in the exact order the model outputs them,
says which joints are joined by "bones", and has a function that draws the stick
figure on a picture.

**Design decisions**
- *Names instead of numbers.* The model calls the left wrist "joint 9". We store it
  as `left_wrist`. If we later try a different model that numbers joints differently,
  we only translate its names – the rest of our code doesn't change.
- *Left side orange, right side blue.* Lets you check at a glance that the model
  didn't swap left and right.
- *Joints below a confidence cut-off aren't drawn.* What you see on the video is
  what the analysis is allowed to trust.

### `video_io.py` – small video helpers
**What it does:** three tiny tools: read a video's size/speed/length, create an
output video file, and write readable text on a picture.

**Design decisions**
- *Try H.264 first.* That's the video format a Mac (QuickTime) plays reliably. If
  unavailable, fall back to an older format so the program still works.
- *Text with a black outline* so labels are readable on light and dark walls.

### `manifest.py` – the inventory of our videos
**What it does:** goes through the dataset folder and writes `data/manifest.csv`:
one row per video with participant, route, size, frame rate, length, license, and a
**hash**.

**Design decisions**
- *Hash = a fingerprint of the file.* If anyone asks "which exact video produced
  this result?", the fingerprint proves it. Changing even one byte changes it.
- *Only the small videos.* The dataset's own labels are written for the small
  720×1280, 25-frames-per-second videos. The 4K versions are 1–2.4 GB each.
- *License recorded on every row* (CC BY 4.0 = we may use it, but must credit the
  authors).
- *p2a and p2b grouped together* – it is the same person filmed twice. Later, when
  we test, a person must not appear in both the "tune" and "test" piles.

### `run_pose.py` – Stage 1: the AI model looks at every frame
**What it does, step by step**
1. Opens the video and skips to the start time you asked for.
2. For every frame: hands it to the model with tracking switched on
   (`model.track(...)`). Back come boxes around people, a track ID for each, and
   17 joints per person.
3. Writes **every person** to `detections.csv`, and one line per frame to
   `frames.csv` (time, how many people, how long the model took).
4. Draws all boxes/IDs/skeletons onto a copy of the frame → `overlay_all.mp4`.
5. At the end writes `meta.json`: which model, which settings, package versions,
   computer, total time.

**Design decisions**
- *Save everyone, choose later.* The model is the slow part. Saving all people means
  choosing the climber (or fixing a wrong choice) never needs the model again.
- *Use the video's own timestamp*, not "frame number ÷ 25". Pause lengths must be
  in real seconds, and some videos don't have perfectly even frames.
- *Skip frames by reading them, not by jumping.* Jumping in a compressed video can
  land on the wrong frame; reading one by one is exact.
- *Record runtime per frame.* So we can honestly say how fast it runs on this laptop.
- *Use the Apple graphics chip (`mps`)* when available – faster. We tested that it
  gives exactly the same joints as the normal processor.
- *Settings are command-line options* (`--model`, `--imgsz`, `--conf` …) so we can
  compare choices without editing code, and every choice is saved in `meta.json`.

### `select_climber.py` – Stage 2: follow only the climber
**What it does**
- `--list` mode: prints every track ID (when it appears, where, how big) and saves a
  picture with the IDs drawn, so a human can see which number is the climber.
- Select mode: you give the climber's ID (or click-style: a pixel point + time). It
  then goes frame by frame:
  - climber's ID present → status `tracked`, copy their box and joints;
  - climber's ID missing → status `lost`, joints left **empty**.
- Writes `selected.csv` (one row per frame), `selection.json` (who chose, how,
  lost periods), and `overlay_selected.mp4` (only the climber, red "LOST" banner).

**Design decisions**
- *A human picks the climber.* Not "biggest person" or "most central person" –
  those rules can grab a bystander or the duplicate boxes we actually saw.
- *Never switch people on its own.* If the tracker loses the climber and later gives
  them a new number, the program only **suggests** it. A person confirms with
  `--link`, and that decision is written down.
- *Lost means empty, not guessed.* Filling gaps would make a missing climber look
  perfectly still – exactly what a pause detector would wrongly call a "pause".
- *Write down who chose* (`--selected-by`) so decisions are traceable.

### `check_run.py` – the automatic inspector
**What it does:** reads a run's files and prints PASS / WARN / FAIL for 15 checks:
- every frame is logged once, times always go forward in equal 0.04 s steps;
- confidences are between 0 and 1, boxes are inside the picture;
- a track ID never appears twice in the same frame;
- every `tracked` row belongs to the chosen person (**never another person**);
- `lost` rows have no joints (**no invented joints**);
- the climber's torso never jumps suddenly (a hint of a person switch);
- output videos have the same frame count, speed and size as the input;
- output video pictures match the original (so the drawings sit on the right frame).

**Design decisions**
- *WARN vs FAIL.* FAIL = definitely broken. WARN = suspicious, a human should look.
- *We tested the tester:* we planted a fake person-switch and confirmed it FAILs.
- *What it can't check:* whether a joint is on the right body part. Only eyes can.

### `__init__.py`
Empty. It just tells Python "this folder is a package called `wingman`", which lets
us run `python -m wingman.run_pose`.

---

## `configs/bytetrack.yaml` – tracker settings
The tracker's settings, copied unchanged from Ultralytics' defaults. The one to
know: `track_buffer: 30` = remember a vanished person for 30 frames (1.2 s) before
giving up on their ID.

**Decision:** keep defaults until we *measure* a problem; keep a copy in our project
so every run records exactly what was used.

## `data/manifest.csv`
The inventory made by `manifest.py`: 22 videos, 10 participants.

## `outputs/<run name>/` – results of one run
Folder name = video, time range, model letter (`n` = nano, `s` = small).
`p1_orange_16-36_s` = participant 1, orange route, 16–36 s, small model.
The CSV/JSON files are in git; the `.mp4` and `.jpg` are not (regenerate them by
rerunning, or share via drive).

## `labels/`
Empty for now. Will hold **human** labels of pauses, kept separate from the
computer's predictions so we can compare them fairly.

## `models/`
Where model weights download to. Not stored in git (downloaded automatically).

## `docs/`
- `PROGRESS_LOG.md` – diary: what was planned vs built vs run vs verified.
- `WALKTHROUGH.md` – step-by-step for running and testing it yourself.
- `FILE_GUIDE.md` – this file.

## Top-level files
- `README.md` – setup and run commands, and what each output column means.
- `pyproject.toml` – tells Python this is a package named `wingman` and which
  libraries it needs.
- `requirements.lock.txt` – the *exact* versions of all 44 installed libraries, so
  a teammate's computer gets the identical setup.
- `.gitignore` – list of things git should not store (environment, weights, videos).

## Libraries we use, and why
| Library | Job |
| --- | --- |
| ultralytics | Runs the YOLO pose model and the ByteTrack tracker |
| torch | The maths engine underneath the model (uses the Mac graphics chip) |
| opencv-python (`cv2`) | Read and write video, draw on pictures |
| pandas | Tables (CSV files) – filtering, counting, grouping |
| numpy | Fast number arrays |
| lap | Helper the tracker needs for matching boxes between frames |
| pyyaml | Reads the tracker settings file |
