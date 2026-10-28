# Wingman – laptop vision prototype

Offline pipeline: climbing video → pretrained pose model + ByteTrack → one selected
climber → saved joint measurements → (next) torso-pause bookmarks.
No model is trained here; we run official pretrained weights.

## Setup (macOS, Apple Silicon; tested on M3 Pro, 18 GB)

```bash
cd ~/wingman
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python -e .
```

The dataset is The Way Up (Zenodo 15196867, CC BY 4.0 – credit the authors). It is
expected, unmodified, at `~/Downloads/wingman dataset`. Use the small `route.mp4`
files (720×1280, 25 FPS); annotations are in their frame numbers. Ignore `*_large.mp4`.

## Run

Commands are run from `models/` so model weights download/load there.

```bash
# 0. Inventory every small video (hash, size, fps, participant, license)
.venv/bin/python -m wingman.manifest --dataset ~/Downloads/"wingman dataset"

# 1. Pose + tracking on an excerpt (all people saved)
cd models && ../.venv/bin/python -m wingman.run_pose \
  --video ~/Downloads/"wingman dataset/p1/orange.mp4" --start 16 --end 36 \
  --model yolo11s-pose.pt --tracker ../configs/bytetrack.yaml \
  --out ../outputs/p1_orange_16-36_s && cd ..

# 2a. Inspect tracks; writes a preview JPEG with IDs drawn
.venv/bin/python -m wingman.select_climber --run outputs/p1_orange_16-36_s --list --at 20
# 2b. Select the climber (by ID, or by a pixel inside their box at a time)
.venv/bin/python -m wingman.select_climber --run outputs/p1_orange_16-36_s --track-id 1 --selected-by YOUR_INITIALS

# 3. Automatic consistency checks (PASS / WARN / FAIL)
.venv/bin/python -m wingman.check_run --run outputs/p1_orange_16-36_s
```

Hands-on guide with explanations: `docs/WALKTHROUGH.md`. Plain-language explanation of every file: `docs/FILE_GUIDE.md`.

## Output files (per run folder)

| File | Contents |
| --- | --- |
| `detections.csv` | Every tracked person, every frame: `track_id`, box `x1 y1 x2 y2` (px), `box_conf`, and for each of 17 COCO joints `<joint>_x`, `<joint>_y` (px, origin top-left) and `<joint>_conf` (0–1). |
| `frames.csv` | One row per processed frame: decoder timestamp `t_s`, people tracked, inference ms. |
| `meta.json` | Video, excerpt, model file, imgsz, device, tracker settings, package versions, runtime. |
| `overlay_all.mp4` | Inspection video: all tracks, IDs, skeletons (joints with conf ≥ 0.5). |
| `selected.csv` | One row per frame for the chosen climber: `status` (`tracked` / `lost` / `before_selection`) plus box and joints; joints empty when lost. |
| `selection.json` | Who selected, how, ID chain incl. human-confirmed re-links, lost intervals, unconfirmed link suggestions. |
| `overlay_selected.mp4` | Only the selected climber; "LOST" banner when not found. |

Joint names are in `src/wingman/keypoints.py`. Joint "confidence" is the model's own
score, not measured accuracy.

## Licensing note
Ultralytics code/weights are AGPL-3.0 (enterprise license available). Fine for this
academic prototype; review before any proprietary product.
