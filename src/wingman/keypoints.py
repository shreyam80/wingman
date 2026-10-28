"""Joint names and skeleton drawing shared by every stage.

YOLO pose models output the 17 COCO keypoints in this fixed order. Keeping the
names here (instead of bare indices) lets a different model plug in later via an
adapter that maps its joints onto these names.
"""
import cv2

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pairs of joint names connected by a bone line in the overlay.
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
]

# Colors are BGR (OpenCV order). Left side orange, right side blue, centre white.
_LEFT, _RIGHT, _MID = (0, 140, 255), (255, 140, 0), (255, 255, 255)


def _side_color(name):
    if name.startswith("left"):
        return _LEFT
    if name.startswith("right"):
        return _RIGHT
    return _MID


def kp_columns():
    """Column names for one person's joints in the frame-level CSV."""
    cols = []
    for n in KEYPOINT_NAMES:
        cols += [f"{n}_x", f"{n}_y", f"{n}_conf"]
    return cols


def draw_skeleton(img, kps, min_conf, thickness=3):
    """Draw joints whose confidence >= min_conf.

    kps: dict name -> (x, y, conf). Joints below the threshold are not drawn at
    all, so the overlay only shows what the analysis would be allowed to use.
    """
    ok = {n: v for n, v in kps.items() if v[2] >= min_conf}
    for a, b in SKELETON:
        if a in ok and b in ok:
            pa = (int(ok[a][0]), int(ok[a][1]))
            pb = (int(ok[b][0]), int(ok[b][1]))
            color = _side_color(a) if _side_color(a) == _side_color(b) else _MID
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for n, (x, y, _) in ok.items():
        cv2.circle(img, (int(x), int(y)), thickness + 2, _side_color(n), -1, cv2.LINE_AA)
