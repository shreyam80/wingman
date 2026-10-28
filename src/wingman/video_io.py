"""Small helpers for reading video properties and writing a playable MP4."""
import cv2


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] else None
    return info


def open_writer(path, fps, width, height):
    """Prefer H.264 ('avc1', plays in QuickTime); fall back to MPEG-4 ('mp4v')."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if w.isOpened():
            return w, codec
        w.release()
    raise RuntimeError(f"Could not open a video writer for {path}")


def put_label(img, text, org, scale=0.8, color=(255, 255, 255)):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
