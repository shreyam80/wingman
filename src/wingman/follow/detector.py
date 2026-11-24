"""Find people in each picture and keep the lock on the chosen climber.

TargetLock is a small state machine:
  WAITING -> LOCKED   first person near the image centre (pilot centred them, then locked)
  LOCKED  -> LOST     locked track id not seen this time
  LOST    -> LOCKED   same id comes back, or (within reacquire_window_s) a person appears
                      close to where the climber was last seen, with a similar size
  LOST stays LOST     after the window: the drone keeps holding and the pilot must re-lock.
It never switches to whoever happens to be nearest or largest.
"""
from ultralytics import YOLO

from wingman.follow.config import PROJECT


class PersonTracker:
    def __init__(self, cfg, device):
        d = cfg.detector
        self.model = YOLO(str(PROJECT / "models" / d.model))
        self.kw = dict(persist=True, tracker=str(PROJECT / d.tracker), imgsz=d.imgsz,
                       conf=d.conf, classes=[0], device=device, verbose=False)  # class 0 = person

    def people(self, image):
        """List of (track_id, (x1, y1, x2, y2)) in image pixels; untracked boxes get id -1."""
        r = self.model.track(image, **self.kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        return list(zip(ids, [tuple(b) for b in r.boxes.xyxy.tolist()]))


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class TargetLock:
    def __init__(self, cfg, img_w, img_h, tap_box=None):
        c = cfg.lock
        self.tap_box = tap_box
        self.lock_r = c.lock_radius_frac * img_h
        self.window, self.re_r, self.size_ratio = (c.reacquire_window_s,
                                                   c.reacquire_radius_frac * img_h,
                                                   c.reacquire_size_ratio)
        self.cx, self.cy = img_w / 2, img_h / 2
        self.state, self.track_id, self.box, self.lost_since = "WAITING", None, None, None

    def update(self, t, people):
        """Returns (box or None, event or None). event in {"lock", "lost", "relock",
        "needs_pilot"} marks state changes for the log."""
        tracked = [(i, b) for i, b in people if i >= 0]
        if self.state == "WAITING" and self.tap_box is not None:  # user selected the climber
            hits = [(_iou(b, self.tap_box), i, b) for i, b in tracked]
            hits = [h for h in hits if h[0] >= 0.3]
            if not hits:
                return None, None
            _, self.track_id, self.box = max(hits)
            self.state = "LOCKED"
            return self.box, "lock"
        if self.state == "WAITING":
            near = [(abs(_centre(b)[0] - self.cx) + abs(_centre(b)[1] - self.cy), i, b) for i, b in tracked]
            near = [n for n in near if n[0] <= self.lock_r]
            if not near:
                return None, None
            _, self.track_id, self.box = min(near)
            self.state = "LOCKED"
            return self.box, "lock"

        same = [b for i, b in tracked if i == self.track_id]
        if same:
            event = "relock" if self.state in ("LOST", "NEEDS_PILOT") else None
            self.state, self.box = "LOCKED", same[0]
            return self.box, event

        if self.state == "LOCKED":
            self.state, self.lost_since = "LOST", t
            return None, "lost"

        # LOST: look for the climber reappearing near the last known box (new id)
        if t - self.lost_since <= self.window:
            lx, ly = _centre(self.box)
            lh = self.box[3] - self.box[1]
            for i, b in tracked:
                bx, by = _centre(b)
                h = b[3] - b[1]
                if (abs(bx - lx) + abs(by - ly) <= self.re_r
                        and 1 / self.size_ratio <= h / lh <= self.size_ratio):
                    self.state, self.track_id, self.box = "LOCKED", i, b
                    return b, "relock"
            return None, None
        if self.state != "NEEDS_PILOT":
            self.state = "NEEDS_PILOT"
            return None, "needs_pilot"
        return None, None


class DetectorFollower:
    """Baseline follower: person detector + ByteTrack ids every check, plus TargetLock.
    Same interface as HybridFollower: update(t, image) -> (box, event); .people; .state."""

    def __init__(self, cfg, img_w, img_h, device, tap_box=None):
        self.tracker = PersonTracker(cfg, device)
        self.lock = TargetLock(cfg, img_w, img_h, tap_box)
        self.people = []

    @property
    def state(self):
        return self.lock.state

    @property
    def track_id(self):
        return self.lock.track_id

    def update(self, t, image):
        found = self.tracker.people(image)
        self.people = [b for _, b in found]
        return self.lock.update(t, found)
