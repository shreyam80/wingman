"""Hybrid follower: a single-object tracker follows the selected climber every check;
the person detector confirms it every few checks.

Why: the person detector alone sometimes misses a climber for a second or two (e.g. seen
from behind), which made the drone stop. A single-object tracker follows "whatever looks
like the box we selected" and doesn't need to recognise a person, so it bridges those
gaps - but on its own it can drift onto the wall. The detector catches drift.

States (same meaning as TargetLock): LOCKED, LOST (drone holds), NEEDS_PILOT.
  LOCKED: tracker box used. If a detected person overlaps it -> confirmed, and the
          tracker is re-centred on that person (fixes drift). If no person has confirmed
          it for max_unconfirmed_s, or the tracker's own score drops -> LOST.
  LOST:   hold. Within reacquire_window_s, re-lock only onto a person close to the last
          confirmed position with a similar size. Otherwise -> NEEDS_PILOT.
Never switches to a person far from the last confirmed position.
"""
import cv2
from ultralytics import YOLO

from wingman.follow.config import PROJECT


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _xyxy(xywh):
    x, y, w, h = xywh
    return (x, y, x + w, y + h)


def _xywh(xyxy):
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    return (x1, y1, x2 - x1, y2 - y1)


class PersonDetector:
    """Person boxes only (no tracking ids) - the tracker handles identity."""

    def __init__(self, cfg, device):
        d = cfg.detector
        self.model = YOLO(str(PROJECT / "models" / d.model))
        self.kw = dict(imgsz=d.imgsz, conf=d.conf, classes=[0], device=device, verbose=False)

    def people(self, image):
        r = self.model.predict(image, **self.kw)[0]
        return [tuple(b) for b in r.boxes.xyxy.tolist()] if r.boxes is not None else []


class HybridFollower:
    def __init__(self, cfg, img_w, img_h, device, tap_box):
        h, c = cfg.hybrid, cfg.lock
        self.detector = PersonDetector(cfg, device)
        params = cv2.TrackerVit_Params()
        params.net = str(PROJECT / h.tracker_model)
        self._params = params
        self.min_score, self.every = h.min_tracker_score, h.detect_every_n
        self.agree, self.max_unconf = h.agree_iou, h.max_unconfirmed_s
        self.window = c.reacquire_window_s
        self.re_r, self.size_ratio = c.reacquire_radius_frac * img_h, c.reacquire_size_ratio
        self.tap_box = tap_box
        self.state, self.box, self.track_id = "WAITING", None, 1
        self.confirmed_box, self.last_confirm, self.lost_since = None, None, None
        self.people, self._n, self.tracker = [], 0, None

    def _start_tracker(self, image, box):
        self.tracker = cv2.TrackerVit_create(self._params)
        self.tracker.init(image, _xywh(box))

    def _confirm(self, t, image, person):
        self.box = self.confirmed_box = person
        self.last_confirm = t
        self._start_tracker(image, person)  # re-centre the tracker on the real person

    def _near_last(self, b):
        lb = self.confirmed_box
        lh, h = lb[3] - lb[1], b[3] - b[1]
        dist = abs((b[0] + b[2]) / 2 - (lb[0] + lb[2]) / 2) + abs((b[1] + b[3]) / 2 - (lb[1] + lb[3]) / 2)
        return dist <= self.re_r and 1 / self.size_ratio <= h / lh <= self.size_ratio

    def update(self, t, image):
        """Returns (box or None, event or None), like TargetLock.update."""
        self._n += 1
        run_detector = self.state != "LOCKED" or self._n % self.every == 0
        self.people = self.detector.people(image) if run_detector else []

        if self.state == "WAITING":  # start from the user's selection
            cand = [p for p in self.people if _iou(p, self.tap_box) >= self.agree]
            start = max(cand, key=lambda p: _iou(p, self.tap_box)) if cand else self.tap_box
            self._confirm(t, image, start)
            self.state = "LOCKED"
            return self.box, "lock"

        if self.state == "LOCKED":
            ok, xywh = self.tracker.update(image)
            score = self.tracker.getTrackingScore()
            tbox = _xyxy(xywh) if ok and score >= self.min_score else None
            if tbox is not None and run_detector:
                match = max(self.people, key=lambda p: _iou(p, tbox), default=None)
                if match is not None and _iou(match, tbox) >= self.agree:
                    self._confirm(t, image, match)
                    return self.box, None
            if tbox is not None and t - self.last_confirm <= self.max_unconf:
                self.box = tbox  # bridge: no person confirmed this time, trust the tracker briefly
                return self.box, None
            self.state, self.lost_since, self.box = "LOST", t, None
            return None, "lost"

        # LOST / NEEDS_PILOT: hold; look for the climber close to where they were confirmed
        if self.state == "LOST" and t - self.lost_since <= self.window:
            near = [p for p in self.people if self._near_last(p)]
            if near:
                self._confirm(t, image, min(near, key=lambda p: abs(p[1] - self.confirmed_box[1])))
                self.state = "LOCKED"
                return self.box, "relock"
            return None, None
        if self.state == "LOST":
            self.state = "NEEDS_PILOT"
            return None, "needs_pilot"
        return None, None
