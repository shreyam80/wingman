"""Where pictures come from. The follow loop only calls camera.view_size() and
camera.picture(...), so the simulated camera can later be swapped for the real one."""
import math

import cv2
import numpy as np

from wingman.follow.geometry import View


def image_size(cfg):
    """(width, height) in pixels of the mounted camera, before any sim scaling."""
    w, h = cfg.camera.sensor_width_px, cfg.camera.sensor_height_px
    return (min(w, h), max(w, h)) if cfg.camera.orientation == "portrait" else (max(w, h), min(w, h))


class SimCamera:
    """What a drone camera would see of a stabilized dataset video.

    Assumes the camera faces the wall squarely from `distance_to_wall_m` (tilt 0) and
    treats the climber as lying on the wall plane. The picture is cut out of the
    dataset frame using the WallMap, so its sharpness is limited by the original video
    (about 2 pixels per cm), not by the real camera's sensor.
    """

    def __init__(self, cfg, wall_map):
        if cfg.camera.tilt_deg != 0:
            raise NotImplementedError("Simulated camera assumes tilt_deg = 0")
        W, H = image_size(cfg)
        self.W, self.H = int(W * cfg.sim.render_scale), int(H * cfg.sim.render_scale)
        d = cfg.flight.distance_to_wall_m
        self.view_h = 2 * d * math.tan(math.radians(cfg.camera.fov_vertical_deg) / 2)
        self.view_w = self.view_h * self.W / self.H  # square pixels
        self.wall_map = wall_map

    def view(self, cx, cy):
        return View(cx, cy, self.view_w, self.view_h, self.W, self.H)

    def picture(self, dataset_frame, cx, cy):
        """Render the drone's view, aimed at wall point (cx, cy) metres."""
        view_px_to_img = self.wall_map.wall_to_img @ self.view(cx, cy).px_to_wall_matrix()
        img_to_view = np.linalg.inv(view_px_to_img)
        return cv2.warpPerspective(dataset_frame, img_to_view, (self.W, self.H))


class LiveCamera:
    """The real camera on the Jetson. NOT TESTED - no camera yet.

    The Arducam B0429 connects by MIPI ribbon cable; on a Jetson that usually means
    OpenCV reading a GStreamer pipeline provided by Arducam's driver. Fill in
    `pipeline` once the driver is installed (see Arducam's Jetson instructions).
    """

    def __init__(self, pipeline=None, device=0):
        self.cap = (cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) if pipeline
                    else cv2.VideoCapture(device))
        if not self.cap.isOpened():
            raise RuntimeError("Camera did not open - check cable, driver and pipeline")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None
