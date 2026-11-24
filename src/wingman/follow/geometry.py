"""Convert between dataset video pixels and real positions on the climbing wall.

Wall coordinates: x = metres across the wall, y = metres up the wall (0 = the lowest
row of reference bolt holes), as in each video's <route>_bolt_holes.json.

Dataset quirk (checked 2026-11-23): in most 4-point files the two TOP image points are
listed in the opposite left/right order to their wall coordinates, which folds the
provided homography and gives nonsense (climbers hundreds of metres tall). The one
360-point grid file (p2a green) is consistent. We therefore pair points row by row,
both sorted left to right, and fit our own homography. With this, climber boxes come
out ~1.4-1.5 m tall from floor to top on the videos checked - plausible, not verified.
"""
import json

import cv2
import numpy as np


class WallMap:
    def __init__(self, bolt_json):
        b = json.load(open(bolt_json))
        img = np.array(b["img_coords"], float)
        wall = np.array(b["world_coords"], float)[:, :2] / 100.0  # cm -> m
        src, dst = [], []
        for row_y in np.unique(wall[:, 1]):  # one row of bolts at a time
            rows = wall[:, 1] == row_y
            src.append(img[rows][np.argsort(img[rows][:, 0])])
            dst.append(wall[rows][np.argsort(wall[rows][:, 0])])
        self.img_to_wall, _ = cv2.findHomography(np.vstack(src), np.vstack(dst))
        self.wall_to_img = np.linalg.inv(self.img_to_wall)

    def to_wall(self, pts_px):
        p = np.asarray(pts_px, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.img_to_wall).reshape(-1, 2)

    def box_to_wall(self, x1, y1, x2, y2):
        """Pixel box -> wall box (left, bottom, right, top) in metres."""
        w = self.to_wall([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
        return w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()


class View:
    """The patch of wall a camera sees: centre (cx, cy) and size (w, h) in wall metres,
    shown as an image of W x H pixels. Image row 0 is the TOP of the view."""

    def __init__(self, cx, cy, w, h, W, H):
        self.cx, self.cy, self.w, self.h, self.W, self.H = cx, cy, w, h, W, H

    def wall_to_px(self, x, y):
        u = (x - (self.cx - self.w / 2)) * self.W / self.w
        v = ((self.cy + self.h / 2) - y) * self.H / self.h
        return u, v

    def px_to_wall_matrix(self):
        """3x3 matrix taking view pixels (u, v, 1) to wall metres (x, y, 1)."""
        return np.array([[self.w / self.W, 0, self.cx - self.w / 2],
                         [0, -self.h / self.H, self.cy + self.h / 2],
                         [0, 0, 1]])

    def box_px(self, left, bottom, right, top):
        u1, v1 = self.wall_to_px(left, top)
        u2, v2 = self.wall_to_px(right, bottom)
        return u1, v1, u2, v2
