"""Pixel <-> plate-plane coordinate mapping.

The ball detector naturally works in pixel coordinates. The control loop
(Persona D's PID/LQR) wants the ball's position in the same physical
frame as the plate's tilt commands (theta_x, theta_y), typically meters
from the plate center. A homography computed from a few known
correspondences (image pixel <-> real-world point on the flat plate)
converts between the two, assuming the plate is planar and the camera
is fixed — both true here (static overhead camera, ball-on-plate).

Two ways to get the homography, both supported:
  1. Manual: click/measure the 4 plate corners once (see
     scripts/calibrate_homography.py) and pass them in with the known
     physical plate size.
  2. Automatic: if the platform-pose ArUco extra is in use, its 4
     corner markers give the same correspondences for free every frame
     (see platform_pose.py -> PixelToPlaneMapper.from_corners).
"""
from __future__ import annotations

import numpy as np
import cv2


class PixelToPlaneMapper:
    """Wraps a fixed 3x3 homography from image pixels to plate-plane coords."""

    def __init__(self, homography: np.ndarray):
        if homography.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        self.H = homography

    @classmethod
    def from_point_correspondences(
        cls, image_points: np.ndarray, plane_points: np.ndarray
    ) -> "PixelToPlaneMapper":
        """Build a mapper from >=4 (pixel <-> plane) point pairs.

        image_points, plane_points: Nx2 arrays, N >= 4.
        Uses cv2.findHomography (robust to more than 4 points / noise).
        """
        image_points = np.asarray(image_points, dtype=np.float64)
        plane_points = np.asarray(plane_points, dtype=np.float64)
        if len(image_points) < 4 or len(plane_points) < 4:
            raise ValueError("need at least 4 point correspondences")
        H, _mask = cv2.findHomography(image_points, plane_points, method=0)
        if H is None:
            raise RuntimeError("cv2.findHomography failed to converge")
        return cls(H)

    @classmethod
    def from_rectangular_plate(
        cls,
        image_corners_px: np.ndarray,
        plate_width_m: float,
        plate_height_m: float,
        corner_order: str = "TL,TR,BR,BL",
    ) -> "PixelToPlaneMapper":
        """Build a mapper assuming the plate is a centered rectangle.

        image_corners_px: 4x2 array of pixel coordinates for the plate
        corners, in the order given by `corner_order` (default:
        top-left, top-right, bottom-right, bottom-left, matching how
        the 4 platform-pose ArUco markers are conventionally placed).
        The plate frame origin is the plate center, x to the right,
        y "up" in image terms (so plane y is flipped vs. image v).
        """
        if corner_order != "TL,TR,BR,BL":
            raise NotImplementedError("only TL,TR,BR,BL order is implemented")
        hw, hh = plate_width_m / 2.0, plate_height_m / 2.0
        plane_corners = np.array([
            [-hw, hh],
            [hw, hh],
            [hw, -hh],
            [-hw, -hh],
        ])
        return cls.from_point_correspondences(image_corners_px, plane_corners)

    def to_plane(self, u: float, v: float) -> tuple[float, float]:
        """Map one pixel coordinate to plate-plane (x, y) in meters."""
        pt = np.array([[[u, v]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(pt, self.H)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def save(self, path: str) -> None:
        np.save(path, self.H)

    @classmethod
    def load(cls, path: str) -> "PixelToPlaneMapper":
        return cls(np.load(path))
