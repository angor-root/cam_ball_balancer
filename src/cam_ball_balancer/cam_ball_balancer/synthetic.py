"""Synthetic frame generation, so the whole pipeline can be unit-tested
without the real camera or printed tags (which haven't arrived yet).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import cv2.aruco as aruco


def blank_frame(width: int = 640, height: int = 480, color=(40, 90, 40)) -> np.ndarray:
    """A plain plate-colored background (default: dull green, like felt)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def draw_ball(
    frame: np.ndarray,
    center_px: tuple[float, float],
    radius_px: float = 15.0,
    color=(0, 140, 255),  # orange, BGR
) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (int(round(center_px[0])), int(round(center_px[1]))), int(round(radius_px)), color, -1)
    # A little blur mimics real camera softness / motion blur.
    return cv2.GaussianBlur(out, (5, 5), 0)


def circular_trajectory(
    num_frames: int, dt: float, center_px: tuple[float, float], radius_px: float, period_s: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Returns [(t, x_px, y_px), ...] for a ball moving in a circle."""
    out = []
    for i in range(num_frames):
        t = i * dt
        omega = 2 * np.pi / period_s
        x = center_px[0] + radius_px * np.cos(omega * t)
        y = center_px[1] + radius_px * np.sin(omega * t)
        out.append((t, x, y))
    return out


def generate_ball_sequence(
    num_frames: int = 60,
    dt: float = 1.0 / 30.0,
    width: int = 640,
    height: int = 480,
    drop_frames: Optional[set[int]] = None,
) -> list[tuple[float, np.ndarray, tuple[float, float]]]:
    """Returns [(t, frame_bgr, ground_truth_px), ...].

    `drop_frames` is a set of frame indices where the ball is NOT drawn
    (simulating occlusion/detection failure), to test that the Kalman
    filter keeps predicting sensibly through gaps.
    """
    drop_frames = drop_frames or set()
    traj = circular_trajectory(num_frames, dt, center_px=(width / 2, height / 2), radius_px=100)
    out = []
    for i, (t, x, y) in enumerate(traj):
        frame = blank_frame(width, height)
        if i not in drop_frames:
            frame = draw_ball(frame, (x, y))
        out.append((t, frame, (x, y)))
    return out


@dataclass
class SyntheticPlateConfig:
    width: int = 640
    height: int = 480
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    marker_length_px: int = 60
    plate_margin_px: int = 80


def render_plate_with_markers(config: SyntheticPlateConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Renders a synthetic top-down view of the plate with 4 ArUco
    markers at its corners (axis-aligned, no real tilt — good enough to
    unit-test detection + the flat-pose case).

    Returns (frame_bgr, corner_pixel_centers) where corner_pixel_centers
    is a 4x2 array in the same TL,TR,BR,BL order as the markers.
    """
    cfg = config or SyntheticPlateConfig()
    frame = blank_frame(cfg.width, cfg.height, color=(60, 100, 60))
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, cfg.dictionary_name)) \
        if hasattr(aruco, "getPredefinedDictionary") \
        else aruco.Dictionary_get(getattr(aruco, cfg.dictionary_name))

    m = cfg.plate_margin_px
    half = cfg.marker_length_px // 2
    centers = [
        (m, m),
        (cfg.width - m, m),
        (cfg.width - m, cfg.height - m),
        (m, cfg.height - m),
    ]
    for marker_id, (cx, cy) in zip(cfg.marker_ids, centers):
        if hasattr(aruco, "generateImageMarker"):
            marker_img = aruco.generateImageMarker(dictionary, marker_id, cfg.marker_length_px)
        else:
            marker_img = aruco.drawMarker(dictionary, marker_id, cfg.marker_length_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        y0, y1 = cy - half, cy - half + marker_bgr.shape[0]
        x0, x1 = cx - half, cx - half + marker_bgr.shape[1]
        frame[y0:y1, x0:x1] = marker_bgr

    return frame, np.array(centers, dtype=np.float64)


def render_plate_with_markers_perspective(
    plate_positions_m: np.ndarray,
    marker_ids: tuple[int, ...],
    marker_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    width: int = 640,
    height: int = 480,
    dictionary_name: str = "DICT_4X4_50",
) -> np.ndarray:
    """Renders markers with a *real* pinhole projection (unlike
    ``render_plate_with_markers``, which just pastes flat bitmaps).

    Each marker's 4 corners are projected from the plate frame into the
    image via the given camera intrinsics + extrinsics (rvec/tvec —
    same convention `platform_pose.py` estimates), and the flat marker
    bitmap is warped onto those projected corners. This makes it
    possible to unit-test pose *recovery* against a known ground-truth
    tilt, not just "did we detect something roughly plate-shaped".

    `plate_positions_m`: Nx2 array, marker center positions in the plate
    frame (same image-aligned convention as PlatformPoseConfig).
    """
    frame = blank_frame(width, height, color=(60, 100, 60))
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name)) \
        if hasattr(aruco, "getPredefinedDictionary") \
        else aruco.Dictionary_get(getattr(aruco, dictionary_name))

    half = marker_length_m / 2.0
    bitmap_size = 200

    for marker_id, (cx, cy) in zip(marker_ids, plate_positions_m):
        # Object-space corners in the SAME order used by platform_pose.py:
        # TL, TR, BR, BL of this marker, image-aligned plate frame.
        obj_corners = np.array([
            [cx - half, cy - half, 0.0],
            [cx + half, cy - half, 0.0],
            [cx + half, cy + half, 0.0],
            [cx - half, cy + half, 0.0],
        ])
        img_corners, _ = cv2.projectPoints(obj_corners, rvec, tvec, camera_matrix, dist_coeffs)
        img_corners = img_corners.reshape(4, 2).astype(np.float32)

        if hasattr(aruco, "generateImageMarker"):
            marker_img = aruco.generateImageMarker(dictionary, marker_id, bitmap_size)
        else:
            marker_img = aruco.drawMarker(dictionary, marker_id, bitmap_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        bitmap_corners = np.array([
            [0, 0], [bitmap_size - 1, 0], [bitmap_size - 1, bitmap_size - 1], [0, bitmap_size - 1],
        ], dtype=np.float32)
        H = cv2.getPerspectiveTransform(bitmap_corners, img_corners)
        warped = cv2.warpPerspective(marker_bgr, H, (width, height), borderValue=(60, 100, 60))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, img_corners.astype(np.int32), 255)
        frame[mask > 0] = warped[mask > 0]

    return frame
