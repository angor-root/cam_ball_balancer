"""Ball detection + tracking — the deliverable Persona D depends on.

Team-plan contract (frozen interface, see README):

    obtener_posicion_pelota() -> (x, y, t, valido)

Implemented here as ``BallTracker.update(frame, timestamp) -> BallMeasurement``.

Pipeline: HSV color threshold -> morphological cleanup -> largest
plausible contour -> pixel centroid -> (optional) homography to plate
coordinates -> Kalman filter (constant velocity) for smoothing and for
predicting through frames where detection fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .kalman import ConstantVelocityKalman2D, KalmanConfig
from .homography import PixelToPlaneMapper


@dataclass
class BallDetectorConfig:
    # Default HSV range targets a bright orange ball (common choice for
    # this kind of project — high contrast against most plate colors).
    # Tune with scripts/tune_hsv.py once the real camera/ball arrive.
    hsv_lower: tuple[int, int, int] = (5, 120, 120)
    hsv_upper: tuple[int, int, int] = (18, 255, 255)
    # Ruido sal-y-pimienta visto en la camara real (2026-09-14): un blur
    # gaussiano no lo quita bien (es ruido impulsivo, no ruido gaussiano
    # de fondo) y de hecho lo esparce. cv2.medianBlur si lo quita, porque
    # reemplaza cada pixel por la mediana de su vecindad -> un pixel
    # blanco/negro aislado en medio de un fondo uniforme desaparece sin
    # emborronar bordes reales. Se aplica ANTES del gaussiano (que sigue
    # sirviendo para el ruido de fondo normal de la camara).
    median_kernel: int = 5
    blur_kernel: int = 7
    morph_kernel: int = 5
    min_radius_px: float = 4.0
    max_radius_px: float = 200.0
    min_circularity: float = 0.6  # 1.0 = perfect circle


@dataclass
class BallMeasurement:
    x: float
    y: float
    t: float
    valid: bool
    pixel: Optional[tuple[float, float]] = None
    radius_px: Optional[float] = None


class BallDetector:
    """Single-frame color-blob ball detector. No temporal state."""

    def __init__(self, config: BallDetectorConfig | None = None):
        self.config = config or BallDetectorConfig()

    def detect(self, frame_bgr: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Returns (u, v, radius_px) in pixel coordinates, or None."""
        cfg = self.config
        denoised = cv2.medianBlur(frame_bgr, cfg.median_kernel)
        blurred = cv2.GaussianBlur(denoised, (cfg.blur_kernel, cfg.blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))

        kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            (u, v), radius = cv2.minEnclosingCircle(c)
            if not (cfg.min_radius_px <= radius <= cfg.max_radius_px):
                continue
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0.0
            if circularity < cfg.min_circularity:
                continue
            if area > best_area:
                best_area = area
                best = (u, v, radius)
        return best


class BallTracker:
    """Stateful tracker: detector + Kalman filter + optional plane mapping.

    Call ``update()`` once per frame, in order, with a monotonically
    increasing timestamp (seconds).
    """

    def __init__(
        self,
        detector_config: BallDetectorConfig | None = None,
        kalman_config: KalmanConfig | None = None,
        plane_mapper: Optional[PixelToPlaneMapper] = None,
        max_consecutive_misses: int = 15,
    ):
        self.detector = BallDetector(detector_config)
        self.kalman = ConstantVelocityKalman2D(kalman_config)
        self.plane_mapper = plane_mapper
        self.max_consecutive_misses = max_consecutive_misses

        self._last_t: Optional[float] = None
        self._misses = 0

    def _to_output_coords(self, u: float, v: float) -> tuple[float, float]:
        if self.plane_mapper is not None:
            return self.plane_mapper.to_plane(u, v)
        # Fallback with no calibration yet: normalized pixel coords in
        # [-1, 1], origin at frame center — NOT physical units. Fine for
        # bring-up/testing; switch to a real plane_mapper before this
        # feeds a control loop.
        return u, v

    def update(self, frame_bgr: np.ndarray, timestamp: float) -> BallMeasurement:
        dt = 0.0 if self._last_t is None else max(timestamp - self._last_t, 0.0)
        self._last_t = timestamp

        self.kalman.predict(dt)

        detection = self.detector.detect(frame_bgr)

        if detection is not None:
            u, v, radius = detection
            x, y = self._to_output_coords(u, v)
            self.kalman.update(x, y)
            self._misses = 0
        else:
            self._misses += 1
            u = v = radius = None

        if not self.kalman.initialized:
            return BallMeasurement(x=float("nan"), y=float("nan"), t=timestamp, valid=False)

        track_lost = self._misses > self.max_consecutive_misses
        fx, fy = self.kalman.position
        return BallMeasurement(
            x=fx,
            y=fy,
            t=timestamp,
            valid=not track_lost,
            pixel=(u, v) if detection is not None else None,
            radius_px=radius,
        )

    def reset(self) -> None:
        self.kalman = ConstantVelocityKalman2D(self.kalman.config)
        self._last_t = None
        self._misses = 0
