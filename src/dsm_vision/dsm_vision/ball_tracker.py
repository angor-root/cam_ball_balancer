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
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .kalman import ConstantVelocityKalman2D, KalmanConfig
from .homography import PixelToPlaneMapper


class TrackState(Enum):
    """Explicit track lifecycle, so noisy/intermittent detections (motion
    blur, lighting glare, brief occlusion) are handled the same way every
    time instead of by a single ad-hoc miss counter.

    SEARCHING -> no track yet; any detection (ungated, nothing to compare
                 against) starts one.
    TRACKING  -> track established; only detections that pass the Kalman
                 gate update it, so an outlier (e.g. a reflection) can't
                 yank the estimate.
    COASTING  -> a frame or few had no accepted detection; keep reporting
                 the Kalman prediction (still "valid") instead of dropping
                 output, but don't extend this indefinitely.
    LOST      -> too many consecutive misses; stop trusting the stale
                 extrapolation and require a fresh, ungated detection to
                 re-acquire (avoids drifting far from reality and then
                 snapping badly once the ball reappears).
    """

    SEARCHING = "searching"
    TRACKING = "tracking"
    COASTING = "coasting"
    LOST = "lost"


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
    # Diagnostic only — NOT part of the frozen obtener_posicion_pelota()
    # contract (x, y, t, valido). Useful for logging/debugging why a given
    # frame was (in)valid; consumers should keep depending on `valid` alone.
    state: str = ""


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
        gate_max_speed: float = 3000.0,
        gate_min_jump: float = 40.0,
    ):
        self.detector = BallDetector(detector_config)
        self.kalman = ConstantVelocityKalman2D(kalman_config)
        self.plane_mapper = plane_mapper
        self.max_consecutive_misses = max_consecutive_misses
        # Kinematic outlier gate, in tracker output units (pixels in the
        # no-plane_mapper fallback, plate units once one is configured):
        # a detection is only accepted if it's within `gate_min_jump +
        # gate_max_speed * dt` of the current prediction. This is
        # deliberately NOT a statistical (Mahalanobis) gate against the
        # Kalman's own P — the constant-velocity model systematically
        # lags on curved motion (see KalmanConfig), so P underestimates
        # real per-frame residuals and a statistical gate ends up
        # rejecting good detections, not just outliers. A plain "how far
        # could the ball plausibly have moved" bound is more robust and
        # easier to tune against the real plate size/ball speed once
        # known. `gate_min_jump` is slack for near-zero-speed jitter;
        # `gate_max_speed` should be set comfortably above the ball's real
        # max speed once that's known (defaults generous for bring-up).
        self.gate_max_speed = gate_max_speed
        self.gate_min_jump = gate_min_jump

        self._last_t: Optional[float] = None
        self._misses = 0
        self.state = TrackState.SEARCHING

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
        accepted = False

        if detection is not None:
            u, v, radius = detection
            x, y = self._to_output_coords(u, v)
            if not self.kalman.initialized:
                # SEARCHING -> TRACKING: no prior track to gate against.
                self.kalman.update(x, y)
                accepted = True
            else:
                pred_x, pred_y = self.kalman.position
                jump = float(np.hypot(x - pred_x, y - pred_y))
                allowed = self.gate_min_jump + self.gate_max_speed * dt
                if jump <= allowed:
                    self.kalman.update(x, y)
                    accepted = True
            # else: rejected as an outlier (e.g. a lighting reflection the
            # HSV mask briefly latched onto) — treated exactly like a
            # missing detection below, so the filter coasts instead of
            # jumping to a point the ball almost certainly isn't at.
        else:
            u = v = radius = None

        if accepted:
            self._misses = 0
        else:
            self._misses += 1

        if not self.kalman.initialized:
            self.state = TrackState.TRACKING if accepted else TrackState.SEARCHING
        elif accepted:
            self.state = TrackState.TRACKING
        elif self._misses > self.max_consecutive_misses:
            self.state = TrackState.LOST
            # Stop trusting the stale extrapolation — require a fresh,
            # ungated detection to re-acquire rather than snapping from
            # wherever a long coast drifted to.
            self.kalman.initialized = False
        else:
            self.state = TrackState.COASTING

        if not self.kalman.initialized:
            return BallMeasurement(
                x=float("nan"), y=float("nan"), t=timestamp, valid=False,
                state=self.state.value,
            )

        fx, fy = self.kalman.position
        return BallMeasurement(
            x=fx,
            y=fy,
            t=timestamp,
            valid=self.state != TrackState.LOST,
            pixel=(u, v) if accepted else None,
            radius_px=radius if accepted else None,
            state=self.state.value,
        )

    def reset(self) -> None:
        self.kalman = ConstantVelocityKalman2D(self.kalman.config)
        self._last_t = None
        self._misses = 0
        self.state = TrackState.SEARCHING
