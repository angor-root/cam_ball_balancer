"""Constant-velocity 2D Kalman filter.

Used to turn noisy, occasionally-missing ball detections into a smooth
position *and* velocity estimate, and to keep predicting through frames
where the ball detector fails (motion blur, the gripper/servo arm
crossing the frame, etc.) — this is the "predict during dynamic motion,
not just when static" requirement from the team plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    # Process noise: how much we trust the constant-velocity model.
    # Larger = filter adapts faster to real trajectory changes but is noisier.
    process_noise_pos: float = 1.0e-2
    # Tuned against a synthetic circular trajectory (see test_ball_tracker.py):
    # too small and the filter lags noticeably on curved motion (the ball
    # rolling/bouncing on the plate is never truly constant-velocity);
    # too large and it just repeats the raw noisy detections. Re-tune once
    # real footage is available (scripts/tune_hsv.py can be extended for this).
    process_noise_vel: float = 20.0
    # Measurement noise: how much we trust a single detector reading (in
    # the same units as the tracker's output, e.g. meters or normalized
    # pixel units — see ball_tracker.py).
    measurement_noise: float = 5.0e-3
    # Initial state uncertainty.
    initial_pos_var: float = 1.0e-2
    initial_vel_var: float = 1.0


class ConstantVelocityKalman2D:
    """Tracks state [x, y, vx, vy] with position-only measurements.

    This is a small, dependency-free implementation (numpy only) instead
    of pulling in filterpy, since the whole point is that Persona C
    should be able to run this with nothing but opencv-python + numpy.
    """

    STATE_DIM = 4
    MEAS_DIM = 2

    def __init__(self, config: KalmanConfig | None = None):
        self.config = config or KalmanConfig()
        self.x = np.zeros((self.STATE_DIM, 1))
        self.P = np.eye(self.STATE_DIM)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(self.MEAS_DIM) * self.config.measurement_noise
        self.initialized = False

    def reset(self, x: float, y: float) -> None:
        """(Re)initialize the filter at a known position with zero velocity."""
        self.x = np.array([[x], [y], [0.0], [0.0]])
        self.P = np.diag([
            self.config.initial_pos_var,
            self.config.initial_pos_var,
            self.config.initial_vel_var,
            self.config.initial_vel_var,
        ])
        self.initialized = True

    def _transition_matrix(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def _process_noise(self, dt: float) -> np.ndarray:
        dt = max(dt, 1e-6)
        q_pos = self.config.process_noise_pos * dt
        q_vel = self.config.process_noise_vel * dt
        return np.diag([q_pos, q_pos, q_vel, q_vel])

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        F = self._transition_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)

    def update(self, z_x: float, z_y: float) -> None:
        z = np.array([[z_x], [z_y]])
        if not self.initialized:
            self.reset(z_x, z_y)
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.STATE_DIM)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])
