"""Pipeline de vision unificado: un cuadro -> pose del plato + pelota.

Reemplaza la cadena camara -> topic Image -> 2 nodos Python (cada uno
deserializando la imagen cruda). Aqui el cuadro se procesa una sola vez
en el mismo proceso que lo captura, y hacia ROS solo salen los datos
(pose, pelota, TF, estadisticas). Sin dependencia de ROS: testeable y
reutilizable desde scripts/bench_camera.py.

Por cuadro:
  1. ArUco -> pose del plato (cada `aruco_every_n` cuadros; entre medio
     se reutiliza la ultima pose valida, marcada como `held`).
  2. Pelota por HSV, solo dentro de un ROI alrededor de los tags
     (menos pixeles y menos falsos positivos fuera del plato).
  3. El pixel de la pelota se proyecta al plano del plato con la pose
     actual (interseccion rayo-plano) -> (x, y) en metros en el frame
     del plato, que es lo que necesita el control, y el Kalman/gate
     trabajan en esas unidades.
Cada etapa se cronometra; RateStats mide la frecuencia real y el jitter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .ball_tracker import BallTracker, BallDetector, BallDetectorConfig, BallMeasurement
from .kalman import KalmanConfig
from .camera import RateStats  # noqa: F401  (re-export)
from .platform_pose import (
    PlatformPoseEstimator, PlatformPoseConfig, CameraIntrinsics, PlatformPose,
)


class PlateRayMapper:
    """Pixel -> (x, y) en el frame del plato, intersectando el rayo de la
    camara con el plano donde vive el CENTRO de la pelota.

    El frame del plato es el de platform_pose.py: X a la derecha, Y hacia
    abajo en la imagen (image-aligned), por lo tanto Z apunta ALEJANDOSE
    de la camara. El centro de la pelota esta a `ball_radius_m` sobre la
    superficie, o sea en z = -ball_radius_m. Ignorar ese offset mete un
    error de paralaje de ~r * (distancia al eje optico / altura camara).

    Expone `to_plane(u, v)` para enchufarse como `plane_mapper` del
    BallTracker sin tocarlo.
    """

    def __init__(self, intrinsics: CameraIntrinsics, ball_radius_m: float = 0.0):
        self.intrinsics = intrinsics
        self.ball_radius_m = ball_radius_m
        self._R: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self._R is not None

    def set_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        self._R, _ = cv2.Rodrigues(rvec)
        self._t = np.asarray(tvec, dtype=np.float64).reshape(3)

    def to_plane(self, u: float, v: float) -> tuple[float, float]:
        if self._R is None:
            raise RuntimeError("PlateRayMapper sin pose todavia")
        pts = np.array([[[u, v]]], dtype=np.float64)
        xn, yn = cv2.undistortPoints(
            pts, self.intrinsics.camera_matrix, self.intrinsics.dist_coeffs
        ).reshape(2)
        d = np.array([xn, yn, 1.0])
        R, t = self._R, self._t
        n = R[:, 2]
        p0 = t + R @ np.array([0.0, 0.0, -self.ball_radius_m])
        denom = float(n @ d)
        if abs(denom) < 1e-9:
            raise RuntimeError("rayo paralelo al plato")
        s = float(n @ p0) / denom
        p_plate = R.T @ (s * d - t)
        return float(p_plate[0]), float(p_plate[1])


class _RoiDetector:
    """Envuelve un BallDetector para buscar solo dentro de un ROI y
    devolver coordenadas del cuadro completo."""

    def __init__(self, base: BallDetector):
        self.base = base
        self.config = base.config
        self.roi: Optional[tuple[int, int, int, int]] = None  # x0, y0, x1, y1

    def detect(self, frame_bgr: np.ndarray):
        if self.roi is None:
            return self.base.detect(frame_bgr)
        x0, y0, x1, y1 = self.roi
        det = self.base.detect(frame_bgr[y0:y1, x0:x1])
        if det is None:
            return None
        u, v, r = det
        return u + x0, v + y0, r


@dataclass
class PipelineConfig:
    detector: BallDetectorConfig = field(default_factory=BallDetectorConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    pose: PlatformPoseConfig = field(default_factory=PlatformPoseConfig)
    aruco_every_n: int = 1
    roi_margin_px: int = 30
    use_roi: bool = True
    ball_radius_m: float = 0.02
    # Gate cinematico en METROS (la salida ya esta en el frame del plato).
    max_consecutive_misses: int = 15
    gate_max_speed: float = 2.0  # m/s
    gate_min_jump: float = 0.02  # m
    # Cuantos cuadros seguidos se puede reutilizar una pose vieja si los
    # tags dejan de verse, antes de marcar la pose como invalida.
    pose_hold_max_frames: int = 10


@dataclass
class VisionResult:
    t: float
    pose: PlatformPose
    pose_fresh: bool  # False = pose reutilizada de un cuadro anterior
    ball: BallMeasurement
    timings_ms: dict
    roi: Optional[tuple[int, int, int, int]]


class VisionPipeline:
    def __init__(self, config: PipelineConfig, intrinsics: CameraIntrinsics):
        self.config = config
        self.intrinsics = intrinsics
        self.pose_estimator = PlatformPoseEstimator(config=config.pose, intrinsics=intrinsics)
        self.mapper = PlateRayMapper(intrinsics, ball_radius_m=config.ball_radius_m)
        self.tracker = BallTracker(
            detector_config=config.detector,
            kalman_config=config.kalman,
            plane_mapper=self.mapper,
            max_consecutive_misses=config.max_consecutive_misses,
            gate_max_speed=config.gate_max_speed,
            gate_min_jump=config.gate_min_jump,
        )
        self._roi_detector = _RoiDetector(self.tracker.detector)
        self.tracker.detector = self._roi_detector

        self._frame_idx = 0
        self._last_pose: Optional[PlatformPose] = None
        self._pose_age = 0

    def _update_roi(self, pose: PlatformPose, shape) -> None:
        if not self.config.use_roi or pose.rvec is None:
            self._roi_detector.roi = None
            return
        # ROI = proyeccion de las esquinas exteriores de los tags, mas margen.
        half = self.config.pose.marker_length_m / 2.0
        pts3d = []
        for cx, cy in self.config.pose.corner_positions_m:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts3d.append([cx + sx * half, cy + sy * half, 0.0])
        img, _ = cv2.projectPoints(
            np.asarray(pts3d), pose.rvec, pose.tvec,
            self.intrinsics.camera_matrix, self.intrinsics.dist_coeffs,
        )
        img = img.reshape(-1, 2)
        h, w = shape[:2]
        m = self.config.roi_margin_px
        x0 = int(max(0, img[:, 0].min() - m))
        y0 = int(max(0, img[:, 1].min() - m))
        x1 = int(min(w, img[:, 0].max() + m))
        y1 = int(min(h, img[:, 1].max() + m))
        self._roi_detector.roi = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def process(self, frame_bgr: np.ndarray, t: float) -> VisionResult:
        timings = {}
        cfg = self.config

        t0 = time.perf_counter()
        run_aruco = self._last_pose is None or self._frame_idx % max(cfg.aruco_every_n, 1) == 0
        pose_fresh = False
        if run_aruco:
            pose = self.pose_estimator.estimate(frame_bgr, t)
            if pose.valid:
                self._last_pose = pose
                self._pose_age = 0
                pose_fresh = True
                self.mapper.set_pose(pose.rvec, pose.tvec)
                self._update_roi(pose, frame_bgr.shape)
        timings["aruco_ms"] = (time.perf_counter() - t0) * 1e3 if run_aruco else 0.0

        if not pose_fresh:
            self._pose_age += 1
        if self._last_pose is not None and self._pose_age <= cfg.pose_hold_max_frames:
            held = self._last_pose
            pose = PlatformPose(
                t=t, valid=True, theta_x=held.theta_x, theta_y=held.theta_y,
                tx=held.tx, ty=held.ty, tz=held.tz, rvec=held.rvec, tvec=held.tvec,
                corner_pixels=held.corner_pixels,
            )
        else:
            pose = PlatformPose(t=t, valid=False)

        t1 = time.perf_counter()
        if self.mapper.ready and pose.valid:
            ball = self.tracker.update(frame_bgr, t)
        else:
            # Sin pose no hay a donde proyectar la pelota: no se inventa
            # una posicion en pixeles con otras unidades.
            ball = BallMeasurement(x=float("nan"), y=float("nan"), t=t, valid=False, state="no_pose")
        timings["ball_ms"] = (time.perf_counter() - t1) * 1e3

        self._frame_idx += 1
        return VisionResult(
            t=t, pose=pose, pose_fresh=pose_fresh, ball=ball,
            timings_ms=timings, roi=self._roi_detector.roi,
        )


def draw_debug(frame_bgr: np.ndarray, result: VisionResult, stats: Optional[dict] = None) -> np.ndarray:
    out = frame_bgr.copy()
    if result.roi is not None:
        x0, y0, x1, y1 = result.roi
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 200, 0), 1)
    if result.pose.corner_pixels is not None:
        for (u, v) in result.pose.corner_pixels:
            cv2.circle(out, (int(u), int(v)), 4, (255, 0, 255), -1)
    b = result.ball
    if b.pixel is not None:
        cv2.circle(out, (int(b.pixel[0]), int(b.pixel[1])), int(b.radius_px or 5), (0, 255, 0), 2)
    lines = [
        f"tilt x={np.degrees(result.pose.theta_x):+.2f} y={np.degrees(result.pose.theta_y):+.2f} deg"
        + ("" if result.pose.valid else " (INVALID)"),
        f"ball {b.state}: x={b.x * 100:+.1f} y={b.y * 100:+.1f} cm" if b.valid else f"ball {b.state}",
    ]
    if stats:
        lines.append(f"{stats.get('hz', 0):.1f} Hz  jitter {stats.get('jitter_ms', 0):.1f} ms")
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
