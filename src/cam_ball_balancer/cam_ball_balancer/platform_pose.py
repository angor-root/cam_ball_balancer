"""Platform (plate) pose estimation from ArUco tags at its corners.

There is no encoder for the 2-DOF plate's tilt angle, so vision is the
*only* source of truth for how the plate is actually oriented (θx, θy)
— this is just as central to the project as ball tracking, not an
extra. Four ArUco markers are mounted at the plate corners (flat on the
plate, all facing the fixed overhead camera); their combined pose gives
the plate's position and orientation relative to the camera.

Two OpenCV ArUco APIs exist in the wild (the old free-function API from
opencv-python <4.7, and the ``cv2.aruco.ArucoDetector`` object API from
>=4.7). The dev machine here has 4.6 (old API); the Pi's OpenCV version
is unknown. ``_ArucoCompat`` below picks whichever is available so the
same code runs on either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
import cv2.aruco as aruco


@dataclass
class PlatformPoseConfig:
    # ArUco dictionary shared by all 4 corner markers.
    dictionary_name: str = "DICT_4X4_50"
    # Physical side length of each marker's black square, in meters.
    marker_length_m: float = 0.03
    # Marker IDs at each corner, and their center position in the PLATE
    # frame (meters, z=0, origin at plate center). Order must match
    # `corner_marker_ids`. Defaults assume a 0.20m x 0.20m plate with
    # markers set slightly inset from the physical edge — override this
    # in config/params.yaml once the real plate is built.
    #
    # IMPORTANT: this frame is image-aligned, not "math-style" — plate Y
    # grows the same direction as image row/v (i.e. "top" = smaller y),
    # matching the per-marker corner convention below. Getting this
    # backwards silently produces a pose flipped ~180 deg about the
    # plate's own X axis (this is a real planar-pose ambiguity that
    # solvePnP/IPPE will happily "solve" for the wrong correspondence —
    # see test_platform_pose.py for the regression that caught it).
    corner_marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)  # TL, TR, BR, BL
    corner_positions_m: tuple[tuple[float, float], ...] = (
        (-0.09, -0.09),
        (0.09, -0.09),
        (0.09, 0.09),
        (-0.09, 0.09),
    )


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # 1xN or Nx1

    @classmethod
    def load(cls, path: str) -> "CameraIntrinsics":
        data = np.load(path)
        return cls(camera_matrix=data["camera_matrix"], dist_coeffs=data["dist_coeffs"])

    @classmethod
    def identity_guess(cls, image_width: int, image_height: int) -> "CameraIntrinsics":
        """Rough pinhole guess (focal length ~= image width) for bring-up
        *only*, before the real camera is calibrated. Tilt angles from
        this will be approximate — do not trust for the real control
        loop. See scripts/calibrate_camera.py.
        """
        f = float(image_width)
        cx, cy = image_width / 2.0, image_height / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        return cls(camera_matrix=K, dist_coeffs=dist)


@dataclass
class PlatformPose:
    t: float
    valid: bool
    theta_x: float = 0.0  # tilt about plate x-axis, radians
    theta_y: float = 0.0  # tilt about plate y-axis, radians
    tx: float = 0.0  # plate center position in camera frame, meters
    ty: float = 0.0
    tz: float = 0.0
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    corner_pixels: Optional[np.ndarray] = None  # 4x2, order = config.corner_marker_ids


class _ArucoCompat:
    """Wraps whichever ArUco API (old free-function vs new class-based) is installed."""

    def __init__(self, dictionary_name: str):
        dict_id = getattr(aruco, dictionary_name)
        if hasattr(aruco, "ArucoDetector"):
            self._new_api = True
            self.dictionary = aruco.getPredefinedDictionary(dict_id)
            self.params = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.dictionary, self.params)
        else:
            self._new_api = False
            self.dictionary = aruco.Dictionary_get(dict_id)
            self.params = aruco.DetectorParameters_create()
            self.detector = None

    def detect(self, gray: np.ndarray):
        if self._new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        return corners, ids


class PlatformPoseEstimator:
    """Detects the 4 corner ArUco markers and solves for the plate pose."""

    def __init__(
        self,
        config: PlatformPoseConfig | None = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ):
        self.config = config or PlatformPoseConfig()
        self.intrinsics = intrinsics
        self._aruco = _ArucoCompat(self.config.dictionary_name)

        # Precompute the object points (plate frame, z=0) indexed by marker id,
        # each as the 4 corners of that marker's own square (needed by solvePnP).
        self._object_points_by_id: dict[int, np.ndarray] = {}
        half = self.config.marker_length_m / 2.0
        for marker_id, (cx, cy) in zip(self.config.corner_marker_ids, self.config.corner_positions_m):
            # Corner order must match what cv2.aruco.detectMarkers returns
            # for an unrotated marker: index 0/1/2/3 = the marker's own
            # top-left/top-right/bottom-right/bottom-left corner, where
            # "top" means smaller image row. We use a plate frame whose Y
            # axis grows the same direction as image v (down), so "top"
            # (index 0/1) is the smaller-y side. Getting this backwards
            # silently produces a pose flipped ~180 deg about the plate's
            # own X axis (caught by test_platform_pose.py).
            self._object_points_by_id[marker_id] = np.array([
                [cx - half, cy - half, 0.0],
                [cx + half, cy - half, 0.0],
                [cx + half, cy + half, 0.0],
                [cx - half, cy + half, 0.0],
            ])

    def estimate(self, frame_bgr: np.ndarray, timestamp: float) -> PlatformPose:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = self._aruco.detect(gray)

        if ids is None or len(ids) == 0:
            return PlatformPose(t=timestamp, valid=False)

        object_points = []
        image_points = []
        found_corner_pixels = {}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self._object_points_by_id:
                continue
            object_points.append(self._object_points_by_id[marker_id])
            image_points.append(marker_corners.reshape(4, 2))
            found_corner_pixels[marker_id] = marker_corners.reshape(4, 2).mean(axis=0)

        if len(object_points) < 2:
            # Need at least 2 markers (8 points) for a numerically stable
            # planar PnP in practice; with only 1 the pose is unreliable.
            return PlatformPose(t=timestamp, valid=False)

        object_points = np.concatenate(object_points, axis=0)
        image_points = np.concatenate(image_points, axis=0)

        if self.intrinsics is None:
            h, w = frame_bgr.shape[:2]
            self.intrinsics = CameraIntrinsics.identity_guess(w, h)

        # A near-fronto-parallel planar target (our exact case: overhead
        # camera, mostly-flat tiltable plate) has a well-known pose
        # ambiguity — two rotations fit the 2D projection almost equally
        # well (the true one and a ~180 deg "ghost" flip). Plain
        # SOLVEPNP_ITERATIVE can converge to the wrong one. IPPE is the
        # estimator designed for exactly this (planar, near-fronto-parallel)
        # case: it returns both candidate solutions ranked by reprojection
        # error, so we take the best one instead of guessing.
        n_sols, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points, image_points,
            self.intrinsics.camera_matrix, self.intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if n_sols == 0:
            return PlatformPose(t=timestamp, valid=False)

        best = int(np.argmin(errors))
        rvec, tvec = rvecs[best], tvecs[best]

        theta_x, theta_y = _tilt_angles_from_rvec(rvec)

        corner_px = None
        if all(cid in found_corner_pixels for cid in self.config.corner_marker_ids):
            corner_px = np.array([found_corner_pixels[cid] for cid in self.config.corner_marker_ids])

        return PlatformPose(
            t=timestamp,
            valid=True,
            theta_x=theta_x,
            theta_y=theta_y,
            tx=float(tvec[0, 0]), ty=float(tvec[1, 0]), tz=float(tvec[2, 0]),
            rvec=rvec, tvec=tvec,
            corner_pixels=corner_px,
        )


def _tilt_angles_from_rvec(rvec: np.ndarray) -> tuple[float, float]:
    """Extract (theta_x, theta_y) tilt of the plate's normal vs. the
    camera's optical axis from a solvePnP rotation vector.

    theta_x = rotation about the plate's own x-axis (tilts the y-edge
    up/down), theta_y = rotation about the plate's own y-axis. This is
    the natural (θx, θy) pair the ESP32 gimbal command already uses.
    """
    R, _ = cv2.Rodrigues(rvec)
    # ZYX-ish decomposition assuming the plate is close to facing the
    # camera (small-angle regime, which is the whole point of a
    # balance platform): theta_x ~= asin(-R[2,1]) style extraction via
    # the plate's local Z axis (its normal) expressed in camera frame.
    normal = R @ np.array([0.0, 0.0, 1.0])
    # Signs chosen so a pure rvec=[a,0,0] (right-hand rotation about the
    # camera's own X axis) reads back as theta_x=a, and rvec=[0,b,0]
    # reads back as theta_y=b — see TestTiltAngleExtraction.
    theta_x = float(np.arctan2(-normal[1], normal[2]))
    theta_y = float(np.arctan2(normal[0], normal[2]))
    return theta_x, theta_y
