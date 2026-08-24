import numpy as np
import cv2

from dsm_vision.platform_pose import (
    PlatformPoseEstimator,
    PlatformPoseConfig,
    CameraIntrinsics,
    _tilt_angles_from_rvec,
)
from dsm_vision.synthetic import (
    render_plate_with_markers,
    SyntheticPlateConfig,
    render_plate_with_markers_perspective,
)


def _pose_config():
    return PlatformPoseConfig(
        marker_length_m=0.03,
        corner_marker_ids=(0, 1, 2, 3),
        corner_positions_m=((-0.09, -0.09), (0.09, -0.09), (0.09, 0.09), (-0.09, 0.09)),
    )


def _intrinsics(width=640, height=480):
    f = 700.0
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    return CameraIntrinsics(camera_matrix=K, dist_coeffs=np.zeros(5))


class TestTiltAngleExtraction:
    """Unit-tests _tilt_angles_from_rvec in isolation from any rendering,
    since a synthetic renderer that isn't itself a true perspective
    projection can't be trusted to validate exact recovered angles.
    """

    def test_zero_rotation_gives_zero_tilt(self):
        rvec = np.zeros((3, 1))
        theta_x, theta_y = _tilt_angles_from_rvec(rvec)
        assert abs(theta_x) < 1e-9
        assert abs(theta_y) < 1e-9

    def test_rotation_about_camera_x_axis(self):
        a = 0.2  # radians
        rvec = np.array([[a], [0.0], [0.0]])
        theta_x, theta_y = _tilt_angles_from_rvec(rvec)
        assert abs(theta_x - a) < 1e-6
        assert abs(theta_y) < 1e-6

    def test_rotation_about_camera_y_axis(self):
        b = 0.15
        rvec = np.array([[0.0], [b], [0.0]])
        theta_x, theta_y = _tilt_angles_from_rvec(rvec)
        assert abs(theta_x) < 1e-6
        assert abs(theta_y - b) < 1e-6


class TestPerspectiveCorrectRendering:
    """End-to-end: real pinhole-projected markers, known ground-truth
    pose, recovered via the actual detection + solvePnP pipeline.
    """

    def _run(self, theta_x_true: float, theta_y_true: float, tz: float = 0.5):
        cfg = _pose_config()
        intr = _intrinsics()
        rvec_true = np.array([[theta_x_true], [theta_y_true], [0.0]])
        tvec_true = np.array([[0.0], [0.0], [tz]])

        frame = render_plate_with_markers_perspective(
            plate_positions_m=np.array(cfg.corner_positions_m),
            marker_ids=cfg.corner_marker_ids,
            marker_length_m=cfg.marker_length_m,
            camera_matrix=intr.camera_matrix,
            dist_coeffs=intr.dist_coeffs,
            rvec=rvec_true,
            tvec=tvec_true,
        )
        estimator = PlatformPoseEstimator(config=cfg, intrinsics=intr)
        return estimator.estimate(frame, timestamp=0.0)

    def test_flat_plate_recovers_near_zero_tilt(self):
        pose = self._run(0.0, 0.0)
        assert pose.valid
        assert abs(pose.theta_x) < 0.05
        assert abs(pose.theta_y) < 0.05
        assert pose.corner_pixels is not None

    def test_tilted_plate_recovers_known_angles(self):
        pose = self._run(0.2, -0.1)
        assert pose.valid
        assert abs(pose.theta_x - 0.2) < 0.05
        assert abs(pose.theta_y - (-0.1)) < 0.05


def test_invalid_when_no_markers_present():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank[:] = (60, 100, 60)
    estimator = PlatformPoseEstimator()
    pose = estimator.estimate(blank, timestamp=0.0)
    assert not pose.valid


def test_flat_paste_render_is_at_least_detected():
    """The cheap flat-paste renderer (no real projection) isn't suitable
    for validating exact angles (see TestPerspectiveCorrectRendering),
    but detection itself should still succeed on it.
    """
    plate_cfg = SyntheticPlateConfig()
    frame, _ = render_plate_with_markers(plate_cfg)
    estimator = PlatformPoseEstimator(config=_pose_config())
    pose = estimator.estimate(frame, timestamp=0.0)
    assert pose.valid
    assert pose.corner_pixels is not None
