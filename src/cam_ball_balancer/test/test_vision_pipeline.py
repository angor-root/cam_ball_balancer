import numpy as np
import cv2

from cam_ball_balancer.platform_pose import PlatformPoseConfig, CameraIntrinsics
from cam_ball_balancer.synthetic import render_plate_with_markers_perspective
from cam_ball_balancer.vision_pipeline import (
    PipelineConfig, VisionPipeline, PlateRayMapper, RateStats,
)

BALL_R = 0.02


def _intrinsics(width=640, height=480):
    f = 700.0
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    return CameraIntrinsics(camera_matrix=K, dist_coeffs=np.zeros(5))


def _pose_cfg():
    return PlatformPoseConfig(
        marker_length_m=0.03,
        corner_marker_ids=(0, 1, 2, 3),
        corner_positions_m=((-0.09, -0.09), (0.09, -0.09), (0.09, 0.09), (-0.09, 0.09)),
    )


def _render(ball_xy, rvec, tvec, intr, cfg):
    frame = render_plate_with_markers_perspective(
        plate_positions_m=np.array(cfg.corner_positions_m),
        marker_ids=cfg.corner_marker_ids,
        marker_length_m=cfg.marker_length_m,
        camera_matrix=intr.camera_matrix,
        dist_coeffs=intr.dist_coeffs,
        rvec=rvec, tvec=tvec,
    )
    # Centro de la pelota a BALL_R sobre la superficie (z negativo = hacia la camara).
    center = np.array([[ball_xy[0], ball_xy[1], -BALL_R]])
    px, _ = cv2.projectPoints(center, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
    u, v = px.reshape(2)
    radius_px = intr.camera_matrix[0, 0] * BALL_R / float(tvec[2, 0])
    cv2.circle(frame, (int(round(u)), int(round(v))), int(round(radius_px)), (0, 140, 255), -1)
    return frame


def test_ray_mapper_inverts_projection_on_tilted_plate():
    intr = _intrinsics()
    rvec = np.array([[0.15], [-0.1], [0.05]])
    tvec = np.array([[0.01], [-0.02], [0.45]])
    mapper = PlateRayMapper(intr, ball_radius_m=BALL_R)
    mapper.set_pose(rvec, tvec)
    for x, y in [(0.0, 0.0), (0.06, -0.04), (-0.07, 0.05)]:
        px, _ = cv2.projectPoints(np.array([[x, y, -BALL_R]]), rvec, tvec,
                                  intr.camera_matrix, intr.dist_coeffs)
        xr, yr = mapper.to_plane(*px.reshape(2))
        assert abs(xr - x) < 1e-6 and abs(yr - y) < 1e-6


def test_pipeline_ball_in_meters_on_tilted_plate():
    intr = _intrinsics()
    cfg = _pose_cfg()
    rvec = np.array([[0.12], [-0.08], [0.0]])
    tvec = np.array([[0.0], [0.0], [0.5]])
    pipe = VisionPipeline(PipelineConfig(pose=cfg, ball_radius_m=BALL_R), intr)

    truth = (0.04, -0.03)
    frame = _render(truth, rvec, tvec, intr, cfg)
    res = pipe.process(frame, 0.0)

    assert res.pose.valid and res.pose_fresh
    assert res.ball.valid
    # Tolerancia: error de pose por deteccion de esquinas + pixelado del circulo.
    assert abs(res.ball.x - truth[0]) < 0.006
    assert abs(res.ball.y - truth[1]) < 0.006
    assert res.roi is not None


def test_no_pose_means_no_ball():
    intr = _intrinsics()
    pipe = VisionPipeline(PipelineConfig(pose=_pose_cfg()), intr)
    blank = np.full((480, 640, 3), (60, 100, 60), np.uint8)
    cv2.circle(blank, (320, 240), 14, (0, 140, 255), -1)
    res = pipe.process(blank, 0.0)
    assert not res.pose.valid
    assert not res.ball.valid and res.ball.state == "no_pose"


def test_aruco_every_n_holds_pose_between_runs():
    intr = _intrinsics()
    cfg = _pose_cfg()
    rvec = np.array([[0.1], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [0.5]])
    frame = _render((0.0, 0.0), rvec, tvec, intr, cfg)
    pipe = VisionPipeline(PipelineConfig(pose=cfg, aruco_every_n=3), intr)
    fresh = [pipe.process(frame, i / 30).pose_fresh for i in range(6)]
    assert fresh == [True, False, False, True, False, False]


def test_rate_stats():
    r = RateStats()
    for i in range(31):
        r.tick(i / 30.0)
    s = r.summary()
    assert abs(s["hz"] - 30.0) < 1e-6
    assert s["jitter_ms"] < 1e-6
