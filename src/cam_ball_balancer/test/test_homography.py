import numpy as np

from cam_ball_balancer.homography import PixelToPlaneMapper


def test_rectangular_plate_round_trip():
    # A 0.2m x 0.2m plate imaged as a perfect 400x400 px square (2000 px/m),
    # corners in TL,TR,BR,BL order.
    image_corners = np.array([
        [100, 100],
        [500, 100],
        [500, 500],
        [100, 500],
    ], dtype=np.float64)
    mapper = PixelToPlaneMapper.from_rectangular_plate(image_corners, 0.2, 0.2)

    cx, cy = mapper.to_plane(300, 300)  # image center -> plate center
    assert abs(cx) < 1e-6
    assert abs(cy) < 1e-6

    x, y = mapper.to_plane(500, 100)  # top-right corner -> (+0.1, +0.1)
    assert abs(x - 0.1) < 1e-6
    assert abs(y - 0.1) < 1e-6


def test_from_point_correspondences_matches_identity_scale():
    image_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    plane_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    mapper = PixelToPlaneMapper.from_point_correspondences(image_pts, plane_pts)
    x, y = mapper.to_plane(5, 5)
    assert abs(x - 0.5) < 1e-6
    assert abs(y - 0.5) < 1e-6
