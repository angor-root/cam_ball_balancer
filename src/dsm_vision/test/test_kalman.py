import numpy as np

from dsm_vision.kalman import ConstantVelocityKalman2D


def test_converges_to_constant_velocity_line():
    kf = ConstantVelocityKalman2D()
    dt = 1.0 / 30.0
    vx, vy = 50.0, -20.0
    x0, y0 = 100.0, 200.0
    rng = np.random.default_rng(0)

    for i in range(120):
        kf.predict(dt)
        t = i * dt
        true_x = x0 + vx * t
        true_y = y0 + vy * t
        noisy_x = true_x + rng.normal(0, 1.0)
        noisy_y = true_y + rng.normal(0, 1.0)
        kf.update(noisy_x, noisy_y)

    est_x, est_y = kf.position
    true_x = x0 + vx * 119 * dt
    true_y = y0 + vy * 119 * dt
    assert abs(est_x - true_x) < 3.0
    assert abs(est_y - true_y) < 3.0

    # Velocity is intentionally noisier than position: the default gains
    # favor low position lag on curved real-world motion (see
    # test_ball_tracker.py) over a smooth velocity estimate, and velocity
    # isn't part of the public obtener_posicion_pelota() contract anyway.
    est_vx, est_vy = kf.velocity
    assert abs(est_vx - vx) < 30.0
    assert abs(est_vy - vy) < 30.0


def test_predict_without_update_extrapolates():
    kf = ConstantVelocityKalman2D()
    kf.reset(0.0, 0.0)
    kf.x[2, 0] = 10.0  # vx
    kf.x[3, 0] = 0.0
    kf.predict(dt=1.0)
    x, y = kf.position
    assert abs(x - 10.0) < 1e-6
    assert abs(y - 0.0) < 1e-6


def test_uninitialized_predict_is_noop():
    kf = ConstantVelocityKalman2D()
    kf.predict(dt=1.0)  # should not raise, should stay uninitialized
    assert not kf.initialized
