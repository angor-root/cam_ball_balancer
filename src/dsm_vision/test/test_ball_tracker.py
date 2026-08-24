import numpy as np
import pytest

from dsm_vision.ball_tracker import BallTracker
from dsm_vision.synthetic import generate_ball_sequence


def test_tracks_moving_ball_within_tolerance():
    tracker = BallTracker()
    seq = generate_ball_sequence(num_frames=60)
    errors = []
    for t, frame, gt_px in seq:
        meas = tracker.update(frame, t)
        if meas.valid:
            errors.append(np.hypot(meas.x - gt_px[0], meas.y - gt_px[1]))
    assert len(errors) > 40
    assert np.mean(errors) < 5.0  # pixels, no plane mapper configured


def test_predicts_through_dropped_frames():
    drop = set(range(20, 25))  # 5-frame occlusion
    tracker = BallTracker(max_consecutive_misses=15)
    seq = generate_ball_sequence(num_frames=60, drop_frames=drop)
    for i, (t, frame, gt_px) in enumerate(seq):
        meas = tracker.update(frame, t)
        if i in drop:
            # Still "valid" (short occlusion), and prediction should stay
            # reasonably close to ground truth even with no measurement.
            # Some drift is expected: constant-velocity extrapolation on a
            # curved (circular) path necessarily lags behind the turn.
            assert meas.valid
            assert np.hypot(meas.x - gt_px[0], meas.y - gt_px[1]) < 30.0


def test_invalid_until_first_detection():
    tracker = BallTracker()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank[:] = (40, 90, 40)
    meas = tracker.update(blank, 0.0)
    assert not meas.valid


def test_track_lost_after_long_occlusion():
    tracker = BallTracker(max_consecutive_misses=3)
    seq = generate_ball_sequence(num_frames=20, drop_frames=set(range(5, 20)))
    last_valid = None
    for t, frame, _ in seq:
        last_valid = tracker.update(frame, t).valid
    assert last_valid is False
