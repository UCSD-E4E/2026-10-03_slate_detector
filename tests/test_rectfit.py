import numpy as np

from slate_detector import rectfit as R
from slate_detector.segment import mask_iou, polygon_mask

K = np.array([[1130.0, 0, 800], [0, 1140.0, 600], [0, 0, 1]])


def _board_mask(rv, tv, shape=(1202, 1600)):
    poly = R.project(R.board_outline(R.ASPECT, R.RADIUS), rv, tv, K)
    return polygon_mask(shape, poly), poly


def test_fit_recovers_a_board_with_a_hand_over_a_corner():
    rv, tv = np.array([0.15, -0.25, 0.08]), np.array([-0.45, -0.3, 3.2])
    full, _ = _board_mask(rv, tv)
    seen = full.copy()
    ys, xs = np.nonzero(full)
    seen[ys.max() - 60:, xs.max() - 70:] = False  # a hand over the bottom-right corner
    f = R.fit_board(seen, K)
    assert f is not None and R.plausible(f)
    assert mask_iou(polygon_mask(full.shape, f.polygon), full) > 0.97  # the hidden corner is filled back in


def test_clip_to_frame():
    sq = np.array([[-10, -10], [50, -10], [50, 50], [-10, 50]], float)
    c = R.clip_to_frame(sq, 40, 30)
    assert c[:, 0].min() == 0 and c[:, 0].max() == 40 and c[:, 1].min() == 0 and c[:, 1].max() == 30


def test_board_outline_is_a_rounded_rectangle():
    o = R.board_outline(0.7, 0.05)
    assert o[:, 0].min() == 0 and o[:, 0].max() == 1 and np.isclose(o[:, 1].max(), 0.7) and len(o) == 24


def test_portrait_board_is_not_fitted_as_a_tilted_landscape_board():
    # A board held portrait, facing the camera: rotate 90 deg about the camera axis.
    rv, tv = np.array([0.0, 0.0, np.pi / 2]), np.array([0.3, -0.4, 3.0])
    full, _ = _board_mask(rv, tv)
    f = R.fit_board(full, K)
    assert f is not None and R.tilt_deg(f.rvec) < 15
