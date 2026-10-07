import numpy as np

from slate_detector import segment as G


def _box_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def _cands():
    shape = (100, 100)
    # 0: high-scoring reflection at the top; 1: the slate lower down
    masks = np.stack([_box_mask(shape, 10, 5, 40, 20), _box_mask(shape, 50, 60, 80, 90)])
    return G.Candidates(masks=masks, scores=np.array([0.9, 0.6]))


def test_choose_prefers_label_outline_over_score():
    idx, why = G.choose(_cands(), outline=[[50, 60], [80, 60], [80, 90], [50, 90]])
    assert (idx, why) == (1, "label_outline")


def test_choose_uses_laser_dot_without_outline():
    assert G.choose(_cands(), dot=(65.0, 75.0)) == (1, "laser_dot")


def test_choose_falls_back_to_top_score():
    assert G.choose(_cands(), dot=(1.0, 99.0)) == (0, "top_score")
    assert G.choose(G.Candidates(masks=np.zeros((0, 4, 4), bool), scores=np.array([]))) == (-1, "none")


def test_outline_with_no_overlap_does_not_force_a_choice():
    assert G.choose(_cands(), outline=[[0, 95], [5, 95], [5, 99], [0, 99]]) == (0, "top_score")


def test_crop_box_pads_and_clips():
    assert G.crop_box([[40, 40], [60, 50]], (100, 100), pad=1.0) == (20, 30, 80, 60)
    assert G.crop_box([[0, 0], [60, 50]], (100, 100), pad=1.0) == (0, 0, 100, 100)


def test_paste_masks_places_crop_back():
    sub = np.ones((1, 10, 20), bool)
    out = G.paste_masks(sub, (5, 7, 25, 17), (30, 40))
    assert out.shape == (1, 30, 40) and out.sum() == 200 and out[0, 7, 5] and not out[0, 6, 5]


def test_save_load_round_trip(tmp_path):
    c = _cands()
    G.save_candidates(tmp_path / "1.npz", c, 1, "laser_dot")
    back, chosen, why = G.load_candidates(tmp_path / "1.npz")
    assert (back.masks == c.masks).all() and np.allclose(back.scores, c.scores)
    assert (chosen, why, back.crop) == (1, "laser_dot", False)
