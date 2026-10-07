import numpy as np

from slate_detector import labelstudio as L
from slate_detector.segment import Candidates, save_candidates


def test_percent_round_trip():
    pts = [[160.0, 120.2], [1440.0, 120.2], [1440.0, 1081.8], [160.0, 1081.8]]
    back = L.from_percent(L.to_percent(pts, 1600, 1202), 1600, 1202)
    assert np.allclose(back, pts, atol=0.01)


def test_load_secrets_skips_blanks_and_comments(tmp_path):
    p = tmp_path / "s.env"
    p.write_text("# c\nA=1\nB=\n C = x=y \n")
    assert L.load_secrets(p) == {"A": "1", "C": "x=y"}


def test_build_task_puts_quad_prediction(tmp_path):
    m = np.zeros((1, 120, 160), bool)
    m[0, 30:90, 40:120] = True
    save_candidates(tmp_path / "5.npz", Candidates(m, np.array([0.8])), 0, "top_score")
    t = L.build_task(5, 9, "s3://b/k/frames/5.jpg", 160, 120, tmp_path / "5.npz")
    assert t["data"] == {"image": "s3://b/k/frames/5.jpg", "image_id": 5, "dive_id": 9}
    r = t["predictions"][0]["result"][0]
    assert r["type"] == "polygonlabels" and len(r["value"]["points"]) == 4 and r["original_width"] == 160


def test_build_task_without_candidate_has_no_prediction(tmp_path):
    save_candidates(tmp_path / "6.npz", Candidates(np.zeros((0, 10, 10), bool), np.array([])), -1, "none")
    assert "predictions" not in L.build_task(6, 9, "s3://b/x.jpg", 10, 10, tmp_path / "6.npz")


def _ann(result, **kw):
    return {"id": 1, "result": result, "updated_at": "2026-10-06T00:00:00Z", **kw}


def test_annotations_to_rows():
    poly = {"type": "polygonlabels", "from_name": "slate", "original_width": 1600, "original_height": 1202,
            "value": {"points": [[10, 10], [20, 10], [20, 20], [10, 20]], "polygonlabels": ["Slate"]}}
    tasks = [
        {"data": {"image_id": 1}, "annotations": [_ann([poly])]},
        {"data": {"image_id": 2}, "annotations": [_ann([{"type": "choices", "value": {"choices": ["No slate"]}}])]},
        {"data": {"image_id": 3}, "annotations": [_ann([], was_cancelled=True)]},
        {"data": {"image_id": 4}, "annotations": []},
    ]
    rows = {r["image_id"]: r for r in L.annotations_to_rows(tasks)}
    assert set(rows) == {1, 2}
    assert rows[1]["answer"] == "accept" and np.allclose(rows[1]["quad"][0], [160, 120.2])
    assert rows[2]["answer"] == "no_slate" and rows[2]["polygons"] is None


def test_hull_polygon_bridges_a_hand_and_keeps_edges():
    m = np.zeros((100, 140), bool)
    m[20:80, 20:120] = True
    m[20:35, 60:80] = False  # a hand over the top edge
    hull = np.array(L.hull_polygon(m))
    assert hull[:, 1].min() == 20 and hull[:, 0].min() == 20 and hull[:, 0].max() == 119
    assert len(hull) == 4  # straight edges collapse to the corners
