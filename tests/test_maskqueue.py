import json

import numpy as np
import pytest
from PIL import Image

from slate_detector import maskqueue as MQ
from slate_detector.segment import Candidates, save_candidates


def _rect(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_fit_quad_recovers_rectangle_corners_despite_a_notch():
    m = _rect((100, 120), 20, 30, 90, 80)
    m[30:40, 20:30] = False  # a hand over one corner
    quad = np.array(MQ.fit_quad(m))
    assert quad.shape == (4, 2)
    assert {(round(x / 10), round(y / 10)) for x, y in quad} >= {(9, 3), (9, 8), (2, 8)}


@pytest.fixture
def setup(tmp_path):
    shape = (60, 80)
    cache, masks = tmp_path / "frames", tmp_path / "masks"
    cache.mkdir()
    for iid in (7, 8):
        Image.new("RGB", (shape[1], shape[0])).save(cache / f"{iid}.jpg")
    save_candidates(masks / "7.npz", Candidates(np.stack([_rect(shape, 5, 5, 20, 15), _rect(shape, 30, 30, 70, 55)]), np.array([0.9, 0.6])), 0, "top_score")
    save_candidates(masks / "8.npz", Candidates(np.zeros((0, *shape), bool), np.array([])), -1, "none")
    page = MQ.build(tmp_path / "q", [7, 8], cache, masks, "mq", "chris", tmp_path / "m.json", seed=0)
    return tmp_path, masks, page


def test_build_writes_overlays_and_manifest(setup):
    tmp_path, _, page = setup
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert sorted(m["image_id"] for m in manifest["items"]) == [7, 8]
    by = {m["image_id"]: m for m in manifest["items"]}
    pid = by[7]["id"]
    assert (tmp_path / "q" / "ov" / f"{pid}_1.png").is_file() and by[8]["n"] == 0
    assert "__DATA__" not in page.read_text()


def test_ingest_uses_switched_candidate(setup):
    tmp_path, masks, _ = setup
    manifest = json.loads((tmp_path / "m.json").read_text())
    by = {m["image_id"]: m["id"] for m in manifest["items"]}
    answers = {"queue_id": "mq", "answers": {by[7]: {"answer": "accept", "instance": 1}, by[8]: {"answer": "reject"}}}
    rows = {r["image_id"]: r for r in MQ.ingest(answers, manifest, masks)}
    assert rows[7]["instance"] == 1 and rows[7]["auto_instance"] == 0
    xs = [p[0] for p in rows[7]["polygons"][0]]
    assert min(xs) == 30 and rows[7]["quad"] is not None
    assert rows[8]["answer"] == "reject" and rows[8]["polygons"] is None
