import json

from slate_detector import queue as Q
from slate_detector.dataset import build_frames, read_manual


def _pred(i, flag):
    return {"image_id": str(i), "dive_id": "3", "scan_group": "no_template", "flag": flag, "p_slate": "0.9" if flag else "0.0"}


def test_select_items_keeps_all_flagged_and_samples_checks():
    preds = [_pred(i, "likely_slate") for i in range(5)] + [_pred(i, "") for i in range(5, 50)]
    items = Q.select_items(preds, checks=10, seed=0)
    assert len(items) == 15 and sum(bool(r["flag"]) for r in items) == 5
    assert Q.select_items(preds, checks=10, seed=0) == items


def test_build_hides_scores_and_ingest_round_trips(tmp_path):
    cache = tmp_path / "frames"
    cache.mkdir()
    for i in (1, 2, 3):
        (cache / f"{i}.jpg").write_bytes(b"jpg")
    items = [_pred(1, "likely_slate"), _pred(2, ""), _pred(3, "uncertain")]
    page = Q.build(tmp_path / "q", items, cache, "q1", "chris", tmp_path / "m.json")
    html = page.read_text()
    assert "p_slate" not in html and "likely_slate" not in html  # blind
    manifest = json.loads((tmp_path / "m.json").read_text())
    answers = {"queue_id": "q1", "answers": {"i0001": {"answer": "slate"}, "i0003": {"answer": "cant_tell"}}}
    rows = Q.ingest(answers, manifest)
    assert [(r["image_id"], r["answer"]) for r in rows] == [(1, "slate"), (3, "cant_tell")]
    Q.write_labels(rows, tmp_path / "manual" / "chris" / "q1.csv")
    assert read_manual(tmp_path / "manual") == {1: "slate", 3: "cant_tell"}


def test_manual_answers_label_frames_and_overrides_win():
    row = lambda i, a: {"image_id": str(i), "checksum": "c", "path": "p", "dive_id": "1", "camera_id": "1",  # noqa: E731
                        "dive_has_slate": "0", "answers": a}
    rows = [row(1, "Manual (slate)"), row(2, "Manual (no_slate)"), row(3, "Fish, Hogfish"), row(4, "Manual (cant_tell)")]
    frames, dropped = build_frames(rows, overrides={3: 1}, manual={1: "slate", 2: "no_slate", 3: "no_slate", 4: "cant_tell"})
    assert [(f.image_id, f.label) for f in frames] == [(1, 1), (2, 0), (3, 1)]
    assert dropped == 1
