from slate_detector.cli import fetch_order
from slate_detector.scan import flag, select_scan, spread_sample


def _row(i, dive, group):
    return {"image_id": str(i), "checksum": f"c{i}", "path": f"{i}.ORF", "dive_id": str(dive),
            "camera_id": "1", "dive_has_slate": "0", "scan_group": group}


def test_select_scan_takes_all_priority_groups_in_order_then_sample():
    rows = [_row(1, 9, "no_template"), _row(2, 5, "checkerboard_dive"), _row(3, 4, "slate_labelled_dive"),
            _row(4, 3, "template_no_slate_labels"), _row(5, 9, "no_template"), _row(6, 8, "no_template")]
    frames = select_scan(rows, sample=2)
    assert [f.image_id for f in frames[:3]] == [4, 3, 2]
    assert len(frames) == 5 and {f.dive_id for f in frames[3:]} == {8, 9}  # one per dive before a second
    assert all(f.label == -1 for f in frames)


def test_spread_sample_is_seeded_and_round_robin():
    frames = select_scan([_row(i, i % 3, "no_template") for i in range(30)], sample=0)
    pool = [f for f in select_scan([_row(i, i % 3, "no_template") for i in range(30)], sample=30)]
    a, b = spread_sample(pool, 6, seed=1), spread_sample(pool, 6, seed=1)
    assert a == b and sorted(f.dive_id for f in a) == [0, 0, 1, 1, 2, 2]
    assert frames == []


def test_flags():
    assert flag(0.9) == "likely_slate" and flag(0.5) == "likely_slate"
    assert flag(0.2) == "uncertain" and flag(0.01) == ""


def test_fetch_order_keeps_unlabelled_frames_last_in_manifest_order():
    frames = select_scan([_row(7, 1, "checkerboard_dive"), _row(6, 2, "template_no_slate_labels")], sample=0)
    assert [f.image_id for f in fetch_order(frames)] == [6, 7]
