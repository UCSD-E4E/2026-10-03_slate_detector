from pathlib import Path

from slate_detector.cli import fetch_order
from slate_detector.dataset import Frame, build_frames, slate_label
from slate_detector.raws import raw_path


def test_slate_answers_are_positive():
    assert slate_label({"Slate, Laser on slate"}) == 1
    assert slate_label({"Slate, Laser not on slate", "Slate (diveslatelabel)"}) == 1


def test_fish_targets_and_none_are_negative():
    assert slate_label({"Fish, Hogfish (Lachnolaimus maximus)"}) == 0
    assert slate_label({"Fish Model, Snook", "Calibration Targets, Box"}) == 0
    assert slate_label({"None"}) == 0


def test_mixed_or_unknown_answers_are_ambiguous():
    assert slate_label({"Slate, Laser on slate", "Fish, Other (Identifiable but Nontarget)"}) is None
    assert slate_label({"Something new"}) is None


def test_build_frames_drops_ambiguous():
    rows = [
        {"image_id": "1", "checksum": "c1", "path": "a.ORF", "dive_id": "7", "camera_id": "2", "dive_has_slate": "1",
         "answers": "Slate, Laser on slate"},
        {"image_id": "2", "checksum": "c2", "path": "b.ORF", "dive_id": "7", "camera_id": "", "dive_has_slate": "0",
         "answers": "Slate, Laser on slate|Fish"},
    ]
    frames, ambiguous = build_frames(rows)
    assert [(f.image_id, f.label, f.camera_id) for f in frames] == [(1, 1, 2)]
    assert ambiguous == 1


def test_overrides_win_over_answers():
    rows = [
        {"image_id": "1", "checksum": "c1", "path": "a.ORF", "dive_id": "7", "camera_id": "2", "dive_has_slate": "0",
         "answers": "Slate, Laser on slate"},
        {"image_id": "2", "checksum": "c2", "path": "b.ORF", "dive_id": "7", "camera_id": "2", "dive_has_slate": "0",
         "answers": "Slate, Laser on slate"},
    ]
    frames, _ = build_frames(rows, overrides={1: 0})
    assert [(f.image_id, f.label) for f in frames] == [(1, 0), (2, 1)]


def test_checked_in_overrides_parse():
    from slate_detector.dataset import read_overrides
    overrides = read_overrides()
    assert set(overrides.values()) == {0}
    assert {132606, 133795, 81964, 105340} <= set(overrides)  # checkerboard-only, Box, two queue mis-keys


def _frame(i, dive, label):
    return Frame(image_id=i, checksum="", path="", dive_id=dive, camera_id=1, label=label, answers="", dive_has_slate=0)


def test_fetch_order_positives_then_negatives_round_robin():
    frames = [_frame(1, 10, 0), _frame(2, 10, 0), _frame(3, 20, 0), _frame(4, 30, 1)]
    assert [f.image_id for f in fetch_order(frames)] == [4, 1, 3, 2]


def test_raw_path_handles_relative_and_absolute():
    mount = Path("/m")
    assert raw_path("2024.06.20.REEF/x.ORF", mount) == Path("/m/fishsense_data/REEF/data/2024.06.20.REEF/x.ORF")
    assert raw_path("/fishsense_data/__unsorted-data/x.ORF", mount) == Path("/m/fishsense_data/__unsorted-data/x.ORF")
