import hashlib

import pytest

from slate_detector.raws import ChecksumMismatch, fetch_raw


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _nas(tmp_path, content: bytes):
    mount = tmp_path / "mnt"
    f = mount / "fishsense_data/REEF/data/dive/P1.ORF"
    f.parent.mkdir(parents=True)
    f.write_bytes(content)
    return mount


def test_copies_from_nas_and_verifies(tmp_path):
    mount = _nas(tmp_path, b"raw bytes")
    got = fetch_raw(tmp_path / "raw", 5, "dive/P1.ORF", _md5(b"raw bytes"), local_sources=(), mount=mount)
    assert got == tmp_path / "raw/5.ORF" and got.read_bytes() == b"raw bytes"


def test_rejects_nas_file_with_wrong_checksum(tmp_path):
    mount = _nas(tmp_path, b"corrupt")
    with pytest.raises(ChecksumMismatch):
        fetch_raw(tmp_path / "raw", 5, "dive/P1.ORF", _md5(b"raw bytes"), local_sources=(), mount=mount)
    assert not list((tmp_path / "raw").iterdir())


def test_hardlinks_verified_local_copy_without_touching_nas(tmp_path):
    src = tmp_path / "other_cache"
    src.mkdir()
    (src / "5.ORF").write_bytes(b"raw bytes")
    got = fetch_raw(tmp_path / "raw", 5, "dive/P1.ORF", _md5(b"raw bytes"), local_sources=(src,), mount=tmp_path / "no_mount")
    assert got.read_bytes() == b"raw bytes"
    assert got.stat().st_ino == (src / "5.ORF").stat().st_ino


def test_skips_local_copy_with_wrong_checksum(tmp_path):
    src = tmp_path / "other_cache"
    src.mkdir()
    (src / "5.ORF").write_bytes(b"stale")
    mount = _nas(tmp_path, b"raw bytes")
    got = fetch_raw(tmp_path / "raw", 5, "dive/P1.ORF", _md5(b"raw bytes"), local_sources=(src,), mount=mount)
    assert got.read_bytes() == b"raw bytes"
