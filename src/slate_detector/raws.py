"""A local copy of every raw .ORF, so re-rendering never touches the NAS again.

Raws live at ``data/raw/{image_id}.ORF``. Each one is verified against
``image.checksum`` (md5) before it lands. Sources, in order:

1. a local directory that already holds ``{image_id}.ORF`` files (e.g. the
   cscw-fishsense2027 cache). The file is hardlinked, so it costs no disk,
   and our copy survives if that cache is cleared;
2. the NAS mount, at ~15 s per 17 MB file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

MOUNT = Path("~/mnt").expanduser()
LOCAL_SOURCES = (Path("~/.cache/cscw-fishsense2027/tail_stage").expanduser(),)


class ChecksumMismatch(RuntimeError):
    pass


def raw_path(path: str, mount: Path = MOUNT) -> Path:
    """Where a v1 ``image.path`` lives on the NAS mount.

    Most paths are relative to ``fishsense_data/REEF/data``; older pool dives
    store absolute ``/fishsense_data/...`` paths.
    """
    if path.startswith("/fishsense_data/"):
        return mount / path.lstrip("/")
    return mount / "fishsense_data" / "REEF" / "data" / path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_raw(
    dest_dir: Path,
    image_id: int,
    nas_path: str,
    checksum: str,
    local_sources: tuple[Path, ...] = LOCAL_SOURCES,
    mount: Path = MOUNT,
) -> Path:
    """The verified local raw for ``image_id``, fetching it if needed."""
    dest = Path(dest_dir) / f"{image_id}.ORF"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for src_dir in local_sources:
        src = Path(src_dir) / f"{image_id}.ORF"
        if src.is_file() and md5(src) == checksum:
            try:
                os.link(src, dest)
            except OSError:  # another filesystem
                shutil.copyfile(src, dest)
            return dest
    tmp = dest.with_suffix(".tmp")
    shutil.copyfile(raw_path(nas_path, mount), tmp)
    got = md5(tmp)
    if got != checksum:
        tmp.unlink()
        raise ChecksumMismatch(f"image {image_id}: NAS file md5 {got} != image.checksum {checksum}")
    tmp.replace(dest)
    return dest
