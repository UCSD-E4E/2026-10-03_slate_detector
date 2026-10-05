"""Frames rendered from the raw .ORF the way the rest of FishSense renders them.

fishsense-core ``RawImage`` (``DecodeConfig.production()``) then
``RectifiedImage`` with the camera's intrinsics from ``cameraintrinsics``.
This matches fishsense-detector's ``decode.py`` and the fishsense-lite
preprocess stages, so the classifier sees what everything downstream sees.
The camera's embedded JPEG preview is not used: it's rendered differently and
isn't rectified.

Each worker first makes sure the raw is in the local store
(``raws.fetch_raw``: ~15 s per file from the NAS), then decodes it (~18 s of
CPU). Running several workers overlaps one frame's decode with the next
frame's NAS read. The output is a rectified frame downscaled to
``CACHE_LONG_SIDE`` and stored as JPEG, keyed by image_id.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from slate_detector.raws import LOCAL_SOURCES, fetch_raw

CACHE_LONG_SIDE = 1600
JPEG_QUALITY = 95

INTRINSICS_SQL = "SELECT camera_id, camera_matrix, distortion_coefficients FROM cameraintrinsics"

Intrinsics = tuple[list, list]  # (camera_matrix, distortion_coefficients), picklable for workers


def load_intrinsics(runner) -> dict[int, Intrinsics]:
    return {
        int(r["camera_id"]): (json.loads(r["camera_matrix"]), json.loads(r["distortion_coefficients"]))
        for r in runner.query(INTRINSICS_SQL)
    }


def render_rgb(path: Path, intrinsics: Intrinsics):
    """Rectified RGB uint8 array (H, W, 3) at full resolution."""
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from fishsense_core.camera_intrinsics import CameraIntrinsics
    from fishsense_core.image.decode import DecodeConfig
    from fishsense_core.image.raw_image import RawImage
    from fishsense_core.image.rectified_image import RectifiedImage

    matrix, dist = intrinsics
    k = CameraIntrinsics(camera_matrix=np.array(matrix, dtype=float), distortion_coefficients=np.array(dist, dtype=float))
    bgr = RectifiedImage(RawImage(path, config=DecodeConfig.production()), k).data
    return np.ascontiguousarray(bgr[:, :, ::-1])


def render_to(out: Path, raw_file: Path, intrinsics: Intrinsics) -> None:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    im = Image.fromarray(render_rgb(raw_file, intrinsics))
    im.thumbnail((CACHE_LONG_SIDE, CACHE_LONG_SIDE), Image.Resampling.LANCZOS)
    tmp = out.with_suffix(".tmp")
    im.save(tmp, format="JPEG", quality=JPEG_QUALITY)
    tmp.replace(out)


def stage(out: Path, raw_dir: Path, item: tuple[int, str, str, int], intrinsics: Intrinsics,
          local_sources: tuple[Path, ...]) -> None:
    """Worker job: local raw (fetched and verified if missing), then the render if missing."""
    image_id, nas_path, checksum, _ = item
    raw_file = fetch_raw(raw_dir, image_id, nas_path, checksum, local_sources)
    if not out.is_file():
        render_to(out, raw_file, intrinsics)


class RenderCache:
    def __init__(self, directory: Path, raw_dir: Path, local_sources: tuple[Path, ...] = LOCAL_SOURCES):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path(raw_dir)
        self.local_sources = local_sources

    def path(self, image_id: int) -> Path:
        return self.directory / f"{image_id}.jpg"

    def has(self, image_id: int) -> bool:
        return self.path(image_id).is_file()

    def has_raw(self, image_id: int) -> bool:
        return (self.raw_dir / f"{image_id}.ORF").is_file()

    def prefetch(
        self,
        items: list[tuple[int, str, str, int]],
        intrinsics: dict[int, Intrinsics],
        workers: int = 3,
        on_done: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Fetch the raw and render every (image_id, nas_path, checksum, camera_id) missing either, in order.

        Returns failures instead of raising. A worker killed for memory breaks
        the pool; the rest of the batch is then recorded as failed, and a
        rerun picks it up, since cached frames are skipped.
        """
        todo = [it for it in items if not (self.has(it[0]) and self.has_raw(it[0]))]
        failures: list[dict] = []
        # spawn, not fork: workers don't inherit the parent's frame lists.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}
            for item in todo:
                image_id, camera_id = item[0], item[3]
                if camera_id not in intrinsics:
                    failures.append({"image_id": image_id, "error": f"no intrinsics for camera {camera_id}"})
                    continue
                job = pool.submit(stage, self.path(image_id), self.raw_dir, item, intrinsics[camera_id], self.local_sources)
                futures[job] = image_id
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except BrokenProcessPool:
                    failures.append({"image_id": futures[future], "error": "worker killed (likely out of memory)"})
                except Exception as exc:  # pylint: disable=broad-except
                    failures.append({"image_id": futures[future], "error": repr(exc)})
                if on_done:
                    on_done(n, len(futures))
        return failures
