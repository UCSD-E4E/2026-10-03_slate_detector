"""Slate outlines from existing slate labels, with no new annotation.

A completed ``diveslatelabel`` row holds the points a labeller clicked on the
photo, positionally matched to the slate template's ``diveslate.reference_points``
(after removing ``skipped_points``). A homography fitted to those pairs maps
the template page into the photo, and warping the page's four corners gives
the slate's outline as a quad. This is the approach
2026-07-31_slate_training/scripts/make_masks.py took.

Coordinates, checked against the 2026-10-02 backup:

- template points are pixels of the PDF page rendered at ``diveslate.dpi``;
- label points are pixels of the rectified 4014x3016 photo. The July corpus
  stored them in Label Studio's composite canvas (offset by the template
  panel's width), but no row in this backup does (max x 2554), so no repair
  is applied. ``fit_quad`` refuses points outside the photo rather than
  guessing.

Quads are scaled to the 1600-px frame cache (``data/frames``), so they line
up with what the models see. ``rms_px`` is the fit's reprojection error in
photo pixels: a large value means mis-clicked or mis-ordered points.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PHOTO_W, PHOTO_H = 4014, 3016

LABELS_SQL = """
SELECT l.image_id, l.reference_points::text AS points, l.skipped_points::text AS skipped,
       ds.id AS slate_id, ds.name AS slate, ds.path AS slate_pdf, ds.dpi,
       ds.reference_points::text AS template_points, i.dive_id
FROM diveslatelabel l
JOIN image i ON i.id = l.image_id
JOIN dive d ON d.id = i.dive_id
JOIN diveslate ds ON ds.id = d.dive_slate_id
WHERE l.completed AND NOT coalesce(l.superseded, false)
  AND json_typeof(l.reference_points) = 'array' AND i.is_canonical
ORDER BY l.image_id
"""


def drop_skipped(points: list, skipped: list[int]) -> list:
    """Remove ``skipped`` indices, each read against the original list."""
    drop = {int(i) for i in skipped}
    if any(not 0 <= i < len(points) for i in drop):
        raise ValueError(f"skipped index out of range: {sorted(drop)} for {len(points)} points")
    return [p for i, p in enumerate(points) if i not in drop]


@dataclass(frozen=True)
class SlateQuad:
    image_id: int
    dive_id: int
    slate: str
    quad: list[list[float]]  # 4 corners in cache pixels, page order: TL, TR, BR, BL of the template
    rms_px: float  # reprojection error of the label points, photo pixels
    n_points: int


def fit_quad(template_pts, label_pts, skipped, page_size_px: tuple[float, float]) -> tuple[np.ndarray, float]:
    """(4x2 page corners in photo pixels, reprojection RMS in photo pixels)."""
    import cv2  # pylint: disable=import-outside-toplevel

    src = np.array(drop_skipped(template_pts, skipped), dtype=np.float64)
    dst = np.array(label_pts, dtype=np.float64)
    if len(src) != len(dst):
        raise ValueError(f"{len(dst)} label points for {len(src)} visible template points")
    if len(src) < 4:
        raise ValueError(f"only {len(src)} points; a homography needs 4")
    if (dst[:, 0] >= PHOTO_W).any() or (dst[:, 1] >= PHOTO_H).any() or (dst < 0).any():
        raise ValueError("label points outside the photo (composite-canvas coordinates?)")
    h, _ = cv2.findHomography(src, dst, method=0)
    if h is None:
        raise ValueError("degenerate homography")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
    rms = float(np.sqrt(((proj - dst) ** 2).sum(axis=1).mean()))
    w, hgt = page_size_px
    corners = np.array([[0, 0], [w, 0], [w, hgt], [0, hgt]], dtype=np.float64).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, h).reshape(4, 2)
    if not np.isfinite(quad).all():
        raise ValueError("homography sends the page to infinity")
    return quad, rms


def page_size_px(pdf: Path, dpi: int) -> tuple[float, float]:
    import pymupdf  # pylint: disable=import-outside-toplevel

    with pymupdf.open(pdf) as doc:
        rect = doc.load_page(0).rect
    return rect.width / 72 * dpi, rect.height / 72 * dpi


def build_quads(rows: list[dict[str, str]], pdf_for: dict[int, Path], cache_w: int = 1600) -> tuple[list[SlateQuad], list[dict]]:
    """(quads, failures). ``pdf_for`` maps slate_id to a local copy of its PDF."""
    scale = cache_w / PHOTO_W
    sizes: dict[int, tuple[float, float]] = {}
    quads, failures = [], []
    for r in rows:
        sid = int(r["slate_id"])
        if sid not in sizes:
            sizes[sid] = page_size_px(pdf_for[sid], int(r["dpi"]))
        skipped = json.loads(r["skipped"]) if r["skipped"] else None
        try:
            quad, rms = fit_quad(json.loads(r["template_points"]), json.loads(r["points"]),
                                 skipped if isinstance(skipped, list) else [], sizes[sid])
        except ValueError as exc:
            failures.append({"image_id": int(r["image_id"]), "error": str(exc)})
            continue
        quads.append(SlateQuad(
            image_id=int(r["image_id"]), dive_id=int(r["dive_id"]), slate=r["slate"],
            quad=[[round(float(x) * scale, 2), round(float(y) * scale, 2)] for x, y in quad],
            rms_px=round(rms, 2), n_points=len(json.loads(r["points"])),
        ))
    return quads, failures


def write_quads(quads: list[SlateQuad], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(q) for q in quads], indent=1) + "\n")


def read_quads(path: Path) -> list[SlateQuad]:
    return [SlateQuad(**q) for q in json.loads(Path(path).read_text())]


def overlay_sheet(quads: list[SlateQuad], cache: Path, out: Path, cols: int = 5, width: int = 400) -> Path:
    """Contact sheet: each frame with its quad drawn on (corner 0 marked), for checking by eye."""
    from PIL import Image, ImageDraw  # pylint: disable=import-outside-toplevel

    height = int(width * 3016 / 4014)
    rows = (len(quads) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (height + 20)), "white")
    for i, q in enumerate(quads):
        im = Image.open(Path(cache) / f"{q.image_id}.jpg").convert("RGB")
        s = width / im.width
        im = im.resize((width, height))
        d = ImageDraw.Draw(im)
        pts = [(x * s, y * s) for x, y in q.quad]
        d.line(pts + [pts[0]], fill=(255, 40, 40), width=2)
        d.ellipse([pts[0][0] - 4, pts[0][1] - 4, pts[0][0] + 4, pts[0][1] + 4], fill=(255, 220, 0))
        x, y = (i % cols) * width, (i // cols) * (height + 20)
        sheet.paste(im, (x, y + 20))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), f"{q.image_id} d{q.dive_id} {q.slate} rms={q.rms_px:.1f}px", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out
