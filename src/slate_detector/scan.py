"""Run the trained model over unlabelled frames and flag the ones worth labelling.

The scan set is canonical frames with no completed species or slate label,
grouped by how likely their dive is to contain a slate:

- ``template_no_slate_labels``: the dive has a slate template assigned but no
  frame answered ``Slate``. Slate frames are probably sitting unlabelled here.
- ``slate_labelled_dive``: the rest of the dives that already have slate labels.
- ``checkerboard_dive``: checkerboard calibration dives (hard negatives).
- ``no_template``: everything else. This set is too big to scan (~60k frames
  at ~11 s each), so a seeded sample is spread evenly across these dives. It
  measures how often the model fires where we expect no slate.

Frames carry ``label = -1`` (unknown) and the group name in ``answers``.
"""

from __future__ import annotations

import csv
import html
import random
from itertools import zip_longest
from pathlib import Path

from slate_detector.dataset import Frame

SCAN_SQL = """
WITH labelled AS (
    SELECT image_id FROM specieslabel WHERE completed
    UNION
    SELECT image_id FROM diveslatelabel WHERE completed
),
slate_dives AS (
    SELECT DISTINCT i.dive_id
    FROM specieslabel s JOIN image i ON i.id = s.image_id
    WHERE s.completed AND s.content_of_image LIKE 'Slate%'
)
SELECT i.id AS image_id, i.checksum, i.path, i.dive_id, i.camera_id,
       (d.dive_slate_id IS NOT NULL)::int AS dive_has_slate,
       CASE
           WHEN d.dive_slate_id IS NOT NULL AND i.dive_id NOT IN (SELECT dive_id FROM slate_dives)
               THEN 'template_no_slate_labels'
           WHEN d.dive_slate_id IS NOT NULL THEN 'slate_labelled_dive'
           WHEN d.calibration_target_id IS NOT NULL THEN 'checkerboard_dive'
           ELSE 'no_template'
       END AS scan_group
FROM image i JOIN dive d ON d.id = i.dive_id
WHERE i.is_canonical AND NOT EXISTS (SELECT 1 FROM labelled l WHERE l.image_id = i.id)
ORDER BY i.id
"""

FULL_GROUPS = ("template_no_slate_labels", "slate_labelled_dive", "checkerboard_dive")
SAMPLED_GROUP = "no_template"


def _frame(row: dict[str, str]) -> Frame:
    return Frame(
        image_id=int(row["image_id"]),
        checksum=row["checksum"],
        path=row["path"],
        dive_id=int(row["dive_id"]),
        camera_id=int(row["camera_id"]) if row["camera_id"] else -1,
        label=-1,
        answers=row["scan_group"],
        dive_has_slate=int(row["dive_has_slate"]),
    )


def spread_sample(frames: list[Frame], n: int, seed: int) -> list[Frame]:
    """``n`` frames taken round-robin over dives, each dive's frames in seeded random order."""
    rng = random.Random(seed)
    by_dive: dict[int, list[Frame]] = {}
    for f in frames:
        by_dive.setdefault(f.dive_id, []).append(f)
    for dive in sorted(by_dive):
        rng.shuffle(by_dive[dive])
    ordered = [f for batch in zip_longest(*(by_dive[d] for d in sorted(by_dive))) for f in batch if f is not None]
    return ordered[:n]


def select_scan(rows: list[dict[str, str]], sample: int, seed: int = 0) -> list[Frame]:
    """Every frame of the slate and checkerboard groups (in that priority), then the spread sample."""
    frames = [_frame(r) for r in rows]
    chosen = [f for g in FULL_GROUPS for f in frames if f.answers == g]
    return chosen + spread_sample([f for f in frames if f.answers == SAMPLED_GROUP], sample, seed)


def fetch_scan(runner, sample: int, seed: int = 0) -> list[Frame]:
    return select_scan(runner.query(SCAN_SQL), sample, seed)


# Flags. ``likely_slate`` frames need slate labels (reference points) for
# calibration; ``uncertain`` ones are where a label teaches the model most.
LIKELY = 0.5
UNCERTAIN = 0.05


def flag(p: float) -> str:
    if p >= LIKELY:
        return "likely_slate"
    if p >= UNCERTAIN:
        return "uncertain"
    return ""


def write_predictions(frames: list[Frame], probs, path: Path) -> list[dict]:
    rows = [
        {"image_id": f.image_id, "dive_id": f.dive_id, "scan_group": f.answers, "p_slate": f"{p:.5f}",
         "flag": flag(float(p)), "path": f.path}
        for f, p in zip(frames, probs)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def write_review(rows: list[dict], cache: Path, out_dir: Path, dive_names: dict[int, str]) -> Path:
    """Flagged frames as CSV plus a local HTML contact sheet, grouped by dive.

    The page is meant to be opened from disk, never published: the frames show
    identifiable divers. Thumbnails sit next to it in ``thumbs/``.
    """
    from PIL import Image  # pylint: disable=import-outside-toplevel

    flagged = sorted((r for r in rows if r["flag"]), key=lambda r: (int(r["dive_id"]), -float(r["p_slate"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "to_label.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "dive_id", "dive", "scan_group", "flag", "p_slate", "path"])
        w.writeheader()
        for r in flagged:
            w.writerow({**r, "dive": dive_names.get(int(r["dive_id"]), "")})

    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for r in flagged:
        t = thumbs / f"{r['image_id']}.jpg"
        if not t.is_file():
            im = Image.open(Path(cache) / f"{r['image_id']}.jpg")
            im.thumbnail((480, 480))
            im.save(t, quality=85)

    by_dive: dict[int, list[dict]] = {}
    for r in flagged:
        by_dive.setdefault(int(r["dive_id"]), []).append(r)
    total = {g: sum(r["scan_group"] == g for r in rows) for g in {r["scan_group"] for r in rows}}
    counts = {k: sum(r["flag"] == k for r in rows) for k in ("likely_slate", "uncertain")}

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slate Review</title>
<style>
:root {{ --bg:#fff; --fg:#1b1b1b; --muted:#666; --card:#f4f4f4; --likely:#0a7d32; --uncertain:#a15c00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --muted:#999; --card:#222; --likely:#4cc477; --uncertain:#e0a040; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:0 16px 40px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:15px; margin:28px 0 8px; }}
.meta {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:8px; }}
figure {{ margin:0; background:var(--card); border-radius:6px; overflow:hidden; }}
img {{ width:100%; display:block; }}
figcaption {{ padding:4px 6px; font-size:12px; }}
.likely_slate b {{ color:var(--likely); }} .uncertain b {{ color:var(--uncertain); }}
</style></head><body>
<h1>Slate review: frames to label</h1>
<p class="meta">{counts['likely_slate']} likely slate (p ≥ {LIKELY}), {counts['uncertain']} uncertain
({UNCERTAIN} ≤ p &lt; {LIKELY}), out of {len(rows)} scanned
({', '.join(f'{g}: {n}' for g, n in sorted(total.items()))}). List: <code>to_label.csv</code>.</p>"""]
    for dive, items in by_dive.items():
        name = html.escape(dive_names.get(dive, ""))
        parts.append(f'<h2>Dive {dive} <span class="meta">{name} · {items[0]["scan_group"]} · {len(items)} flagged</span></h2><div class="grid">')
        for r in items:
            parts.append(
                f'<figure class="{r["flag"]}"><img loading="lazy" src="thumbs/{r["image_id"]}.jpg" alt="image {r["image_id"]}">'
                f'<figcaption><b>p={float(r["p_slate"]):.2f}</b> · {r["flag"].replace("_", " ")} · id {r["image_id"]}</figcaption></figure>'
            )
        parts.append("</div>")
    parts.append("</body></html>")
    page = out_dir / "index.html"
    page.write_text("\n".join(parts))
    return page
