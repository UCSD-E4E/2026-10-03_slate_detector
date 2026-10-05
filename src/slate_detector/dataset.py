"""Labelled frames for slate presence, from the human Label Studio answers.

Ground truth (see README):

- positive: any ``specieslabel.content_of_image`` starting with ``Slate`` or a
  completed ``diveslatelabel`` row (a human placed slate reference points);
- negative: every answer on the frame is a fish, a fish model, a non-slate
  calibration target, or ``None``;
- anything else (e.g. a frame answered both ``Slate`` and ``Fish``) is
  dropped as ambiguous.

A checkerboard or Box calibration target is *not* a slate. Some 2023 Nathan's
Pool frames showing only the checkerboard were answered ``Slate``. Each of
those was checked by eye and flipped in ``labels/overrides.csv``
(image_id, label, reason), which wins over the rules above.

Answers from our own labelling queue (``labels/manual/<labeller>/*.csv``, see
``queue``) come next: ``slate`` is 1, ``no_slate`` is 0, and ``cant_tell`` frames
are left out. Precedence is overrides, then manual answers, then the rules.

Only canonical images are used: ``image.is_canonical`` picks one copy of each
frame, so each frame belongs to exactly one dive and splits on ``dive_id``
don't leak.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

LABELLED_SQL = """
WITH answers AS (
    SELECT s.image_id, s.content_of_image AS answer
    FROM specieslabel s
    WHERE s.completed AND NOT coalesce(s.superseded, false)
      AND coalesce(s.content_of_image, '') <> ''
    UNION ALL
    SELECT d.image_id, 'Slate (diveslatelabel)'
    FROM diveslatelabel d
    WHERE d.completed AND NOT coalesce(d.superseded, false)
)
SELECT i.id AS image_id, i.checksum, i.path, i.dive_id, i.camera_id,
       (dv.dive_slate_id IS NOT NULL)::int AS dive_has_slate,
       string_agg(DISTINCT a.answer, '|') AS answers
FROM answers a
JOIN image i ON i.id = a.image_id
JOIN dive dv ON dv.id = i.dive_id
WHERE i.is_canonical
GROUP BY i.id, dv.dive_slate_id
ORDER BY i.id
"""

NEGATIVE_PREFIXES = ("Fish", "Calibration Targets", "None")


def slate_label(answers: set[str]) -> int | None:
    """1 slate present, 0 absent, None ambiguous."""
    slate = {a for a in answers if a.startswith("Slate")}
    other = answers - slate
    if slate and not other:
        return 1
    if not slate and other and all(a.startswith(NEGATIVE_PREFIXES) for a in other):
        return 0
    return None


@dataclass(frozen=True)
class Frame:
    image_id: int
    checksum: str  # md5 of the raw file
    path: str
    dive_id: int
    camera_id: int
    label: int
    answers: str
    dive_has_slate: int  # the dive has a slate template assigned (weak, for analysis)


LABELS = Path(__file__).resolve().parents[2] / "labels"
OVERRIDES = LABELS / "overrides.csv"
MANUAL = LABELS / "manual"
MANUAL_LABEL = {"slate": 1, "no_slate": 0}

FRAMES_BY_ID_SQL = """
SELECT i.id AS image_id, i.checksum, i.path, i.dive_id, i.camera_id,
       (dv.dive_slate_id IS NOT NULL)::int AS dive_has_slate
FROM image i JOIN dive dv ON dv.id = i.dive_id
WHERE i.is_canonical AND i.id IN ({ids})
ORDER BY i.id
"""


def read_overrides(path: Path = OVERRIDES) -> dict[int, int]:
    if not path.is_file():
        return {}
    with open(path, newline="") as fh:
        return {int(r["image_id"]): int(r["label"]) for r in csv.DictReader(fh)}


def read_manual(directory: Path = MANUAL) -> dict[int, str]:
    """image_id -> answer from every queue CSV under ``directory``; later files win."""
    out: dict[int, str] = {}
    for path in sorted(Path(directory).glob("*/*.csv")):
        with open(path, newline="") as fh:
            out.update({int(r["image_id"]): r["answer"] for r in csv.DictReader(fh)})
    return out


def build_frames(
    rows: list[dict[str, str]],
    overrides: dict[int, int] | None = None,
    manual: dict[int, str] | None = None,
) -> tuple[list[Frame], int]:
    """(frames, number dropped as ambiguous).

    ``overrides`` maps image_id to a reviewed label; ``manual`` to a queue
    answer. A ``cant_tell`` answer with no other label leaves the frame out.
    """
    overrides, manual = overrides or {}, manual or {}
    frames, ambiguous = [], 0
    for row in rows:
        answers = set(row["answers"].split("|"))
        image_id = int(row["image_id"])
        if image_id in overrides:
            label = overrides[image_id]
        elif manual.get(image_id) in MANUAL_LABEL:
            label = MANUAL_LABEL[manual[image_id]]
        else:
            label = slate_label(answers)
        if label is None:
            ambiguous += 1
            continue
        frames.append(Frame(
            image_id=int(row["image_id"]),
            checksum=row["checksum"],
            path=row["path"],
            dive_id=int(row["dive_id"]),
            camera_id=int(row["camera_id"]) if row["camera_id"] else -1,
            label=label,
            answers="|".join(sorted(answers)),
            dive_has_slate=int(row["dive_has_slate"]),
        ))
    return frames, ambiguous


def fetch_frames(runner, overrides_path: Path = OVERRIDES, manual_dir: Path = MANUAL) -> tuple[list[Frame], int]:
    rows = runner.query(LABELLED_SQL)
    manual = read_manual(manual_dir)
    known = {int(r["image_id"]) for r in rows}
    extra = sorted(i for i, a in manual.items() if i not in known and a in MANUAL_LABEL)
    if extra:
        rows += [{**r, "answers": f"Manual ({manual[int(r['image_id'])]})"}
                 for r in runner.query(FRAMES_BY_ID_SQL.format(ids=",".join(map(str, extra))))]
    return build_frames(rows, read_overrides(overrides_path), manual)


def write_manifest(frames: list[Frame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Frame)])
        w.writeheader()
        w.writerows(asdict(f) for f in frames)


def read_manifest(path: Path) -> list[Frame]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ints = {f.name for f in fields(Frame) if f.type == "int"}
    return [Frame(**{k: int(v) if k in ints else v for k, v in r.items()}) for r in rows]
