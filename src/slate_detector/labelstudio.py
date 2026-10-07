"""Refine SAM 3.1 slate masks in the lab's hosted Label Studio (needs the ``ls`` extra).

Connects the way fishsense-lite and fishsense-services do (their
``populate_utils`` / ``labels.label_studio``):

- hosted Label Studio Enterprise at ``app.heartex.com``, the ``FishSense``
  workspace, the ``label_studio_sdk`` client with a refresh-token API key;
- images are never uploaded to Label Studio. Frames go to the lab's Garage
  S3 (``s3.e4e.ucsd.edu``), tasks reference them as ``s3://`` URIs, and an S3
  *source* storage registered on the project lets Label Studio presign them
  when serving (registered only to presign; never synced);
- model output goes in as *predictions*, so every annotation is the
  labeller's own. ``import_tasks`` with inline predictions also sets the
  project's ``model_version``, without which Label Studio hides predictions
  (fishsense-lite, dive 94, 2026-09-10).

The prediction is the 4-corner outline fitted to SAM's chosen candidate
(``maskqueue.fit_quad``): a 4-point polygon is quick to drag, where SAM's own
edge has hundreds of vertices. A frame with no candidate gets no
prediction, so the labeller draws it.

Credentials come from ``~/.config/slate-detector/secrets.env`` and are never
printed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

SECRETS = Path("~/.config/slate-detector/secrets.env").expanduser()
MODEL_VERSION = "sam3.1-whiteboard-quad-v1"  # first import (2026-10-06); kept on the tasks
HULL_VERSION = "sam3.1-whiteboard-hull-v1"
RECT_V1_VERSION = "sam3.1-board-rect-v1"  # first rectangle fit; some portrait boards fitted as 60-deg-tilted landscape
RECT_V2_VERSION = "sam3.1-board-rect-v2"  # all corner assignments + least tilt among near-ties
RECT_VERSION = "sam3.1-board-rect-v3"  # rounded-rectangle pose fit (rectfit), all corner assignments + tilt prior; hull where implausible
STORAGE_TITLE = "garage-slate-detector"
LABEL = "Slate"

LABEL_CONFIG = f"""<View>
  <Header value="Outline the slate board (H, Tic-Tac-Toe or V). Drag corners to fit; include parts hidden by hands. Not checkerboards or the Box target."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true" contrastControl="true"/>
  <PolygonLabels name="slate" toName="image" strokeWidth="2" pointSize="small" opacity="0.2">
    <Label value="{LABEL}" background="#22c55e"/>
  </PolygonLabels>
  <Choices name="flags" toName="image" choice="multiple" showInline="true">
    <Choice value="Can't tell"/>
    <Choice value="No slate"/>
  </Choices>
</View>"""


def load_secrets(path: Path = SECRETS) -> dict[str, str]:
    """KEY=VALUE lines; blanks and comments skipped. Raises naming missing keys, never values."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v.strip():
            out[k.strip()] = v.strip()
    return out


def require(secrets: dict[str, str], *keys: str) -> None:
    missing = [k for k in keys if not secrets.get(k)]
    if missing:
        raise SystemExit(f"fill in {', '.join(missing)} in {SECRETS}")


def frame_key(prefix: str, image_id: int) -> str:
    return f"{prefix.strip('/')}/frames/{image_id}.jpg" if prefix.strip("/") else f"frames/{image_id}.jpg"


def to_percent(points, width: int, height: int) -> list[list[float]]:
    return [[round(100.0 * x / width, 4), round(100.0 * y / height, 4)] for x, y in points]


def from_percent(points, width: int, height: int) -> list[list[float]]:
    return [[x * width / 100.0, y * height / 100.0] for x, y in points]


def prediction(points_px, width: int, height: int, score: float, model_version: str = MODEL_VERSION) -> dict:
    return {
        "model_version": model_version,
        "score": float(score),
        "result": [{
            "id": "sam3quad",
            "from_name": "slate", "to_name": "image", "type": "polygonlabels",
            "original_width": width, "original_height": height, "image_rotation": 0,
            "value": {"points": to_percent(points_px, width, height), "polygonlabels": [LABEL], "closed": True},
        }],
    }


def build_task(image_id: int, dive_id: int, url: str, width: int, height: int, cands_path: Path) -> dict:
    """One task: the frame's s3 URL, ids for sorting/joining, and the SAM quad as a prediction if any."""
    from slate_detector.maskqueue import fit_quad, mask_polygons  # pylint: disable=import-outside-toplevel
    from slate_detector.segment import load_candidates  # pylint: disable=import-outside-toplevel

    task: dict = {"data": {"image": url, "image_id": int(image_id), "dive_id": int(dive_id)}}
    c, chosen, _ = load_candidates(cands_path)
    if chosen >= 0:
        mask = c.masks[chosen]
        pts = fit_quad(mask)
        if pts is None:
            polys = mask_polygons(mask, eps=2.0)
            pts = polys[0] if polys else None
        if pts is not None:
            task["predictions"] = [prediction(pts, width, height, float(c.scores[chosen]))]
    return task


def hull_polygon(mask: np.ndarray, eps: float = 1.0) -> list[list[float]] | None:
    """SAM's mask outline, convex: tight wherever the board is visible, straight across hands.

    The 4-corner quad (``maskqueue.fit_quad``) is easier to drag but cuts across
    the corners of small, soft boards and can come out rotated on V-slates; the
    labeller found it looser than SAM's own edge. The hull keeps SAM's edge
    and still spans occluders, which matches labelling the whole board.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    return cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(float).tolist()


def hull_prediction(cands_path: Path, width: int, height: int) -> dict | None:
    from slate_detector.segment import load_candidates  # pylint: disable=import-outside-toplevel

    c, chosen, _ = load_candidates(cands_path)
    if chosen < 0:
        return None
    pts = hull_polygon(c.masks[chosen])
    if not pts or len(pts) < 3:
        return None
    return prediction(pts, width, height, float(c.scores[chosen]), model_version=HULL_VERSION)


def rect_prediction(rect: dict | None, cands_path: Path, width: int, height: int) -> tuple[dict | None, bool]:
    """(prediction, fell back to the hull). The fitted board outline, clipped to the frame."""
    from slate_detector.rectfit import clip_to_frame  # pylint: disable=import-outside-toplevel
    from slate_detector.segment import load_candidates  # pylint: disable=import-outside-toplevel

    if rect and rect.get("plausible"):
        poly = clip_to_frame(np.array(rect["polygon"]), width, height)
        if len(poly) >= 3:
            c, chosen, _ = load_candidates(cands_path)
            return prediction(poly.tolist(), width, height, float(c.scores[chosen]), model_version=RECT_VERSION), False
    hull = hull_prediction(cands_path, width, height)
    if hull is not None:
        hull["model_version"] = RECT_VERSION
    return hull, hull is not None


def call(fn, *args, attempts: int = 5, **kwargs):
    """``fn(*args, **kwargs)``, backing off through Label Studio's 429 throttle (it names the wait)."""
    import re  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    from label_studio_sdk.core import ApiError  # pylint: disable=import-outside-toplevel

    for n in range(attempts):
        try:
            return fn(*args, **kwargs)
        except ApiError as exc:
            if getattr(exc, "status_code", None) != 429 or n == attempts - 1:
                raise
            m = re.search(r"(\d+) seconds", str(getattr(exc, "body", "")))
            time.sleep((int(m.group(1)) if m else 30) + 2)
    return None


def annotations_to_rows(tasks: Iterable[dict]) -> list[dict]:
    """Each task's latest submitted (not cancelled) annotation, as rows in frame-cache pixels.

    ``answer`` is ``accept`` when a slate polygon was submitted, otherwise
    ``no_slate`` or ``cant_tell`` from the flags, or ``empty``.
    """
    rows = []
    for t in tasks:
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if not anns:
            continue
        a = max(anns, key=lambda x: (x.get("updated_at") or "", x.get("id") or 0))
        polys, flags = [], set()
        for r in a.get("result") or []:
            if r.get("type") == "polygonlabels" and r.get("from_name") == "slate":
                w, h = r["original_width"], r["original_height"]
                polys.append(from_percent(r["value"]["points"], w, h))
            elif r.get("type") == "choices":
                flags |= set(r["value"].get("choices", []))
        if polys:
            answer = "accept"
        elif "No slate" in flags:
            answer = "no_slate"
        elif "Can't tell" in flags:
            answer = "cant_tell"
        else:
            answer = "empty"
        quad = polys[0] if len(polys) == 1 and len(polys[0]) == 4 else None
        rows.append({
            "image_id": int(t["data"]["image_id"]), "answer": answer, "polygons": polys or None, "quad": quad,
            "flags": sorted(flags), "annotation_id": a.get("id"), "completed_by": a.get("completed_by"),
            "updated_at": a.get("updated_at"),
        })
    return rows


class Garage:
    def __init__(self, secrets: dict[str, str]):
        import boto3  # pylint: disable=import-outside-toplevel

        self.bucket = secrets["SLATE_S3_BUCKET"]
        self.prefix = secrets.get("SLATE_S3_PREFIX", "")
        self.s3 = boto3.client(
            "s3", endpoint_url=secrets["SLATE_S3_ENDPOINT"], region_name=secrets.get("SLATE_S3_REGION", "garage"),
            aws_access_key_id=secrets["SLATE_S3_ACCESS_KEY"], aws_secret_access_key=secrets["SLATE_S3_SECRET_KEY"],
        )

    def url(self, image_id: int) -> str:
        return f"s3://{self.bucket}/{frame_key(self.prefix, image_id)}"

    def existing(self) -> set[str]:
        keys = set()
        pages = self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=frame_key(self.prefix, 0).rsplit("/", 1)[0] + "/")
        for page in pages:
            keys |= {o["Key"] for o in page.get("Contents", [])}
        return keys

    def upload(self, image_id: int, path: Path) -> None:
        self.s3.upload_file(str(path), self.bucket, frame_key(self.prefix, image_id), ExtraArgs={"ContentType": "image/jpeg"})


def client(secrets: dict[str, str]):
    from label_studio_sdk.client import LabelStudio  # pylint: disable=import-outside-toplevel

    return LabelStudio(base_url=secrets["FISHSENSE_LABEL_STUDIO_URL"], api_key=secrets["FISHSENSE_LABEL_STUDIO_API_KEY"])


def workspace_id(ls, name: str) -> int | None:
    if not name:
        return None
    matches = [w for w in ls.workspaces.list() if w.title == name]
    if not matches:
        raise SystemExit(f"Label Studio workspace {name!r} not found")
    return matches[0].id


def get_or_create_project(ls, title: str, ws: int | None) -> tuple[int, bool]:
    projects = ls.projects.list(workspaces=[ws]) if ws is not None else ls.projects.list()
    found = [p for p in projects if p.title == title]
    if found:
        return found[0].id, False
    p = ls.projects.create(title=title, label_config=LABEL_CONFIG, workspace=ws)
    return p.id, True


def ensure_storage(ls, project_id: int, secrets: dict[str, str]) -> bool:
    """Register the Garage source storage (presign only). True if newly created."""
    bucket, prefix = secrets["SLATE_S3_BUCKET"], secrets.get("SLATE_S3_PREFIX", "").strip("/")
    for s in ls.import_storage.s3.list(project=project_id):
        if getattr(s, "bucket", None) == bucket and getattr(s, "title", None) == STORAGE_TITLE:
            return False
    kwargs = dict(
        project=project_id, title=STORAGE_TITLE, bucket=bucket,
        s3endpoint=secrets["SLATE_S3_ENDPOINT"], region_name=secrets.get("SLATE_S3_REGION", "garage"),
        aws_access_key_id=secrets.get("SLATE_S3_PRESIGN_ACCESS_KEY") or secrets["SLATE_S3_ACCESS_KEY"],
        aws_secret_access_key=secrets.get("SLATE_S3_PRESIGN_SECRET_KEY") or secrets["SLATE_S3_SECRET_KEY"],
        presign=True, use_blob_urls=False,
    )
    if prefix:
        kwargs["prefix"] = prefix
    ls.import_storage.s3.create(**kwargs)
    return True


def existing_image_ids(ls, project_id: int) -> set[int]:
    return {int(t.data["image_id"]) for t in ls.tasks.list(project=project_id) if isinstance(getattr(t, "data", None), dict) and "image_id" in t.data}


def import_tasks(ls, project_id: int, tasks: list[dict], batch: int = 100) -> int:
    n = 0
    for i in range(0, len(tasks), batch):
        ls.projects.import_tasks(project_id, request=tasks[i:i + batch], return_task_ids=True)
        n += len(tasks[i:i + batch])
    return n


def export_tasks(ls, project_id: int) -> list[dict]:
    """Tasks with annotations as plain dicts."""
    from pydantic_core import to_jsonable_python  # pylint: disable=import-outside-toplevel

    return [to_jsonable_python(t) for t in ls.tasks.list(project=project_id, fields="all")]


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows) + "\n")
