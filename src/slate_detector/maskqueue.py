"""Review SAM 3.1 slate masks in the browser: accept, reject, or switch to another candidate.

``build`` writes a self-contained page folder (in ~/Downloads, like the slate
queue): each frame as ``img/<id>.jpg`` and each candidate's outline as a
transparent ``ov/<id>_<k>.png``, so switching candidates is just swapping an
overlay. A manifest beside the repo maps page ids back to image_ids.

``ingest`` turns the downloaded answers into ``labels/masks/<labeller>/<queue_id>.json``:
for every accepted frame, the chosen mask as polygons and a fitted
4-corner quad, in frame-cache (1600 px) coordinates. Masks are stored as
polygons so the labels live in git, not only in ``data/``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from slate_detector.segment import load_candidates

ANSWERS = ("accept", "reject", "cant_tell")
COLORS = [(40, 255, 80), (255, 200, 0), (0, 200, 255), (255, 80, 200), (255, 255, 255)]


def outline_png(mask: np.ndarray, color: tuple[int, int, int], width: int = 3):
    """RGBA image: the mask's outline in ``color`` plus a faint fill."""
    import cv2  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask] = (*color, 45)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros((h, w), np.uint8)
    cv2.drawContours(edge, cs, -1, 1, width)
    rgba[edge > 0] = (*color, 255)
    return Image.fromarray(rgba, "RGBA")


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def build(out_dir: Path, image_ids: Sequence[int], cache: Path, masks_dir: Path, queue_id: str, labeller: str,
          manifest_path: Path, seed: int = 0) -> Path:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "ov").mkdir(exist_ok=True)
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    page, manifest = [], []
    for n, image_id in enumerate(ids, 1):
        pid = f"m{n:04d}"
        c, chosen, reason = load_candidates(Path(masks_dir) / f"{image_id}.npz")
        Image.open(Path(cache) / f"{image_id}.jpg").convert("RGB").save(out / "img" / f"{pid}.jpg", quality=88)
        cands = []
        for k, m in enumerate(c.masks):
            outline_png(m, COLORS[k % len(COLORS)]).save(out / "ov" / f"{pid}_{k}.png", optimize=True)
            cands.append({"bbox": bbox(m)})
        page.append({"id": pid, "cands": cands, "chosen": max(chosen, 0) if cands else -1})
        manifest.append({"id": pid, "image_id": int(image_id), "n": len(cands), "chosen": chosen, "reason": reason,
                         "scores": [round(float(s), 3) for s in c.scores], "crop": c.crop})
    data = {"queue_id": queue_id, "labeller": labeller, "items": page}
    (out / "label.html").write_text(_PAGE.replace("__DATA__", json.dumps(data)))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"queue_id": queue_id, "labeller": labeller, "items": manifest}, indent=1) + "\n")
    return out / "label.html"


def mask_polygons(mask: np.ndarray, eps: float = 1.0) -> list[list[list[float]]]:
    """External contours of ``mask``, simplified by ``eps`` px, largest first."""
    import cv2  # pylint: disable=import-outside-toplevel

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)
    return [cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float).tolist() for c in cs if cv2.contourArea(c) >= 4]


def _approx_corners(hull) -> np.ndarray | None:
    import cv2  # pylint: disable=import-outside-toplevel

    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(float)
        if len(ap) < 4:
            break
    return None


def _intersect(l1, l2) -> np.ndarray | None:
    """Intersection of two lines given as (vx, vy, x0, y0)."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-9:
        return None
    t = ((x2 - x1) * (-vy2) - (y2 - y1) * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1])


def fit_quad(mask: np.ndarray, trim: float = 0.2) -> list[list[float]] | None:
    """4 sharp corners of the board, or None.

    The board is a rigid rectangle, so in the photo it's a quadrilateral. SAM's
    mask of a small, soft board has rounded corners, and the convex hull's own
    vertices sit on those curves. So: take rough corners from the hull, fit a
    straight line to each side using only its middle (``trim`` cut from each
    end, away from the rounding), and intersect neighbouring lines. The hull
    also bridges hands held over an edge.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    cnt = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rough = _approx_corners(hull)
    if rough is None:
        return None
    pts = cnt.reshape(-1, 2).astype(float)
    lines = []
    for k in range(4):
        a, b = rough[k], rough[(k + 1) % 4]
        d = b - a
        length = np.hypot(*d)
        if length < 4:
            return rough.tolist()
        t = ((pts - a) @ d) / length**2  # position along the side, 0..1
        dist = np.abs((pts - a) @ np.array([-d[1], d[0]])) / length  # distance from the side
        near = pts[(t > trim) & (t < 1 - trim) & (dist < max(2.0, 0.05 * length))]
        if len(near) < 3:
            return rough.tolist()
        lines.append(cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    corners = [_intersect(lines[k - 1], lines[k]) for k in range(4)]  # corner k joins side k-1 and side k
    if any(c is None for c in corners):
        return rough.tolist()
    corners = np.array(corners)
    span = np.ptp(rough, axis=0).max()
    if np.abs(corners - rough).max() > 0.25 * span:  # a near-parallel pair shot a corner away
        return rough.tolist()
    return corners.tolist()


def ingest(answers: dict, manifest: dict, masks_dir: Path) -> list[dict]:
    if answers["queue_id"] != manifest["queue_id"]:
        raise ValueError(f"answers are for queue {answers['queue_id']}, manifest is {manifest['queue_id']}")
    rows = []
    for m in manifest["items"]:
        a = answers["answers"].get(m["id"], {})
        answer = a.get("answer")
        if answer not in ANSWERS:
            continue
        row = {"image_id": m["image_id"], "answer": answer, "labeller": manifest["labeller"], "queue_id": manifest["queue_id"],
               "instance": None, "auto_instance": m["chosen"], "polygons": None, "quad": None}
        if answer == "accept":
            k = int(a.get("instance", m["chosen"]))
            c, _, _ = load_candidates(Path(masks_dir) / f"{m['image_id']}.npz")
            if not 0 <= k < len(c.masks):
                raise ValueError(f"image {m['image_id']}: accepted instance {k} of {len(c.masks)}")
            row.update(instance=k, polygons=mask_polygons(c.masks[k]), quad=fit_quad(c.masks[k]))
        rows.append(row)
    return rows


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slate Mask Review</title>
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--muted:#5d5c56;--line:#e4e3dc;--accent:#2a78d6;--accent-ink:#ffffff;--chip:#f1f0ea;--yes:#1f8a4c;--no:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222220;--ink:#ffffff;--muted:#c3c2b7;--line:#3a3a36;--accent:#3987e5;--accent-ink:#ffffff;--chip:#2c2c29;--yes:#3fb36f;--no:#e5675f}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222220;--ink:#ffffff;--muted:#c3c2b7;--line:#3a3a36;--accent:#3987e5;--accent-ink:#ffffff;--chip:#2c2c29;--yes:#3fb36f;--no:#e5675f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{max-width:1400px;margin:0 auto;padding:12px 16px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-weight:600}.muted{color:var(--muted)}
.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--accent)}
main{max-width:1400px;margin:0 auto;padding:12px 16px 24px}
.view{background:#000;border-radius:8px;overflow:hidden;max-height:72vh;display:flex;justify-content:center;cursor:zoom-in}
.stage{position:relative;display:inline-block;transform-origin:0 0;transition:transform .15s}
.stage img{display:block;max-width:100%;max-height:72vh}
.stage img.ov{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font:inherit;color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:10px 16px;cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.ans{font-size:16px;font-weight:600;min-width:130px}
.ans[data-a="accept"][aria-pressed="true"]{background:var(--yes);border-color:var(--yes);color:#fff}
.ans[data-a="reject"][aria-pressed="true"]{background:var(--no);border-color:var(--no);color:#fff}
.ans[data-a="cant_tell"][aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.cand[aria-pressed="true"]{outline:2px solid var(--ink)}
.cand{min-width:44px;padding:8px 10px;font-weight:600}
.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;margin-left:6px;color:var(--muted)}
.ans[aria-pressed="true"] kbd{color:inherit;border-color:currentColor}
.spacer{flex:1}.note{font-size:13px;color:var(--muted);margin:10px 0 0}
</style></head><body>
<header><h1>Slate mask review</h1><span class="muted" id="count"></span><div class="bar" aria-hidden="true"><div id="prog"></div></div>
<button id="dl" class="primary">Download answers</button></header>
<main>
 <div class="view" id="view" title="Click to zoom to the outlined board"><div class="stage" id="stage"><img id="img" alt="Frame"><img id="ov" class="ov" alt=""></div></div>
 <div class="controls">
  <button class="ans" data-a="accept">Accept<kbd>Y</kbd></button>
  <button class="ans" data-a="reject">Reject<kbd>N</kbd></button>
  <button class="ans" data-a="cant_tell">Can't tell<kbd>C</kbd></button>
  <span class="muted" id="candlabel">Candidate:</span><span id="cands"></span>
  <span class="spacer"></span>
  <button id="prev">&larr; Back</button><button id="skip">Next unanswered</button><button id="next">Next &rarr;</button>
 </div>
 <p class="note">Accept if the outline covers the slate board (H, Tic-Tac-Toe or V), apart from hands or gear in front of it, and nothing else.
 If another candidate is right, press its number (<kbd>1</kbd>&ndash;<kbd>5</kbd>) and then <kbd>Y</kbd>. Reject if no candidate is the slate, or the outline is badly off.
 <kbd>Z</kbd> or a click zooms to the board. Saved in this browser as you go; press Download answers when you stop.</p>
</main>
<script>
const D = __DATA__;
const store = `slatemaskqueue-${D.queue_id}`;
let A = {}; try { A = JSON.parse(localStorage.getItem(store) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(store, JSON.stringify(A)); } catch (e) {} };
const answered = id => !!(A[id] && A[id].answer);
let i = Math.max(0, D.items.findIndex(x => !answered(x.id)));
let cur = 0, zoom = false;
const img = document.getElementById("img"), ov = document.getElementById("ov"), stage = document.getElementById("stage");
function applyZoom() {
  const it = D.items[i], c = it.cands[cur];
  if (!zoom || !c || !img.naturalWidth) { stage.style.transform = ""; return; }
  const s = img.clientWidth / img.naturalWidth, [x0, y0, x1, y1] = c.bbox;
  const bw = (x1 - x0) * s, bh = (y1 - y0) * s;
  const z = Math.max(1, Math.min(6, 0.6 * Math.min(img.clientWidth / bw, img.clientHeight / bh)));
  const cx = (x0 + x1) / 2 * s, cy = (y0 + y1) / 2 * s;
  stage.style.transform = `translate(${img.clientWidth / 2 - cx * z}px, ${img.clientHeight / 2 - cy * z}px) scale(${z})`;
}
function showCand(k) {
  const it = D.items[i]; if (!it.cands.length) { ov.removeAttribute("src"); return; }
  cur = Math.max(0, Math.min(it.cands.length - 1, k));
  ov.src = `ov/${it.id}_${cur}.png`;
  document.querySelectorAll(".cand").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.k === cur)));
  applyZoom();
}
function render() {
  const it = D.items[i], a = A[it.id] || {};
  img.src = `img/${it.id}.jpg`;
  const box = document.getElementById("cands"); box.replaceChildren();
  it.cands.forEach((c, k) => { const b = document.createElement("button"); b.className = "cand"; b.dataset.k = k; b.textContent = k + 1; b.onclick = () => showCand(k); box.append(b); });
  document.getElementById("candlabel").textContent = it.cands.length ? "Candidate:" : "No board found";
  showCand(a.instance !== undefined ? a.instance : it.chosen);
  document.querySelectorAll(".ans").forEach(b => b.setAttribute("aria-pressed", String(a.answer === b.dataset.a)));
  const done = D.items.filter(x => answered(x.id)).length;
  document.getElementById("count").textContent = `${i + 1} / ${D.items.length} · ${done} answered`;
  document.getElementById("prog").style.width = `${100 * done / D.items.length}%`;
  const n = D.items[i + 1]; if (n) new Image().src = `img/${n.id}.jpg`;
}
img.onload = applyZoom;
const go = d => { i = Math.max(0, Math.min(D.items.length - 1, i + d)); render(); };
function answer(a) {
  const it = D.items[i];
  if (a === "accept" && !it.cands.length) return;
  A[it.id] = {answer: a, at: new Date().toISOString()};
  if (a === "accept") A[it.id].instance = cur;
  save();
  if (i < D.items.length - 1) go(1); else render();
}
function nextUnanswered() {
  for (let k = 1; k <= D.items.length; k++) { const j = (i + k) % D.items.length; if (!answered(D.items[j].id)) { i = j; render(); return; } }
}
function download() {
  const blob = new Blob([JSON.stringify({queue_id: D.queue_id, labeller: D.labeller, answers: A, saved_at: new Date().toISOString()}, null, 1)], {type: "application/json"});
  const u = URL.createObjectURL(blob), l = document.createElement("a");
  l.href = u; l.download = `slate-masks-${D.queue_id}-answers.json`; document.body.append(l); l.click(); l.remove(); URL.revokeObjectURL(u);
}
document.getElementById("view").addEventListener("click", () => { zoom = !zoom; applyZoom(); });
window.addEventListener("resize", applyZoom);
document.querySelectorAll(".ans").forEach(b => b.addEventListener("click", () => answer(b.dataset.a)));
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("skip").onclick = nextUnanswered;
document.getElementById("dl").onclick = download;
document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") answer("accept"); else if (k === "n") answer("reject"); else if (k === "c") answer("cant_tell");
  else if (k === "z") { zoom = !zoom; applyZoom(); }
  else if (k >= "1" && k <= "5") showCand(+k - 1);
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else return;
  e.preventDefault();
});
render();
</script></body></html>
"""
