"""A slate/no-slate labelling queue for one labeller, adapted from the species detector's.

``build`` writes a self-contained page folder (served from ~/Downloads so a
sandboxed browser can open it): ``label.html`` with the data inlined, and
``img/`` holding the frames under opaque ids. Beside the repo it writes a
manifest mapping those ids back to image_ids.

The page is blind: it never shows the model's score, so the score can't anchor
the answer. Items are shuffled (seeded) for the same reason: a run of
confident slates shouldn't set up the next answer. Answers are kept in the
browser's localStorage as you go and exported with "Download answers".

``ingest`` turns the downloaded answers into ``labels/manual/<labeller>/<queue_id>.csv``
(image_id, answer), which ``dataset`` folds into the training manifest.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Sequence

ANSWERS = ("slate", "no_slate", "cant_tell")


def select_items(predictions: Sequence[dict], checks: int, seed: int) -> list[dict]:
    """Every flagged frame plus ``checks`` random unflagged ones (to catch misses), shuffled."""
    rng = random.Random(seed)
    flagged = [r for r in predictions if r["flag"]]
    rest = [r for r in predictions if not r["flag"]]
    items = flagged + rng.sample(rest, min(checks, len(rest)))
    rng.shuffle(items)
    return items


def build(out_dir: Path, items: Sequence[dict], cache: Path, queue_id: str, labeller: str, manifest_path: Path) -> Path:
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    page_items, manifest = [], []
    for n, it in enumerate(items, 1):
        iid = f"i{n:04d}"
        shutil.copyfile(Path(cache) / f"{it['image_id']}.jpg", out / "img" / f"{iid}.jpg")
        page_items.append({"id": iid})
        manifest.append({"id": iid, "image_id": int(it["image_id"]), "dive_id": int(it["dive_id"]),
                         "scan_group": it["scan_group"], "flag": it["flag"], "p_slate": float(it["p_slate"])})
    data = {"queue_id": queue_id, "labeller": labeller, "items": page_items}
    (out / "label.html").write_text(_PAGE.replace("__DATA__", json.dumps(data)))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"queue_id": queue_id, "labeller": labeller, "items": manifest}, indent=1) + "\n")
    return out / "label.html"


def ingest(answers: dict, manifest: dict) -> list[dict]:
    """One row per answered item: image_id, answer, plus the model's view for later comparison."""
    if answers["queue_id"] != manifest["queue_id"]:
        raise ValueError(f"answers are for queue {answers['queue_id']}, manifest is {manifest['queue_id']}")
    rows = []
    for m in manifest["items"]:
        a = answers["answers"].get(m["id"], {}).get("answer")
        if a not in ANSWERS:
            continue
        rows.append({"image_id": m["image_id"], "answer": a, "labeller": manifest["labeller"],
                     "queue_id": manifest["queue_id"], "model_flag": m["flag"], "model_p_slate": m["p_slate"]})
    return rows


def write_labels(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "answer", "labeller", "queue_id", "model_flag", "model_p_slate"])
        w.writeheader()
        w.writerows(rows)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slate Labelling Queue</title>
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--muted:#5d5c56;--line:#e4e3dc;--accent:#2a78d6;--accent-ink:#ffffff;--chip:#f1f0ea;--yes:#1f8a4c;--no:#b3261e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222220;--ink:#ffffff;--muted:#c3c2b7;--line:#3a3a36;--accent:#3987e5;--accent-ink:#ffffff;--chip:#2c2c29;--yes:#3fb36f;--no:#e5675f}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222220;--ink:#ffffff;--muted:#c3c2b7;--line:#3a3a36;--accent:#3987e5;--accent-ink:#ffffff;--chip:#2c2c29;--yes:#3fb36f;--no:#e5675f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{max-width:1400px;margin:0 auto;padding:12px 16px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-weight:600}.muted{color:var(--muted)}
.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--accent)}
main{max-width:1400px;margin:0 auto;padding:12px 16px 24px}
.photo{background:#000;border-radius:8px;overflow:auto;max-height:74vh;display:flex;justify-content:center;align-items:flex-start;cursor:zoom-in}
.photo img{max-width:100%;max-height:74vh;display:block}
.photo.zoom{cursor:zoom-out;justify-content:flex-start}.photo.zoom img{max-width:none;max-height:none}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{font:inherit;color:var(--ink);background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:10px 16px;cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.ans{font-size:16px;font-weight:600;min-width:150px}
.ans[data-a="slate"][aria-pressed="true"]{background:var(--yes);border-color:var(--yes);color:#fff}
.ans[data-a="no_slate"][aria-pressed="true"]{background:var(--no);border-color:var(--no);color:#fff}
.ans[data-a="cant_tell"][aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;margin-left:6px;color:var(--muted)}
.ans[aria-pressed="true"] kbd{color:inherit;border-color:currentColor}
.spacer{flex:1}.note{font-size:13px;color:var(--muted);margin:10px 0 0}
</style></head><body>
<header><h1>Slate labelling</h1><span class="muted" id="count"></span><div class="bar" aria-hidden="true"><div id="prog"></div></div>
<button id="dl" class="primary">Download answers</button></header>
<main>
 <div class="photo" id="photo" title="Click to zoom"><img id="img" alt="Frame to label"></div>
 <div class="controls">
  <button class="ans" data-a="slate">Slate<kbd>Y</kbd></button>
  <button class="ans" data-a="no_slate">No slate<kbd>N</kbd></button>
  <button class="ans" data-a="cant_tell">Can't tell<kbd>C</kbd></button>
  <span class="spacer"></span>
  <button id="prev">&larr; Back</button><button id="skip">Next unanswered</button><button id="next">Next &rarr;</button>
 </div>
 <p class="note">Is a dive slate (H, Tic-Tac-Toe or V) anywhere in the frame, including small, distant or reflected? A checkerboard or the Box target is <b>not</b> a slate.
 Click the photo (or press <kbd>Z</kbd>) to zoom. Answering moves to the next frame. Saved in this browser as you go; press Download answers when you stop.</p>
</main>
<script>
const D = __DATA__;
const store = `slatequeue-${D.queue_id}`;
let A = {}; try { A = JSON.parse(localStorage.getItem(store) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(store, JSON.stringify(A)); } catch (e) {} };
const answered = id => !!(A[id] && A[id].answer);
let i = Math.max(0, D.items.findIndex(x => !answered(x.id)));
const photo = document.getElementById("photo"), img = document.getElementById("img");
function render() {
  const it = D.items[i];
  img.src = `img/${it.id}.jpg`; photo.classList.remove("zoom"); photo.scrollTo(0, 0);
  const a = (A[it.id] || {}).answer;
  document.querySelectorAll(".ans").forEach(b => b.setAttribute("aria-pressed", String(a === b.dataset.a)));
  const done = D.items.filter(x => answered(x.id)).length;
  document.getElementById("count").textContent = `${i + 1} / ${D.items.length} · ${done} answered`;
  document.getElementById("prog").style.width = `${100 * done / D.items.length}%`;
  const n = D.items[i + 1]; if (n) new Image().src = `img/${n.id}.jpg`;
}
const go = d => { i = Math.max(0, Math.min(D.items.length - 1, i + d)); render(); };
function answer(a) {
  A[D.items[i].id] = {answer: a, at: new Date().toISOString()}; save();
  if (i < D.items.length - 1) go(1); else render();
}
function nextUnanswered() {
  for (let k = 1; k <= D.items.length; k++) { const j = (i + k) % D.items.length; if (!answered(D.items[j].id)) { i = j; render(); return; } }
}
function download() {
  const blob = new Blob([JSON.stringify({queue_id: D.queue_id, labeller: D.labeller, answers: A, saved_at: new Date().toISOString()}, null, 1)], {type: "application/json"});
  const u = URL.createObjectURL(blob), l = document.createElement("a");
  l.href = u; l.download = `slate-queue-${D.queue_id}-answers.json`; document.body.append(l); l.click(); l.remove(); URL.revokeObjectURL(u);
}
photo.addEventListener("click", () => photo.classList.toggle("zoom"));
document.querySelectorAll(".ans").forEach(b => b.addEventListener("click", () => answer(b.dataset.a)));
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("skip").onclick = nextUnanswered;
document.getElementById("dl").onclick = download;
document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") answer("slate"); else if (k === "n") answer("no_slate"); else if (k === "c") answer("cant_tell");
  else if (k === "z") photo.classList.toggle("zoom");
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else return;
  e.preventDefault();
});
render();
</script></body></html>
"""
