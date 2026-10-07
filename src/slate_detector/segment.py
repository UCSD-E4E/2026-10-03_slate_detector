"""Slate masks from SAM 3.1, prompted with text, then checked by a person (needs the ``sam3`` extra).

SAM 3.1 is a concept segmenter. The text prompt ``"white board"`` finds dive
slates; ``"slate"`` and ``"dive slate"`` return nothing (also found by
cscw-fishsense2027's e1j_size_constancy/sam_sizes.py). It isn't a presence
detector: it also finds checkerboards, the Box target and boat decks, so it
only runs on frames the classifier, or a person, already calls slate.

It returns several candidates, and the top-scoring one is sometimes the wrong
white thing: the slate's reflection on the surface, or a second board. The
instance is chosen by, in order:

1. best overlap with the slate's outline from its point label (``masks``),
   when the frame has one;
2. containing the human laser dot (``laserlabel``), since the dot is usually
   on the slate;
3. the highest score.

SAM resizes every input to 1008 px, so distant slates can vanish. When
nothing is found and the frame has a label outline, SAM is retried on a crop
around it, and the masks are mapped back.

Every candidate is kept (``data/masks/sam3/{image_id}.npz``) so the review
page can switch to another one with a single key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROMPT = "white board"
CHECKPOINT_ENV = "FISHSENSE_SAM3_CHECKPOINT"
DEFAULT_CHECKPOINT = Path("~/Downloads/sam3.1_multiplex.pt").expanduser()
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"  # as pinned in fishsense-detector
SAM_SIDE = 1008
MAX_INSTANCES = 5
MIN_SCORE = 0.1
CROP_PAD = 1.0  # crop = outline bbox grown by this fraction on each side
PHOTO_W = 4014

LASER_SQL = """
SELECT DISTINCT ON (image_id) image_id, x, y
FROM laserlabel
WHERE completed AND NOT coalesce(superseded, false) AND x IS NOT NULL AND image_id IN ({ids})
ORDER BY image_id, updated_at DESC
"""


def checkpoint_path() -> Path:
    return Path(os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)).expanduser()


@dataclass
class Candidates:
    masks: np.ndarray  # (n, h, w) bool, frame-cache resolution
    scores: np.ndarray  # (n,)
    crop: bool = False  # found on the retry crop


def mask_iou(mask: np.ndarray, other: np.ndarray) -> float:
    return float((mask & other).sum() / max((mask | other).sum(), 1))


def polygon_mask(shape: tuple[int, int], poly) -> np.ndarray:
    import cv2  # pylint: disable=import-outside-toplevel

    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(poly))], 1)
    return m.astype(bool)


def choose(c: Candidates, outline=None, dot=None) -> tuple[int, str]:
    """(index of the chosen candidate, why). -1 when there are none."""
    if len(c.scores) == 0:
        return -1, "none"
    if outline is not None:
        ref = polygon_mask(c.masks.shape[1:], outline)
        ious = [mask_iou(m, ref) for m in c.masks]
        if max(ious) > 0.3:
            return int(np.argmax(ious)), "label_outline"
    if dot is not None:
        x, y = int(round(dot[0])), int(round(dot[1]))
        h, w = c.masks.shape[1:]
        if 0 <= x < w and 0 <= y < h:
            hit = [i for i in range(len(c.scores)) if c.masks[i, y, x]]
            if hit:
                return max(hit, key=lambda i: c.scores[i]), "laser_dot"
    return int(np.argmax(c.scores)), "top_score"


def crop_box(outline, shape: tuple[int, int], pad: float = CROP_PAD) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) around the outline, grown by ``pad`` of its size on each side, clipped."""
    pts = np.asarray(outline, dtype=float)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    w, h = x1 - x0, y1 - y0
    H, W = shape
    return (max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h)), min(W, int(np.ceil(x1 + pad * w))), min(H, int(np.ceil(y1 + pad * h))))


def paste_masks(masks: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = np.zeros((len(masks), *shape), bool)
    out[:, y0:y1, x0:x1] = masks
    return out


class Sam3Boards:
    def __init__(self, checkpoint: Path | None = None, prompt: str = PROMPT):
        from fishsense_core.fish.sam3 import build_segmenter  # pylint: disable=import-outside-toplevel

        self.processor = build_segmenter(checkpoint or checkpoint_path())
        self.processor.set_confidence_threshold(MIN_SCORE)
        self.prompt = prompt

    def candidates(self, rgb: np.ndarray, upscale: bool = False) -> Candidates:
        """Up to MAX_INSTANCES masks at the input's resolution.

        SAM sees the input with its long side at most 1008 px, or exactly
        1008 px when ``upscale`` (used for crops, so a small board gets more pixels).
        """
        # pylint: disable=import-outside-toplevel
        import torch
        import torch.nn.functional as F
        from PIL import Image

        h, w = rgb.shape[:2]
        s = SAM_SIDE / max(h, w) if upscale else min(1.0, SAM_SIDE / max(h, w))
        image = Image.fromarray(rgb)  # a numpy array silently gives broken masks (fishsense-lite)
        if s != 1.0:
            image = image.resize((round(w * s), round(h * s)), Image.BILINEAR)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(self.prompt, state)
            probs = state["masks_logits"].squeeze(1)
            scores = state["scores"].float().cpu().numpy()
            keep = np.argsort(-scores, kind="stable")[:MAX_INSTANCES]
            if len(keep):
                sel = probs[torch.as_tensor(keep, device=probs.device, dtype=torch.long)][:, None].float()
                masks = (F.interpolate(sel, size=(h, w), mode="bilinear", align_corners=False)[:, 0] > 0.5).cpu().numpy()
            else:
                masks = np.zeros((0, h, w), bool)
        del state, probs
        torch.cuda.empty_cache()
        return Candidates(masks=masks, scores=scores[keep])

    def segment(self, rgb: np.ndarray, outline=None) -> Candidates:
        """Candidates for the whole frame; if none and an outline is known, retry on a crop around it."""
        c = self.candidates(rgb)
        if len(c.scores) == 0 and outline is not None:
            box = crop_box(outline, rgb.shape[:2])
            x0, y0, x1, y1 = box
            sub = self.candidates(np.ascontiguousarray(rgb[y0:y1, x0:x1]), upscale=True)
            if len(sub.scores):
                c = Candidates(masks=paste_masks(sub.masks, box, rgb.shape[:2]), scores=sub.scores, crop=True)
        return c


def save_candidates(path: Path, c: Candidates, chosen: int, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, masks=np.packbits(c.masks, axis=-1), shape=np.array(c.masks.shape),
                        scores=c.scores, chosen=chosen, reason=reason, crop=c.crop)


def load_candidates(path: Path) -> tuple[Candidates, int, str]:
    z = np.load(path)
    n, h, w = z["shape"]
    masks = np.unpackbits(z["masks"], axis=-1, count=w).astype(bool).reshape(n, h, w)
    return Candidates(masks=masks, scores=z["scores"], crop=bool(z["crop"])), int(z["chosen"]), str(z["reason"])


def laser_dots(runner, image_ids: list[int], cache_w: int = 1600) -> dict[int, tuple[float, float]]:
    """Human laser-dot positions, scaled from photo to frame-cache pixels."""
    s = cache_w / PHOTO_W
    out = {}
    for i in range(0, len(image_ids), 1000):
        ids = ",".join(map(str, image_ids[i:i + 1000]))
        for r in runner.query(LASER_SQL.format(ids=ids)):
            out[int(r["image_id"])] = (float(r["x"]) * s, float(r["y"]) * s)
    return out


def summary_row(image_id: int, c: Candidates, chosen: int, reason: str) -> dict:
    return {"image_id": image_id, "n": int(len(c.scores)), "scores": [round(float(x), 3) for x in c.scores],
            "chosen": chosen, "reason": reason, "crop": c.crop,
            "area_frac": round(float(c.masks[chosen].mean()), 5) if chosen >= 0 else 0.0}


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
