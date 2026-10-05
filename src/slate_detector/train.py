"""Binary slate-presence classifier (needs the ``train`` extra).

A pretrained CNN over the whole frame at ~1024 px on the long side. The slate
is ~1% of the frame (2026-07-31_slate_training/docs/data_report.md), so:

- frames are resized, never cropped (a crop can cut the slate out and flip the label);
- global average pooling is replaced by GeM, which leans toward the strongest
  local response instead of diluting it over the water.

Evaluation is cross-validation grouped by dive: frames of one dive are near
duplicates, so a frame-level split would overstate accuracy.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2 as T

from slate_detector.dataset import Frame

MIN_FREE_GPU_MIB = 4500


@dataclass
class TrainConfig:
    arch: str = "efficientnet_b0"
    epochs: int = 12
    image_size: int = 1024  # long side; frames are 4:3
    batch_size: int = 4  # 2.8 GiB at 1024 px; batch 8 OOMs on the 6 GB laptop GPU
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 6
    seed: int = 0


def require_free_gpu(min_free_mib: int = MIN_FREE_GPU_MIB) -> None:
    """The laptop GPU is shared: refuse to start rather than contend."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    total, used = (int(v) for v in out.strip().splitlines()[0].split(","))
    if total - used < min_free_mib:
        raise RuntimeError(f"GPU has {total - used} MiB free, need {min_free_mib}")


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1), keepdim=True).pow(1.0 / self.p)


def build_model(arch: str) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.avgpool = GeM()
        m.classifier[0].inplace = False  # in-place dropout would overwrite GeM's output, which backward needs
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.avgpool = GeM()
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


class ChannelGain(nn.Module):
    """Scale each RGB channel independently: a cheap stand-in for underwater colour casts.

    Hue jitter would do a similar job but costs ~360 ms per 1024 px frame on
    the CPU (an HSV round trip), which starved the GPU.
    """

    def __init__(self, low: float = 0.8, high: float = 1.2):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(3, 1, 1).uniform_(self.low, self.high)
        return (x * gain).clamp_(0.0, 1.0)


def transforms(image_size: int, train: bool) -> T.Compose:
    size = (image_size * 3 // 4, image_size)  # (h, w), landscape 4:3
    steps: list = [T.ToImage(), T.Resize(size, antialias=True)]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3)], p=0.8),  # no hue: see ChannelGain
        ]
    steps += [T.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [ChannelGain()]
    steps += [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(steps)


class FrameDataset(Dataset):
    def __init__(self, frames: list[Frame], cache: Path, tf: T.Compose):
        self.frames, self.cache, self.tf = frames, Path(cache), tf

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        f = self.frames[i]
        im = Image.open(self.cache / f"{f.image_id}.jpg").convert("RGB")
        return self.tf(im), torch.tensor(float(f.label))


def balanced_sampler(frames: list[Frame], seed: int) -> WeightedRandomSampler:
    """Half slate, half not, in expectation; one epoch = one pass's worth of draws."""
    labels = np.array([f.label for f in frames])
    w = np.where(labels == 1, 0.5 / max(labels.sum(), 1), 0.5 / max((1 - labels).sum(), 1))
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(frames), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def cached(frames: list[Frame], cache: Path) -> list[Frame]:
    return [f for f in frames if (Path(cache) / f"{f.image_id}.jpg").is_file()]


def fit(train: list[Frame], cache: Path, cfg: TrainConfig, device: torch.device, log=print) -> nn.Module:
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.arch).to(device).to(memory_format=torch.channels_last)
    loader = DataLoader(
        FrameDataset(train, cache, transforms(cfg.image_size, True)),
        batch_size=cfg.batch_size, sampler=balanced_sampler(train, cfg.seed),
        num_workers=cfg.workers, pin_memory=True, drop_last=True, persistent_workers=cfg.workers > 0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16):
                loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        log(f"  epoch {epoch + 1}/{cfg.epochs} loss {total / n:.4f}")
    return model


@torch.no_grad()
def predict(model: nn.Module, frames: list[Frame], cache: Path, cfg: TrainConfig, device: torch.device) -> np.ndarray:
    """P(slate) per frame; test-time augmentation averages the frame and its horizontal flip."""
    model.eval()
    loader = DataLoader(FrameDataset(frames, cache, transforms(cfg.image_size, False)),
                        batch_size=cfg.batch_size, num_workers=cfg.workers)
    out = []
    for x, _ in loader:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16):
            logits = (model(x) + model(torch.flip(x, dims=[3]))) / 2
        out.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def summarize(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score  # pylint: disable=import-outside-toplevel

    pred = probs >= threshold
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    return {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probs)), "average_precision": float(average_precision_score(labels, probs)),
        "threshold": threshold, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "false_positives": fp, "false_negatives": fn,
    }


def cross_validate(frames: list[Frame], cache: Path, out: Path, cfg: TrainConfig, folds: int = 5) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold  # pylint: disable=import-outside-toplevel

    require_free_gpu()
    device = torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)
    frames = cached(frames, cache)
    labels = np.array([f.label for f in frames])
    groups = np.array([f.dive_id for f in frames])
    print(f"{len(frames)} cached frames ({labels.sum()} slate) over {len(set(groups))} dives")

    probs = np.full(len(frames), np.nan)
    fold_of = np.full(len(frames), -1)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    for k, (tr, te) in enumerate(splitter.split(np.zeros(len(frames)), labels, groups)):
        print(f"fold {k}: train {len(tr)} ({labels[tr].sum()} slate), test {len(te)} ({labels[te].sum()} slate)")
        model = fit([frames[i] for i in tr], cache, cfg, device)
        probs[te] = predict(model, [frames[i] for i in te], cache, cfg, device)
        fold_of[te] = k
        print(f"fold {k}: " + json.dumps({k2: round(v, 4) if isinstance(v, float) else v
                                           for k2, v in summarize(labels[te], probs[te]).items()}))
        del model
        torch.cuda.empty_cache()

    with open(out / "oof_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "dive_id", "fold", "label", "p_slate", "answers", "path"])
        for f, k, p in zip(frames, fold_of, probs):
            w.writerow([f.image_id, f.dive_id, k, f.label, f"{p:.5f}", f.answers, f.path])
    report = {"config": asdict(cfg), "overall": summarize(labels, probs), "per_dive": per_dive(frames, probs)}
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    return report


def per_dive(frames: list[Frame], probs: np.ndarray, threshold: float = 0.5) -> dict:
    rows: dict[int, dict] = {}
    for f, p in zip(frames, probs):
        r = rows.setdefault(f.dive_id, {"slate": 0, "slate_found": 0, "no_slate": 0, "false_alarms": 0})
        if f.label:
            r["slate"] += 1
            r["slate_found"] += int(p >= threshold)
        else:
            r["no_slate"] += 1
            r["false_alarms"] += int(p >= threshold)
    return {str(k): v for k, v in sorted(rows.items())}


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, TrainConfig]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.arch)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last), cfg


def train_final(frames: list[Frame], cache: Path, out: Path, cfg: TrainConfig) -> Path:
    require_free_gpu()
    out.mkdir(parents=True, exist_ok=True)
    frames = cached(frames, cache)
    model = fit(frames, cache, cfg, torch.device("cuda"))
    path = out / f"slate_{cfg.arch}.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, path)
    print(f"-> {path}")
    return path
