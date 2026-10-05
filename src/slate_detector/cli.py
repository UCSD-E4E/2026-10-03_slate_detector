"""slate-detector: manifest -> prefetch -> cv / train; scan-manifest -> prefetch -> predict -> review -> queue -> ingest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import zip_longest
from pathlib import Path

from slate_detector.dataset import Frame, fetch_frames, read_manifest, write_manifest

DATA = Path("data")
MANIFEST = DATA / "manifest.csv"
CACHE = DATA / "frames"
RAWS = DATA / "raw"
SCAN_MANIFEST = DATA / "scan_manifest.csv"
MODEL = Path("runs/final-q1/slate_efficientnet_b0.pt")  # trained with queue slate_q1 (sha256 b8d377ba...cf78)


def fetch_order(frames: list[Frame]) -> list[Frame]:
    """Positives first, then negatives round-robin over dives, so a partial cache is balanced.

    Unlabelled scan frames (label -1) follow, in manifest order (already prioritised).
    """
    pos = [f for f in frames if f.label == 1]
    by_dive: dict[int, list[Frame]] = {}
    for f in frames:
        if f.label == 0:
            by_dive.setdefault(f.dive_id, []).append(f)
    neg = [f for batch in zip_longest(*by_dive.values()) for f in batch if f is not None]
    return pos + neg + [f for f in frames if f.label == -1]


def cmd_manifest(args) -> None:
    from slate_detector.db import DockerPsqlRunner  # pylint: disable=import-outside-toplevel

    frames, ambiguous = fetch_frames(DockerPsqlRunner(args.container, args.database))
    write_manifest(frames, args.manifest)
    pos = sum(f.label for f in frames)
    print(f"{len(frames)} frames ({pos} slate, {len(frames) - pos} no slate), "
          f"{ambiguous} ambiguous dropped, {len({f.dive_id for f in frames})} dives -> {args.manifest}")


def cmd_prefetch(args) -> None:
    # pylint: disable=import-outside-toplevel
    from slate_detector.db import DockerPsqlRunner
    from slate_detector.render import RenderCache, load_intrinsics

    frames = fetch_order(read_manifest(args.manifest))
    if args.limit:
        frames = frames[: args.limit]
    intrinsics = load_intrinsics(DockerPsqlRunner(args.container, args.database))
    cache = RenderCache(args.cache, args.raws)
    start = time.time()

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            rate = (time.time() - start) / done
            print(f"{done}/{total}  {rate:.1f}s/frame  eta {rate * (total - done) / 3600:.1f}h", flush=True)

    items = [(f.image_id, f.path, f.checksum, f.camera_id) for f in frames]
    failures = cache.prefetch(items, intrinsics, workers=args.workers, on_done=progress)
    if failures:
        out = args.cache / "failures.json"
        out.write_text(json.dumps(failures, indent=1))
        print(f"{len(failures)} failures -> {out}", file=sys.stderr)


def cmd_scan_manifest(args) -> None:
    from collections import Counter  # pylint: disable=import-outside-toplevel

    from slate_detector.db import DockerPsqlRunner  # pylint: disable=import-outside-toplevel
    from slate_detector.scan import fetch_scan  # pylint: disable=import-outside-toplevel

    frames = fetch_scan(DockerPsqlRunner(args.container, args.database), args.sample, args.seed)
    write_manifest(frames, args.out)
    groups = Counter(f.answers for f in frames)
    print(f"{len(frames)} unlabelled frames over {len({f.dive_id for f in frames})} dives "
          f"({', '.join(f'{g} {n}' for g, n in groups.items())}) -> {args.out}")


def cmd_predict(args) -> None:
    # pylint: disable=import-outside-toplevel
    import torch

    from slate_detector.scan import write_predictions
    from slate_detector.train import cached, load_model, predict, require_free_gpu

    require_free_gpu()
    device = torch.device("cuda")
    model, cfg = load_model(args.model, device)
    frames = cached(read_manifest(args.manifest), args.cache)
    rows = write_predictions(frames, predict(model, frames, args.cache, cfg, device), args.out)
    print(f"{len(rows)} frames scored -> {args.out}")


def cmd_review(args) -> None:
    # pylint: disable=import-outside-toplevel
    import csv as _csv

    from slate_detector.db import DockerPsqlRunner
    from slate_detector.scan import write_review

    with open(args.predictions, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    names = {int(r["id"]): r["name"] or r["path"]
             for r in DockerPsqlRunner(args.container, args.database).query("SELECT id, name, path FROM dive")}
    page = write_review(rows, args.cache, args.out, names)
    print(f"-> {page}")


def downloads_dir() -> Path:
    import subprocess  # pylint: disable=import-outside-toplevel

    out = subprocess.run(["xdg-user-dir", "DOWNLOAD"], capture_output=True, text=True, check=False).stdout.strip()
    return Path(out) if out else Path.home() / "Downloads"


def cmd_queue(args) -> None:
    # pylint: disable=import-outside-toplevel
    import csv as _csv

    from slate_detector import queue as Q

    with open(args.predictions, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    items = Q.select_items(rows, args.checks, args.seed)
    out = (args.out or downloads_dir()) / f"slate-queue-{args.queue_id}"
    page = Q.build(out, items, args.cache, args.queue_id, args.labeller, Path("labels/queues") / f"{args.queue_id}_manifest.json")
    print(f"{len(items)} frames ({sum(bool(r['flag']) for r in items)} flagged, rest random checks) -> {page}")


def cmd_ingest(args) -> None:
    from collections import Counter  # pylint: disable=import-outside-toplevel

    from slate_detector import queue as Q  # pylint: disable=import-outside-toplevel

    answers = json.loads(Path(args.answers).read_text())
    manifest = json.loads((Path("labels/queues") / f"{answers['queue_id']}_manifest.json").read_text())
    rows = Q.ingest(answers, manifest)
    out = Path("labels/manual") / manifest["labeller"] / f"{manifest['queue_id']}.csv"
    Q.write_labels(rows, out)
    print(f"{len(rows)}/{len(manifest['items'])} answered -> {out}: {dict(Counter(r['answer'] for r in rows))}")
    agree = Counter((r["model_flag"] == "likely_slate", r["answer"]) for r in rows)
    print("model flagged likely_slate vs answer:", {f"{'flagged' if k[0] else 'not flagged'}/{k[1]}": v for k, v in sorted(agree.items())})
    print("Next: slate-detector manifest, then scripts/run cv")


def cmd_cv(args) -> None:
    from slate_detector.train import TrainConfig, cross_validate  # pylint: disable=import-outside-toplevel

    cfg = TrainConfig(epochs=args.epochs, image_size=args.image_size, batch_size=args.batch_size, arch=args.arch)
    cross_validate(read_manifest(args.manifest), args.cache, args.out, cfg, folds=args.folds)


def cmd_train(args) -> None:
    from slate_detector.train import TrainConfig, train_final  # pylint: disable=import-outside-toplevel

    cfg = TrainConfig(epochs=args.epochs, image_size=args.image_size, batch_size=args.batch_size, arch=args.arch)
    train_final(read_manifest(args.manifest), args.cache, args.out, cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="slate-detector")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--raws", type=Path, default=RAWS)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="labelled frames from the restored DB")
    m.add_argument("--container", default="slate-db")
    m.add_argument("--database", default="fishsense")
    m.set_defaults(func=cmd_manifest)

    f = sub.add_parser("prefetch", help="copy raws locally (md5-verified) and render them into the cache")
    f.add_argument("--workers", type=int, default=3, help="overlaps decode with the next NAS read")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--container", default="slate-db")
    f.add_argument("--database", default="fishsense")
    f.set_defaults(func=cmd_prefetch)

    s = sub.add_parser("scan-manifest", help="unlabelled frames to run the model on")
    s.add_argument("--out", type=Path, default=SCAN_MANIFEST)
    s.add_argument("--sample", type=int, default=2000, help="frames sampled across dives with no slate template")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--container", default="slate-db")
    s.add_argument("--database", default="fishsense")
    s.set_defaults(func=cmd_scan_manifest)

    pr = sub.add_parser("predict", help="score cached frames of --manifest with the final model")
    pr.add_argument("--model", type=Path, default=MODEL)
    pr.add_argument("--out", type=Path, default=Path("runs/scan/predictions.csv"))
    pr.set_defaults(func=cmd_predict)

    rv = sub.add_parser("review", help="flagged frames as CSV + local HTML contact sheet")
    rv.add_argument("--predictions", type=Path, default=Path("runs/scan/predictions.csv"))
    rv.add_argument("--out", type=Path, default=Path("runs/scan/review"))
    rv.add_argument("--container", default="slate-db")
    rv.add_argument("--database", default="fishsense")
    rv.set_defaults(func=cmd_review)

    q = sub.add_parser("queue", help="blind slate/no-slate labelling page (in ~/Downloads)")
    q.add_argument("--predictions", type=Path, default=Path("runs/scan/predictions.csv"))
    q.add_argument("--queue-id", default="slate_q1")
    q.add_argument("--labeller", default="chris")
    q.add_argument("--checks", type=int, default=100, help="random unflagged frames mixed in to catch misses")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", type=Path, default=None, help="parent folder (default: XDG Downloads)")
    q.set_defaults(func=cmd_queue)

    ig = sub.add_parser("ingest", help="downloaded queue answers -> labels/manual/<labeller>/<queue_id>.csv")
    ig.add_argument("answers", type=Path)
    ig.set_defaults(func=cmd_ingest)

    for name, func, default_out in (("cv", cmd_cv, "runs/cv"), ("train", cmd_train, "runs/final")):
        t = sub.add_parser(name)
        t.add_argument("--out", type=Path, default=Path(default_out))
        t.add_argument("--arch", default="efficientnet_b0")
        t.add_argument("--epochs", type=int, default=12)
        t.add_argument("--image-size", type=int, default=1024, help="long side in pixels")
        t.add_argument("--batch-size", type=int, default=4)
        if name == "cv":
            t.add_argument("--folds", type=int, default=5)
        t.set_defaults(func=func)

    args = p.parse_args(argv)
    args.func(args)
