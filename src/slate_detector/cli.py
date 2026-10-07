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


def cmd_quads(args) -> None:
    # pylint: disable=import-outside-toplevel
    import shutil

    import numpy as np

    from slate_detector import masks as M
    from slate_detector.db import DockerPsqlRunner
    from slate_detector.raws import raw_path

    db = DockerPsqlRunner(args.container, args.database)
    pdfs = {}
    for r in db.query("SELECT id, path FROM diveslate"):
        dst = DATA / "templates" / f"{r['id']}.pdf"
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_path(r["path"]), dst)
        pdfs[int(r["id"])] = dst
    quads, failures = M.build_quads(db.query(M.LABELS_SQL), pdfs)
    M.write_quads(quads, args.out)
    rms = np.array([q.rms_px for q in quads])
    print(f"{len(quads)} outlines ({len(failures)} failed) -> {args.out}; fit rms median {np.median(rms):.1f}px, max {rms.max():.1f}px")


def cmd_sam3_masks(args) -> None:
    # pylint: disable=import-outside-toplevel
    from collections import Counter

    import numpy as np
    from PIL import Image

    from slate_detector import segment as G
    from slate_detector.db import DockerPsqlRunner
    from slate_detector.masks import read_quads
    from slate_detector.train import require_free_gpu

    frames = [f for f in read_manifest(args.manifest) if f.label == 1]
    if args.limit:
        frames = frames[: args.limit]
    # --label-free: no slate labels at all. The candidate is chosen by the laser dot, then
    # the top score, and the crop retry (which needs a label outline) is skipped. For
    # label-free calibration (cscw-fishsense2027 e1j); the default prefers the label outline.
    outlines = {} if args.label_free else ({q.image_id: q.quad for q in read_quads(args.quads)} if args.quads.is_file() else {})
    dots = G.laser_dots(DockerPsqlRunner(args.container, args.database), [f.image_id for f in frames])
    print(f"{len(frames)} slate frames; {sum(f.image_id in outlines for f in frames)} with a label outline, "
          f"{sum(f.image_id in dots for f in frames)} with a laser dot", flush=True)
    require_free_gpu()
    sam = G.Sam3Boards()
    rows, start = [], time.time()
    for n, f in enumerate(frames, 1):
        out = args.out / f"{f.image_id}.npz"
        if out.is_file():
            c, chosen, reason = G.load_candidates(out)
        else:
            rgb = np.asarray(Image.open(args.cache / f"{f.image_id}.jpg").convert("RGB"))
            c = sam.segment(rgb, outlines.get(f.image_id))
            chosen, reason = G.choose(c, outlines.get(f.image_id), dots.get(f.image_id))
            G.save_candidates(out, c, chosen, reason)
        rows.append(G.summary_row(f.image_id, c, chosen, reason))
        if n % 50 == 0:
            print(f"{n}/{len(frames)}  {(time.time() - start) / n:.2f}s/frame", flush=True)
    G.write_jsonl(rows, args.out / "summary.jsonl")
    print(f"done: {dict(Counter(r['reason'] for r in rows))}; crop retries that found a board: {sum(r['crop'] for r in rows)}")


def _rect_job(job):
    """Worker: fit the board to one frame's chosen SAM mask (module-level so it pickles)."""
    # pylint: disable=import-outside-toplevel
    from slate_detector import rectfit as R
    from slate_detector.segment import load_candidates

    image_id, npz, camera_matrix = job
    c, k, _ = load_candidates(Path(npz))
    if k < 0:
        return image_id, None
    f = R.fit_board(c.masks[k], R.scaled_K(camera_matrix))
    if f is None:
        return image_id, None
    return image_id, {"polygon": f.polygon.round(2).tolist(), "corners": f.corners.round(2).tolist(),
                      "covered": round(f.covered, 4), "extra": round(f.extra, 4), "cost": round(f.cost, 5),
                      "plausible": R.plausible(f), "rvec": f.rvec.tolist(), "tvec": f.tvec.tolist()}


def cmd_rect_fit(args) -> None:
    # pylint: disable=import-outside-toplevel
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from slate_detector.db import DockerPsqlRunner
    from slate_detector.render import load_intrinsics

    intr = load_intrinsics(DockerPsqlRunner(args.container, args.database))
    cams = {f.image_id: f.camera_id for f in read_manifest(args.manifest)}
    done = json.loads(args.out.read_text()) if args.out.is_file() else {}
    jobs = [(int(p.stem), str(p), intr[cams[int(p.stem)]][0]) for p in sorted(args.masks.glob("*.npz"))
            if p.stem not in done and int(p.stem) in cams]
    start = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for n, (iid, res) in enumerate(pool.map(_rect_job, jobs, chunksize=4), 1):
            done[str(iid)] = res
            if n % 100 == 0 or n == len(jobs):
                args.out.write_text(json.dumps(done))
                print(f"{n}/{len(jobs)}  {(time.time() - start) / n:.2f}s/frame", flush=True)
    args.out.write_text(json.dumps(done))
    fits = [v for v in done.values() if v]
    print(f"{len(fits)} fitted, {sum(v['plausible'] for v in fits)} plausible -> {args.out}")


def cmd_mask_queue(args) -> None:
    from slate_detector import maskqueue as MQ  # pylint: disable=import-outside-toplevel

    ids = sorted(int(p.stem) for p in args.masks.glob("*.npz"))
    out = (args.out or downloads_dir()) / f"slate-masks-{args.queue_id}"
    page = MQ.build(out, ids, args.cache, args.masks, args.queue_id, args.labeller,
                    Path("labels/queues") / f"{args.queue_id}_manifest.json", seed=args.seed)
    print(f"{len(ids)} frames -> {page}")


def cmd_mask_ingest(args) -> None:
    from collections import Counter  # pylint: disable=import-outside-toplevel

    from slate_detector import maskqueue as MQ  # pylint: disable=import-outside-toplevel

    answers = json.loads(Path(args.answers).read_text())
    manifest = json.loads((Path("labels/queues") / f"{answers['queue_id']}_manifest.json").read_text())
    rows = MQ.ingest(answers, manifest, args.masks)
    out = Path("labels/masks") / manifest["labeller"] / f"{manifest['queue_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows) + "\n")
    acc = [r for r in rows if r["answer"] == "accept"]
    print(f"{len(rows)}/{len(manifest['items'])} answered -> {out}: {dict(Counter(r['answer'] for r in rows))}")
    print(f"accepted with a different candidate than SAM's choice: {sum(r['instance'] != r['auto_instance'] for r in acc)}; "
          f"without a 4-corner fit: {sum(r['quad'] is None for r in acc)}")


def _ls_frames(args):
    from PIL import Image  # pylint: disable=import-outside-toplevel

    frames = sorted((f for f in read_manifest(args.manifest) if f.label == 1 and (args.masks / f"{f.image_id}.npz").is_file()),
                    key=lambda f: (f.dive_id, f.image_id))
    sizes = {f.image_id: Image.open(args.cache / f"{f.image_id}.jpg").size for f in frames}
    return frames, sizes


def cmd_ls_push(args) -> None:
    # pylint: disable=import-outside-toplevel
    from slate_detector import labelstudio as L

    frames, sizes = _ls_frames(args)
    if args.dry_run:
        tasks = [L.build_task(f.image_id, f.dive_id, f"s3://labels-fishsense-lite/{L.frame_key('slate-detector', f.image_id)}",
                              *sizes[f.image_id], args.masks / f"{f.image_id}.npz") for f in frames]
        n_pred = sum("predictions" in t for t in tasks)
        n_quad = sum(len(t["predictions"][0]["result"][0]["value"]["points"]) == 4 for t in tasks if "predictions" in t)
        print(f"dry run: {len(tasks)} tasks, {n_pred} with a prediction ({n_quad} 4-corner); first task:")
        print(json.dumps(tasks[0])[:600])
        return
    secrets = L.load_secrets()
    L.require(secrets, "FISHSENSE_LABEL_STUDIO_URL", "FISHSENSE_LABEL_STUDIO_API_KEY",
              "SLATE_S3_ENDPOINT", "SLATE_S3_BUCKET", "SLATE_S3_ACCESS_KEY", "SLATE_S3_SECRET_KEY")
    garage = L.Garage(secrets)
    have = garage.existing()
    todo = [f for f in frames if L.frame_key(garage.prefix, f.image_id) not in have]
    for n, f in enumerate(todo, 1):
        garage.upload(f.image_id, args.cache / f"{f.image_id}.jpg")
        if n % 100 == 0 or n == len(todo):
            print(f"uploaded {n}/{len(todo)} frames", flush=True)
    ls = L.client(secrets)
    ws = L.workspace_id(ls, secrets.get("FISHSENSE_LABEL_STUDIO_WORKSPACE", ""))
    pid, created = L.get_or_create_project(ls, args.title, ws)
    new_storage = L.ensure_storage(ls, pid, secrets)
    known = L.existing_image_ids(ls, pid)
    tasks = [L.build_task(f.image_id, f.dive_id, garage.url(f.image_id), *sizes[f.image_id], args.masks / f"{f.image_id}.npz")
             for f in frames if f.image_id not in known]
    n = L.import_tasks(ls, pid, tasks)
    print(f"project {pid} ({'created' if created else 'existing'}; storage {'registered' if new_storage else 'already there'}): "
          f"imported {n} tasks ({len(known)} already present) -> {secrets['FISHSENSE_LABEL_STUDIO_URL'].rstrip('/')}/projects/{pid}")


def cmd_ls_predict(args) -> None:
    """Add a new prediction set to every task of a project and make it the visible one."""
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    from slate_detector import labelstudio as L

    secrets = L.load_secrets()
    L.require(secrets, "FISHSENSE_LABEL_STUDIO_URL", "FISHSENSE_LABEL_STUDIO_API_KEY")
    ls = L.client(secrets)
    version = L.HULL_VERSION if args.kind == "hull" else L.RECT_VERSION
    rects = json.loads(args.rects.read_text()) if args.kind == "rect" else {}
    tasks = list(ls.tasks.list(project=args.project, fields="all"))
    added = skipped = fallback = 0
    for n, t in enumerate(tasks, 1):
        if any((p.get("model_version") if isinstance(p, dict) else getattr(p, "model_version", None)) == version
               for p in (t.predictions or [])):
            skipped += 1
            continue
        iid = int(t.data["image_id"])
        w, h = Image.open(args.cache / f"{iid}.jpg").size
        if args.kind == "rect":
            pred, fb = L.rect_prediction(rects.get(str(iid)), args.masks / f"{iid}.npz", w, h)
            fallback += fb
        else:
            pred = L.hull_prediction(args.masks / f"{iid}.npz", w, h)
        if pred is None:
            continue
        try:
            L.call(ls.predictions.create, task=t.id, result=pred["result"], score=pred["score"], model_version=pred["model_version"])
        except Exception as exc:  # pylint: disable=broad-except
            # The access token minted from the refresh key expires during a long run
            # (2026-10-07: a 401 after ~800 tasks); a fresh client mints a new one.
            if getattr(exc, "status_code", None) != 401:
                raise
            ls = L.client(secrets)
            L.call(ls.predictions.create, task=t.id, result=pred["result"], score=pred["score"], model_version=pred["model_version"])
        added += 1
        if n % 100 == 0:
            print(f"{n}/{len(tasks)} tasks", flush=True)
    L.call(ls.projects.update, id=args.project, model_version=version)
    print(f"added {added} {args.kind} predictions ({skipped} already had one, {fallback} fell back to the hull); "
          f"project now shows {version}")


def cmd_ls_pull(args) -> None:
    # pylint: disable=import-outside-toplevel
    from collections import Counter

    from slate_detector import labelstudio as L

    secrets = L.load_secrets()
    L.require(secrets, "FISHSENSE_LABEL_STUDIO_URL", "FISHSENSE_LABEL_STUDIO_API_KEY")
    ls = L.client(secrets)
    rows = L.annotations_to_rows(L.export_tasks(ls, args.project))
    out = Path("labels/masks") / args.labeller / f"ls_project_{args.project}.json"
    L.write_rows(rows, out)
    print(f"{len(rows)} annotated tasks -> {out}: {dict(Counter(r['answer'] for r in rows))}; "
          f"4-corner: {sum(r['quad'] is not None for r in rows)}")


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

    qd = sub.add_parser("quads", help="slate outlines warped from diveslatelabel points (prompts for sam3-masks)")
    qd.add_argument("--out", type=Path, default=DATA / "masks" / "quads.json")
    qd.add_argument("--container", default="slate-db")
    qd.add_argument("--database", default="fishsense")
    qd.set_defaults(func=cmd_quads)

    sm = sub.add_parser("sam3-masks", help="SAM 3.1 'white board' candidates for every slate frame (resumable)")
    sm.add_argument("--out", type=Path, default=DATA / "masks" / "sam3")
    sm.add_argument("--quads", type=Path, default=DATA / "masks" / "quads.json")
    sm.add_argument("--limit", type=int, default=0)
    sm.add_argument("--label-free", action="store_true",
                    help="ignore slate labels: choose by laser dot, then score; no label-guided crop retry (use with --out data/masks/sam3_labelfree)")
    sm.add_argument("--container", default="slate-db")
    sm.add_argument("--database", default="fishsense")
    sm.set_defaults(func=cmd_sam3_masks)

    mq = sub.add_parser("mask-queue", help="review page for SAM 3.1 slate masks (in ~/Downloads)")
    mq.add_argument("--masks", type=Path, default=DATA / "masks" / "sam3")
    mq.add_argument("--queue-id", default="masks_q1")
    mq.add_argument("--labeller", default="chris")
    mq.add_argument("--seed", type=int, default=0)
    mq.add_argument("--out", type=Path, default=None, help="parent folder (default: XDG Downloads)")
    mq.set_defaults(func=cmd_mask_queue)

    mi = sub.add_parser("mask-ingest", help="downloaded mask answers -> labels/masks/<labeller>/<queue_id>.json")
    mi.add_argument("answers", type=Path)
    mi.add_argument("--masks", type=Path, default=DATA / "masks" / "sam3")
    mi.set_defaults(func=cmd_mask_ingest)

    lp = sub.add_parser("ls-push", help="frames to Garage + a Label Studio project with SAM quads as predictions")
    lp.add_argument("--masks", type=Path, default=DATA / "masks" / "sam3")
    lp.add_argument("--title", default="Slate masks (research) q1")
    lp.add_argument("--dry-run", action="store_true", help="build the tasks locally; no network")
    lp.set_defaults(func=cmd_ls_push)

    lh = sub.add_parser("ls-predict", help="add a prediction set (hull or fitted rectangle) to a project and show it")
    lh.add_argument("project", type=int)
    lh.add_argument("--kind", choices=("hull", "rect"), default="rect")
    lh.add_argument("--masks", type=Path, default=DATA / "masks" / "sam3")
    lh.add_argument("--rects", type=Path, default=DATA / "masks" / "rect.json")
    lh.set_defaults(func=cmd_ls_predict)

    rf = sub.add_parser("rect-fit", help="fit the board (rounded rectangle, known camera) to every SAM mask")
    rf.add_argument("--masks", type=Path, default=DATA / "masks" / "sam3")
    rf.add_argument("--out", type=Path, default=DATA / "masks" / "rect.json")
    rf.add_argument("--workers", type=int, default=12)
    rf.add_argument("--container", default="slate-db")
    rf.add_argument("--database", default="fishsense")
    rf.set_defaults(func=cmd_rect_fit)

    lq = sub.add_parser("ls-pull", help="Label Studio annotations -> labels/masks/<labeller>/ls_project_<id>.json")
    lq.add_argument("project", type=int)
    lq.add_argument("--labeller", default="chris")
    lq.set_defaults(func=cmd_ls_pull)

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
