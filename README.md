# slate-detector

Is a FishSense dive slate anywhere in this frame? A binary classifier, meant to
find slate frames in dives nobody has labelled, so laser calibration can find them
without a person scanning every image.

Earlier slate work (`../2026-07-31_slate_training`, the retired
`predict_slate_image` in fishsense-lite) estimated slate *pose* and only ever
saw frames that contained a slate. This repo answers presence only.

## Data

**Labels.** Labels come from human Label Studio answers in a restored v1 backup:

- **slate**: `specieslabel.content_of_image` starting with `Slate` (Laser on
  slate, Laser not on slate, Slate not in list), or a completed `diveslatelabel`.
- **no slate**: every answer is `Fish, …`, `Fish Model, …`,
  `Calibration Targets, …` or `None`.
- **ambiguous**: anything mixed is dropped.

**Frames.** Only `image.is_canonical` frames are used. The same dive was
imported under several folders, and the canonical flag keeps one copy of each
frame, so splits on `dive_id` don't leak.

**Current count (2026-10-02 backup).** 5,938 frames: 644 slate and 5,294 no
slate (after `labels/overrides.csv`), across 76 dives.

**Known label noise.** About half the negatives come from dives that also have
slate frames. "Fish" there means "the answer was about the fish", so a slate
may still be visible in the background. Check high-scoring negatives in
`oof_predictions.csv` before trusting false-positive counts.

**Pixels.** Frames are rendered from the raw `.ORF` with fishsense-core
(`RawImage` with `DecodeConfig.production()`, then `RectifiedImage` with the
camera's `cameraintrinsics`). This is how fishsense-lite and
fishsense-detector render frames, so the classifier sees what the rest of the
pipeline sees.

We don't use the camera's embedded JPEG preview: it's rendered differently and
isn't rectified. We also don't use the checksum-named JPEGs in
`~/mnt/fishsense_process_work/preprocess_jpeg`: they mix pipeline versions,
some have the laser-region outline drawn on, and they cover only 3,800 of the
5,938 frames.

Every raw is first copied to `data/raw/{image_id}.ORF` and checked against
`image.checksum` (md5), so re-rendering never touches the NAS again. Raws
already in `~/.cache/cscw-fishsense2027/tail_stage` (375 of our frames, all
negatives) are hardlinked instead of copied. All the raws take about 100 GB.

Each frame takes ~15 s to read over the `~/mnt` FUSE mount and ~18 s to
decode. With 3 worker processes that comes to ~11 s/frame, so about 18 hours
for all frames. Frames are cached at 1600×1202 in
`data/frames/{image_id}.jpg`. The prefetch can be resumed, and it fetches slate
frames first, so cross-validation can run on a partial cache.

## Run

```sh
# 1. Restore a backup into a throwaway Postgres
docker run -d --name slate-db -e POSTGRES_PASSWORD=pw -v "$PWD/backup":/backup postgres:17
cp ~/mnt/fishsense_process_work/database_backups/fishsense/<latest>.dump backup/fs.dump
docker exec slate-db psql -U postgres -c 'create database fishsense'
docker exec slate-db pg_restore -U postgres -d fishsense --no-owner --no-acl /backup/fs.dump

# 2. Labelled frames -> data/manifest.csv
uv run slate-detector manifest

# 3. Copy raws locally (md5-verified) and render them into the cache (~18 h; resumable; slate frames first, then negatives round-robin by dive)
scripts/run prefetch

# 4. Cross-validation grouped by dive -> runs/cv/{cv_report.json,oof_predictions.csv}
scripts/run cv

# 5. Final model on everything (the current one was trained with --out runs/final-q1)
scripts/run train --out runs/final-q1
```

`scripts/run` sets the NixOS environment variables torch needs and caps memory.

## Model

The model is ImageNet EfficientNet-B0 with GeM pooling in place of average
pooling, because the slate is about 1% of the frame. Input is the whole frame
resized to 1024×768, never cropped. Training uses balanced sampling,
flips, colour jitter and AMP.

Batch 4 uses 2.8 GiB on the 6 GB laptop GPU. Batch 8 at 1024 px runs out of
memory.

## Results (2026-10-05)

The current model is `runs/final-q1/slate_efficientnet_b0.pt` (sha256
`b8d377ba22d155e7056a5e9ae747fdd0970c7c73dee981bbee17d95c8156cf78`), the
default for `predict`. It was trained on 6,828 frames (1,405 slate) from 240
dives: the database labels plus queue `slate_q1`.

5-fold CV grouped by dive (`runs/cv-q1/`) gives these numbers, excluding the
one-off cutting-board test slate (dives 508 and 510, "Slate not in list"):

| ROC AUC | AP | Precision @0.5 | Recall @0.5 | False alarms | Missed |
|---|---|---|---|---|---|
| 0.9994 | 0.9986 | 0.999 | 0.993 | 1 / 5,423 | 10 / 1,352 |

On new dives, 756 of the 758 frames flagged "likely slate" in queue
`slate_q1` were confirmed as slates. 0 of 100 random low-scoring frames were
missed slates (after correcting two mis-keys).

## Tests

`uv run pytest`

## Find and label slates in unlabelled dives

```sh
uv run slate-detector scan-manifest           # unlabelled frames: slate/checkerboard dives + 2,000 sampled elsewhere
scripts/run --manifest data/scan_manifest.csv prefetch
scripts/run --manifest data/scan_manifest.csv predict   # -> runs/scan/predictions.csv
scripts/run review                             # -> runs/scan/review/{to_label.csv,index.html} (local only)
uv run slate-detector queue                    # -> ~/Downloads/slate-queue-slate_q1/label.html
uv run slate-detector ingest ~/Downloads/slate-queue-slate_q1-answers.json
uv run slate-detector manifest && scripts/run cv
```

The queue page is blind: it never shows the model's score, and frames are
shuffled. Keys: `Y` slate, `N` no slate, `C` can't tell, `Z` zoom, arrows to
move. Answers are saved in the browser until you press "Download answers".
`ingest` writes `labels/manual/<labeller>/<queue_id>.csv`, which the manifest
uses ahead of the database answers. Only `labels/overrides.csv` takes
precedence over it. The id → image map is `labels/queues/<queue_id>_manifest.json`.

## Slate masks (SAM 3.1 + review)

```sh
uv run --extra masks slate-detector quads      # data/masks/quads.json: outlines warped from diveslatelabel points
FISHSENSE_SAM3_CHECKPOINT=~/Downloads/sam3.1_multiplex.pt scripts/run sam3-masks   # -> data/masks/sam3/{image_id}.npz
uv run --extra masks slate-detector mask-queue          # -> ~/Downloads/slate-masks-masks_q1/label.html
uv run --extra masks slate-detector mask-ingest ~/Downloads/slate-masks-masks_q1-answers.json
```

SAM 3.1 with the text prompt **"white board"** finds the slate board. "slate" and
"dive slate" return nothing; cscw-fishsense2027 found the same. SAM keeps up
to 5 candidates per frame, and one is chosen, in this order:
1. best overlap with the outline from the frame's point label;
2. the one containing the human laser dot;
3. the top score.

When nothing is found and an outline exists, SAM is retried on a crop around
it. SAM is not a presence detector (it also finds checkerboards, the Box
target and boat decks), so it only runs on frames labelled slate.

The review page shows the chosen candidate: `Y` accept, `N` reject, `C` can't
tell, `1`–`5` switch candidate, `Z` zoom. `mask-ingest` writes accepted masks
as polygons plus a fitted 4-corner quad (frame-cache 1600 px coordinates) to
`labels/masks/<labeller>/<queue_id>.json`.

Outlines warped from the label points alone are not accurate enough to train
on. The PDF page is smaller than the physical board, and on V-slates all 6
reference points sit on the V tape, so the corners extrapolate badly.

### Board fit and Label Studio

```sh
scripts/run rect-fit                       # data/masks/rect.json: rounded-rectangle pose per frame
uv run --extra masks --extra ls slate-detector ls-push                 # frames -> Garage, project + SAM quads (first time)
uv run --extra masks --extra ls --extra train slate-detector ls-predict <project> --kind rect   # board-fit outlines as predictions
uv run --extra masks --extra ls slate-detector ls-pull <project>       # -> labels/masks/<labeller>/ls_project_<id>.json
```

**Shape.** Every slate is the same 8.5 × 12 in board with rounded corners:
aspect 0.708, corner radius 3.1% of the width, measured on lightly occluded
frames of all families.

**Fit (`rectfit.fit_board`).** The board's 3D pose is fitted through the
camera's known intrinsics so that its outline covers SAM's mask. Area SAM didn't
see is penalised only lightly, because that's where hands are, so the outline
completes the edges hands cover. The fit tries every starting quad and corner
assignment, then picks the fit minimising cost + 0.04·(1 − cos tilt).
Without that prior, portrait-held boards could fit as 60°-tilted landscape
ones (cos 60° = 0.708²), and boards with hands on both edges could follow the
hand outline. `rect_v1.json` and `rect_v2.json` keep the earlier fits.

**Label Studio.** The lab's hosted Label Studio (`app.heartex.com`, workspace
FishSense) is used the way fishsense-lite does: frames in Garage
(`s3://labels-fishsense-lite/slate-detector/frames/`) are presigned through a
source storage, and model output is imported as predictions. Credentials live
in `~/.config/slate-detector/secrets.env`, outside git. Rule: outline the
whole board, including parts hidden by hands.

**For laser calibration (apparent size), use √(SAM mask area), not the board
fit.** Scored against tape with cscw-fishsense2027's e1j scorer: √(SAM area)
2.8%, board-fit depth at the dot 3.9%, √(fitted outline area) 3.5%.
`sam3-masks --label-free` chooses candidates by the laser dot and score only,
with no slate labels.
