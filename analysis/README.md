# crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose
estimation. The maze is a white cross/T-shaped binary decision tree
(4 arms) on a dark background, plus an entry corridor at the bottom of
the rig, filmed from below.

To locate each of 24 maze landmarks in a new video the script doesn't
re-detect the maze geometry from scratch — it searches for small
image patches taken from a calibration frame around where each
landmark was, and finds their best matches in the new frame using
**normalised cross-correlation template matching** (`cv2.matchTemplate`
with `TM_CCOEFF_NORMED`). A few robustness layers sit on top of that
core idea:

- **Median reference frame** instead of a single middle frame: 11
  frames are sampled evenly through the video and per-pixel medianed,
  so the moving mouse averages out and templates match the empty maze.
- **Phase-correlation global pre-alignment** between the calibration
  frame and each video's median frame, so per-landmark searches are
  anchored at the *shifted* position when the camera moved between
  sessions.
- **RANSAC homography + iterative refinement**: outliers are
  re-searched around their homography-predicted positions, recovering
  templates that initially locked onto the wrong nearby corner.
- **Multi-calibration** support for datasets that span more than one
  rig/camera setup — calibrate once per setup, drop the JSONs in a
  directory, and the batch picks the best fit per video.
- **`--skip_existing`** (default on) skips videos whose
  `*_cropped.mp4` is already on disk, so re-runs only touch new work.

Once you've clicked the 24 landmarks on one frame (per rig setup),
every subsequent video is processed without any thresholding, contour
finding, or corner detection.

## Workflow

```
        +------------+
        | calibrate  |   click 24 maze corners on the median frame
        |  per rig   |   -> calibration[_<name>].json
        +-----+------+      + calibration[_<name>]_frame.png
              |
              v
      +---------------+    median frame  ->  phase-correlation shift
      |     batch     |    ->  per-landmark template match
      |   processing  |    ->  RANSAC homography (inlier mask)
      +-------+-------+    ->  refinement (re-search non-inliers)
              |            ->  warp video
              |
              v
          +-------+
          | flagged|   any failed/medium/low video can be rescued
          | videos |   with --manual_video, which saves
          +---+----+   <stem>_landmarks.json next to the cropped
              |        output. Future batch runs pick those up
              v        automatically.
      +---------------+
      |   re-run      |   --skip_existing (default on) skips the
      |   batch       |   videos that were already cropped
      +---------------+
```

### Stage 1 — Calibration (`--calibrate`)

```
python crop_and_align_maze.py --calibrate \
    --input_dir  <path> \
    --output_dir <path> \
    [--patch_size 80] \
    [--calibration_file <output_dir>/calibration_<name>.json]
```

The script picks one video (the first found under `--input_dir`, or
the file you pass via `--calibrate_video`), computes its median frame,
opens it in an OpenCV window, and asks you to click all 24 maze
corners. Clicking on the median frame (instead of a raw frame) means
the mouse never occludes a landmark during calibration.

**Click order (clockwise from top-left):**

| #  | Landmark                                                          |
| -- | ----------------------------------------------------------------- |
| 1  | TL arm, outer top-left corner                                     |
| 2  | TL arm, outer top-right corner                                    |
| 3  | inner corner: TL arm meets central junction (top side)            |
| 4  | inner corner: TR arm meets central junction (top side)            |
| 5  | TR arm, outer top-left corner                                     |
| 6  | TR arm, outer top-right corner                                    |
| 7  | TR arm, outer bottom-right corner                                 |
| 8  | TR arm, outer bottom-left corner                                  |
| 9  | inner corner: TR arm meets central junction (bottom side)         |
| 10 | inner corner: junction meets corridor right (top)                 |
| 11 | inner corner: BR arm meets corridor (top side)                    |
| 12 | BR arm, outer top-right corner                                    |
| 13 | BR arm, outer bottom-right corner                                 |
| 14 | BR arm, outer bottom-left corner                                  |
| 15 | inner corner: BR arm meets corridor (bottom side)                 |
| 16 | corridor, right side, bottom corner                               |
| 17 | corridor, left side, bottom corner                                |
| 18 | inner corner: BL arm meets corridor (bottom side)                 |
| 19 | BL arm, outer bottom-right corner                                 |
| 20 | BL arm, outer bottom-left corner                                  |
| 21 | BL arm, outer top-left corner                                     |
| 22 | inner corner: BL arm meets corridor (top side)                    |
| 23 | inner corner: junction meets corridor left (top)                  |
| 24 | inner corner: TL arm meets central junction (bottom side)         |

Window controls:
- left-click — place the next point
- **r** — reset all clicks (works in placement and confirm phases)
- **y** / **Y** — confirm once 24 points are placed
- **ESC** — cancel without saving

On confirm the script writes (filenames derive from the JSON stem so
multiple calibrations can live in the same directory):

- `<stem>.json` — the 24 landmark positions, the patch size, the
  frame dimensions, and the basename of the saved frame.
- `<stem>_frame.png` — the *unannotated* median frame. Batch matching
  reads this to extract template patches; do not modify it.
- `<stem>_frame_annotated.png` — the same frame with the 24 landmarks
  numbered and the `--patch_size` patch boxes drawn around each point,
  for human reference.

For the default `calibration.json` the frame files keep their legacy
names (`calibration_frame.png`, `calibration_frame_annotated.png`).
For e.g. `calibration_rig2.json` they become
`calibration_rig2_frame.png` etc., so multiple calibrations in one
directory don't overwrite each other.

The script exits after calibration without processing any videos. If
`--patch_size` is too large for one of the clicked landmarks (the patch
falls outside the frame), a warning lists the affected landmarks;
those landmarks will be unmatched in batch mode unless you
re-calibrate with a smaller patch size.

### Stage 2 — Batch processing

```
python crop_and_align_maze.py \
    --input_dir  <path> \
    --output_dir <path>
```

For every video under `--input_dir`:

1. **Skip-if-existing.** If `<stem>_cropped.mp4` already exists in the
   output tree and `--skip_existing` is on (the default), the video is
   skipped entirely — no median sampling, no detection, no warp. Use
   `--no-skip_existing` to force everything to re-run, or `--redo_video
   <name>` to force one specific video.
2. **Median frame.** Read `--n_median_frames` frames (default 11)
   evenly through the video and take the per-pixel median. The mouse
   is in a different place in each sample, so the median collapses to
   the empty maze.
3. **Global pre-alignment.** Run `cv2.phaseCorrelate` between the
   calibration frame and the median frame to estimate a global
   `(dx, dy)` translation. Shift the 24 calibration landmark positions
   by that vector to get the predicted search centres in the new
   frame. (Set `--global_align off` to skip this.) The estimated shift
   is capped at ~35 % of the smaller image dimension; anything beyond
   that is treated as spurious and falls back to `(0, 0)`.
4. **Per-landmark template matching.** For each of the 24 landmarks,
   extract the saved `patch_size × patch_size` patch from the
   calibration frame and run `cv2.matchTemplate` (`TM_CCOEFF_NORMED`)
   inside a `±--search_radius` window around the *shifted* search
   centre. Take the peak as the detected position; mark the landmark
   matched if its score >= `--match_threshold`.
5. **RANSAC homography.** Compute
   `cv2.findHomography(method=cv2.RANSAC,
   ransacReprojThreshold=--ransac_threshold)` over the matched detected
   points → canonical positions. RANSAC returns both the homography
   and an inlier mask; outliers (template-matched points that don't
   fit the homography) are tracked separately from the matched mask.
6. **Iterative refinement.** Up to three rounds. Each round projects
   every *non-inlier* (RANSAC outliers + truly unmatched) landmark
   into the new frame via the current homography, re-runs template
   matching in a tight `--refine_radius` window around the prediction,
   and accepts the result if its score clears `--refine_threshold`
   (looser than `--match_threshold` because the search window is
   tight, so false positives are unlikely). The homography is then
   re-solved with the updated point set.
7. **Confidence & output.** If fewer than 12 RANSAC inliers survive,
   the video is marked `failed` and the warp is skipped. Otherwise,
   every frame is warped with the final homography and written as
   `<stem>_cropped.mp4` (mp4v codec, source FPS). A debug review PNG
   is written for `medium` / `low` / `failed` cases.

The 24 patches are extracted **once** per calibration at the start of
the batch run, not per video — the same set of templates is reused
across every video.

### Stage 2b — Multi-calibration (`--calibration_dir`)

When your dataset spans more than one rig/camera setup, calibrate once
per setup and put the JSONs (plus their frames) in a single directory:

```
python crop_and_align_maze.py \
    --input_dir       <path> \
    --output_dir      <path> \
    --calibration_dir <path>
```

The script loads every valid `*.json` from `--calibration_dir`, and
for each video runs Pass 1 detection with each calibration in turn.
The result with the **best fit** is chosen: higher confidence tier
wins (high > medium > low > failed), ties broken by more RANSAC
inliers, then by higher mean template score. The chosen calibration's
name is recorded in `alignment_summary.csv` as the `calibration_used`
column and printed alongside each video's log line.

`--calibration_dir` takes precedence over `--calibration_file` if both
are supplied. Calibrating an additional rig is a one-liner:

```
python crop_and_align_maze.py --calibrate \
    --calibrate_video  <a video from the new rig> \
    --output_dir       <existing output dir> \
    --calibration_file <existing output dir>/calibration_rig2.json
```

This saves `calibration_rig2.json` + `calibration_rig2_frame.png`
alongside the original `calibration.json`. Re-running the batch with
`--calibration_dir <existing output dir>` then auto-selects per video.

### Manual fallback (`--manual_video <path>`)

```
python crop_and_align_maze.py \
    --manual_video     <path> \
    --output_dir       <path> \
    --calibration_file <path>/calibration.json
```

For a video where automatic detection fails or looks wrong, this opens
the 24-click UI on that video's median frame. The clicked points are
saved as `<output_dir>/<relative path>/<stem>_landmarks.json`, the
homography is computed directly from those 24 points, and that one
video is warped + written.

**Subsequent batch runs pick up saved per-video landmarks
automatically.** When detection starts for a video it first looks for
`<stem>_landmarks.json` in the output tree; if found, it skips
template matching entirely and uses the saved 24-point assignment.
The console prints `using saved manual landmarks: ...`.

To force re-running automatic detection on a video that already has
saved manual landmarks, pass `--redo_video <name>` where the name
matches either the bare filename or the relative path
(`<mouse>/<session>/<file>.mp4`).

## Canonical layout

The canonical positions of the 24 landmarks are derived from the
calibration: each calibration point is translated so the bounding box
of all 24 sits at `(--padding, --padding)`, and the output canvas size
is `(bbox_width + 2*padding, bbox_height + 2*padding)`. Spacing and
proportions are preserved exactly — every video is warped so its 24
detected corners align with these canonical targets. When using
`--calibration_dir`, each calibration defines its own canonical layout,
so videos cropped against different calibrations may have different
output sizes.

## Confidence scoring

Confidence combines three signals:

- the **reprojection error** of the RANSAC homography, measured *over
  the inlier subset only*;
- the **number of RANSAC inliers** (template matches that fit the
  homography within `--ransac_threshold`);
- the **mean template correlation score** across the template-matched
  points.

| Confidence | Reproj. error (inliers) | Inliers | Mean match score |
| ---------- | ----------------------- | ------- | ---------------- |
| `high`     | < 10 px                 | >= 20   | > 0.5            |
| `medium`   | < 25 px                 | >= 16   | (any)            |
| `low`      | otherwise, with inliers >= 12 |   |                  |
| `failed`   | inliers < 12, or earlier failure  |       |                  |
| `skipped`  | `<stem>_cropped.mp4` already on disk and `--skip_existing` is on |  |  |

A `failed` video has no cropped output. `medium` and `low` videos do
have a cropped output but get a debug review PNG so you can decide
whether to keep them or rescue them with `--manual_video`.

For videos processed via saved manual landmarks, confidence is judged
on reprojection error and inlier count only — the mean template score
is not relevant because no template matching ran. (In practice, a
manual run produces 24 inliers and a very small reprojection error.)

## CLI

```
python crop_and_align_maze.py \
    --input_dir <path> \
    --output_dir <path> \
    [--padding 50] \
    [--patch_size 80] \
    [--search_radius 60] \
    [--match_threshold 0.3] \
    [--n_median_frames 11] \
    [--refine_radius 35] \
    [--refine_threshold 0.2] \
    [--ransac_threshold 10] \
    [--global_align phase|off] \
    [--exclude_dirs segments new_segments segments_detected deeplabcut habituation] \
    [--include_pattern "*.mp4,*.avi"] \
    [--calibrate] \
    [--calibrate_video <path>] \
    [--calibration_file <path>] \
    [--calibration_dir <path>] \
    [--manual_video <path>] \
    [--redo_video <name>] \
    [--skip_existing | --no-skip_existing]
```

| Argument              | Default                                          | Meaning                                                                                |
| --------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `--input_dir`         | —                                                | Folder to search recursively for videos. Required for batch mode.                      |
| `--output_dir`        | —                                                | Where cropped videos, calibration files, per-video manual landmarks, the review folder, and the CSV are written. |
| `--padding`           | 50                                               | Padding (px) around the bounding box of the canonical landmarks.                       |
| `--patch_size`        | 80                                               | Square template patch size in pixels. Used during `--calibrate`; the saved value is read back at batch time. |
| `--search_radius`     | 60                                               | How far (px) from each (shifted) calibration landmark to search for the template.      |
| `--match_threshold`   | 0.3                                              | Minimum normalised cross-correlation for an initial template match.                    |
| `--n_median_frames`   | 11                                               | Frames sampled evenly through each video for the per-pixel median reference. Set to 1 for a single middle frame. |
| `--refine_radius`     | 35                                               | Window (px) around the homography-predicted position when re-searching non-inliers.    |
| `--refine_threshold`  | 0.2                                              | Looser match threshold used during refinement re-search.                               |
| `--ransac_threshold`  | 10                                               | RANSAC reprojection threshold (px). A template match is an inlier if its reprojection error is below this. |
| `--global_align`      | `phase`                                          | Use phase correlation to estimate a global `(dx, dy)` between calibration and video before per-landmark search. Use `off` to disable. |
| `--exclude_dirs`      | `segments new_segments segments_detected deeplabcut habituation` | Directory names to skip when walking `input_dir` (case-insensitive).                   |
| `--include_pattern`   | `*.mp4,*.avi`                                    | Comma-separated `fnmatch` globs used to pick which filenames count as videos.          |
| `--calibrate`         | off                                              | Enter calibration mode. Exits without processing videos.                               |
| `--calibrate_video`   | first video in `--input_dir`                     | In `--calibrate` mode, the specific video to calibrate on.                             |
| `--calibration_file`  | `<output_dir>/calibration.json`                  | Path to a single calibration JSON. Read in batch / manual modes, written in calibrate mode. Ignored when `--calibration_dir` is set. |
| `--calibration_dir`   | off                                              | Directory of multiple calibration JSONs. Batch tries each per video and picks the best fit. |
| `--manual_video`      | off                                              | Click 24 landmarks for one specific video, save `<stem>_landmarks.json`, and process that video. |
| `--redo_video`        | off                                              | In batch mode, force re-detection for the named video (bypasses both saved manual landmarks and `--skip_existing`). |
| `--skip_existing`     | on                                               | Skip videos whose `<stem>_cropped.mp4` already exists. Use `--no-skip_existing` to force re-processing. |

If you run batch mode without a calibration (`--calibration_file`
doesn't exist and `--calibration_dir` wasn't given) the script prints
a clear error pointing you to `--calibrate`.

## Outputs

```
<output_dir>/
├── calibration.json                          # default-name calibration
├── calibration_frame.png                     # used by batch matching
├── calibration_frame_annotated.png           # for human reference
├── calibration_rig2.json                     # (optional) extra rigs
├── calibration_rig2_frame.png                # (one set per calibration)
├── calibration_rig2_frame_annotated.png
├── alignment_summary.csv                     # one row per input video
├── <mouse>/<session>/<name>_cropped.mp4
├── <mouse>/<session>/<name>_landmarks.json   # only if --manual_video was used
├── ...
└── review/
    └── <mouse>/<session>/<name>_review.png   # for medium/low/failed
```

`alignment_summary.csv` columns:

| Column                | Meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `filename`            | Path of the input video, relative to `--input_dir`.           |
| `calibration_used`    | Name (JSON stem) of the calibration chosen for this video. Blank for `skipped` rows. |
| `num_matched_points`  | Number of landmarks whose template peak score met `--match_threshold`. |
| `num_inlier_points`   | Number of those matches that survived RANSAC at `--ransac_threshold`. Drives the confidence rating. |
| `mean_match_score`    | Mean normalised cross-correlation score across template-matched points (NaN for manual). |
| `reprojection_error`  | Mean reprojection error over the RANSAC inlier subset (px).   |
| `confidence`          | `high` / `medium` / `low` / `failed` / `skipped`.             |
| `output_path`         | Path of the cropped video, or `error: ...` on failure.        |

After the CSV is saved, the script prints a confidence breakdown to
the console (counts of `high` / `medium` / `low` / `failed` /
`skipped`).

## Dependencies

- Python 3.9+ (for `argparse.BooleanOptionalAction`)
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

## Reviewing flagged videos

The review PNG for a flagged or failed video shows:

- **Green filled circles** + landmark numbers at *detected* positions
  for landmarks that ended up matched.
- **Red hollow circles** + landmark numbers at the *calibration*
  positions for unmatched landmarks (so you can see exactly which
  corner the matcher couldn't find).
- **Yellow lines** from each calibration position to its detected
  position for matched landmarks — the visible "shift" tells you
  whether the camera moved a lot (long lines) or barely at all (short
  lines).
- A header showing `auto` or `manual` source, matched count, mean
  template score, reprojection error, and the confidence label.

Per-video log lines also show:

- the estimated global shift `(dx, dy)` when phase correlation found a
  non-trivial translation, and
- the calibration chosen (`calib=<name>`) when running in multi-
  calibration mode.

Triage workflow:

1. **`high` and `medium` videos** are usually fine. Sample a few
   medium ones in the cropped folder to confirm the maze ended up
   where you expect.
2. **`low` videos** still have a cropped output. Open the review PNG:
   if a few unmatched red circles are concentrated in one part of the
   frame, the alignment is probably fine and the low label came from
   stragglers. If the matched yellow lines are long or inconsistent,
   rescue with `--manual_video`.
3. **`failed` videos** have no cropped output. By far the most common
   cause in a multi-rig dataset is **wrong calibration** — entire
   mouse cohorts failing together is the signature. Calibrate a new
   rig from one of their videos, drop the JSON into
   `--calibration_dir`, and re-run. Other causes:
     - Mouse was sitting on a critical landmark in *every* sampled
       frame — try a larger `--n_median_frames` to denoise more.
     - Camera shift exceeded the search window — raise
       `--search_radius` and/or `--refine_radius`.
     - Very noisy templates — lower `--match_threshold` (e.g. 0.2) at
       the risk of more false matches; the RANSAC inlier filter will
       drop them but you may end up with fewer inliers.
     - Rescue with `--manual_video <path>`. Clicks are saved next to
       the cropped output and future batch runs pick them up.
4. **Re-run automatic detection** on a video that already has saved
   manual landmarks with `--redo_video <name>`. This also bypasses
   `--skip_existing`.
5. **`*_frame_annotated.png`** is the canonical reference for each
   calibration: if the patch boxes look mis-clicked, re-calibrate.

## Tuning notes

- **`--patch_size`** controls how distinctive each template is. Larger
  patches are more discriminative (fewer false matches) but more
  sensitive to small camera rotation/scale changes — and they're more
  likely to clip outside the frame near the edges of the maze. 80 px
  is a reasonable starting point for ~720 p ventral recordings; halve
  or double it if your videos are much smaller or larger.
- **`--search_radius`** is the per-landmark search window *after* the
  global phase-correlation shift. Because the global shift already
  takes care of most of the camera translation, this can stay tight
  (60 px). If you've disabled `--global_align` you may want to raise
  it to ~150 px.
- **`--ransac_threshold`** controls how strict "is this match
  consistent with the homography" is. Smaller values (e.g. 5) give a
  tighter fit at the cost of fewer inliers; larger values (e.g. 15-20)
  keep more inliers but loosen the fit. 10 is a sensible default for
  noisy template matches with ~1-2 px quantisation.
- **`--refine_radius`** is the window for the refinement re-search
  around homography-predicted positions. If the first homography is
  built from few inliers it may extrapolate badly to far landmarks;
  raising this from 35 to 50-60 gives the re-search more room to
  recover them.
- **`--n_median_frames`** trades off median-frame quality vs runtime.
  11 is fine for most videos; raise to 21 if a mouse is unusually
  stationary in the sampled frames, drop to 1 for a single middle
  frame if you want to debug something quickly.
- **Multi-rig datasets**: if a coherent cohort of mice all fail with
  similar `num_inlier_points` counts, that's the signal to add
  another calibration via `--calibration_dir`, not to keep tuning
  parameters.
