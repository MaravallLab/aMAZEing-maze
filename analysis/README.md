# crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose
estimation. The maze is a white cross/T-shaped binary decision tree
(4 arms) on a dark background, plus an entry corridor at the bottom of
the rig, filmed from below.

The camera position only shifts slightly between sessions, so to find
each of 24 maze landmarks in a new frame the script doesn't need to
re-detect the maze geometry from scratch — it just searches for a
small image patch from a single calibration frame around where each
landmark was, and locates its best match in the new frame using
**normalised cross-correlation template matching** (`cv2.matchTemplate`
with `TM_CCOEFF_NORMED`).

This is a one-time-calibration approach. Once you've clicked the 24
landmarks on a single frame, every subsequent video is processed
without any thresholding, contour finding, or corner detection.

## Workflow

```
                +------------+
                | calibrate  |   click 24 maze corners on one frame
                |  (once)    |   -> calibration.json + calibration_frame.png
                +-----+------+
                      |
                      v
              +---------------+
              |     batch     |   for each video, for each landmark:
              |   processing  |   matchTemplate( saved patch ,
              +-------+-------+                  ±search_radius window )
                      |        -> RANSAC homography over matches
                      |        -> warp every frame
                      v
                  +-------+
                  | flagged|   any failed/medium/low video can be
                  | videos |   rescued with --manual_video, which
                  +---+----+   saves <stem>_landmarks.json next to
                      |        the cropped output. Future batch runs
                      v        pick those up automatically.
              +---------------+
              |   re-run      |
              |   batch       |   uses saved manual landmarks
              +---------------+
```

### Stage 1 — Calibration (`--calibrate`)

```
python crop_and_align_maze.py --calibrate \
    --input_dir  <path> \
    --output_dir <path> \
    [--patch_size 80]
```

The script picks one video (the first found under `--input_dir`, or
the file passed via `--calibrate_video`), opens its middle frame in
an OpenCV window, and asks you to click all 24 maze corners. A small
reference diagram in the top-right corner shows the numbered click
order so you don't have to memorise it.

**Click order (clockwise from top-left):**

```
        1________2                      5________6
        |        \                     /         |
        |         3___________________4          |
        |         |                   |          |
       24         |                   |          7
         \        |    central        |        /
          \       |    junction       |       /
           +======23 ================10======+
           |      |                   |      |
          BL     22                  11     BR
          arm     |                   |     arm
           |      |                   |      |
          21      |                   |     12
           \      |     entry         |     /
            20    |    corridor       |    13
             \    |                   |    /
             19   |                   |   14
              \   |                   |   /
               18__|                   |__15
                  17_________________16
```

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

On confirm, the script writes:
- `calibration.json` — the 24 landmark positions, the patch size,
  the calibration frame's dimensions, and the basename of the saved
  calibration frame.
- `calibration_frame.png` — the *unannotated* middle frame. This is
  what batch matching reads to extract template patches; do not
  modify it.
- `calibration_frame_annotated.png` — the same frame with the 24
  landmarks numbered and the `--patch_size` patch boxes drawn around
  each point, for human reference.

The script then exits without processing any videos. If `--patch_size`
is too large for one of the clicked landmarks (the patch falls outside
the frame), a warning is printed listing the affected landmarks; those
landmarks will be unmatched in batch mode unless you re-calibrate with
a smaller patch size.

### Stage 2 — Batch processing

```
python crop_and_align_maze.py \
    --input_dir  <path> \
    --output_dir <path> \
    [--search_radius 150] \
    [--match_threshold 0.3]
```

For every video under `--input_dir`:

1. Sample the middle frame.
2. Convert it to grayscale.
3. For each of the 24 calibration landmarks:
   - Extract the corresponding `patch_size`×`patch_size` patch from
     `calibration_frame.png` (centred on the calibration point).
   - Build a search window in the new frame: a square region centred
     on the calibration point that lets the patch's *centre* land
     anywhere within ±`--search_radius` pixels (clipped to the frame).
   - Run `cv2.matchTemplate` with `TM_CCOEFF_NORMED` and take the
     peak correlation as the detected position.
   - If the peak score is below `--match_threshold`, mark the
     landmark as unmatched.
4. Solve a RANSAC homography
   (`cv2.findHomography(method=cv2.RANSAC, ransacReprojThreshold=5.0)`)
   over the matched detected points → canonical positions.
5. If fewer than 12 of the 24 landmarks matched, mark the video
   `failed` and skip the warp.
6. Otherwise, warp every frame of the video with the homography and
   write the cropped output to
   `<output_dir>/<relative session path>/<name>_cropped.mp4` at the
   source FPS using the `mp4v` codec.
7. Save a debug PNG under `review/` for any `medium` / `low` /
   `failed` video showing matched and unmatched landmarks.
8. Write `alignment_summary.csv` and print a confidence breakdown.

The 24 patches are extracted **once** at the start of the batch run,
not per-video — the same set of templates is reused across every
video. Only the search step (one `matchTemplate` call per landmark per
video) actually depends on the new frame.

### Manual fallback (`--manual_video <path>`)

```
python crop_and_align_maze.py \
    --manual_video      <path> \
    --calibration_file  calibration.json \
    --output_dir        <path>
```

For a video where automatic template matching fails or looks wrong,
this opens the same 24-click UI on that video. The clicked points are
saved as `<output_dir>/<relative path>/<stem>_landmarks.json`, the
homography is computed directly from those 24 points, and that one
video is warped + written.

**Subsequent batch runs pick up saved per-video landmarks
automatically.** When `detect_one` starts a video it first looks for
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
detected corners align with these canonical targets.

## Confidence scoring

Confidence combines three signals: the **reprojection error** of the
RANSAC homography, the **number of matched landmarks** (peak score
above threshold), and the **mean template correlation score** across
all 24 points (matched and unmatched).

| Confidence | Reproj. error | Matched points | Mean match score |
| ---------- | ------------- | -------------- | ---------------- |
| `high`     | < 10 px       | >= 20          | > 0.5            |
| `medium`   | < 25 px       | >= 16          | (any)            |
| `low`      | otherwise, with matched >= 12 |        |                  |
| `failed`   | matched < 12, or earlier failure   |        |                  |

A `failed` video has no cropped output. `medium` and `low` videos do
have a cropped output but get a debug review PNG so you can decide
whether to keep them or rescue them with `--manual_video`.

For videos processed via saved manual landmarks, confidence is judged
on `reprojection_error` and `num_matched_points` only — the mean
template score is not relevant because no template matching ran. (In
practice, a manual run produces `num_matched=24` and a very small
reprojection error.)

## CLI

```
python crop_and_align_maze.py \
    --input_dir <path> \
    --output_dir <path> \
    [--padding 50] \
    [--patch_size 80] \
    [--search_radius 150] \
    [--match_threshold 0.3] \
    [--exclude_dirs segments new_segments segments_detected deeplabcut habituation] \
    [--include_pattern "*.mp4,*.avi"] \
    [--calibrate] \
    [--calibrate_video <path>] \
    [--calibration_file <path>] \
    [--manual_video <path>] \
    [--redo_video <name>]
```

| Argument              | Default                                 | Meaning                                                                                |
| --------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| `--input_dir`         | —                                       | Folder to search recursively for videos. Required for batch mode.                      |
| `--output_dir`        | —                                       | Where cropped videos, calibration files, per-video manual landmarks, the review folder, and the CSV are written. |
| `--padding`           | 50                                      | Padding (px) around the bounding box of the canonical landmarks.                       |
| `--patch_size`        | 80                                      | Square template patch size in pixels. Used during `--calibrate`; the saved value is read back at batch time. |
| `--search_radius`     | 150                                     | How far (px) from each calibration landmark to search for the template in a new frame. |
| `--match_threshold`   | 0.3                                     | Minimum normalised cross-correlation score for a template match to count as "matched". |
| `--exclude_dirs`      | `segments new_segments segments_detected deeplabcut habituation` | Directory names to skip when walking `input_dir` (case-insensitive).                   |
| `--include_pattern`   | `*.mp4,*.avi`                           | Comma-separated `fnmatch` globs used to pick which filenames count as videos.          |
| `--calibrate`         | off                                     | Enter calibration mode. Exits without processing videos.                               |
| `--calibrate_video`   | first video in `--input_dir`            | In `--calibrate` mode, the specific video to calibrate on.                             |
| `--calibration_file`  | `<output_dir>/calibration.json`         | Path to the calibration JSON. Read in batch / manual modes, written in calibrate mode. |
| `--manual_video`      | off                                     | Click 24 landmarks for one specific video, save `<stem>_landmarks.json`, and process that video. |
| `--redo_video`        | off                                     | In batch mode, force template-matching re-detection for the named video.               |

If you run batch mode without a calibration file the script prints a
clear error pointing you to `--calibrate`.

## Outputs

```
<output_dir>/
├── calibration.json                       # 24 landmarks + patch_size + frame size
├── calibration_frame.png                  # unannotated middle frame (used by batch matching)
├── calibration_frame_annotated.png        # same frame with landmarks + patch boxes drawn
├── alignment_summary.csv                  # one row per input video
├── <mouse>/<session>/<name>_cropped.mp4
├── <mouse>/<session>/<name>_landmarks.json     # only if --manual_video was used
├── ...
└── review/
    └── <mouse>/<session>/<name>_review.png     # for medium/low/failed
```

`alignment_summary.csv` columns:

| Column                | Meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `filename`            | Path of the input video, relative to `--input_dir`.           |
| `num_matched_points`  | Number of landmarks whose template peak score met the threshold. |
| `mean_match_score`    | Mean normalised cross-correlation score across all 24 landmarks (NaN for videos processed via saved manual landmarks). |
| `reprojection_error`  | Mean reprojection error of the matched landmarks (px).        |
| `confidence`          | `high` / `medium` / `low` / `failed`.                         |
| `output_path`         | Path of the cropped video, or an `error: ...` string on failure. |

After the CSV is saved, the script also prints a confidence breakdown
to the console (counts of `high` / `medium` / `low` / `failed`).

## Dependencies

- Python 3
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

## Reviewing flagged videos

The review PNG for a flagged or failed video shows:

- **Green filled circles** + landmark numbers at *detected* positions
  for landmarks where the peak template score met the match threshold.
- **Red hollow circles** + landmark numbers at *calibration* positions
  for unmatched landmarks (so you can see exactly which corner the
  matcher couldn't find).
- **Yellow lines** from each calibration position to its detected
  position for matched landmarks — the visible "shift" tells you
  whether the camera moved a lot (long lines) or barely at all
  (short lines).
- A header showing `auto` or `manual` source, the matched count, the
  mean template score, the reprojection error, and the confidence
  label.

Triage workflow:

1. **`high` and `medium` videos** are usually fine. Sample a few
   medium ones in the cropped folder to make sure the maze ended up
   where you expect.
2. **`low` videos** still have a cropped output. Open the review PNG
   first: if a few unmatched red circles are concentrated in one part
   of the frame (e.g. where the mouse was at the sampled middle
   frame), the alignment is probably fine and the low label came
   from those few stragglers. If the matched yellow lines are long
   or inconsistent, rescue with `--manual_video`.
3. **`failed` videos** have no cropped output. Most often this means
   `<12` landmarks matched — usually because the mouse was sitting on
   top of a critical landmark in the sampled middle frame, dropping
   that template's score below `--match_threshold`. Either:
     - Rerun with a lower `--match_threshold` (e.g. 0.2) and see if
       the matches recover. Be careful — too low a threshold lets
       in noisy peaks.
     - Increase `--search_radius` if the camera shift is larger than
       150 px in any direction.
     - Rescue with `--manual_video <path>`. The clicks are saved next
       to the cropped output and future batch runs pick them up.
4. **Re-run automatic detection** on a video that already has saved
   manual landmarks with `--redo_video <name>` (e.g. after tuning
   `--search_radius` or `--match_threshold`).
5. **`calibration_frame_annotated.png`** is the canonical reference:
   if it doesn't match the actual maze pose you want (e.g. the patch
   boxes look mis-clicked), re-run `--calibrate` and click more
   carefully.

## Tuning notes

- **`--patch_size`** controls how distinctive each template is. Larger
  patches are more discriminative (fewer false matches) but more
  sensitive to small camera rotation/scale changes — and they're more
  likely to clip outside the frame near the edges of the maze. The
  default of 80 px is a reasonable starting point for ~720p ventral
  recordings; halve or double it if your videos are much smaller or
  larger.
- **`--search_radius`** is the upper bound on how far you expect the
  camera to have shifted between sessions, in pixels. If you see
  matched landmarks pinned to the edge of the search window in the
  review PNG, raise it.
- **`--match_threshold`** is the cross-correlation cutoff. 0.3 is
  permissive on purpose so a mouse occlusion on one landmark doesn't
  cascade into a `failed` for the whole video. If you see the
  homography being yanked around by an obviously-bad match (long
  outlier yellow line in the review PNG), raise it to ~0.5; if you're
  losing too many to mouse occlusion, drop to ~0.2.
