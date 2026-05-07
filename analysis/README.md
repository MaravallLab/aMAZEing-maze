# crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose
estimation. The maze is a white cross/T-shaped binary decision tree
(4 arms) on a dark background, plus an entry corridor at the bottom of
the rig, filmed from below.

A naive "threshold and grab the largest bright contour" detector locks
onto the wrong thing: room walls, ceiling panels, and other bright
surfaces around the maze are all over the frame. The detector ends up
tracing the enclosure instead of the maze, and the homography goes
sideways.

This script avoids that by using a one-time manual calibration to
establish:

1. A **24-point reference outline** of the maze (the user clicks every
   corner of the maze perimeter clockwise from the top-left), and
2. A **search ROI** — the bounding box of those 24 points expanded by
   150 px on each side. All thresholding and corner detection in batch
   mode happens **inside that ROI only**, so room features can never
   contaminate the binary mask.

In batch mode, every video is corner-detected within the ROI, those
corners are matched to the calibration's 24 points using the Hungarian
algorithm, and a RANSAC homography is fit through the matched pairs.

## Workflow

```
                +------------+
                | calibrate  |   click 24 maze corners on one frame
                |  (once)    |   -> calibration.json (+ search ROI)
                +-----+------+
                      |
                      v
              +---------------+
              |     batch     |   ROI-thresh -> goodFeaturesToTrack
              |   processing  |   -> Hungarian match -> RANSAC homography
              +-------+-------+   -> warp every frame
                      |
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
    --output_dir <path>
```

The script picks one video (the first found under `--input_dir`, or
the file you pass via `--calibrate_video`), opens its middle frame in
an OpenCV window, and asks you to click all 24 maze corners.

A small reference diagram in the top-right of the window shows the
numbered click order so you don't have to memorize it.

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

On confirm the script writes:
- `calibration.json` — the 24 landmark positions, the search ROI, and
  the calibration frame's dimensions.
- `calibration_frame.png` — the calibration frame with the 24
  landmarks numbered and the search ROI drawn on top.

The script then exits without processing any videos.

### Stage 2 — Batch processing

```
python crop_and_align_maze.py \
    --input_dir  <path> \
    --output_dir <path>
```

For every video under `--input_dir` the script:

1. Samples the middle frame.
2. **Crops the frame to the calibration's search ROI before
   thresholding** — this is the critical fix: room walls, ceiling
   panels, light reflections etc. are all outside the ROI and never
   reach the binary mask.
3. Thresholds the cropped region with Otsu and applies morphological
   closing (`--morph_kernel_size`, default 50 px) to fill the
   mouse-shaped hole the mouse's dark body would otherwise punch into
   the threshold.
4. Runs `cv2.goodFeaturesToTrack` on the cropped binary mask
   (`maxCorners=60`, `qualityLevel=0.01`, `minDistance=20`) to detect
   the maze's sharp corners. Detected corner coordinates are offset
   from ROI-local back to full-frame coordinates.
5. Matches the detected corners to the 24 calibration points using
   the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`)
   on a Euclidean distance cost matrix. Pairs whose distance exceeds
   60 px are discarded as unmatched. If fewer than 16 of the 24
   landmarks survive matching, the video is marked `failed`.
6. Computes a homography from the matched detected corners to the
   canonical layout via `cv2.findHomography(method=cv2.RANSAC,
   ransacReprojThreshold=5.0)`.
7. Warps every frame of the original video with that homography and
   writes the cropped output to
   `<output_dir>/<relative session path>/<name>_cropped.mp4` at the
   source FPS using the `mp4v` codec.
8. Saves a debug PNG under `review/` for any `medium` / `low` /
   `failed` video showing detected corners (green), all 24 calibration
   points (red), and lines connecting matched pairs.
9. Writes `alignment_summary.csv` and prints a confidence breakdown.

### Manual fallback (`--manual_video <path>`)

```
python crop_and_align_maze.py \
    --manual_video      <path> \
    --calibration_file  calibration.json \
    --output_dir        <path>
```

For a video where automatic detection fails or looks wrong, this opens
the same 24-click UI on that video. The clicked points are saved to
`<output_dir>/<relative path>/<stem>_landmarks.json`, the homography
is computed directly from those 24 points, and that one video is
warped + written.

**Subsequent batch runs pick up saved per-video landmarks
automatically.** When `detect_one` starts a video it first looks for
`<stem>_landmarks.json` in the output tree; if found, it skips ROI
detection entirely and uses the saved 24-point assignment. The console
print is `Using saved manual landmarks for <video>`.

To force re-detection of a specific video despite saved landmarks, use
`--redo_video <name>` where the name matches either the bare filename
or the full `<mouse>/<session>/<file>.mp4` relative path.

## Canonical layout

The canonical positions of the 24 landmarks are derived from the
calibration: each calibration point is translated so the bounding box
of all 24 sits at `(--padding, --padding)`, and the output canvas size
is `(bbox_width + 2*padding, bbox_height + 2*padding)`. Spacing and
proportions of the calibration are preserved exactly — every video is
warped so its 24 detected corners line up with these canonical targets.

## Confidence scoring

Confidence combines two signals: the **reprojection error** of the
RANSAC homography and the **number of matched landmarks**.

| Confidence | Reproj. error | Matched points    |
| ---------- | ------------- | ----------------- |
| `high`     | < 10 px       | >= 20             |
| `medium`   | < 25 px       | >= 16             |
| `low`      | otherwise, but matched >= 16 |     |
| `failed`   | matched < 16, or detection blew up earlier |  |

A `failed` video has no cropped output. `medium` and `low` videos do
have a cropped output but get a debug review PNG so you can decide
whether to keep them or rescue them with `--manual_video`.

## CLI

```
python crop_and_align_maze.py \
    --input_dir <path> \
    --output_dir <path> \
    [--padding 50] \
    [--morph_kernel_size 50] \
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
| `--output_dir`        | —                                       | Where cropped videos, the calibration file (default), per-video manual landmarks, the review folder, and the CSV are written. |
| `--padding`           | 50                                      | Padding (px) around the bounding box of the canonical landmarks.                       |
| `--morph_kernel_size` | 50                                      | Square kernel size (px) for `MORPH_CLOSE` on the ROI threshold (fills the mouse-shaped hole). |
| `--exclude_dirs`      | `segments new_segments segments_detected deeplabcut habituation` | Directory names to skip when walking `input_dir` (case-insensitive). Pruned in-place during the walk. |
| `--include_pattern`   | `*.mp4,*.avi`                           | Comma-separated `fnmatch` globs used to pick which filenames count as videos.          |
| `--calibrate`         | off                                     | Enter calibration mode: click 24 maze corners and save calibration.json. Exits without processing videos. |
| `--calibrate_video`   | first video in `--input_dir`            | In `--calibrate` mode, the specific video to calibrate on.                             |
| `--calibration_file`  | `<output_dir>/calibration.json`         | Path to the calibration JSON. Read in batch / manual modes, written in calibrate mode. |
| `--manual_video`      | off                                     | Click 24 landmarks for one specific video, save `<stem>_landmarks.json`, and process that video. |
| `--redo_video`        | off                                     | In batch mode, force automatic re-detection for the named video (matched against either the bare filename or the relative path). |

If you run batch mode without a calibration file the script prints a
clear error pointing you to `--calibrate`.

## Outputs

```
<output_dir>/
├── calibration.json                     # 24 landmarks + ROI + frame size
├── calibration_frame.png                # frame with landmarks + ROI drawn
├── alignment_summary.csv                # one row per input video
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
| `num_matched_points`  | Number of detected corners matched to calibration landmarks.  |
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
- [SciPy](https://scipy.org/) — for `scipy.optimize.linear_sum_assignment`
  (the Hungarian matcher).

## Reviewing flagged videos

The review PNG for a flagged or failed video shows:

- All **24 calibration points** as hollow **red circles** at their
  source-image positions.
- All detected corners. Matched ones are bright **green** filled
  circles; unmatched detections are dim green hollow circles.
- A **yellow line** between each calibration point and its matched
  detected corner — long lines mean the assignment is straining; lots
  of short, neat lines mean a clean match.
- A header showing `auto` or `manual` source, matched count, mean
  reprojection error, and the resulting confidence label.

Triage workflow:

1. **`high` and `medium` videos** are usually fine. Sample a few of
   the medium ones in the cropped folder to make sure the maze ended
   up where you expect.
2. **`low` videos** still have a cropped output. Open the review PNG
   first: if a few unmatched detections are off in a corner of the
   ROI but the matched-pair lines are short, the homography is
   probably good and the low label came from one or two stragglers.
   If the matched-pair lines are long or the layout is clearly wrong,
   rescue with `--manual_video`.
3. **`failed` videos** have no cropped output. Reasons typically are
   `<16` corners matched (mouse was on top of a critical landmark in
   the sampled frame) or the file couldn't be opened. The review PNG
   shows whatever did get detected. Either:
     - Rerun with a different `--morph_kernel_size` if the mouse keeps
       breaking the maze outline.
     - Rescue with `--manual_video <path>`. The clicks are saved next
       to the cropped output and future batch runs pick them up.
4. **Re-detect a single video** with `--redo_video <name>` if you
   manually corrected one earlier but want to retry the auto path
   (e.g. after tuning kernel size).
5. **`calibration_frame.png`** is the canonical reference: if it
   doesn't match the actual maze pose you want, re-run `--calibrate`
   and click more carefully.
