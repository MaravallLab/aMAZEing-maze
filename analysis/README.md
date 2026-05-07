# crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose
estimation. The maze is a white cross/T-shaped binary decision tree
(4 arms) on a dark background, plus an entry corridor at the bottom of
the rig, filmed from below. Each session can have a slightly different
camera position, framing, and rotation, which makes pose estimation
harder and lets the mouse occupy fewer pixels than it should.

This script detects 6 anatomical landmarks on the maze in every video,
computes a homography that maps those landmarks to a canonical layout
defined by a one-time manual calibration, and writes a cropped/aligned
copy of every input video. After this step, every recording has the
maze in the same place, the mouse appears larger, and SLEAP doesn't
have to learn through that variability.

## Workflow

The pipeline runs in two stages: a one-time **calibration** that you
run once per camera setup, then a **batch** run that processes every
video.

### Stage 1 — Calibration (`--calibrate`)

```
python crop_and_align_maze.py --calibrate \
    --input_dir  <path> \
    --output_dir <path>
```

The script picks one video (the first one it finds under `--input_dir`,
or the file you pass via `--calibrate_video`), opens its middle frame in
an OpenCV window, and asks you to click 6 landmarks **in this exact
order**:

| # | Landmark                                       | Color    |
| - | ---------------------------------------------- | -------- |
| 1 | Tip of top-left arm                            | red      |
| 2 | Tip of top-right arm                           | orange   |
| 3 | Tip of bottom-left arm                         | yellow   |
| 4 | Tip of bottom-right arm                        | green    |
| 5 | Bottom-left corner of the entry corridor       | magenta  |
| 6 | Bottom-right corner of the entry corridor      | cyan     |

Controls inside the calibration window:
- Left-click to place the next landmark.
- **r** resets the click set so you can start over.
- After all 6 are placed, **y** confirms; **r** redoes from scratch.
- **ESC** quits without saving.

On confirm, the script writes:
- `calibration.json` (or whatever you pass via `--calibration_file`) —
  the 6 landmark positions and the calibration frame's dimensions.
- `calibration_frame.png` — the calibration frame with the 6 landmarks
  drawn on it, for your records.

The script then exits. Calibration mode does not process any videos.

### Stage 2 — Batch processing

```
python crop_and_align_maze.py \
    --input_dir  <path> \
    --output_dir <path>
```

For every video under `--input_dir`, the script:

1. **Samples the middle frame** (50% of total frame count).
2. **Thresholds with Otsu** to separate the bright maze from the dark
   background.
3. **Applies morphological closing** with a square kernel of size
   `--morph_kernel_size` (default 50 px) to fill the mouse-shaped hole
   the mouse's dark body would otherwise punch into the threshold.
   The closing happens on the binary mask before the contour is
   re-extracted, so the mouse can't drag landmark detection off course.
4. **Re-extracts the largest contour** from the closed binary — that is
   the maze.
5. **Detects the same 6 landmarks** automatically (see "Automatic
   landmark detection" below).
6. **Solves a homography** from the detected landmarks to the canonical
   layout derived from the calibration, using `cv2.findHomography`'s
   least-squares fit over 6 correspondences.
7. **Warps every frame** of the original video with that homography and
   writes the cropped output to
   `<output_dir>/<relative session path>/<name>_cropped.mp4` at the
   source FPS using the `mp4v` codec.
8. **Scores confidence** from the reprojection error (see "Confidence
   scoring") and saves a review PNG for any flagged or failed video.
9. **Writes `alignment_summary.csv`** in the output directory with one
   row per input video and prints a confidence breakdown to the console.

The pipeline is two-pass internally: pass 1 detects landmarks for every
video, pass 2 writes the cropped videos. Output mirrors the input
directory structure so filenames don't collide across sessions.

## Automatic landmark detection

The 6 landmarks are extracted from the closed maze contour as follows.

### 4 arm tips

1. Compute the convex hull of the maze contour.
2. Apply `cv2.approxPolyDP` to the hull with decreasing epsilon until
   the vertex count is in the range `[8, 12]` (the convex hull of a
   cross typically has ~8 corners — roughly two per arm tip region).
3. Cluster the resulting hull vertices into 4 groups by which quadrant
   of the contour's axis-aligned bounding box they fall in (top-left,
   top-right, bottom-left, bottom-right).
4. Within each quadrant, pick the vertex furthest from the contour's
   centroid as that arm's tip. The arm tips reach further out than any
   other hull vertex, so this consistently picks the tip even though
   the corridor base may also live in the bottom quadrants.

### 2 corridor base points

1. Take all contour points whose `y` coordinate is within 10 px of the
   maximum `y` (i.e. the very bottom strip of the maze contour).
2. The leftmost and rightmost points of that strip are the bottom-left
   and bottom-right corridor base corners respectively.

The 6 detected landmarks are then placed in the same order as the
calibration (`top_left_arm`, `top_right_arm`, `bottom_left_arm`,
`bottom_right_arm`, `corridor_base_left`, `corridor_base_right`) so the
homography solve has a consistent point-to-point correspondence.

If a quadrant contains no hull vertex, or the bottom strip has fewer
than 2 contour points, detection fails for that video.

## Canonical layout

The canonical positions of the 6 landmarks are derived from the
calibration: each calibration point is translated so the bounding box
of all 6 sits at `(--padding, --padding)`, and the output canvas size is
`(bbox_width + 2*padding, bbox_height + 2*padding)`. The relative
spacing and proportions of the calibration are preserved exactly — the
warp simply centers the maze in a padded canvas.

This is what gives every cropped video the same maze pose: every video
is warped onto the same target landmarks.

## Confidence scoring

Confidence is based on **reprojection error**: for each video, after
solving the homography, the 6 detected landmarks are projected through
H and compared to the 6 canonical targets. The mean Euclidean distance
(in pixels) is the reprojection error.

| Reprojection error | Base confidence |
| ------------------ | --------------- |
| < 5 px             | high            |
| 5 – 15 px          | medium          |
| > 15 px            | low             |

A **contour-area sanity check** runs alongside: if the detected contour
covers less than 5% or more than 50% of the frame, an area flag is
raised. An area flag downgrades the base label by one tier (high →
medium, medium → low). The label `failed` is reserved for videos where
the file couldn't be opened, no contour could be found, or fewer than 6
landmarks could be detected.

| Label    | What it means                                                     |
| -------- | ----------------------------------------------------------------- |
| high     | Reprojection error < 5 px and the contour size is sensible.       |
| medium   | Reprojection error in [5, 15] px, or area-flag downgrade from high. |
| low      | Reprojection error > 15 px, or area-flag downgrade from medium.   |
| failed   | File couldn't be processed; no cropped video is written.          |

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
    [--calibration_file <path>]
```

| Argument              | Default                                 | Meaning                                                                                |
| --------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| `--input_dir`         | —                                       | Folder to search recursively for videos. Required for batch mode; in `--calibrate` mode it's used to find a video if `--calibrate_video` isn't given. |
| `--output_dir`        | —                                       | Where cropped videos, the calibration file (default), the review folder, and the CSV are written. |
| `--padding`           | 50                                      | Pixels of padding added around the bounding box of the canonical landmarks.            |
| `--morph_kernel_size` | 50                                      | Square kernel size (px) for `MORPH_CLOSE` on the threshold. Should be comfortably larger than the mouse but smaller than the maze. |
| `--exclude_dirs`      | `segments new_segments segments_detected deeplabcut habituation` | Directory names to skip when walking `input_dir` (case-insensitive). Pruned in-place during the walk so the script never even descends into them. |
| `--include_pattern`   | `*.mp4,*.avi`                           | Comma-separated `fnmatch` globs used to pick which filenames count as videos.          |
| `--calibrate`         | off                                     | Enter calibration mode: click 6 landmarks on a sampled frame and save them. Exits without processing any videos. |
| `--calibrate_video`   | — (uses first video in `--input_dir`)   | In `--calibrate` mode, the specific video to calibrate on.                             |
| `--calibration_file`  | `<output_dir>/calibration.json`         | Path to the calibration JSON file. Read in batch mode, written in calibrate mode.      |

`input_dir` is searched recursively, so you can point it at the top
`simplermaze/` folder and it will pick up `mouse<id>/<session>/<file>.mp4`
automatically. The defaults for `--exclude_dirs` skip per-trial segment
folders, DLC outputs, and habituation videos so only the raw session
videos are processed.

If you run batch mode without a calibration file, the script will print
a clear error pointing you to `--calibrate`.

## Outputs

```
<output_dir>/
├── calibration.json                   # 6 landmarks + frame size
├── calibration_frame.png              # frame with calibration landmarks drawn
├── alignment_summary.csv              # one row per input video
├── <mouse>/<session>/<name>_cropped.mp4
├── ...
└── review/
    └── <mouse>/<session>/<name>_review.png   # for medium/low/failed
```

`alignment_summary.csv` columns:

| Column              | Meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| `filename`          | Path of the input video, relative to `--input_dir`.           |
| `contour_area`      | Area of the detected maze contour (px²).                      |
| `reprojection_error`| Mean reprojection error of the 6 landmarks (px).              |
| `confidence`        | `high` / `medium` / `low` / `failed`.                         |
| `output_path`       | Path of the cropped video, or an `error: ...` string on failure. |

After it's saved, the script also prints a confidence breakdown to the
console (counts of `high` / `medium` / `low` / `failed`) so you don't
have to count rows by hand.

## Dependencies

- Python 3
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

## Reviewing flagged videos

Anything labeled `medium`, `low`, or `failed` in `alignment_summary.csv`
gets a debug PNG under `review/`. The PNG shows:

- the closed maze contour outline in **cyan**,
- the convex-hull approximation vertices as small **white dots**,
- the bottom strip used for corridor detection as a thin **green line**,
- per-quadrant arm-tip candidates as hollow **teal circles**, and
- the 6 final landmarks in their per-landmark colors (red / orange /
  yellow / green / magenta / cyan, numbered 1–6).

How to triage:

1. Open the review PNG for the flagged video. If all 6 colored markers
   are sitting on the right anatomical points, the alignment is fine
   and the flag came from a slightly noisy reprojection — usually
   tolerable for downstream pose estimation.
2. If the markers are clearly off (e.g. an arm tip landed on the
   contour where the mouse broke the maze outline), inspect the white
   hull-approximation dots and the teal candidates. They tell you
   whether the convex-hull approximation degenerated or whether the
   mouse was too close to a tip when the frame was sampled.
3. Open the corresponding cropped video and confirm the maze ends up in
   the expected canonical pose. The cropped video is written even for
   `low` confidence — it's still usable in many cases.
4. `failed` rows have no cropped output. Re-check `--morph_kernel_size`
   if the mouse was breaking the maze outline; raise it until the
   closing fully repairs the cross.
5. `calibration_frame.png` is your reference for what the canonical
   layout should look like. If the canonical pose looks wrong, re-run
   `--calibrate` and click more carefully.

If a particular video keeps failing automatic detection no matter what,
you can re-run calibration on that video specifically (via
`--calibrate_video`) and use a separate output directory to crop just
that file under its own calibration.
