# crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose
estimation. The maze is a white cross/T-shaped binary decision tree (4
arms) on a dark background, filmed from below. Each session can have a
slightly different camera position, framing, and rotation, which makes
pose estimation harder and lets the mouse occupy fewer pixels than it
needs to.

This script detects the maze in every video, computes a perspective
transform that warps it into a canonical axis-aligned rectangle (with a
fixed amount of padding around the edges), and writes a cropped/aligned
copy of every input video. After this step, every recording has the maze
in the same place, the mouse appears larger, and SLEAP doesn't need to
learn through that variability.

## Two-pass pipeline

The camera position can shift between sessions (e.g. after the rig is
cleaned), so picking any one video as the "reference" is unfair: every
later recording would be measured against a single arbitrary frame. The
script therefore runs in two passes and builds a consensus template from
the whole dataset.

### Pass 1 — Detection only

For every video:
- Sample the frame at 50% of the total frame count.
- Convert to grayscale and binarize with Otsu's threshold (the maze is
  bright, the background is dark).
- Find the largest external contour — that is the maze.
- Compute the rotated bounding rectangle (`cv2.minAreaRect`) and the
  perspective transform that maps its 4 corners to a canonical
  axis-aligned rectangle plus padding.
- Run the contour-level sanity checks (area ratio, aspect ratio,
  rotation magnitude) and remember which ones failed.
- Warp the binary maze mask using that transform and keep it in memory.

The canonical rectangle size and the reference aspect ratio are taken
from the first successfully detected video, so every warped mask lives
in the same pixel grid and can be compared like-for-like.

If a video can't be opened, has no frames, has no detectable contour,
or yields a degenerate bounding rectangle, it is marked `failed` and
skipped from then on.

### Build the consensus template

All successful warped masks are averaged pixel-wise (each mask is 0/1,
so the average is a value in [0, 1] expressing how many recordings agree
that this pixel is "inside" the maze). The result is thresholded at 0.5
to produce a single binary template. Because the average is taken across
the whole dataset, a few unusual recordings can't drag the template
toward themselves — the template represents what the maze typically
looks like in canonical coordinates.

The template is saved as `consensus_template.png` in the output
directory so you can sanity-check it visually.

### Pass 2 — IoU comparison and video writing

For every video that passed pass 1:
- Compute the IoU between its warped mask and the consensus template.
- Combine all four checks (area, aspect, IoU, rotation) into an overall
  `high` / `medium` / `low` confidence label.
- Warp every frame of the original video with the same transform and
  write the cropped video to `output_dir/<relative session path>/
  <name>_cropped.mp4` at the source FPS using the `mp4v` codec.
- For `medium` / `low` confidence videos, save the sampled frame with
  the detected contour drawn on it under `review/`.

A summary CSV (`alignment_summary.csv`) is written with one row per
input video.

## Confidence scoring

Each video is checked against four conditions. The number of failing
checks determines the confidence label.

| Metric              | What it measures                                               | Flag condition                                                     |
| ------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| Contour area ratio  | Detected contour area as a fraction of total frame area.       | Outside `[0.05, 0.5]` (maze fills <5% or >50% of frame).           |
| Aspect ratio        | Long/short side ratio of the rotated bounding rectangle.       | Deviates more than 20% from the reference (first successful video).|
| Mask IoU            | Overlap of the warped mask with the consensus template.        | IoU below `--iou_threshold` (default 0.85).                        |
| Rotation magnitude  | Signed deviation (in degrees) from axis-aligned, in [-45, 45]. | `abs(angle)` greater than `--rotation_threshold` (default 15°).    |

Confidence buckets:
- **high** — all four checks pass. Auto-accept.
- **medium** — exactly one check fails. Worth a quick look in `review/`.
- **low** — two or more checks fail. Inspect carefully before using.
- **manual** — you accepted the alignment by clicking the 4 corners in
  `--manual` mode. The cropped video is written and a review PNG is
  saved (showing your clicked quad) for the record.
- **failed** — detection itself failed (bad file, no contour, etc.) and
  you didn't override it in `--manual` mode. No output video is written.

## CLI

```
python crop_and_align_maze.py \
    --input_dir <path> \
    --output_dir <path> \
    [--padding 50] \
    [--iou_threshold 0.85] \
    [--rotation_threshold 15] \
    [--manual]
```

| Argument              | Default | Meaning                                                                                |
| --------------------- | ------- | -------------------------------------------------------------------------------------- |
| `--input_dir`         | —       | Folder to search recursively for `.mp4` and `.avi` videos. Subfolders preserved in output. |
| `--output_dir`        | —       | Where cropped videos, the consensus template, the review folder, and the CSV are written. |
| `--padding`           | 50      | Pixels of padding added around the canonical maze rectangle on every side.             |
| `--iou_threshold`     | 0.85    | Minimum mask IoU vs. consensus template to count as a passing check.                   |
| `--rotation_threshold`| 15      | Maximum allowed `abs(rotation)` in degrees before the rotation check is flagged.       |
| `--manual`            | off     | After automatic scoring, open a window for each `medium` / `low` / `failed` video so you can click the 4 maze corners by hand. |

`input_dir` is searched recursively, so you can point it at the top-level
`simplermaze/` folder and it will pick up videos in
`mouse<id>/<session>/<file>.mp4` automatically. The output mirrors the
relative path to keep filenames from colliding across sessions.

## Dependencies

- Python 3
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

## Reviewing flagged videos

Anything labeled `medium`, `low`, or `failed` in `alignment_summary.csv`
deserves a look:

1. Open the corresponding image in `output_dir/review/<session>/
   <name>_review.png`. The detected maze contour is drawn in green over
   the sampled frame. If the contour looks right (it traces the outline
   of the maze), the alignment is probably fine and the flag came from
   the IoU drifting against the consensus — usually a real but tolerable
   camera shift.
2. If the contour is clearly wrong (it includes reflections, room
   features, the mouse's home cage, etc.), discard the cropped video
   and either re-record or pre-mask the frame before re-running this
   script.
3. Open the cropped video in `output_dir/<session>/<name>_cropped.mp4`
   and confirm the maze is centered and the mouse stays inside the
   frame for the whole recording.
4. `consensus_template.png` shows the canonical maze shape used for the
   IoU check. If it looks like a cross/T with crisp arms, the dataset
   is internally consistent. If it looks blurry or has multiple ghost
   outlines, the camera moved a lot across sessions and the IoU check
   will be conservative — consider raising `--iou_threshold` or
   inspecting more videos by hand.
5. `failed` rows mean detection itself didn't run; check whether the
   file opens, whether it has frames, and whether the maze is visibly
   bright against the background in a sampled frame.

## Manual correction (`--manual`)

When you re-run the script with `--manual`, after the automatic scoring
step the script opens an OpenCV window for every `medium`, `low`, or
`failed` video and asks you to click the 4 maze corners.

Workflow per video:

1. A window pops up showing the sampled middle frame, with the filename
   and current confidence label in the title.
2. Click the corners **in this order**: top-left, top-right,
   bottom-right, bottom-left (clockwise from TL). Markers and connecting
   lines are drawn as you click so you can see the quad form.
3. Press **ENTER** to confirm — the script rebuilds the perspective
   transform from your clicks, recomputes the warped mask, marks the
   confidence as `manual`, and moves on to the next video.
4. Press **r** at any time to reset the 4 clicks and start over for the
   current video.
5. Press **ESC** to skip the current video without correcting it
   (its row stays `medium` / `low` / `failed`).

Notes:
- The corners you click define the perspective transform; you don't need
  to match the automatic detection. If you'd rather crop tighter or
  looser than the auto detection did, click accordingly.
- Manually corrected videos are written to the same output path as
  automatic ones (`<output_dir>/<session>/<name>_cropped.mp4`) and a
  review PNG with your clicked quad is saved under `review/`.
- The consensus template is built from the automatic detections only
  (manual entries don't fold back in), so the IoU printed for manual
  rows is informational. The `manual` label means "the user has visually
  verified the corners" and overrides the IoU check.
- If literally no video passed automatic detection, manual mode still
  works: the first manual rectangle defines the canonical size, and
  every subsequent click is fit into that same coordinate system.
