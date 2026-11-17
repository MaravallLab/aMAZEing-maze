# Cropping and aligning the videos

!!! info "At a glance"
    **Time:** five minutes of clicking per rig; then a few minutes of computing per video.
    **You need:** the ventral videos of the tactile maze, Python with OpenCV, NumPy and pandas.
    **You get:** a cropped video per session with the maze in one fixed place, and a table rating each alignment.

## What it does

Stage 1 of the [pose estimation pipeline](pose.md). It warps every video so the maze lands in the same place. That is what lets the later stages tell a video's rig from its size, and filter detections against the maze outline.

You click the 24 corners of the maze once, on one frame per rig. For every other video, the script finds those corners again by matching small patches of the calibration frame, fits a homography, and warps the video. No thresholds or contour finding are involved.

??? note "How the corners are found"
    - **A median frame**, not a single frame: 11 frames are sampled through the video and medianed pixel by pixel, so the moving mouse averages out and the patches match the empty maze.
    - **A global shift first**: phase correlation between the calibration frame and the median frame estimates how far the camera moved, so each corner is searched for around its shifted position.
    - **Template matching**: each of the 24 patches is searched for with normalised cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`).
    - **RANSAC and refinement**: a homography is fitted with RANSAC, and corners that do not fit are searched for again around where the homography puts them, up to three times.
    - **Several rigs**: calibrate once per rig, put the calibrations in one folder, and the script picks the best for each video.

## What you need

- Python 3.9 or later with `opencv-python`, `numpy` and `pandas` (`pip install opencv-python numpy pandas`). The sleap-nn Python of the [pose pipeline](pose.md#what-you-need) has them already.
- The videos, in folders by mouse and session.
- A screen, for the calibration clicks.

## Quick start

From the repository root, with `<videos>` the input folder and `<out>` the output folder:

**1. Calibrate once per rig.** Pick a video with a clear view of the maze.

```bash
python analysis/crop_and_align_maze.py --calibrate --input_dir <videos> --output_dir <out>
```

A window opens on the median frame. Left-click the 24 maze corners in [the order below](#click-order), starting from the top-left arm and going clockwise. Press `y` to confirm, `r` to start again, `Esc` to cancel. This writes `calibration.json` and `calibration_frame.png` into `<out>` and stops.

**2. Run the batch.**

```bash
python analysis/crop_and_align_maze.py --input_dir <videos> --output_dir <out>
```

It walks every subfolder of `<videos>` and writes a cropped video for each into the same place under `<out>`. Videos already cropped are skipped, so you can stop and start again.

**3. Check the results.** The script ends with a count:

```
Confidence summary: high=141, medium=26, low=0, failed=238, skipped=0 (total 405)
```

`<out>/alignment_summary.csv` has the rating of every video. For any `medium`, `low` or `failed` video, open `<out>/review/<mouse>/<session>/<name>_review.png` to see what the matcher found. [Troubleshooting](#troubleshooting) says what to do about failures.

## Details

### Click order

| # | Corner |
|---|---|
| 1 | Top-left arm, outer top-left |
| 2 | Top-left arm, outer top-right |
| 3 | Inner: top-left arm meets the central junction (top side) |
| 4 | Inner: top-right arm meets the central junction (top side) |
| 5 | Top-right arm, outer top-left |
| 6 | Top-right arm, outer top-right |
| 7 | Top-right arm, outer bottom-right |
| 8 | Top-right arm, outer bottom-left |
| 9 | Inner: top-right arm meets the central junction (bottom side) |
| 10 | Inner: junction meets the corridor, right (top) |
| 11 | Inner: bottom-right arm meets the corridor (top side) |
| 12 | Bottom-right arm, outer top-right |
| 13 | Bottom-right arm, outer bottom-right |
| 14 | Bottom-right arm, outer bottom-left |
| 15 | Inner: bottom-right arm meets the corridor (bottom side) |
| 16 | Corridor, right side, bottom |
| 17 | Corridor, left side, bottom |
| 18 | Inner: bottom-left arm meets the corridor (bottom side) |
| 19 | Bottom-left arm, outer bottom-right |
| 20 | Bottom-left arm, outer bottom-left |
| 21 | Bottom-left arm, outer top-left |
| 22 | Inner: bottom-left arm meets the corridor (top side) |
| 23 | Inner: junction meets the corridor, left (top) |
| 24 | Inner: top-left arm meets the central junction (bottom side) |

### Calibration files

A calibration writes three files, named after the JSON so several can share a folder:

| File | Contents |
|---|---|
| `<stem>.json` | The 24 corner positions, the patch size, the frame size and the name of the frame image |
| `<stem>_frame.png` | The median frame, unmarked. The batch cuts its patches from it, so do not edit it. |
| `<stem>_frame_annotated.png` | The same frame with the corners numbered and the patches drawn, for checking |

For the default `calibration.json` the images are `calibration_frame.png` and `calibration_frame_annotated.png`. Pick the video with `--calibrate_video PATH` (default: the first video in `--input_dir`), and the file with `--calibration_file PATH`. If a patch would fall outside the frame, a warning lists the corners affected; calibrate again with a smaller `--patch_size`.

### Several rigs

When the videos come from more than one rig, calibrate each and keep the calibrations in one folder:

```bash
python analysis/crop_and_align_maze.py --calibrate --calibrate_video <a video from the second rig> --output_dir <out> --calibration_file <out>/calibration_rig2.json
python analysis/crop_and_align_maze.py --input_dir <videos> --output_dir <out> --calibration_dir <out>
```

With `--calibration_dir`, every calibration is tried on every video and the best fit is kept: the higher rating wins, then more RANSAC inliers, then the higher mean match score. The one chosen is recorded in the `calibration_used` column. `--calibration_dir` takes precedence over `--calibration_file`.

### Rescuing one video by hand

```bash
python analysis/crop_and_align_maze.py --manual_video <path to the video> --output_dir <out> --calibration_file <out>/calibration.json
```

This opens the 24-click window on that video's median frame, warps the video from your clicks, and saves them as `<out>/<relative path>/<stem>_landmarks.json`. Later batch runs use the saved clicks for that video automatically (the console says `using saved manual landmarks`). To make the batch detect a video again anyway, pass `--redo_video <name>`, with the file name or the path `<mouse>/<session>/<file>.mp4`.

??? note "What the batch does to each video"
    1. **Skips it** if `<stem>_cropped.mp4` already exists (unless `--no-skip_existing` or `--redo_video`).
    2. **Median frame** from `--n_median_frames` frames (default 11).
    3. **Global shift** by phase correlation (`--global_align off` to skip). A shift over about 35% of the smaller image side is treated as spurious and ignored.
    4. **Template matching** of each corner within `--search_radius` pixels of its shifted position. A corner counts as matched if its score reaches `--match_threshold`.
    5. **RANSAC homography** with `--ransac_threshold` pixels.
    6. **Refinement**, up to three rounds: corners that do not fit are searched for within `--refine_radius` of where the homography puts them, accepted at the looser `--refine_threshold`, and the homography is fitted again.
    7. **Rating and output.** With fewer than 12 inliers the video is `failed` and not warped. Otherwise every frame is warped and written as `<stem>_cropped.mp4` (mp4v, the source frame rate), with a review image for `medium`, `low` and `failed` videos.

    The patches are cut once per calibration at the start, not per video.

??? note "The fixed frame each video is warped to"
    The 24 calibration corners are moved so their bounding box starts at (`--padding`, `--padding`), and the output is the bounding box plus the padding on every side. Spacing and proportions are kept exactly. With several calibrations, each defines its own frame, so videos cropped against different rigs come out at different sizes.

??? note "How a video is rated"
    | Rating | Reprojection error (inliers) | Inliers | Mean match score |
    |---|---|---|---|
    | `high` | under 10 px | 20 or more | over 0.5 |
    | `medium` | under 25 px | 16 or more | any |
    | `low` | otherwise, with 12 or more inliers | | |
    | `failed` | fewer than 12 inliers, or an earlier failure | | |
    | `skipped` | the cropped video already exists | | |

    A `failed` video has no cropped output. `medium` and `low` videos do, with a review image so you can decide whether to keep them. A video done from saved manual clicks is rated on reprojection error and inliers only.

??? note "All options"
    | Option | Default | Meaning |
    |---|---|---|
    | `--input_dir DIR` | | Folder searched for videos; required for the batch |
    | `--output_dir DIR` | | Where everything is written; required |
    | `--padding PX` | 50 | Padding around the corners in the fixed frame |
    | `--patch_size PX` | 80 | Size of each patch, set at calibration and read back in the batch |
    | `--search_radius PX` | 60 | Search window around each shifted corner |
    | `--match_threshold F` | 0.3 | Minimum score for a first match |
    | `--n_median_frames N` | 11 | Frames in the median; 1 takes the middle frame |
    | `--refine_radius PX` | 35 | Search window during refinement |
    | `--refine_threshold F` | 0.2 | Minimum score during refinement |
    | `--ransac_threshold PX` | 10 | Reprojection error under which a match is an inlier |
    | `--global_align` | `phase` | `off` skips the global shift |
    | `--exclude_dirs NAMES` | `segments new_segments segments_detected deeplabcut habituation` | Folder names to skip |
    | `--include_pattern GLOBS` | `*.mp4,*.avi` | Which files count as videos |
    | `--calibrate` | off | Calibrate and stop |
    | `--calibrate_video PATH` | the first video | The video to calibrate on |
    | `--calibration_file PATH` | `<output_dir>/calibration.json` | One calibration, read or written |
    | `--calibration_dir DIR` | off | A folder of calibrations; the best is chosen per video |
    | `--manual_video PATH` | off | Click the corners of one video and process it |
    | `--redo_video NAME` | off | Detect this video again, ignoring saved clicks and existing output |
    | `--skip_existing`, `--no-skip_existing` | on | Skip videos already cropped |

??? note "Output files"
    ```
    <output_dir>/
      calibration.json                         the default calibration
      calibration_frame.png                    used by the batch
      calibration_frame_annotated.png          for checking
      calibration_rig2.json                    further rigs, one set each
      alignment_summary.csv                    one row per input video
      <mouse>/<session>/<name>_cropped.mp4
      <mouse>/<session>/<name>_landmarks.json  only after --manual_video
      review/<mouse>/<session>/<name>_review.png   medium, low and failed videos
    ```

    | Column of `alignment_summary.csv` | Meaning |
    |---|---|
    | `filename` | The input video, relative to `--input_dir` |
    | `calibration_used` | The calibration chosen (blank when skipped) |
    | `num_matched_points` | Corners whose score reached `--match_threshold` |
    | `num_inlier_points` | Matches that survived RANSAC; this sets the rating |
    | `mean_match_score` | Mean score of the matched corners (blank for manual clicks) |
    | `reprojection_error` | Mean reprojection error of the inliers, in pixels |
    | `confidence` | `high`, `medium`, `low`, `failed` or `skipped` |
    | `output_path` | The cropped video, or `error: ...` |

## Troubleshooting

**Reading a review image.** Green filled circles mark corners found, red hollow circles mark corners not found (at their calibration position), and yellow lines join each calibration position to where the corner was found: long lines mean the camera moved a lot. The header gives the source (auto or manual), the counts, the mean score, the error and the rating.

??? question "A whole mouse or cohort failed together"
    Those videos almost certainly come from another rig. Calibrate that rig from one of the failed videos and run the batch again with `--calibration_dir` ([several rigs](#several-rigs)). Videos already cropped are skipped. If the failures all have similar `num_inlier_points`, add a calibration rather than tuning the settings.

??? question "One video failed"
    Rescue it [by hand](#rescuing-one-video-by-hand). Other causes:

    - The mouse sat on a corner in every sampled frame: raise `--n_median_frames` (for example 21).
    - The camera moved further than the search window: raise `--search_radius` or `--refine_radius`.
    - Noisy patches: lower `--match_threshold` (for example 0.2); RANSAC drops the false matches, at the cost of fewer inliers.

??? question "A video is rated low"
    It still has a cropped video. If the red circles are few and in one part of the frame, the alignment is probably fine. If the yellow lines are long or point different ways, rescue it by hand.

??? question "The batch says there is no calibration"
    `--calibration_file` does not exist and no `--calibration_dir` was given. Calibrate first, or point at the folder that holds `calibration*.json`.

??? question "The patch boxes in the annotated frame look wrong"
    A corner was mis-clicked. Calibrate again.

??? note "Tuning"
    - **`--patch_size`**: larger patches are more distinctive but more sensitive to small rotations, and more likely to fall off the edge of the frame. 80 px suits videos of about 720p; halve or double it for much smaller or larger videos.
    - **`--search_radius`**: the global shift already absorbs most camera movement, so 60 px is enough. With `--global_align off`, raise it to about 150.
    - **`--ransac_threshold`**: smaller (5) fits tighter with fewer inliers; larger (15 to 20) keeps more inliers with a looser fit.
    - **`--refine_radius`**: if the first homography rests on few inliers it can place far corners badly; 50 to 60 gives the search more room.
    - **`--n_median_frames`**: 21 if a mouse sat still in the sampled frames; 1 for a quick check.
