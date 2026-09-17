#!/usr/bin/env python3

# python plot_mouse_trajectories_with_maze.py --dlc "C:/Users/shahd/OneDrive - University of Sussex/DLC_MOSEQ/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv" --segments "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_segments_manifest.csv" --keypoint nose --min-likelihood 0.8 --background-video "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_trial_019.mp4" --bg-frame mid --output-dir "./trajectories_maze_video_nose"


"""
Plot mouse trajectories per trial from a DeepLabCut CSV and segments_manifest, overlaid on the maze.

Features
--------
1) Background overlay (choose ONE):
   - --background-image: path to a still image (same camera/scene as DLC)
   - --background-video: path to the source video; use --bg-frame to pick a frame
     * --bg-frame accepts an integer frame index OR the strings: "start", "mid", "end"
       relative to the trial's start/end_frame

2) Optional maze geometry overlay:
   - --maze-geometry path/to/maze.json  (polygons/lines will be outlined)
   JSON schema example:
   {
     "polygons": [
       {"name": "Arm1", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]},
       {"name": "Center", "points": [[...], ...]}
     ],
     "lines": [
       {"name": "Boundary", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
     ],
     "labels": true  // optional: draw names near centroids
   }

3) Optional homography rectification (warp both background and DLC points to a template):
   - --warp-dst-size WIDTH HEIGHT   (e.g., 800 800 pixels)
   - --warp-src-pts x1 y1 x2 y2 x3 y3 x4 y4  (order: TL, TR, BR, BL in camera space)
   The destination quad is implicitly: (0,0), (W,0), (W,H), (0,H).
   If provided, both the background image and the (x,y) coordinates are warped before plotting.

4) Likelihood filtering, axis inversion, and per-trial PNG export.

Usage
-----
python plot_mouse_trajectories_with_maze.py \
  --dlc DLC.csv \
  --segments segments_manifest.csv \
  --keypoint nose \
  --min-likelihood 0.9 \
  --background-video video.mp4 --bg-frame mid \
  --maze-geometry maze.json \
  --invert-y \
  --output-dir ./trajectories_maze

# With rectification
python plot_mouse_trajectories_with_maze.py \
  --dlc DLC.csv --segments segments_manifest.csv --keypoint nose \
  --background-image frame.png \
  --warp-dst-size 900 900 \
  --warp-src-pts 120 80  830 75  845 845  110 850 \
  --invert-y --output-dir ./trajectories_rectified
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2 as cv


# ---------------------- DLC utilities ----------------------
def read_dlc_csv_multiindex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2])
    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) != 3:
        raise ValueError("DLC CSV does not appear to have a 3-level header.")
    return df


def get_scorer_and_bodyparts(df: pd.DataFrame) -> Tuple[str, List[str]]:
    level0 = sorted(set([lvl0 for (lvl0, _, _) in df.columns]))
    scorer_candidates = [s for s in level0 if s.lower() != "scorer"]
    if not scorer_candidates:
        raise ValueError("Could not find DLC 'scorer' level (level 0) other than placeholder 'scorer'.")
    scorer = scorer_candidates[0]
    bodyparts = sorted(set([bp for (_, bp, _) in df.columns if bp.lower() != "bodyparts"]))
    return scorer, bodyparts


def extract_frame_series(df: pd.DataFrame) -> pd.Series:
    if ('scorer', 'bodyparts', 'coords') in df.columns:
        frames = df[('scorer', 'bodyparts', 'coords')].astype(int)
        frames.name = 'frame'
        return frames
    return pd.Series(np.arange(len(df)), name='frame')


def pick_keypoint(bodyparts: List[str], desired: Optional[str]) -> str:
    if desired and desired in bodyparts:
        return desired
    for candidate in ["nose", "mid", "front_mid", "back_mid", "tailbase", "center", "centroid", "body_center"]:
        if candidate in bodyparts:
            return candidate
    return bodyparts[0]


# ---------------------- Background utilities ----------------------
def read_image_rgb(path: str) -> np.ndarray:
    im_bgr = cv.imread(path, cv.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read background image: {path}")
    return cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)


def read_video_frame_rgb(path: str, frame_idx: int) -> np.ndarray:
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(0, total - 1)))
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {path}")
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def choose_trial_frame(start_frame: int, end_frame: int, spec: str) -> int:
    if isinstance(spec, int):
        return spec
    spec = (spec or "mid").lower()
    if spec == "start":
        return int(start_frame)
    if spec == "end":
        return int(end_frame)
    return int((start_frame + end_frame) // 2)


# ---------------------- Geometry overlay ----------------------
def load_maze_geometry(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_maze_geometry(ax, geom, homography: Optional[np.ndarray] = None, label: bool = True):
    def maybe_warp(points_np: np.ndarray) -> np.ndarray:
        if homography is None:
            return points_np
        pts = points_np.reshape(-1, 1, 2).astype(np.float32)
        warped = cv.perspectiveTransform(pts, homography)
        return warped.reshape(-1, 2)

    if "polygons" in geom:
        for poly in geom["polygons"]:
            pts = np.array(poly["points"], dtype=float)
            pts_w = maybe_warp(pts)
            patch = Polygon(pts_w, closed=True, fill=False, linewidth=1.0)
            ax.add_patch(patch)
            if geom.get("labels", True):
                cx, cy = pts_w.mean(axis=0)
                ax.text(cx, cy, poly.get("name", ""), fontsize=8, ha="center", va="center")

    if "lines" in geom:
        for line in geom["lines"]:
            pts = np.array(line["points"], dtype=float)
            pts_w = maybe_warp(pts)
            ax.plot(pts_w[:, 0], pts_w[:, 1], linewidth=1.0)
            if geom.get("labels", True):
                cx, cy = pts_w.mean(axis=0)
                ax.text(cx, cy, line.get("name", ""), fontsize=8, ha="center", va="center")


# ---------------------- Homography ----------------------
def compute_homography(src_pts: List[float], w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return H and dst quad given 4 src pts (TL,TR,BR,BL) and destination size (w,h)."""
    if len(src_pts) != 8:
        raise ValueError("--warp-src-pts requires 8 numbers: x1 y1 x2 y2 x3 y3 x4 y4")
    src = np.array(src_pts, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv.getPerspectiveTransform(src, dst)
    return H, dst


def warp_points(x: np.ndarray, y: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2).astype(np.float32)
    warped = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    return warped[:, 0], warped[:, 1]


def warp_image(img: np.ndarray, H: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv.warpPerspective(img, H, (w, h))


# ---------------------- Plotting core ----------------------
def plot_trial(ax,
               df: pd.DataFrame,
               frames: pd.Series,
               scorer: str,
               keypoint: str,
               start_frame: int,
               end_frame: int,
               min_likelihood: float,
               invert_y: bool,
               background: Optional[np.ndarray],
               maze_geom: Optional[dict],
               H: Optional[np.ndarray],
               warp_size: Optional[Tuple[int, int]]):
    # Slice points
    xcol = (scorer, keypoint, 'x')
    ycol = (scorer, keypoint, 'y')
    lkcol = (scorer, keypoint, 'likelihood') if (scorer, keypoint, 'likelihood') in df.columns else None

    mask = (frames >= start_frame) & (frames <= end_frame)
    x = df.loc[mask, xcol].astype(float).to_numpy()
    y = df.loc[mask, ycol].astype(float).to_numpy()
    if lkcol is not None:
        lk = df.loc[mask, lkcol].astype(float).to_numpy()
        valid = lk >= min_likelihood
        x, y = x[valid], y[valid]

    # Remove NaNs
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    # Optional homography
    if H is not None and len(x) > 0:
        x, y = warp_points(x, y, H)

    # Background
    if background is not None:
        ax.imshow(background, alpha=1.0)  # do not force any specific colormap/colors
    # Maze geometry
    if maze_geom is not None:
        draw_maze_geometry(ax, maze_geom, homography=H, label=maze_geom.get("labels", True))

    # Trajectory
    if len(x) > 0:
        ax.plot(x, y, linewidth=1.5)
        ax.scatter([x[0]], [y[0]], s=30, marker='o')
        ax.scatter([x[-1]], [y[-1]], s=30, marker='x')
    else:
        ax.set_title("No valid points in this interval (after filtering).")

    ax.set_aspect('equal', adjustable='datalim')
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")


def main():
    ap = argparse.ArgumentParser(description="Plot mouse trajectories per trial on maze background.")
    ap.add_argument("--dlc", required=True, help="Path to DLC CSV with 3-row header.")
    ap.add_argument("--segments", required=True, help="Path to segments_manifest CSV with start_frame/end_frame.")
    ap.add_argument("--keypoint", default=None, help="Bodypart to plot (default: tries 'nose' or first available).")
    ap.add_argument("--min-likelihood", type=float, default=0.0, help="Minimum DLC likelihood to include a point.")
    ap.add_argument("--invert-y", action="store_true", help="Invert y-axis (image-style coordinates).")
    ap.add_argument("--output-dir", default="./trajectories_maze", help="Directory for per-trial PNGs.")
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N trials.")

    # Background options
    ap.add_argument("--background-image", default=None, help="Path to a still image to show the maze.")
    ap.add_argument("--background-video", default=None, help="Path to the source video; use --bg-frame to pick a frame.")
    ap.add_argument("--bg-frame", default="mid", help='Frame index or one of {"start","mid","end"} relative to the trial.')

    # Maze geometry overlay
    ap.add_argument("--maze-geometry", default=None, help="JSON file with polygons/lines to draw (camera pixel coords).")

    # Homography rectification
    ap.add_argument("--warp-dst-size", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="Destination (width height) for rectified view.")
    ap.add_argument("--warp-src-pts", nargs=8, type=float, default=None,
                    help="Eight numbers for TL TR BR BL corners in camera coords: x1 y1 x2 y2 x3 y3 x4 y4")

    args = ap.parse_args()

    # Read DLC & segments
    df = read_dlc_csv_multiindex(args.dlc)
    scorer, bodyparts = get_scorer_and_bodyparts(df)
    frames = extract_frame_series(df)
    keypoint = pick_keypoint(bodyparts, args.keypoint)
    print(f"Using scorer='{scorer}', keypoint='{keypoint}'. Bodyparts: {', '.join(bodyparts)}")

    seg = pd.read_csv(args.segments)
    # normalize names (case-insensitive)
    lower_map = {c.lower(): c for c in seg.columns}
    required = ["trial_index", "start_frame", "end_frame"]
    for name in required:
        if name not in lower_map:
            raise ValueError(f"Segments manifest is missing '{name}' column. Found: {list(seg.columns)}")
    tcol, scol, ecol = (lower_map[n] for n in required)
    trials = seg[[tcol, scol, ecol]].sort_values(by=tcol).reset_index(drop=True)
    if args.limit is not None:
        trials = trials.head(args.limit)

    # Load optional maze geometry
    maze_geom = None
    if args.maze_geometry:
        maze_geom = load_maze_geometry(args.maze_geometry)

    # Prepare homography if requested
    H = None
    warp_size = None
    if args.warp_dst_size and args.warp_src_pts:
        W, Hh = args.warp_dst_size
        H, _dst = compute_homography(args.warp_src_pts, W, Hh)
        warp_size = (W, Hh)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Iterate trials
    for _, row in trials.iterrows():
        trial_idx = int(row[tcol])
        sframe = int(row[scol])
        eframe = int(row[ecol])

        # Build background for this trial
        bg = None
        if args.background_image:
            bg = read_image_rgb(args.background_image)
            if warp_size and H is not None:
                bg = warp_image(bg, H, *warp_size)
        elif args.background_video:
            # pick a frame based on --bg-frame spec
            idx = None
            try:
                idx = int(args.bg_frame)
            except ValueError:
                pass
            if isinstance(idx, int):
                frame_idx = idx
            else:
                frame_idx = choose_trial_frame(sframe, eframe, args.bg_frame)
            bg = read_video_frame_rgb(args.background_video, frame_idx)
            if warp_size and H is not None:
                bg = warp_image(bg, H, *warp_size)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_trial(ax, df, frames, scorer, keypoint, sframe, eframe,
                   min_likelihood=args.min_likelihood,
                   invert_y=args.invert_y,
                   background=bg,
                   maze_geom=maze_geom,
                   H=H,
                   warp_size=warp_size)
        ax.set_title(f"Trial {trial_idx} | frames {sframe}–{eframe} | keypoint='{keypoint}'")
        fig.tight_layout()

        outpath = str(outdir / f"trajectory_trial_{trial_idx:03d}_{keypoint}.png")
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
