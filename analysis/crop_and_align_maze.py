"""
crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose estimation.

The maze is a white cross/T-shaped binary decision tree (4 arms) on a dark
background, plus an entry corridor at the bottom of the rig, filmed from
below. This script detects 6 anatomical landmarks on the maze in a sampled
frame from each video, computes a homography that maps those landmarks to a
canonical layout learned from a one-time manual calibration, and warps every
frame so the maze appears identically positioned across recordings.

Workflow:

  1. Calibration (run once, --calibrate):
        python crop_and_align_maze.py --calibrate \
            --input_dir <path> --output_dir <path>
     A window opens on the middle frame of one video and you click 6
     landmarks in this order:
        1. tip of top-left arm
        2. tip of top-right arm
        3. tip of bottom-left arm
        4. tip of bottom-right arm
        5. bottom-left corner of the entry corridor
        6. bottom-right corner of the entry corridor
     The clicked points + frame size are saved to calibration.json (or
     --calibration_file). The script then exits.

  2. Batch processing (no --calibrate):
        python crop_and_align_maze.py \
            --input_dir <path> --output_dir <path>
     For every video the script:
       - thresholds the frame and finds the largest contour,
       - applies morphological closing to fill the mouse-shaped hole,
       - detects the same 6 landmarks automatically from contour geometry,
       - computes a homography (cv2.findHomography, least-squares over 6
         points) from detected landmarks to the canonical layout derived
         from the calibration,
       - warps every frame and writes the cropped/aligned output video.

Confidence per video is based on the homography's reprojection error
(mean Euclidean distance between detected landmarks projected through H
and the canonical targets):
    < 5 px       -> high
    5 - 15 px    -> medium
    > 15 px      -> low
A contour-area sanity check downgrades the label by one tier when the
detected contour is implausibly small or large.
"""

import argparse
import fnmatch
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_DIRS = [
    "segments",
    "new_segments",
    "segments_detected",
    "deeplabcut",
    "habituation",
]
DEFAULT_INCLUDE_PATTERN = "*.mp4,*.avi"
DEFAULT_CALIBRATION_FILENAME = "calibration.json"
DEFAULT_CALIBRATION_FRAME_FILENAME = "calibration_frame.png"

LANDMARK_NAMES = [
    "top_left_arm",
    "top_right_arm",
    "bottom_left_arm",
    "bottom_right_arm",
    "corridor_base_left",
    "corridor_base_right",
]
LANDMARK_LABELS = [
    "tip of top-left arm",
    "tip of top-right arm",
    "tip of bottom-left arm",
    "tip of bottom-right arm",
    "bottom-left corner of entry corridor",
    "bottom-right corner of entry corridor",
]
# Distinct BGR colors so each landmark is visually identifiable.
LANDMARK_COLORS = [
    (0, 0, 255),     # top_left_arm        - red
    (0, 165, 255),   # top_right_arm       - orange
    (0, 255, 255),   # bottom_left_arm     - yellow
    (0, 255, 0),     # bottom_right_arm    - green
    (255, 0, 255),   # corridor_base_left  - magenta
    (255, 255, 0),   # corridor_base_right - cyan
]

# Confidence thresholds (mean reprojection error in pixels).
HIGH_REPROJ_PX = 5.0
MEDIUM_REPROJ_PX = 15.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def sample_middle_frame(cap):
    """Read the frame at 50% of the video's total frame count."""
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    middle = n_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, frame = cap.read()
    return frame if ok else None


def find_videos(input_dir, include_pattern, exclude_dirs):
    """
    Recursively walk `input_dir` and return matching video paths.

    `include_pattern` is a comma-separated list of fnmatch globs (e.g.
    "*.mp4,*.avi"). `exclude_dirs` is a list of directory names (any
    component matching is pruned, case-insensitive). Excluded directories
    are pruned from os.walk in-place so the script doesn't even descend
    into them.
    """
    patterns = [p.strip() for p in str(include_pattern).split(",") if p.strip()]
    excluded = {name.strip().lower() for name in exclude_dirs if name.strip()}

    matches = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in excluded]
        for fname in files:
            lower = fname.lower()
            for pat in patterns:
                if fnmatch.fnmatch(lower, pat.lower()):
                    matches.append(Path(root) / fname)
                    break
    return sorted(matches)


def warp_video(video_path, H, dst_size, output_path, fps):
    """Apply H to every frame of the input video and write to output_path."""
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, dst_size)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            warped = cv2.warpPerspective(frame, H, dst_size)
            writer.write(warped)
    finally:
        writer.release()
        cap.release()


# ---------------------------------------------------------------------------
# Maze detection / mask cleanup
# ---------------------------------------------------------------------------

def detect_maze_contour(frame):
    """Binarize the frame and return (largest_contour, binary)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, binary
    largest = max(contours, key=cv2.contourArea)
    return largest, binary


def close_mask(mask, kernel_size):
    """Morphological closing with a square kernel; no-op for size <= 1."""
    k = int(kernel_size)
    if k <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def closed_maze_contour(frame, kernel_size):
    """
    Threshold + close + re-extract the largest contour.

    Closing is applied to the threshold *before* re-extracting the contour
    so the mouse-shaped hole / mouse-induced gap in the maze outline gets
    filled before any landmark detection sees the shape.
    """
    contour, binary = detect_maze_contour(frame)
    if contour is None:
        return None
    closed = close_mask(binary, kernel_size)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# Calibration UI
# ---------------------------------------------------------------------------

def calibrate_landmarks(frame, title="Maze calibration"):
    """
    Show `frame` and collect 6 landmark clicks in LANDMARK_NAMES order.

    Returns a (6, 2) float32 array in source-image coordinates, or None
    if the user pressed ESC. Provides:
      - 'r' to reset clicks during placement
      - 'y' / 'Y' to confirm once 6 clicks are placed
      - 'r' to redo from confirmation step

    Large frames are downscaled for display; clicks are scaled back to
    the original resolution.
    """
    h, w = frame.shape[:2]
    max_dim = 1200
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
    base_disp = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 6:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_w, disp_h)
    cv2.setMouseCallback(title, on_mouse)

    def render(stage_msg):
        disp = base_disp.copy()
        for i, (cx, cy) in enumerate(clicks):
            px, py = int(cx * scale), int(cy * scale)
            cv2.circle(disp, (px, py), 8, LANDMARK_COLORS[i], -1)
            cv2.putText(disp, f"{i + 1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, LANDMARK_COLORS[i], 2)
        # Connect arm tips into a quad if all 4 are placed.
        if len(clicks) >= 4:
            for i in range(4):
                p1 = (int(clicks[i][0] * scale), int(clicks[i][1] * scale))
                p2 = (int(clicks[(i + 1) % 4][0] * scale),
                      int(clicks[(i + 1) % 4][1] * scale))
                cv2.line(disp, p1, p2, (0, 255, 0), 1)
        if len(clicks) >= 6:
            p1 = (int(clicks[4][0] * scale), int(clicks[4][1] * scale))
            p2 = (int(clicks[5][0] * scale), int(clicks[5][1] * scale))
            cv2.line(disp, p1, p2, (255, 0, 255), 2)
        cv2.putText(disp, stage_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, title, (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return disp

    while True:
        # Phase 1: collect 6 clicks.
        while len(clicks) < 6:
            n = len(clicks)
            msg = f"Click point {n + 1}/6: {LANDMARK_LABELS[n]}.  'r' reset, ESC quit."
            cv2.imshow(title, render(msg))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(title); cv2.waitKey(1)
                return None
            if key == ord('r'):
                clicks.clear()

        # Phase 2: confirm.
        confirmed = None
        while confirmed is None:
            cv2.imshow(title, render("All 6 placed.  'y' confirm, 'r' redo, ESC quit."))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(title); cv2.waitKey(1)
                return None
            if key in (ord('y'), ord('Y')):
                confirmed = True
            elif key == ord('r'):
                clicks.clear()
                confirmed = False  # falls back to phase 1 loop

        if confirmed:
            break

    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return np.asarray(clicks, dtype=np.float32)


def save_calibration(path, calibration_video, frame_shape, landmarks):
    """Write calibration JSON. landmarks is a (6, 2) array in source pixels."""
    payload = {
        "calibration_video": str(calibration_video),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],  # [h, w]
        "landmark_names": LANDMARK_NAMES,
        "landmarks": [[float(p[0]), float(p[1])] for p in landmarks],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path):
    """Load calibration JSON, returning the (6, 2) landmark array."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("landmark_names") and payload["landmark_names"] != LANDMARK_NAMES:
        raise ValueError(
            f"Calibration file landmark_names {payload['landmark_names']} "
            f"do not match expected {LANDMARK_NAMES}."
        )
    landmarks = np.asarray(payload["landmarks"], dtype=np.float32)
    if landmarks.shape != (6, 2):
        raise ValueError(f"Calibration must contain 6x2 landmarks; got {landmarks.shape}.")
    return landmarks, payload


def canonical_positions(calibration_landmarks, padding):
    """
    Translate the calibration landmarks so the bounding box sits at
    (padding, padding). Returns (canonical_pts (6,2), (out_w, out_h)).

    The calibration's relative spacing and proportions are preserved
    exactly — the warp just centers the maze in a padded canvas.
    """
    pts = np.asarray(calibration_landmarks, dtype=np.float32).copy()
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pts[:, 0] -= xmin - padding
    pts[:, 1] -= ymin - padding
    out_w = int(round(xmax - xmin + 2 * padding))
    out_h = int(round(ymax - ymin + 2 * padding))
    return pts, (out_w, out_h)


# ---------------------------------------------------------------------------
# Automatic landmark detection
# ---------------------------------------------------------------------------

def _approx_hull(contour, target_min=8, target_max=12):
    """
    cv2.approxPolyDP on the convex hull, decreasing epsilon until the
    vertex count lands in [target_min, target_max]. Returns an Nx2 array.
    Falls back to the full hull if no epsilon produces a count in range.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    best = hull.reshape(-1, 2)
    for eps_frac in (0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002):
        approx = cv2.approxPolyDP(hull, eps_frac * perimeter, True).reshape(-1, 2)
        n = len(approx)
        if target_min <= n <= target_max:
            return approx
        if n > target_max:
            best = approx  # remember the densest candidate before overshooting
            break
    return best


def _contour_centroid(contour):
    """(cx, cy) from image moments, or None if degenerate."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


def detect_landmarks(contour):
    """
    Detect the 6 maze landmarks from a (closed) maze contour.

    Returns (landmarks_or_none, debug). landmarks is a (6, 2) float32 in
    LANDMARK_NAMES order on success, None on failure. `debug` carries
    intermediate detection results (per-quadrant arm tip candidates and
    the corridor near-bottom strip) so failed cases can still draw what
    was found.
    """
    debug = {"arm_tips": {}, "near_bottom": None, "approx_hull": None}
    if contour is None or len(contour) < 6:
        return None, debug

    pts = contour.reshape(-1, 2).astype(np.float32)
    centroid = _contour_centroid(contour)
    if centroid is None:
        return None, debug
    cx, cy = centroid

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    mid_x = float((xmin + xmax) / 2.0)
    mid_y = float((ymin + ymax) / 2.0)

    # --- 4 arm tips: cluster approx-hull vertices by bbox quadrant, then
    #     pick the vertex farthest from the contour centroid in each.
    approx = _approx_hull(contour)
    debug["approx_hull"] = approx
    quads = {"tl": [], "tr": [], "bl": [], "br": []}
    for p in approx:
        x, y = float(p[0]), float(p[1])
        if x < mid_x and y < mid_y:
            quads["tl"].append((x, y))
        elif x >= mid_x and y < mid_y:
            quads["tr"].append((x, y))
        elif x < mid_x and y >= mid_y:
            quads["bl"].append((x, y))
        else:
            quads["br"].append((x, y))

    arm_tips = {}
    for q in ("tl", "tr", "bl", "br"):
        if not quads[q]:
            return None, debug  # missing a quadrant -> can't form 4 tips
        farthest = max(quads[q], key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        arm_tips[q] = farthest
    debug["arm_tips"] = arm_tips

    # --- 2 corridor base points: leftmost & rightmost contour points
    #     within 10 px of the maximum y (the very bottom of the maze).
    near_bottom = pts[pts[:, 1] >= ymax - 10.0]
    debug["near_bottom"] = near_bottom
    if len(near_bottom) < 2:
        return None, debug
    leftmost = near_bottom[near_bottom[:, 0].argmin()]
    rightmost = near_bottom[near_bottom[:, 0].argmax()]

    landmarks = np.array([
        arm_tips["tl"],
        arm_tips["tr"],
        arm_tips["bl"],
        arm_tips["br"],
        (float(leftmost[0]), float(leftmost[1])),
        (float(rightmost[0]), float(rightmost[1])),
    ], dtype=np.float32)
    return landmarks, debug


# ---------------------------------------------------------------------------
# Homography + scoring
# ---------------------------------------------------------------------------

def compute_homography(detected_pts, canonical_pts):
    """
    Least-squares homography mapping `detected_pts` -> `canonical_pts`.
    Returns the 3x3 H or None if the solve fails.
    """
    src = np.asarray(detected_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(canonical_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def reprojection_error(detected_pts, canonical_pts, H):
    """Mean Euclidean distance between H(detected_pts) and canonical_pts."""
    src = np.asarray(detected_pts, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    dst = np.asarray(canonical_pts, dtype=np.float32)
    diffs = projected - dst
    return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())


def classify_confidence(reproj_px, area_flag):
    """
    Confidence label from reprojection error (px) and area sanity flag.
    Reprojection error is the primary signal; an area flag downgrades
    one tier (high -> medium, medium -> low).
    """
    if not np.isfinite(reproj_px):
        return "failed"
    if reproj_px < HIGH_REPROJ_PX:
        base = "high"
    elif reproj_px <= MEDIUM_REPROJ_PX:
        base = "medium"
    else:
        base = "low"
    if area_flag:
        if base == "high":
            return "medium"
        if base == "medium":
            return "low"
    return base


# ---------------------------------------------------------------------------
# Pass 1: per-video detection
# ---------------------------------------------------------------------------

def detect_one(video_path, input_dir, canonical_pts, dst_size, args):
    """
    Sample the middle frame, detect the closed maze contour, detect 6
    landmarks, compute the homography to the canonical layout, and score.
    Returns a dict consumed by pass 2 and the CSV writer.
    """
    rel = video_path.relative_to(input_dir)
    base = {
        "video_path": video_path,
        "rel": rel,
        "filename": str(rel).replace("\\", "/"),
        "status": "failed",
        "frame": None,
        "fps": None,
        "contour": None,
        "detected_landmarks": None,
        "debug": {"arm_tips": {}, "near_bottom": None, "approx_hull": None},
        "H": None,
        "dst_size": dst_size,
        "contour_area": np.nan,
        "reprojection_error": np.nan,
        "area_flag": False,
        "confidence": "failed",
        "error": "",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        base["error"] = "could not open video"
        return base

    frame = sample_middle_frame(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if frame is None:
        base["error"] = "could not sample frame"
        return base
    base["frame"] = frame
    base["fps"] = fps

    contour = closed_maze_contour(frame, args.morph_kernel_size)
    if contour is None:
        base["error"] = "no contour found"
        return base
    base["contour"] = contour

    contour_area = float(cv2.contourArea(contour))
    frame_h, frame_w = frame.shape[:2]
    area_ratio = contour_area / float(frame_h * frame_w)
    base["contour_area"] = contour_area
    base["area_flag"] = not (0.05 <= area_ratio <= 0.5)

    detected, debug = detect_landmarks(contour)
    base["debug"] = debug
    if detected is None:
        base["error"] = "could not detect 6 landmarks"
        return base
    base["detected_landmarks"] = detected

    H = compute_homography(detected, canonical_pts)
    if H is None:
        base["error"] = "homography solve failed"
        return base
    rep_err = reprojection_error(detected, canonical_pts, H)
    base.update({
        "status": "ok",
        "H": H,
        "reprojection_error": rep_err,
        "confidence": classify_confidence(rep_err, base["area_flag"]),
    })
    return base


# ---------------------------------------------------------------------------
# Review image rendering
# ---------------------------------------------------------------------------

def save_review_image(frame, contour, landmarks, debug, review_path):
    """
    Save a debug PNG of the sampled frame with detection overlays:
    contour outline (cyan), the approx-hull vertices (small white dots),
    near-bottom strip (thin green line), and the 6 landmarks colored by
    LANDMARK_COLORS. Whatever fields are present in `debug` are drawn so
    failed detections can still be inspected.
    """
    img = frame.copy()
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (255, 255, 0), 2)

    approx = debug.get("approx_hull")
    if approx is not None:
        for p in approx:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)

    near_bottom = debug.get("near_bottom")
    if near_bottom is not None and len(near_bottom) > 0:
        ys = near_bottom[:, 1]
        y = int(np.median(ys))
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)

    # Per-quadrant arm tips (drawn even on failure with smaller markers).
    for q, p in (debug.get("arm_tips") or {}).items():
        if p is None:
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 200, 200), 2)

    if landmarks is not None:
        for i, p in enumerate(landmarks):
            cv2.circle(img, (int(p[0]), int(p[1])), 9, LANDMARK_COLORS[i], -1)
            cv2.putText(img, f"{i + 1}", (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, LANDMARK_COLORS[i], 2)

    cv2.imwrite(str(review_path), img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=Path,
                   help="Directory containing .mp4/.avi videos (searched recursively). "
                        "Required for batch mode; in --calibrate mode, used to find a "
                        "video if --calibrate_video is not given.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory to write cropped videos, the calibration file (default), "
                        "the review folder, and the summary CSV.")
    p.add_argument("--padding", type=int, default=50,
                   help="Pixels of padding around the bounding box of the canonical landmarks.")
    p.add_argument("--morph_kernel_size", type=int, default=50,
                   help="Square kernel size (px) for MORPH_CLOSE on the threshold mask. "
                        "Should be comfortably larger than the mouse but smaller than the maze.")
    p.add_argument("--exclude_dirs", nargs="*", default=DEFAULT_EXCLUDE_DIRS,
                   help="Directory names to skip when walking input_dir (case-insensitive). "
                        f"Default: {' '.join(DEFAULT_EXCLUDE_DIRS)}.")
    p.add_argument("--include_pattern", type=str, default=DEFAULT_INCLUDE_PATTERN,
                   help="Comma-separated fnmatch globs for video filenames "
                        f"(default: \"{DEFAULT_INCLUDE_PATTERN}\").")
    p.add_argument("--calibrate", action="store_true",
                   help="Enter calibration mode: click 6 maze landmarks on a sampled "
                        "frame and save them to the calibration file. Exits without "
                        "processing any videos.")
    p.add_argument("--calibrate_video", type=Path, default=None,
                   help="In --calibrate mode, the specific video to calibrate on. "
                        "If omitted, the first video found under --input_dir is used.")
    p.add_argument("--calibration_file", type=Path, default=None,
                   help="Path to the calibration JSON. Defaults to "
                        "<output_dir>/calibration.json.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(args, calibration_path):
    """
    Pick a video, sample its middle frame, run the 6-click UI, and save
    the calibration JSON + a calibration_frame.png next to it.
    """
    if args.calibrate_video is not None:
        video_path = args.calibrate_video
        if not video_path.exists():
            print(f"Error: --calibrate_video does not exist: {video_path}")
            return 1
    else:
        if args.input_dir is None:
            print("Error: --calibrate needs either --calibrate_video or --input_dir.")
            return 1
        videos = find_videos(args.input_dir, args.include_pattern, args.exclude_dirs)
        if not videos:
            print(f"Error: no videos found under {args.input_dir} for calibration.")
            return 1
        video_path = videos[0]

    print(f"Calibrating on: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open calibration video: {video_path}")
        return 1
    frame = sample_middle_frame(cap)
    cap.release()
    if frame is None:
        print(f"Error: could not sample a frame from: {video_path}")
        return 1

    landmarks = calibrate_landmarks(frame, title=f"Calibration: {video_path.name}")
    if landmarks is None:
        print("Calibration cancelled.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration_path, video_path, frame.shape[:2], landmarks)

    frame_out = args.output_dir / DEFAULT_CALIBRATION_FRAME_FILENAME
    annotated = frame.copy()
    for i, p in enumerate(landmarks):
        cv2.circle(annotated, (int(p[0]), int(p[1])), 8, LANDMARK_COLORS[i], -1)
        cv2.putText(annotated, f"{i + 1}", (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LANDMARK_COLORS[i], 2)
    cv2.imwrite(str(frame_out), annotated)

    print(f"Saved calibration: {calibration_path}")
    print(f"Saved annotated calibration frame: {frame_out}")
    return 0


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def make_failed_row(rel, error=None):
    return {
        "filename": str(rel).replace("\\", "/"),
        "contour_area": np.nan,
        "reprojection_error": np.nan,
        "confidence": "failed",
        "output_path": f"error: {error}" if error else "",
    }


def run_batch(args, calibration_path):
    """Run the two-pass batch pipeline. Returns a process exit code."""
    if not calibration_path.exists():
        print(
            f"Error: calibration file not found at {calibration_path}.\n"
            f"Run calibration first, e.g.:\n"
            f"    python crop_and_align_maze.py --calibrate "
            f"--input_dir <path> --output_dir <path>"
        )
        return 2
    if args.input_dir is None:
        print("Error: --input_dir is required for batch processing.")
        return 1

    calibration_landmarks, _ = load_calibration(calibration_path)
    canonical_pts, dst_size = canonical_positions(calibration_landmarks, args.padding)
    print(
        f"Loaded calibration from {calibration_path}; "
        f"canonical output size = {dst_size[0]}x{dst_size[1]} px."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(args.input_dir, args.include_pattern, args.exclude_dirs)
    if not videos:
        print(
            f"No videos matching '{args.include_pattern}' found under "
            f"{args.input_dir} (excluded dirs: {args.exclude_dirs})."
        )
        return 0
    total = len(videos)
    print(
        f"Found {total} video(s). Excluding dirs: {args.exclude_dirs}. "
        f"Pattern: {args.include_pattern}."
    )

    # ----------------------------------------------------------------------
    # Pass 1: detect landmarks for every video.
    # ----------------------------------------------------------------------
    print("Pass 1/2: detecting landmarks...")
    detections = []
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(args.input_dir)
        try:
            d = detect_one(video_path, args.input_dir, canonical_pts, dst_size, args)
        except Exception as e:
            d = {
                "video_path": video_path,
                "rel": rel,
                "filename": str(rel).replace("\\", "/"),
                "status": "failed",
                "frame": None,
                "fps": None,
                "contour": None,
                "detected_landmarks": None,
                "debug": {"arm_tips": {}, "near_bottom": None, "approx_hull": None},
                "H": None,
                "dst_size": dst_size,
                "contour_area": np.nan,
                "reprojection_error": np.nan,
                "area_flag": False,
                "confidence": "failed",
                "error": f"exception: {e}",
            }
        detections.append(d)
        if d["status"] == "ok":
            print(f"  [{i}/{total}] {video_path.name} — detection ok "
                  f"(reproj {d['reprojection_error']:.2f} px, {d['confidence']})")
        else:
            print(f"  [{i}/{total}] {video_path.name} — detection failed: {d['error']}")

    # ----------------------------------------------------------------------
    # Pass 2: write cropped videos and review images.
    # ----------------------------------------------------------------------
    print("Pass 2/2: writing cropped videos...")
    rows = []
    for i, d in enumerate(detections, start=1):
        rel = d["rel"]
        if d["status"] != "ok" or d["H"] is None:
            # Save a debug review image for failed detections (showing
            # whatever partial landmarks were found).
            if d["frame"] is not None:
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                review_path.parent.mkdir(parents=True, exist_ok=True)
                save_review_image(d["frame"], d["contour"], d["detected_landmarks"],
                                  d["debug"], review_path)
            rows.append(make_failed_row(rel, d.get("error")))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed")
            continue

        try:
            output_path = args.output_dir / rel.parent / (d["video_path"].stem + "_cropped.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            warp_video(d["video_path"], d["H"], d["dst_size"], output_path, d["fps"])

            if d["confidence"] in ("medium", "low"):
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                review_path.parent.mkdir(parents=True, exist_ok=True)
                save_review_image(d["frame"], d["contour"], d["detected_landmarks"],
                                  d["debug"], review_path)

            rows.append({
                "filename": d["filename"],
                "contour_area": d["contour_area"],
                "reprojection_error": d["reprojection_error"],
                "confidence": d["confidence"],
                "output_path": str(output_path),
            })
            print(f"  [{i}/{total}] {d['filename']} — confidence: {d['confidence']} "
                  f"(reproj {d['reprojection_error']:.2f} px)")
        except Exception as e:
            rows.append(make_failed_row(rel, str(e)))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed ({e})")

    summary = pd.DataFrame(rows, columns=[
        "filename", "contour_area", "reprojection_error", "confidence", "output_path",
    ])
    summary_path = args.output_dir / "alignment_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    counts = summary["confidence"].value_counts().to_dict()
    print(
        "\nConfidence summary: "
        f"high={counts.get('high', 0)}, "
        f"medium={counts.get('medium', 0)}, "
        f"low={counts.get('low', 0)}, "
        f"failed={counts.get('failed', 0)} "
        f"(total {len(summary)})"
    )
    return 0


def main():
    args = parse_args()

    calibration_path = (
        args.calibration_file
        if args.calibration_file is not None
        else args.output_dir / DEFAULT_CALIBRATION_FILENAME
    )

    if args.calibrate:
        return run_calibration(args, calibration_path)
    return run_batch(args, calibration_path)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
