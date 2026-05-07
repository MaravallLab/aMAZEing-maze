"""
crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose estimation.

The maze is a white cross/T-shaped binary decision tree (4 arms) on a dark
background, plus an entry corridor at the bottom of the rig, filmed from
below. The camera also sees room walls, ceiling panels, and other bright
surfaces around the maze, which a naive "largest bright contour" detector
will sometimes lock onto instead of the maze itself.

This script avoids that problem by using a one-time manual calibration to
define both the canonical layout AND a search ROI that excludes everything
outside the maze area:

  Stage 1 — Calibration (--calibrate):
      The user clicks all 24 corners of the maze outline (clockwise from
      the top-left). The bounding box of those 24 points is expanded by
      150 px on each side and saved as the search ROI. The 24 points,
      ROI, and frame size are saved to calibration.json.

  Stage 2 — Batch processing (no --calibrate):
      For every video, the script crops to the ROI BEFORE thresholding
      (so room walls don't pollute the binary mask), morphologically
      closes the mouse-shaped hole, runs cv2.goodFeaturesToTrack on the
      cropped binary mask to detect maze corners, matches detected
      corners to the 24 calibration points via the Hungarian algorithm,
      and computes a RANSAC homography from the matched pairs. Every
      frame is then warped through that homography and the cropped
      output is written.

Per-video fallback (--manual_video <path>):
      For videos where automatic detection fails, the user can re-run
      the 24-point click UI on the failing video. The clicked points
      are saved to <stem>_landmarks.json next to the cropped output and
      the video is processed with those points. On subsequent batch
      runs, the saved per-video landmarks are picked up automatically
      (use --redo_video <name> to force re-detection).

Confidence is based on the homography reprojection error AND the number
of matched corners:
    high   : reproj < 10 px AND matched >= 20
    medium : reproj < 25 px AND matched >= 16
    low    : anything worse with matched >= 16
    failed : matched < 16, or any earlier step blew up.
"""

import argparse
import fnmatch
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


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

NUM_LANDMARKS = 24
ROI_MARGIN_PX = 150
MAX_MATCH_DISTANCE_PX = 60
MIN_MATCHED_POINTS = 16

# Confidence thresholds.
HIGH_REPROJ_PX = 10.0
HIGH_MIN_MATCHED = 20
MEDIUM_REPROJ_PX = 25.0
MEDIUM_MIN_MATCHED = MIN_MATCHED_POINTS

# Click-order labels (1-based from the user's perspective).
LANDMARK_LABELS = [
    "TL arm, outer top-left corner",
    "TL arm, outer top-right corner",
    "inner corner: TL arm meets junction (top side)",
    "inner corner: TR arm meets junction (top side)",
    "TR arm, outer top-left corner",
    "TR arm, outer top-right corner",
    "TR arm, outer bottom-right corner",
    "TR arm, outer bottom-left corner",
    "inner corner: TR arm meets junction (bottom side)",
    "inner corner: junction meets corridor right (top)",
    "inner corner: BR arm meets corridor (top side)",
    "BR arm, outer top-right corner",
    "BR arm, outer bottom-right corner",
    "BR arm, outer bottom-left corner",
    "inner corner: BR arm meets corridor (bottom side)",
    "corridor right side, bottom corner",
    "corridor left side, bottom corner",
    "inner corner: BL arm meets corridor (bottom side)",
    "BL arm, outer bottom-right corner",
    "BL arm, outer bottom-left corner",
    "BL arm, outer top-left corner",
    "inner corner: BL arm meets corridor (top side)",
    "inner corner: junction meets corridor left (top)",
    "inner corner: TL arm meets junction (bottom side)",
]

# Stylized 24-point maze schematic for the on-screen reference diagram.
REFERENCE_POINTS = np.array([
    (45, 15),    (95, 15),    (95, 75),    (145, 75),
    (145, 15),   (195, 15),   (195, 75),   (160, 95),
    (145, 105),  (145, 130),  (160, 145),  (195, 145),
    (195, 195),  (160, 215),  (145, 195),  (145, 230),
    (95, 230),   (95, 195),   (80, 215),   (45, 195),
    (45, 145),   (80, 145),   (95, 130),   (95, 105),
], dtype=np.int32)
REFERENCE_CANVAS_SIZE = 240


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def sample_middle_frame(cap):
    """Read the frame at 50% of the video's total frame count."""
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ok, frame = cap.read()
    return frame if ok else None


def find_videos(input_dir, include_pattern, exclude_dirs):
    """
    Recursively walk `input_dir` and return matching video paths.

    `include_pattern` is a comma-separated list of fnmatch globs.
    `exclude_dirs` is a list of directory names; any directory whose
    name matches (case-insensitive) is pruned in-place during the walk.
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
# Reference diagram for the calibration UI
# ---------------------------------------------------------------------------

def make_reference_diagram(size=REFERENCE_CANVAS_SIZE):
    """
    Render a small schematic of the 24-point click order so the user can
    see which corner is "next" without holding the order in their head.
    Returns a BGR uint8 image of `size`x`size`.
    """
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)

    # Outline connecting the 24 points in click order (closes back to 1).
    for i in range(NUM_LANDMARKS):
        p1 = tuple(int(v) for v in REFERENCE_POINTS[i])
        p2 = tuple(int(v) for v in REFERENCE_POINTS[(i + 1) % NUM_LANDMARKS])
        cv2.line(canvas, p1, p2, (90, 90, 90), 1)

    # Numbered dots.
    for i, p in enumerate(REFERENCE_POINTS):
        ix, iy = int(p[0]), int(p[1])
        cv2.circle(canvas, (ix, iy), 3, (220, 220, 220), -1)
        cv2.putText(canvas, str(i + 1), (ix + 4, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "click order", (8, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), (160, 160, 160), 1)
    return canvas


def overlay_reference(disp, ref):
    """Blit `ref` into the top-right corner of `disp` (in-place)."""
    h, w = disp.shape[:2]
    rh, rw = ref.shape[:2]
    if rh + 20 > h or rw + 20 > w:
        return  # display too small to overlay
    y0, x0 = 10, w - rw - 10
    disp[y0:y0 + rh, x0:x0 + rw] = ref


# ---------------------------------------------------------------------------
# 24-click UI (used by both --calibrate and --manual_video)
# ---------------------------------------------------------------------------

def click_landmarks_ui(frame, title, num_points=NUM_LANDMARKS, with_reference=True):
    """
    Display `frame`, collect `num_points` clicks in click-order, return
    an Nx2 float32 array of source-image coordinates, or None on cancel.

    Controls:
      - left click to place the next point
      - 'r' to reset all clicks (during placement OR confirm phase)
      - 'y' / 'Y' to confirm once all points are placed
      - ESC to cancel
    """
    h, w = frame.shape[:2]
    max_dim = 1400
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
    base_disp = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()
    if with_reference:
        ref = make_reference_diagram()
        overlay_reference(base_disp, ref)

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < num_points:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_w, disp_h)
    cv2.setMouseCallback(title, on_mouse)

    def render(stage_msg):
        disp = base_disp.copy()
        # Outline so far.
        for i in range(len(clicks) - 1):
            p1 = (int(clicks[i][0] * scale), int(clicks[i][1] * scale))
            p2 = (int(clicks[i + 1][0] * scale), int(clicks[i + 1][1] * scale))
            cv2.line(disp, p1, p2, (0, 200, 200), 1)
        if len(clicks) == num_points:
            p1 = (int(clicks[-1][0] * scale), int(clicks[-1][1] * scale))
            p2 = (int(clicks[0][0] * scale), int(clicks[0][1] * scale))
            cv2.line(disp, p1, p2, (0, 200, 200), 1)
        # Numbered dots.
        for i, (cx, cy) in enumerate(clicks):
            color = (
                int(60 + (i / max(1, num_points - 1)) * 195),
                int(255 - (i / max(1, num_points - 1)) * 195),
                int(255),
            )
            px, py = int(cx * scale), int(cy * scale)
            cv2.circle(disp, (px, py), 6, color, -1)
            cv2.putText(disp, str(i + 1), (px + 7, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(disp, stage_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, title, (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return disp

    while True:
        # Phase 1: collect all clicks.
        while len(clicks) < num_points:
            n = len(clicks)
            label = LANDMARK_LABELS[n] if n < len(LANDMARK_LABELS) else f"point {n + 1}"
            msg = f"Click {n + 1}/{num_points}: {label}.  'r' reset, ESC cancel."
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
            cv2.imshow(title, render(
                f"All {num_points} placed.  'y' confirm, 'r' redo, ESC cancel."
            ))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(title); cv2.waitKey(1)
                return None
            if key in (ord('y'), ord('Y')):
                confirmed = True
            elif key == ord('r'):
                clicks.clear()
                confirmed = False

        if confirmed:
            break

    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return np.asarray(clicks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Calibration / per-video landmark JSON I/O
# ---------------------------------------------------------------------------

def compute_search_roi(landmarks, frame_shape, margin=ROI_MARGIN_PX):
    """
    Return [x_min, y_min, x_max, y_max] integer ROI clamped to the frame.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return [
        int(max(0, xmin - margin)),
        int(max(0, ymin - margin)),
        int(min(w, xmax + margin)),
        int(min(h, ymax + margin)),
    ]


def save_calibration(path, calibration_video, frame_shape, landmarks, roi):
    payload = {
        "calibration_video": str(calibration_video),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "num_landmarks": NUM_LANDMARKS,
        "landmarks": [[float(p[0]), float(p[1])] for p in landmarks],
        "search_roi": [int(v) for v in roi],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    landmarks = np.asarray(payload["landmarks"], dtype=np.float32)
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Calibration must contain {NUM_LANDMARKS}x2 landmarks; "
            f"got {landmarks.shape}."
        )
    roi = payload.get("search_roi")
    if roi is None or len(roi) != 4:
        raise ValueError("Calibration JSON missing 'search_roi'.")
    return {
        "landmarks": landmarks,
        "search_roi": [int(v) for v in roi],
        "frame_shape": tuple(int(v) for v in payload["frame_shape"]),
        "raw": payload,
    }


def manual_landmarks_path(output_dir, rel_video_path):
    rel = Path(rel_video_path)
    return output_dir / rel.parent / (rel.stem + "_landmarks.json")


def save_video_landmarks(path, video_path, frame_shape, landmarks):
    payload = {
        "video": str(video_path),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "num_landmarks": NUM_LANDMARKS,
        "landmarks": [[float(p[0]), float(p[1])] for p in landmarks],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_video_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    landmarks = np.asarray(payload["landmarks"], dtype=np.float32)
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Per-video landmarks must contain {NUM_LANDMARKS}x2 points; "
            f"got {landmarks.shape}."
        )
    return landmarks


# ---------------------------------------------------------------------------
# Canonical layout
# ---------------------------------------------------------------------------

def canonical_positions(calibration_landmarks, padding):
    """
    Translate the 24 calibration landmarks so their bounding box sits at
    (padding, padding). Returns (canonical_pts (24,2), (out_w, out_h)).

    Spacing and proportions are preserved exactly; the warp just centers
    the maze in a padded canvas of the same size as the calibration bbox
    plus 2*padding on each side.
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
# ROI-constrained mask + corner detection
# ---------------------------------------------------------------------------

def crop_roi(frame, roi):
    """Slice the frame to [x_min, y_min, x_max, y_max]."""
    x0, y0, x1, y1 = roi
    return frame[y0:y1, x0:x1]


def closed_binary_in_roi(frame, roi, kernel_size):
    """
    Crop to ROI, threshold (Otsu), then morphologically close. Returns
    (binary_roi, contour_roi_or_None) where contour_roi is the largest
    contour found in the closed ROI mask, in ROI-local coordinates.
    """
    sub = crop_roi(frame, roi)
    if sub.size == 0:
        return None, None
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(1, int(kernel_size))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    return binary, largest


def detect_corners_in_roi(binary_roi, roi, max_corners=60, quality=0.01, min_distance=20):
    """
    Run cv2.goodFeaturesToTrack on the ROI binary mask. Returns an Nx2
    float32 array of corner coordinates in **full-frame** pixel space
    (ROI offset already added back).
    """
    if binary_roi is None:
        return np.zeros((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        binary_roi, maxCorners=int(max_corners),
        qualityLevel=float(quality), minDistance=int(min_distance),
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = corners.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += roi[0]
    pts[:, 1] += roi[1]
    return pts


# ---------------------------------------------------------------------------
# Hungarian matching + homography + scoring
# ---------------------------------------------------------------------------

def hungarian_match(detected_pts, calibration_pts, max_distance=MAX_MATCH_DISTANCE_PX):
    """
    Match detected corners to the 24 calibration points.

    Builds a (num_calib, num_detected) Euclidean distance matrix and
    solves the assignment via scipy.optimize.linear_sum_assignment.
    Returns:
        matched_calib_idx  (M,)  : indices into calibration_pts (0..23)
        matched_det_idx    (M,)  : corresponding indices into detected_pts
        matched_distance   (M,)  : per-pair distances
    Pairs whose distance exceeds `max_distance` are dropped.
    """
    if len(detected_pts) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float))
    cal = np.asarray(calibration_pts, dtype=np.float32)
    det = np.asarray(detected_pts, dtype=np.float32)
    # Distance matrix (num_calib, num_detected).
    diff = cal[:, None, :] - det[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    # Hungarian. linear_sum_assignment minimizes total cost.
    row_idx, col_idx = linear_sum_assignment(dist)
    pair_dists = dist[row_idx, col_idx]
    keep = pair_dists <= float(max_distance)
    return row_idx[keep], col_idx[keep], pair_dists[keep]


def compute_homography_ransac(src_pts, dst_pts, ransac_threshold=5.0):
    """
    findHomography with RANSAC over `src_pts` -> `dst_pts`. Returns the
    3x3 H or None on failure.
    """
    if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
        return None
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC,
                              ransacReprojThreshold=float(ransac_threshold))
    return H


def reprojection_error(src_pts, dst_pts, H):
    """Mean Euclidean distance between H(src_pts) and dst_pts."""
    if H is None or len(src_pts) == 0:
        return float("inf")
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32)
    diffs = projected - dst
    return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())


def classify_confidence(reproj_px, num_matched):
    """
    Confidence label from reprojection error and the number of matched
    landmarks. `failed` is signalled by num_matched < MIN_MATCHED_POINTS.
    """
    if not np.isfinite(reproj_px):
        return "failed"
    if num_matched < MIN_MATCHED_POINTS:
        return "failed"
    if reproj_px < HIGH_REPROJ_PX and num_matched >= HIGH_MIN_MATCHED:
        return "high"
    if reproj_px < MEDIUM_REPROJ_PX and num_matched >= MEDIUM_MIN_MATCHED:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Per-video pipeline (pass 1 worker)
# ---------------------------------------------------------------------------

def empty_detection_record(video_path, rel, dst_size):
    return {
        "video_path": video_path,
        "rel": rel,
        "filename": str(rel).replace("\\", "/"),
        "status": "failed",
        "frame": None,
        "fps": None,
        "contour_roi": None,
        "binary_roi": None,
        "detected_corners": None,
        "matched_calib_idx": None,
        "matched_detected_pts": None,
        "matched_canonical_pts": None,
        "calibration_landmarks": None,
        "H": None,
        "dst_size": dst_size,
        "num_matched": 0,
        "reprojection_error": np.nan,
        "source": "auto",
        "confidence": "failed",
        "error": "",
    }


def detect_one(video_path, input_dir, calibration, canonical_pts, dst_size, args,
               manual_landmarks=None):
    """
    Per-video detection. Samples the middle frame, then either:
      - if `manual_landmarks` was supplied (24x2 in source coords), uses
        those directly (all 24 matched, no auto-detection); or
      - crops to the calibration ROI, thresholds + closes, runs
        goodFeaturesToTrack, Hungarian-matches detected corners to the
        calibration's 24 points, and computes a RANSAC homography.

    Returns a dict consumed by pass 2 and the CSV writer.
    """
    rel = video_path.relative_to(input_dir)
    rec = empty_detection_record(video_path, rel, dst_size)
    rec["calibration_landmarks"] = calibration["landmarks"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        rec["error"] = "could not open video"
        return rec
    frame = sample_middle_frame(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if frame is None:
        rec["error"] = "could not sample frame"
        return rec
    rec["frame"] = frame
    rec["fps"] = fps

    calib_pts = calibration["landmarks"]

    # Path A: per-video manual landmarks override automatic detection.
    if manual_landmarks is not None:
        rec["source"] = "manual"
        rec["matched_calib_idx"] = np.arange(NUM_LANDMARKS, dtype=int)
        rec["matched_detected_pts"] = np.asarray(manual_landmarks, dtype=np.float32)
        rec["matched_canonical_pts"] = canonical_pts
        rec["num_matched"] = NUM_LANDMARKS
        H = compute_homography_ransac(manual_landmarks, canonical_pts)
        if H is None:
            rec["error"] = "homography solve failed (manual)"
            return rec
        rec["H"] = H
        rec["reprojection_error"] = reprojection_error(
            manual_landmarks, canonical_pts, H
        )
        rec["status"] = "ok"
        rec["confidence"] = classify_confidence(
            rec["reprojection_error"], rec["num_matched"]
        )
        return rec

    # Path B: automatic detection within the calibration ROI.
    roi = calibration["search_roi"]
    binary_roi, contour_roi = closed_binary_in_roi(frame, roi, args.morph_kernel_size)
    if binary_roi is None:
        rec["error"] = "could not crop to ROI"
        return rec
    rec["binary_roi"] = binary_roi
    rec["contour_roi"] = contour_roi

    detected = detect_corners_in_roi(binary_roi, roi)
    rec["detected_corners"] = detected
    if len(detected) < MIN_MATCHED_POINTS:
        rec["error"] = f"only {len(detected)} corners detected in ROI"
        return rec

    calib_idx, det_idx, pair_dists = hungarian_match(detected, calib_pts)
    if len(calib_idx) < MIN_MATCHED_POINTS:
        rec["num_matched"] = int(len(calib_idx))
        rec["error"] = f"only {len(calib_idx)} of {NUM_LANDMARKS} landmarks matched"
        return rec

    src_pts = detected[det_idx]
    dst_pts = canonical_pts[calib_idx]
    H = compute_homography_ransac(src_pts, dst_pts)
    if H is None:
        rec["error"] = "homography solve failed"
        rec["num_matched"] = int(len(calib_idx))
        rec["matched_calib_idx"] = calib_idx
        rec["matched_detected_pts"] = src_pts
        rec["matched_canonical_pts"] = dst_pts
        return rec

    rep_err = reprojection_error(src_pts, dst_pts, H)
    rec.update({
        "status": "ok",
        "matched_calib_idx": calib_idx,
        "matched_detected_pts": src_pts,
        "matched_canonical_pts": dst_pts,
        "num_matched": int(len(calib_idx)),
        "reprojection_error": rep_err,
        "H": H,
        "source": "auto",
        "confidence": classify_confidence(rep_err, int(len(calib_idx))),
    })
    return rec


# ---------------------------------------------------------------------------
# Review image rendering
# ---------------------------------------------------------------------------

def save_review_image(rec, review_path):
    """
    Save a debug PNG showing detected corners (green), all 24 calibration
    points (red), and lines connecting matched pairs. Detected corners
    that didn't match a calibration point are drawn as dim green.
    """
    frame = rec["frame"]
    if frame is None:
        return
    img = frame.copy()

    # Search ROI rectangle.
    calib = rec.get("calibration_landmarks")
    if calib is None:
        return
    # Draw the ROI for context if available on the record.
    # (It's stored on calibration via the caller; included via rec not strictly
    # required since we don't carry it here. Skip if absent.)

    # All calibration points in red.
    for p in calib:
        cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 0, 255), 2)

    # Detected corners in green (dim by default; bright if matched).
    detected = rec.get("detected_corners")
    matched_det_pts = rec.get("matched_detected_pts")
    matched_set = set()
    if detected is not None and matched_det_pts is not None:
        for mp in matched_det_pts:
            # Find the row in detected closest to this matched point.
            d = np.sqrt(((detected - mp) ** 2).sum(axis=1))
            matched_set.add(int(np.argmin(d)))
    if detected is not None:
        for i, p in enumerate(detected):
            if i in matched_set:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
            else:
                cv2.circle(img, (int(p[0]), int(p[1])), 4, (60, 160, 60), 1)

    # Lines between matched pairs.
    calib_idx = rec.get("matched_calib_idx")
    if (calib_idx is not None and matched_det_pts is not None
            and len(calib_idx) == len(matched_det_pts)):
        for ci, det_pt in zip(calib_idx, matched_det_pts):
            cp = calib[int(ci)]
            cv2.line(img,
                     (int(det_pt[0]), int(det_pt[1])),
                     (int(cp[0]), int(cp[1])),
                     (0, 255, 255), 1)

    # Header text.
    src_label = rec.get("source", "auto")
    matched = rec.get("num_matched", 0)
    rep = rec.get("reprojection_error", float("nan"))
    conf = rec.get("confidence", "?")
    err = rec.get("error", "")
    msg = f"{src_label} | matched {matched}/{NUM_LANDMARKS} | reproj {rep:.2f} | {conf}"
    if err:
        msg += f" | {err}"
    cv2.putText(img, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_path), img)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", type=Path,
                   help="Directory of input videos (searched recursively). "
                        "Required for batch mode.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory for cropped videos, calibration file (default), "
                        "per-video manual landmarks, review/, and the summary CSV.")
    p.add_argument("--padding", type=int, default=50,
                   help="Padding (px) added around the bounding box of the canonical landmarks.")
    p.add_argument("--morph_kernel_size", type=int, default=50,
                   help="Square kernel size (px) for MORPH_CLOSE on the ROI threshold "
                        "to fill the mouse-shaped hole.")
    p.add_argument("--exclude_dirs", nargs="*", default=DEFAULT_EXCLUDE_DIRS,
                   help="Directory names to skip when walking input_dir (case-insensitive). "
                        f"Default: {' '.join(DEFAULT_EXCLUDE_DIRS)}.")
    p.add_argument("--include_pattern", type=str, default=DEFAULT_INCLUDE_PATTERN,
                   help=f"Comma-separated fnmatch globs for video filenames "
                        f"(default: \"{DEFAULT_INCLUDE_PATTERN}\").")
    p.add_argument("--calibrate", action="store_true",
                   help="Enter calibration mode: click all 24 maze corners on a sampled "
                        "frame and save them. Exits without processing any videos.")
    p.add_argument("--calibrate_video", type=Path, default=None,
                   help="In --calibrate mode, the specific video to calibrate on. "
                        "Defaults to the first video found under --input_dir.")
    p.add_argument("--calibration_file", type=Path, default=None,
                   help="Path to the calibration JSON. Defaults to "
                        "<output_dir>/calibration.json.")
    p.add_argument("--manual_video", type=Path, default=None,
                   help="Run the 24-click UI on this specific video, save the clicked "
                        "landmarks next to the cropped output as <stem>_landmarks.json, "
                        "and process that one video. Future batch runs will pick up the "
                        "saved landmarks automatically.")
    p.add_argument("--redo_video", type=str, default=None,
                   help="In batch mode, force automatic re-detection for the named video "
                        "(matched against the relative path or filename). Saved manual "
                        "landmarks for that video are ignored.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------

def run_calibration(args, calibration_path):
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

    landmarks = click_landmarks_ui(frame, title=f"Calibration: {video_path.name}")
    if landmarks is None:
        print("Calibration cancelled.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi = compute_search_roi(landmarks, frame.shape[:2])
    save_calibration(calibration_path, video_path, frame.shape[:2], landmarks, roi)

    annotated = frame.copy()
    cv2.rectangle(annotated, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
    for i, p in enumerate(landmarks):
        cv2.circle(annotated, (int(p[0]), int(p[1])), 7, (0, 255, 0), -1)
        cv2.putText(annotated, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    frame_out = args.output_dir / DEFAULT_CALIBRATION_FRAME_FILENAME
    cv2.imwrite(str(frame_out), annotated)

    print(f"Saved calibration: {calibration_path}")
    print(f"Search ROI: {roi}")
    print(f"Saved annotated calibration frame: {frame_out}")
    return 0


# ---------------------------------------------------------------------------
# Manual single-video entry point
# ---------------------------------------------------------------------------

def run_manual_video(args, calibration_path):
    """
    Click 24 landmarks for a specific video, save the per-video JSON,
    then process that one video using those landmarks.
    """
    video_path = args.manual_video
    if not video_path.exists():
        print(f"Error: --manual_video does not exist: {video_path}")
        return 1
    if not calibration_path.exists():
        print(
            f"Error: calibration file not found at {calibration_path}.\n"
            f"Run --calibrate first."
        )
        return 2

    calibration = load_calibration(calibration_path)
    canonical_pts, dst_size = canonical_positions(calibration["landmarks"], args.padding)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open: {video_path}")
        return 1
    frame = sample_middle_frame(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if frame is None:
        print(f"Error: could not sample a frame from: {video_path}")
        return 1

    landmarks = click_landmarks_ui(frame, title=f"Manual: {video_path.name}")
    if landmarks is None:
        print("Manual landmark entry cancelled.")
        return 1

    # Decide where to save the per-video JSON. If we can resolve the video
    # under --input_dir, mirror that relative path under --output_dir;
    # otherwise drop it next to the cropped output by basename.
    if args.input_dir is not None:
        try:
            rel = video_path.relative_to(args.input_dir)
        except ValueError:
            rel = Path(video_path.name)
    else:
        rel = Path(video_path.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_json = manual_landmarks_path(args.output_dir, rel)
    save_video_landmarks(landmarks_json, video_path, frame.shape[:2], landmarks)
    print(f"Saved per-video landmarks: {landmarks_json}")

    # Process this one video with those landmarks.
    H = compute_homography_ransac(landmarks, canonical_pts)
    if H is None:
        print("Error: homography solve failed for the manual landmarks.")
        return 1
    rep_err = reprojection_error(landmarks, canonical_pts, H)

    output_path = args.output_dir / rel.parent / (video_path.stem + "_cropped.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Warping: {video_path.name} -> {output_path}")
    warp_video(video_path, H, dst_size, output_path, fps)
    print(f"Done. reprojection_error={rep_err:.2f} px (24/24 matched).")
    return 0


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def make_failed_row(rel, num_matched=0, error=None):
    return {
        "filename": str(rel).replace("\\", "/"),
        "num_matched_points": int(num_matched),
        "reprojection_error": np.nan,
        "confidence": "failed",
        "output_path": f"error: {error}" if error else "",
    }


def matches_redo_target(rel_path, target):
    """True if target matches either the filename or the relative path."""
    if target is None:
        return False
    target_norm = str(target).replace("\\", "/").strip().strip("/")
    rel_norm = str(rel_path).replace("\\", "/").strip().strip("/")
    return target_norm == rel_norm or target_norm == Path(rel_path).name


def run_batch(args, calibration_path):
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

    calibration = load_calibration(calibration_path)
    canonical_pts, dst_size = canonical_positions(calibration["landmarks"], args.padding)
    print(
        f"Loaded calibration from {calibration_path}; "
        f"canonical output size = {dst_size[0]}x{dst_size[1]} px; "
        f"search ROI = {calibration['search_roi']}."
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
    # Pass 1: per-video detection (auto or manual-landmarks-override).
    # ----------------------------------------------------------------------
    print("Pass 1/2: detecting landmarks per video...")
    detections = []
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(args.input_dir)
        # Look up saved manual landmarks (unless --redo_video targets this one).
        manual_path = manual_landmarks_path(args.output_dir, rel)
        manual_landmarks = None
        if manual_path.exists() and not matches_redo_target(rel, args.redo_video):
            try:
                manual_landmarks = load_video_landmarks(manual_path)
                print(f"  [{i}/{total}] {video_path.name} — using saved manual "
                      f"landmarks: {manual_path.name}")
            except Exception as e:
                print(f"  [{i}/{total}] {video_path.name} — could not read "
                      f"{manual_path.name}: {e}; falling back to auto.")
                manual_landmarks = None

        try:
            d = detect_one(
                video_path, args.input_dir, calibration,
                canonical_pts, dst_size, args,
                manual_landmarks=manual_landmarks,
            )
        except Exception as e:
            d = empty_detection_record(video_path, rel, dst_size)
            d["error"] = f"exception: {e}"
        detections.append(d)

        if d["status"] == "ok":
            print(f"  [{i}/{total}] {video_path.name} — {d['source']}: "
                  f"matched {d['num_matched']}/{NUM_LANDMARKS}, "
                  f"reproj {d['reprojection_error']:.2f} px, {d['confidence']}")
        else:
            print(f"  [{i}/{total}] {video_path.name} — failed: {d['error']}")

    # ----------------------------------------------------------------------
    # Pass 2: write cropped videos and review images.
    # ----------------------------------------------------------------------
    print("Pass 2/2: writing cropped videos...")
    rows = []
    for i, d in enumerate(detections, start=1):
        rel = d["rel"]
        if d["status"] != "ok" or d["H"] is None:
            if d["frame"] is not None:
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                save_review_image(d, review_path)
            rows.append(make_failed_row(rel, d.get("num_matched", 0), d.get("error")))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed")
            continue

        try:
            output_path = args.output_dir / rel.parent / (d["video_path"].stem + "_cropped.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            warp_video(d["video_path"], d["H"], d["dst_size"], output_path, d["fps"])

            if d["confidence"] in ("medium", "low"):
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                save_review_image(d, review_path)

            rows.append({
                "filename": d["filename"],
                "num_matched_points": int(d["num_matched"]),
                "reprojection_error": float(d["reprojection_error"]),
                "confidence": d["confidence"],
                "output_path": str(output_path),
            })
            print(f"  [{i}/{total}] {d['filename']} — confidence: {d['confidence']} "
                  f"(matched {d['num_matched']}/{NUM_LANDMARKS}, "
                  f"reproj {d['reprojection_error']:.2f} px)")
        except Exception as e:
            rows.append(make_failed_row(rel, d.get("num_matched", 0), str(e)))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed ({e})")

    summary = pd.DataFrame(rows, columns=[
        "filename", "num_matched_points", "reprojection_error",
        "confidence", "output_path",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    calibration_path = (
        args.calibration_file
        if args.calibration_file is not None
        else args.output_dir / DEFAULT_CALIBRATION_FILENAME
    )
    if args.calibrate:
        return run_calibration(args, calibration_path)
    if args.manual_video is not None:
        return run_manual_video(args, calibration_path)
    return run_batch(args, calibration_path)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
