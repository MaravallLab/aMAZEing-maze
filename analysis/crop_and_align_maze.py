"""
crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose estimation.

The maze is a white cross/T-shaped binary decision tree (4 arms) on a dark
background, plus an entry corridor at the bottom of the rig, filmed from
below. The camera position only shifts slightly between sessions, so we can
locate each of 24 maze landmarks in a new frame by **template matching**:
take a small patch from a single calibration frame around each landmark,
and search for it in a small window centered on the calibration position
in the new frame.

Workflow:

  Stage 1 — Calibration (--calibrate):
      The user clicks all 24 corners of the maze outline (clockwise from
      the top-left) on the middle frame of one video. The script saves
      the 24 points, the chosen patch size, and the unannotated frame to
      calibration.json + calibration_frame.png. (An annotated copy is
      also saved as calibration_frame_annotated.png for reference.)

  Stage 2 — Batch processing (no --calibrate):
      For every video the script:
        - builds a per-pixel median reference frame across
          --n_median_frames evenly-spaced frames (collapses the moving
          mouse out of the image so the empty maze is visible),
        - for each of the 24 calibration points, extracts the saved
          patch around that point and runs cv2.matchTemplate
          (TM_CCOEFF_NORMED) inside a ±--search_radius search window,
        - keeps the location of the peak correlation as the detected
          landmark, marking it unmatched if the peak is below
          --match_threshold,
        - solves a RANSAC homography (cv2.findHomography) over the
          matched detected points -> canonical positions,
        - iteratively refines: projects unmatched canonical landmarks
          into the frame via the current H and re-searches them in a
          tight ±--refine_radius window with a looser
          --refine_threshold, then re-solves, repeating until nothing
          new is recovered, and
        - warps every frame and writes the cropped/aligned output.

Per-video fallback (--manual_video <path>):
      If automatic template matching fails or looks wrong for a video,
      run the 24-click UI on that video. The clicked points are saved
      next to the cropped output as <stem>_landmarks.json. Future batch
      runs pick those up automatically (use --redo_video <name> to
      force re-running automatic detection).

Confidence is based on three signals: reprojection error of the
homography, the number of matched landmarks, and the mean template
correlation across all 24 points:

  high   : reproj < 10 px AND matched >= 20 AND mean_score > 0.5
  medium : reproj < 25 px AND matched >= 16
  low    : matched >= 12 (anything worse than medium)
  failed : matched <  12, or any earlier step blew up.
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
DEFAULT_CALIBRATION_FRAME_ANNOTATED = "calibration_frame_annotated.png"

NUM_LANDMARKS = 24
MIN_MATCHED_POINTS = 12
DEFAULT_PATCH_SIZE = 80
DEFAULT_SEARCH_RADIUS = 60
DEFAULT_MATCH_THRESHOLD = 0.3
DEFAULT_N_MEDIAN_FRAMES = 11
DEFAULT_REFINE_RADIUS = 35
DEFAULT_REFINE_THRESHOLD = 0.2
DEFAULT_RANSAC_THRESHOLD = 10.0
MAX_REFINE_ITERATIONS = 3

# Confidence thresholds.
HIGH_REPROJ_PX = 10.0
HIGH_MIN_MATCHED = 20
HIGH_MIN_MEAN_SCORE = 0.5
MEDIUM_REPROJ_PX = 25.0
MEDIUM_MIN_MATCHED = 16

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
# Generic I/O helpers
# ---------------------------------------------------------------------------

def sample_middle_frame(cap):
    """Read the frame at 50% of the video's total frame count."""
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ok, frame = cap.read()
    return frame if ok else None


def sample_median_frame(cap, n_samples=DEFAULT_N_MEDIAN_FRAMES):
    """
    Return a per-pixel median across `n_samples` frames evenly spaced
    through the video. The mouse is in a different place in each frame,
    so the median collapses to the empty maze — giving a clean,
    mouse-free image for landmark detection.

    Falls back to the middle frame if only one frame is readable.
    """
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    if n_samples < 1:
        n_samples = 1
    if n_samples == 1 or n_frames < 2:
        return sample_middle_frame(cap)

    n_use = min(n_samples, n_frames)
    indices = np.linspace(0, n_frames - 1, n_use, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    # Stack and take per-pixel median. Frames may differ slightly in
    # shape after decoding on some codecs; trim to the smallest.
    min_h = min(f.shape[0] for f in frames)
    min_w = min(f.shape[1] for f in frames)
    cropped = [f[:min_h, :min_w] for f in frames]
    stack = np.stack(cropped, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def find_videos(input_dir, include_pattern, exclude_dirs):
    """
    Recursively walk `input_dir` and return matching video paths.
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
    # mp4v (MPEG-4 Part 2) is bundled with OpenCV/FFmpeg on Windows;
    # avc1 (H.264) requires the OpenH264 DLL which is often missing.
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


def to_gray(frame):
    """Return frame as 2D uint8 (BGR -> gray, or pass-through if already 2D)."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Reference diagram for the calibration UI
# ---------------------------------------------------------------------------

# def make_reference_diagram(size=REFERENCE_CANVAS_SIZE):
#     """Render a small schematic of the 24-point click order."""
#     canvas = np.full((size, size, 3), 30, dtype=np.uint8)
#     for i in range(NUM_LANDMARKS):
#         p1 = tuple(int(v) for v in REFERENCE_POINTS[i])
#         p2 = tuple(int(v) for v in REFERENCE_POINTS[(i + 1) % NUM_LANDMARKS])
#         cv2.line(canvas, p1, p2, (90, 90, 90), 1)
#     for i, p in enumerate(REFERENCE_POINTS):
#         ix, iy = int(p[0]), int(p[1])
#         cv2.circle(canvas, (ix, iy), 3, (220, 220, 220), -1)
#         cv2.putText(canvas, str(i + 1), (ix + 4, iy - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
#     cv2.putText(canvas, "click order", (8, size - 8),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
#     cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), (160, 160, 160), 1)
#     return canvas


# def overlay_reference(disp, ref):
#     """Blit `ref` into the top-right corner of `disp` (in-place)."""
#     h, w = disp.shape[:2]
#     rh, rw = ref.shape[:2]
#     if rh + 20 > h or rw + 20 > w:
#         return
#     y0, x0 = 10, w - rw - 10
#     disp[y0:y0 + rh, x0:x0 + rw] = ref


# ---------------------------------------------------------------------------
# 24-click UI (used by both --calibrate and --manual_video)
# ---------------------------------------------------------------------------

def click_landmarks_ui(frame, title, num_points=NUM_LANDMARKS, with_reference=False):
    """Display `frame`, collect 24 clicks in click-order, return Nx2 float32."""
    h, w = frame.shape[:2]
    max_dim = 1400
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
    base_disp = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()
    # if with_reference:
    #     overlay_reference(base_disp, make_reference_diagram())

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < num_points:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_w, disp_h)
    cv2.setMouseCallback(title, on_mouse)

    def render(stage_msg):
        disp = base_disp.copy()
        for i in range(len(clicks) - 1):
            p1 = (int(clicks[i][0] * scale), int(clicks[i][1] * scale))
            p2 = (int(clicks[i + 1][0] * scale), int(clicks[i + 1][1] * scale))
            cv2.line(disp, p1, p2, (0, 200, 200), 1)
        if len(clicks) == num_points:
            p1 = (int(clicks[-1][0] * scale), int(clicks[-1][1] * scale))
            p2 = (int(clicks[0][0] * scale), int(clicks[0][1] * scale))
            cv2.line(disp, p1, p2, (0, 200, 200), 1)
        for i, (cx, cy) in enumerate(clicks):
            color = (
                int(60 + (i / max(1, num_points - 1)) * 195),
                int(255 - (i / max(1, num_points - 1)) * 195),
                255,
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

def save_calibration(path, calibration_video, frame_shape, landmarks, patch_size,
                     calibration_frame_filename):
    payload = {
        "calibration_video": str(calibration_video),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "num_landmarks": NUM_LANDMARKS,
        "landmarks": [[float(p[0]), float(p[1])] for p in landmarks],
        "patch_size": int(patch_size),
        "calibration_frame_file": str(calibration_frame_filename),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path):
    """Load calibration JSON and the unannotated calibration frame."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    landmarks = np.asarray(payload["landmarks"], dtype=np.float32)
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Calibration must contain {NUM_LANDMARKS}x2 landmarks; "
            f"got {landmarks.shape}."
        )
    patch_size = int(payload.get("patch_size", DEFAULT_PATCH_SIZE))

    frame_filename = payload.get(
        "calibration_frame_file", DEFAULT_CALIBRATION_FRAME_FILENAME
    )
    frame_path = (path.parent / frame_filename).resolve()
    if not frame_path.exists():
        raise FileNotFoundError(
            f"Calibration frame not found at {frame_path}. Re-run --calibrate."
        )
    calib_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if calib_frame is None:
        raise FileNotFoundError(f"Could not read calibration frame: {frame_path}")
    calib_gray = to_gray(calib_frame)

    return {
        "landmarks": landmarks,
        "patch_size": patch_size,
        "frame_shape": tuple(int(v) for v in payload["frame_shape"]),
        "calibration_frame_path": frame_path,
        "calibration_frame": calib_frame,
        "calibration_frame_gray": calib_gray,
        "raw": payload,
    }


def load_calibrations_from_dir(calibration_dir):
    """
    Load every valid calibration JSON found in `calibration_dir`. A JSON
    is "valid" if it has 24 landmarks and its associated frame file
    exists. Returns a list of (calibration dict, source path) tuples.
    Invalid files are skipped with a warning.
    """
    calibration_dir = Path(calibration_dir)
    results = []
    if not calibration_dir.is_dir():
        return results
    for json_path in sorted(calibration_dir.glob("*.json")):
        try:
            calib = load_calibration(json_path)
        except Exception as e:
            print(f"  Skipping {json_path.name}: {e}")
            continue
        calib["name"] = json_path.stem
        calib["source_path"] = str(json_path)
        results.append(calib)
    return results


def prepare_calibration_set(calibration, padding):
    """Extract patches + compute canonical layout for one calibration."""
    patches = extract_patches(
        calibration["calibration_frame_gray"],
        calibration["landmarks"],
        calibration["patch_size"],
    )
    canonical_pts, dst_size = canonical_positions(
        calibration["landmarks"], padding
    )
    n_valid = sum(1 for p in patches if p is not None)
    return {
        "calibration": calibration,
        "patches": patches,
        "canonical_pts": canonical_pts,
        "dst_size": dst_size,
        "n_valid_patches": n_valid,
    }


_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "failed": 0}


def detection_quality_key(rec):
    """Sort key: better detections compare greater. Used for picking
    the best calibration in multi-calibration mode."""
    status_ok = 1 if rec.get("status") == "ok" else 0
    conf = _CONFIDENCE_RANK.get(rec.get("confidence", "failed"), 0)
    return (status_ok, conf, int(rec.get("num_inliers", 0)),
            float(rec.get("mean_match_score") or 0.0))


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
    (padding, padding). Returns (canonical_pts (24, 2), (out_w, out_h)).
    """
    pts = np.asarray(calibration_landmarks, dtype=np.float32).copy()
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pts[:, 0] -= xmin - padding
    pts[:, 1] -= ymin - padding
    out_w = int(round(xmax - xmin + 2 * padding))
    out_h = int(round(ymax - ymin + 2 * padding))

    if out_w % 2 != 0:
        out_w += 1
    if out_h % 2 != 0:
        out_h += 1
    return pts, (out_w, out_h)


# ---------------------------------------------------------------------------
# Template-matching landmark detection
# ---------------------------------------------------------------------------

def extract_patches(calib_gray, landmarks, patch_size):
    """
    Extract a `patch_size`x`patch_size` patch from `calib_gray` centered
    on each landmark. Returns a list of length 24; entries are uint8
    arrays of shape (patch_size, patch_size) or None if the patch would
    extend outside the calibration frame.
    """
    h, w = calib_gray.shape[:2]
    patch_size = int(patch_size)
    half = patch_size // 2
    patches = []
    for pt in landmarks:
        cx, cy = float(pt[0]), float(pt[1])
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            patches.append(None)
        else:
            patches.append(calib_gray[y0:y1, x0:x1].copy())
    return patches


def match_one_landmark(new_gray, patch, calib_pt, search_radius):
    """
    Run cv2.matchTemplate (TM_CCOEFF_NORMED) on a region around
    `calib_pt`. Returns (detected_pt or None, score).

    The search region is sized so the *center* of the patch can land
    anywhere within ±search_radius of calib_pt. detected_pt is the
    full-frame position of that center.
    """
    if patch is None:
        return None, 0.0
    ph, pw = patch.shape[:2]
    half_w = pw // 2
    half_h = ph // 2
    h, w = new_gray.shape[:2]
    cx, cy = float(calib_pt[0]), float(calib_pt[1])

    sx0 = max(0, int(round(cx - search_radius - half_w)))
    sy0 = max(0, int(round(cy - search_radius - half_h)))
    sx1 = min(w, int(round(cx + search_radius + half_w)))
    sy1 = min(h, int(round(cy + search_radius + half_h)))

    if sx1 - sx0 < pw or sy1 - sy0 < ph:
        return None, 0.0
    region = new_gray[sy0:sy1, sx0:sx1]

    result = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    detected = np.array([
        float(sx0 + max_loc[0] + half_w),
        float(sy0 + max_loc[1] + half_h),
    ], dtype=np.float32)
    return detected, float(max_val)


def estimate_global_translation(calib_gray, new_gray, max_shift_fraction=0.35):
    """
    Estimate the global (dx, dy) translation that maps calibration-frame
    coordinates into new-frame coordinates, via phase correlation. So if
    a landmark sits at (x, y) in the calibration frame, it is expected
    near (x + dx, y + dy) in the new frame.

    Returns (dx, dy, response). If the detected shift exceeds
    `max_shift_fraction` of the smaller image dimension (likely a
    spurious peak), falls back to (0, 0).
    """
    if calib_gray is None or new_gray is None:
        return 0.0, 0.0, 0.0
    h = min(calib_gray.shape[0], new_gray.shape[0])
    w = min(calib_gray.shape[1], new_gray.shape[1])
    if h < 16 or w < 16:
        return 0.0, 0.0, 0.0
    c = calib_gray[:h, :w].astype(np.float32)
    n = new_gray[:h, :w].astype(np.float32)
    try:
        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        c = c * hann
        n = n * hann
        (dx, dy), response = cv2.phaseCorrelate(c, n)
    except cv2.error:
        return 0.0, 0.0, 0.0
    cap = max_shift_fraction * min(w, h)
    if not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0, float(response)
    if abs(dx) > cap or abs(dy) > cap:
        return 0.0, 0.0, float(response)
    return float(dx), float(dy), float(response)


def refine_with_homography(new_gray, patches, canonical_pts, H,
                           detected, scores, matched, inlier_mask,
                           refine_radius, refine_threshold):
    """
    One round of iterative refinement. Uses the current homography H
    (frame -> canonical) to predict where each *non-inlier* canonical
    landmark should sit in the new frame, then re-runs template matching
    in a tight window around that predicted position.

    "Non-inlier" means: not a RANSAC inlier — either truly unmatched OR
    a template match that RANSAC rejected as inconsistent with H (i.e.,
    the template most likely locked onto a wrong nearby corner). Both
    cases get the same treatment: re-search around the predicted
    position; if a peak >= refine_threshold is found, replace the
    landmark's position and mark it matched.

    Returns (detected, scores, matched, n_changed) where n_changed is
    the number of landmarks whose state was updated this round.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return detected, scores, matched, 0

    canonical_h = canonical_pts.reshape(-1, 1, 2).astype(np.float32)
    predicted = cv2.perspectiveTransform(canonical_h, H_inv).reshape(-1, 2)

    n_changed = 0
    for i in range(NUM_LANDMARKS):
        if inlier_mask[i]:
            continue
        if patches[i] is None:
            continue
        pred = predicted[i]
        if not (np.isfinite(pred[0]) and np.isfinite(pred[1])):
            continue
        det_pt, score = match_one_landmark(
            new_gray, patches[i], pred, refine_radius
        )
        if det_pt is None:
            continue
        if score < refine_threshold:
            if score > scores[i]:
                scores[i] = score
            continue
        # Promote / relocate this landmark. Count as changed if the
        # match is new or the position moved.
        was_matched = bool(matched[i])
        prev_pt = detected[i].copy() if was_matched else None
        detected[i] = det_pt
        scores[i] = score
        matched[i] = True
        if (not was_matched) or prev_pt is None or not np.allclose(
            prev_pt, det_pt, atol=0.5
        ):
            n_changed += 1
    return detected, scores, matched, n_changed


def detect_via_template(new_frame, calibration, patches, search_radius, match_threshold,
                        search_centers=None):
    """
    Match all 24 landmarks. Returns:
        detected_pts  (24, 2) float32 — NaN for points where no patch
                                        was extractable
        scores        (24,)   float32 — peak correlation per point (0
                                        if patch missing or region too
                                        small)
        matched_mask  (24,)   bool    — score >= match_threshold AND
                                        patch was extractable

    If `search_centers` is given (Nx2 float), it is used as the center of
    each per-landmark search window instead of the raw calibration
    positions. This is how the optional global pre-alignment step
    steers per-landmark searches near the right place when the camera
    has shifted between calibration and the new video.
    """
    new_gray = to_gray(new_frame)
    detected = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    scores = np.zeros(NUM_LANDMARKS, dtype=np.float32)
    matched = np.zeros(NUM_LANDMARKS, dtype=bool)
    centers = (
        np.asarray(search_centers, dtype=np.float32)
        if search_centers is not None else calibration["landmarks"]
    )
    for i in range(NUM_LANDMARKS):
        det_pt, score = match_one_landmark(
            new_gray, patches[i], centers[i], search_radius
        )
        scores[i] = score
        if det_pt is not None:
            detected[i] = det_pt
            if score >= match_threshold:
                matched[i] = True
    return detected, scores, matched


# ---------------------------------------------------------------------------
# Homography + scoring
# ---------------------------------------------------------------------------

def compute_homography_ransac(src_pts, dst_pts,
                              ransac_threshold=DEFAULT_RANSAC_THRESHOLD,
                              return_mask=False):
    """
    findHomography with RANSAC. By default returns just H for backward
    compatibility. With return_mask=True, returns (H, inlier_mask) where
    inlier_mask is a bool array over the input points (True = RANSAC
    inlier). Returns (None, None) / None on solve failure.
    """
    if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
        return (None, None) if return_mask else None
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                 ransacReprojThreshold=float(ransac_threshold))
    if return_mask:
        if H is None or mask is None:
            return H, None
        return H, mask.ravel().astype(bool)
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


def classify_confidence(reproj_px, num_inliers, mean_match_score, source="auto"):
    """
    Confidence from reprojection error (over RANSAC inliers), inlier
    count, and mean template correlation (over template-matched points).
    Manual-source videos are scored without the mean-score requirement
    (the user has visually verified the points).
    """
    if num_inliers < MIN_MATCHED_POINTS:
        return "failed"
    if reproj_px is None or not np.isfinite(reproj_px):
        return "failed"
    if source == "manual":
        if reproj_px < HIGH_REPROJ_PX and num_inliers >= HIGH_MIN_MATCHED:
            return "high"
        if reproj_px < MEDIUM_REPROJ_PX and num_inliers >= MEDIUM_MIN_MATCHED:
            return "medium"
        return "low"
    if (reproj_px < HIGH_REPROJ_PX
            and num_inliers >= HIGH_MIN_MATCHED
            and mean_match_score > HIGH_MIN_MEAN_SCORE):
        return "high"
    if reproj_px < MEDIUM_REPROJ_PX and num_inliers >= MEDIUM_MIN_MATCHED:
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
        "detected_pts": None,    # (24, 2) NaN-filled for unmatched
        "match_scores": None,    # (24,)
        "matched_mask": None,    # (24,) bool — template-matched points
        "inlier_mask": None,     # (24,) bool — RANSAC inliers (subset of matched)
        "calibration_landmarks": None,
        "H": None,
        "dst_size": dst_size,
        "num_matched": 0,
        "num_inliers": 0,
        "mean_match_score": np.nan,
        "reprojection_error": np.nan,
        "global_shift": (0.0, 0.0),
        "source": "auto",
        "confidence": "failed",
        "error": "",
    }


def detect_one(video_path, input_dir, calibration, patches, canonical_pts, dst_size,
               args, manual_landmarks=None):
    """
    Per-video detection. Either uses pre-saved manual landmarks (skip
    template matching, all 24 matched) or runs template matching against
    the calibration patches.
    """
    rel = video_path.relative_to(input_dir)
    rec = empty_detection_record(video_path, rel, dst_size)
    rec["calibration_landmarks"] = calibration["landmarks"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        rec["error"] = "could not open video"
        return rec
    frame = sample_median_frame(cap, getattr(args, "n_median_frames",
                                             DEFAULT_N_MEDIAN_FRAMES))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 30.0
    cap.release()
    if frame is None:
        rec["error"] = "could not sample frame"
        return rec
    rec["frame"] = frame
    rec["fps"] = fps

    # Path A: per-video manual landmarks override automatic detection.
    if manual_landmarks is not None:
        rec["source"] = "manual"
        rec["detected_pts"] = np.asarray(manual_landmarks, dtype=np.float32)
        rec["match_scores"] = np.full(NUM_LANDMARKS, np.nan, dtype=np.float32)
        rec["matched_mask"] = np.ones(NUM_LANDMARKS, dtype=bool)
        rec["inlier_mask"] = np.ones(NUM_LANDMARKS, dtype=bool)
        rec["num_matched"] = NUM_LANDMARKS
        rec["num_inliers"] = NUM_LANDMARKS
        rec["mean_match_score"] = np.nan
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
            rec["reprojection_error"], rec["num_inliers"],
            rec["mean_match_score"], source="manual",
        )
        return rec

    # Path B: automatic template matching.
    # Step 1 (optional): global pre-alignment. The camera frequently
    # shifts by tens-to-hundreds of pixels between sessions; if we
    # search around the raw calibration positions, individual templates
    # can lock onto similar-looking *neighbouring* corners. A quick
    # phase-correlation gives us the global (dx, dy) that maps
    # calibration coordinates into the new frame; we then anchor each
    # per-landmark template search at the shifted position.
    new_gray = to_gray(frame)
    global_align = getattr(args, "global_align", "phase")
    if global_align == "phase":
        dx, dy, _ = estimate_global_translation(
            calibration["calibration_frame_gray"], new_gray
        )
    else:
        dx, dy = 0.0, 0.0
    rec["global_shift"] = (dx, dy)
    search_centers = calibration["landmarks"] + np.array(
        [dx, dy], dtype=np.float32
    )
    detected, scores, matched = detect_via_template(
        frame, calibration, patches,
        search_radius=args.search_radius,
        match_threshold=args.match_threshold,
        search_centers=search_centers,
    )
    rec["detected_pts"] = detected
    rec["match_scores"] = scores
    rec["matched_mask"] = matched
    rec["num_matched"] = int(matched.sum())
    rec["mean_match_score"] = float(np.mean(scores))

    if rec["num_matched"] < MIN_MATCHED_POINTS:
        rec["error"] = f"only {rec['num_matched']} of {NUM_LANDMARKS} landmarks matched"
        return rec

    ransac_threshold = getattr(args, "ransac_threshold", DEFAULT_RANSAC_THRESHOLD)
    src_pts = detected[matched]
    dst_pts = canonical_pts[matched]
    H, inlier_mask_local = compute_homography_ransac(
        src_pts, dst_pts, ransac_threshold=ransac_threshold, return_mask=True
    )
    if H is None:
        rec["error"] = "homography solve failed"
        return rec
    # Map the local inlier mask (over only the matched subset) back to a
    # global (24,) mask over all landmarks. Note: we do NOT demote
    # outliers from `matched` — they're template-matched but just don't
    # fit the homography tightly. Tracking them separately lets us
    # report num_matched honestly while computing reproj over the
    # RANSAC-consistent subset.
    inlier_mask = np.zeros(NUM_LANDMARKS, dtype=bool)
    matched_idx = np.where(matched)[0]
    if inlier_mask_local is not None:
        for k, gi in enumerate(matched_idx):
            inlier_mask[gi] = bool(inlier_mask_local[k])
    else:
        inlier_mask[matched_idx] = True

    # Iterative refinement: predict positions of every *non-inlier*
    # landmark (RANSAC outliers + truly unmatched) from the current H,
    # re-search in a tight window around the prediction, re-solve.
    # Handles the case where the initial template match locked onto a
    # wrong nearby corner — the homography from the inliers points us
    # at the right spot, which the re-search can then snap to.
    refine_radius = getattr(args, "refine_radius", DEFAULT_REFINE_RADIUS)
    refine_threshold = getattr(args, "refine_threshold", DEFAULT_REFINE_THRESHOLD)
    for _ in range(MAX_REFINE_ITERATIONS):
        if int(inlier_mask.sum()) >= NUM_LANDMARKS:
            break
        detected, scores, matched, n_changed = refine_with_homography(
            new_gray, patches, canonical_pts, H,
            detected, scores, matched, inlier_mask,
            refine_radius, refine_threshold,
        )
        if n_changed == 0:
            break
        src_pts = detected[matched]
        dst_pts = canonical_pts[matched]
        H_new, inlier_mask_local = compute_homography_ransac(
            src_pts, dst_pts, ransac_threshold=ransac_threshold, return_mask=True
        )
        if H_new is None:
            break
        H = H_new
        inlier_mask = np.zeros(NUM_LANDMARKS, dtype=bool)
        matched_idx = np.where(matched)[0]
        if inlier_mask_local is not None:
            for k, gi in enumerate(matched_idx):
                inlier_mask[gi] = bool(inlier_mask_local[k])
        else:
            inlier_mask[matched_idx] = True

    num_matched = int(matched.sum())
    num_inliers = int(inlier_mask.sum())
    rec["detected_pts"] = detected
    rec["match_scores"] = scores
    rec["matched_mask"] = matched
    rec["inlier_mask"] = inlier_mask
    rec["num_matched"] = num_matched
    rec["num_inliers"] = num_inliers
    # Score over template-matched points (what the templates actually
    # locked onto, including outliers).
    rec["mean_match_score"] = (
        float(np.mean(scores[matched])) if num_matched > 0 else 0.0
    )
    # Reproj over the RANSAC-inlier subset (the points that actually
    # constrain H, so this measures the fit's true quality).
    if num_inliers >= 4:
        src_in = detected[inlier_mask]
        dst_in = canonical_pts[inlier_mask]
        rep_err = reprojection_error(src_in, dst_in, H)
    else:
        rep_err = reprojection_error(
            detected[matched], canonical_pts[matched], H
        )

    if num_inliers < MIN_MATCHED_POINTS:
        rec["error"] = (
            f"only {num_inliers} of {NUM_LANDMARKS} RANSAC inliers "
            f"(out of {num_matched} template matches)"
        )
        return rec

    rec.update({
        "status": "ok",
        "H": H,
        "reprojection_error": rep_err,
        "confidence": classify_confidence(
            rep_err, num_inliers, rec["mean_match_score"], source="auto"
        ),
    })
    return rec


# ---------------------------------------------------------------------------
# Review image rendering
# ---------------------------------------------------------------------------

def save_review_image(rec, review_path):
    """
    Save a debug PNG showing template-matching results:
    - Green filled circles + numbers at *detected* positions for matched landmarks.
    - Red hollow circles at *calibration* positions for unmatched landmarks.
    - Yellow lines from each calibration position to its detected position
      (the per-landmark shift) for matched points.
    """
    frame = rec["frame"]
    if frame is None:
        return
    img = frame.copy()

    calib = rec.get("calibration_landmarks")
    detected = rec.get("detected_pts")
    matched = rec.get("matched_mask")
    scores = rec.get("match_scores")
    if calib is None:
        return

    if matched is not None and detected is not None:
        for i in range(NUM_LANDMARKS):
            cp = calib[i]
            cpx, cpy = int(cp[0]), int(cp[1])
            if not bool(matched[i]):
                # Unmatched: red hollow circle at the expected (calibration) spot.
                cv2.circle(img, (cpx, cpy), 8, (0, 0, 255), 2)
                cv2.putText(img, str(i + 1), (cpx + 9, cpy - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),
                            1, cv2.LINE_AA)
                continue
            dp = detected[i]
            dpx, dpy = int(dp[0]), int(dp[1])
            # Line from expected -> detected (showing the per-point shift).
            cv2.line(img, (cpx, cpy), (dpx, dpy), (0, 255, 255), 1)
            # Detected position: green filled with the landmark number.
            cv2.circle(img, (dpx, dpy), 6, (0, 255, 0), -1)
            cv2.putText(img, str(i + 1), (dpx + 8, dpy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2, cv2.LINE_AA)
    else:
        # No detection at all — just show the calibration positions in red.
        for i, p in enumerate(calib):
            cv2.circle(img, (int(p[0]), int(p[1])), 8, (0, 0, 255), 2)

    src_label = rec.get("source", "auto")
    matched_n = rec.get("num_matched", 0)
    rep = rec.get("reprojection_error", float("nan"))
    mean_s = rec.get("mean_match_score", float("nan"))
    conf = rec.get("confidence", "?")
    err = rec.get("error", "")
    msg = (f"{src_label} | matched {matched_n}/{NUM_LANDMARKS} | "
           f"mean_score {mean_s:.2f} | reproj {rep:.2f} | {conf}")
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
                   help="Directory for cropped videos, calibration files, "
                        "per-video manual landmarks, review/, and the summary CSV.")
    p.add_argument("--padding", type=int, default=50,
                   help="Padding (px) added around the bounding box of the canonical landmarks.")
    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
                   help="Square template patch size in pixels (default %(default)s). "
                        "Used during --calibrate; the saved value is read back at batch time.")
    p.add_argument("--search_radius", type=int, default=DEFAULT_SEARCH_RADIUS,
                   help="How far (px) from each calibration landmark to search for "
                        "the template in a new frame. Default %(default)s.")
    p.add_argument("--match_threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                   help="Minimum normalized cross-correlation score for a template "
                        "match to count as 'matched'. Default %(default)s.")
    p.add_argument("--n_median_frames", type=int, default=DEFAULT_N_MEDIAN_FRAMES,
                   help="Number of frames evenly sampled through each video for the "
                        "per-pixel median reference frame. Removes the moving mouse so "
                        "templates can match the empty maze. Use 1 to revert to a single "
                        "middle frame. Default %(default)s.")
    p.add_argument("--refine_radius", type=int, default=DEFAULT_REFINE_RADIUS,
                   help="After the first homography solve, unmatched landmarks are "
                        "re-searched in a window of ±this radius (px) around the "
                        "homography-predicted position. Default %(default)s.")
    p.add_argument("--refine_threshold", type=float, default=DEFAULT_REFINE_THRESHOLD,
                   help="Minimum correlation score during refinement re-search. Looser "
                        "than --match_threshold because the search window is tight, so "
                        "false positives are unlikely. Default %(default)s.")
    p.add_argument("--ransac_threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD,
                   help="RANSAC reprojection threshold (px) for cv2.findHomography. "
                        "A template-matched point is treated as a homography inlier "
                        "if its reprojection error is below this. Larger values keep "
                        "more landmarks as inliers at the cost of looser fit; smaller "
                        "values give a tighter fit at the cost of fewer inliers. "
                        "Default %(default)s.")
    p.add_argument("--global_align", choices=["phase", "off"], default="phase",
                   help="Global pre-alignment between the calibration frame and each "
                        "video's median frame before per-landmark template matching. "
                        "'phase' uses phase correlation to estimate a (dx, dy) "
                        "translation, so per-landmark searches are anchored at the "
                        "shifted positions instead of the raw calibration positions "
                        "(critical when the camera moved between sessions). 'off' "
                        "uses raw calibration positions. Default %(default)s.")
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
                        "<output_dir>/calibration.json. Ignored when "
                        "--calibration_dir is set.")
    p.add_argument("--calibration_dir", type=Path, default=None,
                   help="Directory containing one or more calibration JSON files "
                        "(plus their associated *_frame.png files). When set, the "
                        "batch tries every calibration per video and picks the best "
                        "fit (highest confidence tier, then most RANSAC inliers). "
                        "Use this when the dataset spans multiple rig/camera setups: "
                        "calibrate once per setup with "
                        "`--calibrate --calibration_file <dir>/calibration_<name>.json` "
                        "and drop them all in one directory.")
    p.add_argument("--manual_video", type=Path, default=None,
                   help="Run the 24-click UI on this specific video, save the clicked "
                        "landmarks next to the cropped output as <stem>_landmarks.json, "
                        "and process that one video. Future batch runs pick up the saved "
                        "landmarks automatically.")
    p.add_argument("--redo_video", type=str, default=None,
                   help="In batch mode, force template-matching re-detection for the "
                        "named video (matched against the bare filename or the relative "
                        "path). Saved manual landmarks for that video are ignored.")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Skip videos whose <stem>_cropped.mp4 already exists in the "
                        "output tree (default on). Use --no-skip_existing to force "
                        "re-detection and re-warping for all videos. --redo_video "
                        "always re-processes its target regardless of this flag.")
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
    frame = sample_median_frame(cap, args.n_median_frames)
    cap.release()
    if frame is None:
        print(f"Error: could not sample a frame from: {video_path}")
        return 1

    landmarks = click_landmarks_ui(frame, title=f"Calibration: {video_path.name}")
    if landmarks is None:
        print("Calibration cancelled.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Derive frame filenames from the JSON stem so that multiple
    # calibrations in the same output dir don't overwrite each other's
    # frames. The default "calibration.json" maps to "calibration_frame.png",
    # which is backward-compatible.
    stem = calibration_path.stem
    if stem == "calibration":
        frame_filename = DEFAULT_CALIBRATION_FRAME_FILENAME
        annotated_filename = DEFAULT_CALIBRATION_FRAME_ANNOTATED
    else:
        frame_filename = f"{stem}_frame.png"
        annotated_filename = f"{stem}_frame_annotated.png"

    # Save the unannotated frame (this is what batch matching reads).
    clean_path = args.output_dir / frame_filename
    cv2.imwrite(str(clean_path), frame)

    # Save the calibration JSON.
    save_calibration(
        calibration_path, video_path, frame.shape[:2],
        landmarks, args.patch_size, frame_filename,
    )

    # Save an annotated copy with patch boxes + numbers, for human reference.
    annotated = frame.copy()
    half = int(args.patch_size) // 2
    for i, p in enumerate(landmarks):
        x, y = int(p[0]), int(p[1])
        cv2.rectangle(annotated, (x - half, y - half),
                      (x + half, y + half), (255, 255, 0), 1)
        cv2.circle(annotated, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(annotated, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),
                    2, cv2.LINE_AA)
    annotated_path = args.output_dir / annotated_filename
    cv2.imwrite(str(annotated_path), annotated)

    # Warn about any landmark whose patch falls outside the frame.
    h, w = frame.shape[:2]
    edge_warnings = []
    for i, p in enumerate(landmarks):
        cx, cy = float(p[0]), float(p[1])
        if (cx - half < 0 or cy - half < 0 or
                cx + half > w or cy + half > h):
            edge_warnings.append(i + 1)
    if edge_warnings:
        print(
            f"Warning: patch_size={args.patch_size} is too large for landmarks "
            f"{edge_warnings} (patch falls outside the frame). Those landmarks "
            f"will be unmatched in batch mode unless you re-calibrate with a "
            f"smaller --patch_size."
        )

    print(f"Saved calibration: {calibration_path}")
    print(f"Saved clean calibration frame: {clean_path}")
    print(f"Saved annotated calibration frame: {annotated_path}")
    print(f"Patch size: {args.patch_size} px")
    return 0


# ---------------------------------------------------------------------------
# Manual single-video entry point
# ---------------------------------------------------------------------------

def run_manual_video(args, calibration_path):
    """Click 24 landmarks for a specific video and process that one video."""
    video_path = args.manual_video
    if not video_path.exists():
        print(f"Error: --manual_video does not exist: {video_path}")
        return 1
    if not calibration_path.exists():
        print(f"Error: calibration file not found at {calibration_path}.\n"
              f"Run --calibrate first.")
        return 2

    calibration = load_calibration(calibration_path)
    canonical_pts, dst_size = canonical_positions(calibration["landmarks"], args.padding)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open: {video_path}")
        return 1
    frame = sample_median_frame(cap, args.n_median_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if frame is None:
        print(f"Error: could not sample a frame from: {video_path}")
        return 1

    landmarks = click_landmarks_ui(frame, title=f"Manual: {video_path.name}")
    if landmarks is None:
        print("Manual landmark entry cancelled.")
        return 1

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

def make_failed_row(rel, num_matched=0, num_inliers=0, mean_score=np.nan,
                    error=None, calibration_used=""):
    return {
        "filename": str(rel).replace("\\", "/"),
        "calibration_used": str(calibration_used),
        "num_matched_points": int(num_matched),
        "num_inlier_points": int(num_inliers),
        "mean_match_score": float(mean_score) if np.isfinite(mean_score) else np.nan,
        "reprojection_error": np.nan,
        "confidence": "failed",
        "output_path": f"error: {error}" if error else "",
    }


def matches_redo_target(rel_path, target):
    if target is None:
        return False
    target_norm = str(target).replace("\\", "/").strip().strip("/")
    rel_norm = str(rel_path).replace("\\", "/").strip().strip("/")
    return target_norm == rel_norm or target_norm == Path(rel_path).name


def run_batch(args, calibration_path):
    if args.input_dir is None:
        print("Error: --input_dir is required for batch processing.")
        return 1

    # Build the list of calibration sets to try per video. With
    # --calibration_dir we try every calibration found there and pick
    # the best fit per video. Otherwise we fall back to the single
    # --calibration_file (default <output_dir>/calibration.json).
    calibration_sets = []
    if args.calibration_dir is not None:
        calibrations = load_calibrations_from_dir(args.calibration_dir)
        if not calibrations:
            print(f"Error: no valid calibrations found in {args.calibration_dir}.")
            return 2
        for c in calibrations:
            cset = prepare_calibration_set(c, args.padding)
            if cset["n_valid_patches"] < MIN_MATCHED_POINTS:
                print(
                    f"  Skipping {c['name']}: only {cset['n_valid_patches']}/"
                    f"{NUM_LANDMARKS} patches usable."
                )
                continue
            calibration_sets.append(cset)
        if not calibration_sets:
            print("Error: no usable calibrations after patch validation.")
            return 1
    else:
        if not calibration_path.exists():
            print(
                f"Error: calibration file not found at {calibration_path}.\n"
                f"Run calibration first, e.g.:\n"
                f"    python crop_and_align_maze.py --calibrate "
                f"--input_dir <path> --output_dir <path>"
            )
            return 2
        calibration = load_calibration(calibration_path)
        calibration["name"] = calibration_path.stem
        calibration["source_path"] = str(calibration_path)
        cset = prepare_calibration_set(calibration, args.padding)
        if cset["n_valid_patches"] < MIN_MATCHED_POINTS:
            print(
                f"Error: only {cset['n_valid_patches']} of {NUM_LANDMARKS} "
                f"calibration patches could be extracted (the rest fall outside "
                f"the calibration frame). Re-run --calibrate with a smaller "
                f"--patch_size."
            )
            return 1
        calibration_sets.append(cset)

    print(f"Loaded {len(calibration_sets)} calibration(s):")
    for cset in calibration_sets:
        c = cset["calibration"]
        print(
            f"  - {c['name']}: canonical {cset['dst_size'][0]}x{cset['dst_size'][1]} px, "
            f"patch_size={c['patch_size']}, "
            f"valid patches {cset['n_valid_patches']}/{NUM_LANDMARKS}"
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
        f"Pattern: {args.include_pattern}.  "
        f"search_radius={args.search_radius}, match_threshold={args.match_threshold}."
    )

    # ----------------------------------------------------------------------
    # Pass 1: per-video detection (auto template-match, or manual override).
    # ----------------------------------------------------------------------
    print("Pass 1/2: locating landmarks per video...")
    detections = []
    skipped_rows = []
    n_skipped = 0
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(args.input_dir)
        force_redo = matches_redo_target(rel, args.redo_video)

        # Skip-if-already-cropped: avoids redoing the slow median-sample +
        # warp passes for videos that already produced an output file.
        if args.skip_existing and not force_redo:
            existing_output = (
                args.output_dir / rel.parent / (video_path.stem + "_cropped.mp4")
            )
            if existing_output.exists():
                n_skipped += 1
                skipped_rows.append({
                    "filename": str(rel).replace("\\", "/"),
                    "calibration_used": "",
                    "num_matched_points": 0,
                    "num_inlier_points": 0,
                    "mean_match_score": np.nan,
                    "reprojection_error": np.nan,
                    "confidence": "skipped",
                    "output_path": str(existing_output),
                })
                print(f"  [{i}/{total}] {video_path.name} — skipped "
                      f"(already cropped)")
                continue

        manual_path = manual_landmarks_path(args.output_dir, rel)
        manual_landmarks = None
        if manual_path.exists() and not force_redo:
            try:
                manual_landmarks = load_video_landmarks(manual_path)
                print(f"  [{i}/{total}] {video_path.name} — using saved manual "
                      f"landmarks: {manual_path.name}")
            except Exception as e:
                print(f"  [{i}/{total}] {video_path.name} — could not read "
                      f"{manual_path.name}: {e}; falling back to auto.")
                manual_landmarks = None

        d = None
        for cset in calibration_sets:
            try:
                cand = detect_one(
                    video_path, args.input_dir,
                    cset["calibration"], cset["patches"],
                    cset["canonical_pts"], cset["dst_size"], args,
                    manual_landmarks=manual_landmarks,
                )
            except Exception as e:
                cand = empty_detection_record(video_path, rel, cset["dst_size"])
                cand["error"] = f"exception: {e}"
            cand["calibration_used"] = cset["calibration"].get("name", "?")
            # Manual landmarks are calibration-independent — first try wins.
            if manual_landmarks is not None:
                d = cand
                break
            if d is None or detection_quality_key(cand) > detection_quality_key(d):
                d = cand
        if d is None:
            # Defensive: no calibration sets (should be unreachable).
            d = empty_detection_record(video_path, rel, (0, 0))
            d["error"] = "no calibrations available"
        detections.append(d)

        shift = d.get("global_shift", (0.0, 0.0))
        shift_str = (
            f", shift ({shift[0]:+.0f},{shift[1]:+.0f})"
            if (abs(shift[0]) >= 1.0 or abs(shift[1]) >= 1.0) else ""
        )
        calib_str = (
            f", calib={d.get('calibration_used', '?')}"
            if len(calibration_sets) > 1 else ""
        )
        if d["status"] == "ok":
            mean_s = d["mean_match_score"]
            mean_str = "n/a" if not np.isfinite(mean_s) else f"{mean_s:.2f}"
            print(f"  [{i}/{total}] {video_path.name} — {d['source']}: "
                  f"matched {d['num_matched']}/{NUM_LANDMARKS} "
                  f"(inliers {d['num_inliers']}){shift_str}{calib_str}, "
                  f"mean_score {mean_str}, "
                  f"reproj {d['reprojection_error']:.2f} px, {d['confidence']}")
        else:
            print(f"  [{i}/{total}] {video_path.name} — failed{shift_str}{calib_str}: "
                  f"{d['error']}")

    # ----------------------------------------------------------------------
    # Pass 2: write cropped videos and review images.
    # ----------------------------------------------------------------------
    print("Pass 2/2: writing cropped videos...")
    rows = []
    for i, d in enumerate(detections, start=1):
        rel = d["rel"]
        calib_used = d.get("calibration_used", "")
        if d["status"] != "ok" or d["H"] is None:
            if d["frame"] is not None:
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                save_review_image(d, review_path)
            rows.append(make_failed_row(
                rel, d.get("num_matched", 0), d.get("num_inliers", 0),
                d.get("mean_match_score", np.nan), d.get("error"),
                calibration_used=calib_used,
            ))
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
                "calibration_used": calib_used,
                "num_matched_points": int(d["num_matched"]),
                "num_inlier_points": int(d["num_inliers"]),
                "mean_match_score": (
                    float(d["mean_match_score"])
                    if np.isfinite(d["mean_match_score"]) else np.nan
                ),
                "reprojection_error": float(d["reprojection_error"]),
                "confidence": d["confidence"],
                "output_path": str(output_path),
            })
            mean_s = d["mean_match_score"]
            mean_str = "n/a" if not np.isfinite(mean_s) else f"{mean_s:.2f}"
            print(f"  [{i}/{total}] {d['filename']} — confidence: {d['confidence']} "
                  f"(matched {d['num_matched']}/{NUM_LANDMARKS}, "
                  f"inliers {d['num_inliers']}, "
                  f"mean_score {mean_str}, "
                  f"reproj {d['reprojection_error']:.2f} px)")
        except Exception as e:
            rows.append(make_failed_row(
                rel, d.get("num_matched", 0), d.get("num_inliers", 0),
                d.get("mean_match_score", np.nan), str(e),
                calibration_used=calib_used,
            ))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed ({e})")

    # Re-attach the skipped rows so the summary CSV still reflects the
    # full set of videos discovered under --input_dir.
    summary = pd.DataFrame(skipped_rows + rows, columns=[
        "filename", "calibration_used", "num_matched_points", "num_inlier_points",
        "mean_match_score", "reprojection_error", "confidence", "output_path",
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
        f"failed={counts.get('failed', 0)}, "
        f"skipped={counts.get('skipped', 0)} "
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
