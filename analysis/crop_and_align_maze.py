"""
crop_and_align_maze.py

Preprocess ventral video recordings of a mouse maze for SLEAP pose estimation.

The maze is a white cross/T-shaped binary decision tree (4 arms) on a dark
background, filmed from below. This script detects the maze in a sampled
frame from each video, computes a homography to a canonical axis-aligned
rectangle, and warps every frame so the maze appears identically positioned
across recordings (and the mouse appears larger).

Two-pass design:
  Pass 1 — detect the maze in every video and record the warped binary mask
           plus contour-level sanity checks. No output videos written yet.
  Build  — average all successful warped masks and threshold at 0.5 to make
           a consensus template that doesn't depend on any single recording.
  Score  — compare each warped mask to the consensus and assign a
           high/medium/low/failed confidence label.
  Manual — (optional, --manual) for any flagged or failed video, pop up the
           sampled frame and let the user click the 4 maze corners to
           override the automatic transform.
  Pass 2 — warp every frame and write the cropped/aligned output video.

Usage:
    python crop_and_align_maze.py \
        --input_dir <path> \
        --output_dir <path> \
        [--padding 50] \
        [--iou_threshold 0.85] \
        [--rotation_threshold 15] \
        [--manual]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = (".mp4", ".avi")


def sample_middle_frame(cap):
    """Read the frame at 50% of the video's total frame count."""
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return None
    middle = n_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, frame = cap.read()
    return frame if ok else None


def detect_maze_contour(frame):
    """Binarize the frame and return the largest contour (the bright maze)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Otsu picks a sensible split between bright maze and dark background.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, binary
    largest = max(contours, key=cv2.contourArea)
    return largest, binary


def order_corners(pts):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]   # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]   # bottom-left: largest y-x
    return ordered


def canonical_dst(canonical_size, padding):
    """Return the canonical destination rectangle (4x2) and the output size."""
    cw, ch = canonical_size
    out_w = cw + 2 * padding
    out_h = ch + 2 * padding
    dst = np.array([
        [padding,         padding],
        [padding + cw,    padding],
        [padding + cw,    padding + ch],
        [padding,         padding + ch],
    ], dtype=np.float32)
    return dst, (out_w, out_h)


def compute_transform(contour, canonical_size, padding):
    """
    Compute a perspective transform mapping the contour's rotated bounding
    rectangle to a canonical axis-aligned rectangle with padding.

    Returns (H, dst_size, rotation_angle).
    """
    rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    src = order_corners(box)

    dst, dst_size = canonical_dst(canonical_size, padding)
    H = cv2.getPerspectiveTransform(src, dst)

    # Normalize OpenCV's minAreaRect angle to a signed deviation from
    # axis-aligned in [-45, 45]; a perfectly aligned maze yields ~0.
    angle = rect[2]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    return H, dst_size, angle


def transform_from_clicks(corners, canonical_size, padding):
    """Build a transform from user-clicked corners (already ordered TL,TR,BR,BL)."""
    dst, dst_size = canonical_dst(canonical_size, padding)
    src = np.asarray(corners, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H, dst_size


def warped_mask_from_corners(corners, frame_shape, H, dst_size):
    """Render a filled quad mask of the corners and warp it to canonical coords."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 255)
    return cv2.warpPerspective(mask, H, dst_size)


def compute_iou(mask_a, mask_b):
    """Intersection-over-union for two binary uint8 masks."""
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def classify_confidence(flags):
    """Combine boolean flag list into 'high' / 'medium' / 'low'."""
    n = sum(flags)
    if n == 0:
        return "high"
    if n == 1:
        return "medium"
    return "low"


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


def save_review_image(frame, contour, review_path):
    """Save the sampled frame with the detected contour overlaid."""
    annotated = frame.copy()
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 3)
    cv2.imwrite(str(review_path), annotated)


def manual_correct_corners(frame, title="Click maze corners"):
    """
    Interactively collect 4 maze corner clicks from the user.

    Order required: top-left, top-right, bottom-right, bottom-left
    (clockwise from TL). Keys: 'r' resets, ESC skips, ENTER confirms.

    Returns 4x2 float32 array of corners in original-image pixel
    coordinates, or None if the user skipped (ESC) or closed the window.

    Large frames are downscaled for display; clicks are scaled back to the
    original image so the resulting transform is in source-pixel space.
    """
    h, w = frame.shape[:2]
    max_dim = 1200
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))

    state = {"clicks": [], "skip": False, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["clicks"]) < 4:
            # Convert click back to original-image coordinates.
            state["clicks"].append((x / scale, y / scale))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_w, disp_h)
    cv2.setMouseCallback(title, on_mouse)

    labels = ["TL (1)", "TR (2)", "BR (3)", "BL (4)"]
    base_disp = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()

    while True:
        disp = base_disp.copy()
        # Draw existing clicks and connecting lines for visual feedback.
        for i, (cx, cy) in enumerate(state["clicks"]):
            px, py = int(cx * scale), int(cy * scale)
            cv2.circle(disp, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(disp, labels[i], (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(state["clicks"]) >= 2:
            for i in range(len(state["clicks"]) - 1):
                p1 = (int(state["clicks"][i][0] * scale), int(state["clicks"][i][1] * scale))
                p2 = (int(state["clicks"][i + 1][0] * scale), int(state["clicks"][i + 1][1] * scale))
                cv2.line(disp, p1, p2, (0, 255, 0), 2)
            if len(state["clicks"]) == 4:
                p1 = (int(state["clicks"][3][0] * scale), int(state["clicks"][3][1] * scale))
                p2 = (int(state["clicks"][0][0] * scale), int(state["clicks"][0][1] * scale))
                cv2.line(disp, p1, p2, (0, 255, 0), 2)

        n = len(state["clicks"])
        if n < 4:
            msg = f"Click corner {n + 1}/4 ({labels[n]}). 'r' reset, ESC skip."
        else:
            msg = "ENTER confirm, 'r' reset, ESC skip."
        cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, title, (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(title, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                         # ESC -> skip this video
            state["skip"] = True
            break
        if key == ord('r'):                   # reset clicks
            state["clicks"] = []
        if key in (13, 10) and len(state["clicks"]) == 4:  # ENTER -> confirm
            state["done"] = True
            break

    cv2.destroyWindow(title)
    cv2.waitKey(1)  # let the window actually close on some platforms
    if state["skip"] or not state["done"]:
        return None
    return np.asarray(state["clicks"], dtype=np.float32)


def detect_one(video_path, input_dir, state):
    """
    Pass 1 worker. Detect the maze in `video_path` and return a dict with
    everything needed by pass 2 (transform, warped mask, sampled frame,
    contour-level metrics, contour-level flags). Returns a dict whose
    `status` is either "ok" or "failed". Even on `failed`, the sampled
    frame and FPS are kept when available so manual mode can use them.

    The first successful detection populates `state["canonical_size"]` and
    `state["reference_aspect_ratio"]`, which are reused for all subsequent
    videos so every warped mask lives in the same coordinate system.
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
        "H": None,
        "dst_size": None,
        "warped_mask": None,
        "contour_area": np.nan,
        "aspect_ratio": np.nan,
        "rotation_angle": np.nan,
        "area_flag": False,
        "aspect_flag": False,
        "rotation_flag": False,
        "manual": False,
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return base

    frame = sample_middle_frame(cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if frame is None:
        return base
    base["frame"] = frame
    base["fps"] = fps

    frame_h, frame_w = frame.shape[:2]
    contour, _ = detect_maze_contour(frame)
    if contour is None:
        return base

    contour_area = float(cv2.contourArea(contour))
    area_ratio = contour_area / float(frame_h * frame_w)
    if area_ratio <= 0:
        return base

    rect = cv2.minAreaRect(contour)
    w_rect, h_rect = rect[1]
    if w_rect <= 0 or h_rect <= 0:
        return base

    longer, shorter = max(w_rect, h_rect), min(w_rect, h_rect)
    aspect_ratio = longer / shorter

    # First successful video sets the canonical rectangle size and the
    # reference aspect ratio for the aspect_flag check.
    if state.get("canonical_size") is None:
        state["canonical_size"] = (int(round(longer)), int(round(shorter)))
        state["reference_aspect_ratio"] = aspect_ratio

    H, dst_size, rotation_angle = compute_transform(
        contour, state["canonical_size"], state["padding"]
    )

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    warped_mask = cv2.warpPerspective(mask, H, dst_size)

    ref_ar = state["reference_aspect_ratio"]
    base.update({
        "status": "ok",
        "contour": contour,
        "H": H,
        "dst_size": dst_size,
        "warped_mask": warped_mask,
        "contour_area": contour_area,
        "aspect_ratio": aspect_ratio,
        "rotation_angle": rotation_angle,
        # Contour-level sanity flags computed in pass 1.
        "area_flag": not (0.05 <= area_ratio <= 0.5),
        "aspect_flag": abs(aspect_ratio - ref_ar) / ref_ar > 0.20,
        "rotation_flag": abs(rotation_angle) > state["rotation_threshold"],
    })
    return base


def build_consensus_template(detections, dst_size):
    """
    Average all successful warped masks pixel-wise and threshold at 0.5
    to produce a single consensus template mask (uint8, 0/255).
    """
    out_w, out_h = dst_size
    accumulator = np.zeros((out_h, out_w), dtype=np.float32)
    count = 0
    for d in detections:
        if d["status"] != "ok" or d["warped_mask"] is None:
            continue
        accumulator += (d["warped_mask"] > 0).astype(np.float32)
        count += 1
    if count == 0:
        return None
    averaged = accumulator / float(count)
    return (averaged >= 0.5).astype(np.uint8) * 255


def score_detection(d, consensus, args):
    """
    Compute IoU vs the consensus template and assign a confidence label.
    Updates `d` in place with `mask_iou`, `iou_flag`, and `confidence`.
    No-op for failed detections.
    """
    if d["status"] != "ok":
        d["mask_iou"] = np.nan
        d["iou_flag"] = False
        d["confidence"] = "failed"
        return
    iou = compute_iou(d["warped_mask"], consensus)
    d["mask_iou"] = iou
    d["iou_flag"] = iou < args.iou_threshold
    d["confidence"] = classify_confidence([
        d["area_flag"], d["aspect_flag"], d["iou_flag"], d["rotation_flag"],
    ])


def run_manual_pass(detections, state, consensus, args):
    """
    For each medium / low / failed detection, prompt the user to click 4
    maze corners on the sampled frame. On confirmation: rebuild H,
    warped_mask, mask_iou and mark the detection's confidence as 'manual'.
    On skip: leave the detection unchanged. If no automatic detection ever
    succeeded, the first manual rectangle defines the canonical size.
    """
    candidates = [d for d in detections
                  if d["confidence"] in ("medium", "low", "failed")
                  and d["frame"] is not None]
    if not candidates:
        print("Manual pass: no flagged or failed videos with sampled frames.")
        return consensus

    print(f"Manual pass: {len(candidates)} video(s) need review. "
          f"Click 4 corners (TL, TR, BR, BL). 'r' reset, ESC skip, ENTER confirm.")

    updated_any = False
    for i, d in enumerate(candidates, start=1):
        title = f"[{i}/{len(candidates)}] {d['filename']} (auto: {d['confidence']})"
        corners = manual_correct_corners(d["frame"], title=title)
        if corners is None:
            print(f"  skipped: {d['filename']}")
            continue

        # If we never had an automatic canonical size, infer from this rect.
        if state.get("canonical_size") is None:
            side1 = np.linalg.norm(corners[0] - corners[1])
            side2 = np.linalg.norm(corners[1] - corners[2])
            longer, shorter = max(side1, side2), min(side1, side2)
            state["canonical_size"] = (int(round(longer)), int(round(shorter)))
            state["reference_aspect_ratio"] = float(longer / shorter)

        H, dst_size = transform_from_clicks(corners, state["canonical_size"], args.padding)
        warped_mask = warped_mask_from_corners(corners, d["frame"].shape, H, dst_size)

        d["H"] = H
        d["dst_size"] = dst_size
        d["warped_mask"] = warped_mask
        d["status"] = "ok"
        d["manual"] = True
        # Replace the automatic contour with the user's quad for the review image.
        d["contour"] = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        # IoU vs the existing consensus is informational; confidence is forced
        # to "manual" because the user has visually verified the corners.
        if consensus is not None and consensus.shape[:2] == warped_mask.shape[:2]:
            d["mask_iou"] = compute_iou(warped_mask, consensus)
        d["confidence"] = "manual"
        updated_any = True
        print(f"  corrected: {d['filename']}")

    # If we got our first canonical size from manual mode, build a consensus now
    # so subsequent runs (and the saved PNG) reflect those corrections.
    if updated_any and consensus is None:
        any_size = next((d["dst_size"] for d in detections
                         if d["status"] == "ok" and d["dst_size"] is not None), None)
        if any_size is not None:
            consensus = build_consensus_template(detections, any_size)
    return consensus


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, type=Path,
                   help="Directory containing .mp4/.avi videos (searched recursively).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory to write cropped videos and the summary CSV.")
    p.add_argument("--padding", type=int, default=50,
                   help="Pixels of padding around the canonical maze rectangle.")
    p.add_argument("--iou_threshold", type=float, default=0.85,
                   help="Mask IoU below this value is flagged for review.")
    p.add_argument("--rotation_threshold", type=float, default=15.0,
                   help="abs(rotation angle) above this (deg) is flagged.")
    p.add_argument("--manual", action="store_true",
                   help="After automatic scoring, interactively click 4 corners "
                        "for any medium/low/failed video to override the transform.")
    return p.parse_args()


def make_failed_row(rel, error=None):
    return {
        "filename": str(rel).replace("\\", "/"),
        "contour_area": np.nan,
        "aspect_ratio": np.nan,
        "rotation_angle": np.nan,
        "mask_iou": np.nan,
        "confidence": "failed",
        "output_path": f"error: {error}" if error else "",
    }


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in args.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No videos with extensions {VIDEO_EXTENSIONS} found under {args.input_dir}.")
        return

    total = len(videos)
    state = {
        "canonical_size": None,
        "reference_aspect_ratio": None,
        "padding": args.padding,
        "rotation_threshold": args.rotation_threshold,
    }

    # ------------------------------------------------------------------
    # Pass 1: detection only.
    # ------------------------------------------------------------------
    print(f"Pass 1/2: detecting maze in {total} video(s)...")
    detections = []
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(args.input_dir)
        try:
            d = detect_one(video_path, args.input_dir, state)
        except Exception as e:
            d = {
                "video_path": video_path,
                "rel": rel,
                "filename": str(rel).replace("\\", "/"),
                "status": "failed",
                "error": str(e),
                "frame": None,
                "fps": None,
                "contour": None,
                "H": None,
                "dst_size": None,
                "warped_mask": None,
                "contour_area": np.nan,
                "aspect_ratio": np.nan,
                "rotation_angle": np.nan,
                "area_flag": False,
                "aspect_flag": False,
                "rotation_flag": False,
                "manual": False,
            }
        detections.append(d)
        print(f"  [{i}/{total}] {video_path.name} — detection: {d['status']}")

    n_ok = sum(1 for d in detections if d["status"] == "ok")

    # ------------------------------------------------------------------
    # Build consensus + initial scoring (only if any auto-detect succeeded).
    # ------------------------------------------------------------------
    consensus = None
    if n_ok > 0:
        dst_size = next(d["dst_size"] for d in detections if d["status"] == "ok")
        consensus = build_consensus_template(detections, dst_size)
        template_path = args.output_dir / "consensus_template.png"
        cv2.imwrite(str(template_path), consensus)
        print(f"Built consensus template from {n_ok} mask(s) -> {template_path}")
        for d in detections:
            score_detection(d, consensus, args)
    else:
        # Nothing detected automatically; everyone starts as 'failed'.
        for d in detections:
            d["mask_iou"] = np.nan
            d["confidence"] = "failed"
        if not args.manual:
            print("No videos passed detection. Writing failed-only summary and exiting.")
            rows = [make_failed_row(d["rel"], d.get("error")) for d in detections]
            pd.DataFrame(rows).to_csv(args.output_dir / "alignment_summary.csv", index=False)
            return
        print("No videos passed automatic detection; falling through to --manual mode.")

    # ------------------------------------------------------------------
    # Optional manual correction pass.
    # ------------------------------------------------------------------
    if args.manual:
        consensus = run_manual_pass(detections, state, consensus, args)
        # Save (or re-save) the consensus template now that manual entries exist.
        if consensus is not None:
            template_path = args.output_dir / "consensus_template.png"
            cv2.imwrite(str(template_path), consensus)

    # ------------------------------------------------------------------
    # Pass 2: write cropped videos and review images.
    # ------------------------------------------------------------------
    print(f"Pass 2/2: writing cropped videos...")
    rows = []
    for i, d in enumerate(detections, start=1):
        rel = d["rel"]
        if d["status"] != "ok" or d["H"] is None:
            rows.append(make_failed_row(rel, d.get("error")))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed")
            continue
        try:
            output_path = args.output_dir / rel.parent / (d["video_path"].stem + "_cropped.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            warp_video(d["video_path"], d["H"], d["dst_size"], output_path, d["fps"])

            # Save a review PNG for anything that wasn't a clean automatic 'high'.
            if d["confidence"] in ("medium", "low", "manual"):
                review_path = review_dir / rel.parent / (d["video_path"].stem + "_review.png")
                review_path.parent.mkdir(parents=True, exist_ok=True)
                save_review_image(d["frame"], d["contour"], review_path)

            rows.append({
                "filename": d["filename"],
                "contour_area": d["contour_area"],
                "aspect_ratio": d["aspect_ratio"],
                "rotation_angle": d["rotation_angle"],
                "mask_iou": d.get("mask_iou", np.nan),
                "confidence": d["confidence"],
                "output_path": str(output_path),
            })
            print(f"  [{i}/{total}] {d['filename']} — confidence: {d['confidence']}")
        except Exception as e:
            rows.append(make_failed_row(rel, str(e)))
            print(f"  [{i}/{total}] {d['filename']} — confidence: failed ({e})")

    summary = pd.DataFrame(rows, columns=[
        "filename", "contour_area", "aspect_ratio",
        "rotation_angle", "mask_iou", "confidence", "output_path",
    ])
    summary_path = args.output_dir / "alignment_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
