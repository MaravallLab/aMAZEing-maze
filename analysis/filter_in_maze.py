"""
filter_in_maze.py

Remove out-of-maze false detections from sleap-nn predictions.

Why this exists
---------------
sleap-nn (like DeepLabCut) reports a per-point confidence (`score`). But a high
score only means "this looks like a <node>", NOT "this is a real, in-bounds
mouse". On frames where the animal is absent, the model can place a confident,
internally-consistent pose on background clutter (cables, hardware, reflections)
-- scoring 0.9+ while being completely wrong. No score threshold removes these.

The fix is spatial: the mouse can only ever be on the maze, so we drop any
detection whose body falls outside the maze outline.

How the maze outline is known (no hand-drawing)
-----------------------------------------------
`crop_and_align_maze.py` warps every video so the maze lands at fixed
`canonical` landmark positions. Those 24 landmarks (clicked in outline order)
ARE the maze-boundary polygon, in the exact coordinate system of the `.slp`
keypoints. Each calibration ("rig") warps to a unique canonical frame size, so
a cropped video's WxH identifies its rig -> its polygon. This works for every
video, including ones missing from alignment_summary.csv.

Note: `score` is a confidence, not a correctness check. When the mouse is
absent the model still emits a mouse-shaped pose, sometimes confidently. Those
hallucinations land outside the maze, so we gate on geometry AND confidence.

What it does
------------
For each `*.predictions.slp`:
  1. Look up the rig polygon by the cropped video's frame size.
  2. For each frame's instance, test its body centre (`--anchor`, default
     midbody) against the polygon and drop it when it is either
       - clearly outside (dist < -`--margin_px`), at any confidence, OR
       - outside at all AND below `--min_score_outside` (mouse-absent
         hallucination). A real edge pose (just outside, high score) is kept.
  3. Temporal absent-block sweep: a surviving (confident) detection sitting
     inside a sustained run of mostly-dropped frames is itself a mouse-absent
     hallucination, so it is dropped too. Real activity is untouched. Disable
     with `--no_absent_block`.
  4. Dropped instances leave that frame as "no mouse".
  4. (optional) `--clip_points` also NaNs individual stray points outside the
     polygon on otherwise-kept instances.
  5. Write non-destructive outputs:
       <name>.predictions.filtered.slp   (raw .slp is untouched)
       <name>.keypoints.filtered.csv
     and print how many detections were dropped.

Run with the sleap-nn interpreter, e.g.:

  & "$env:APPDATA\\uv\\tools\\sleap-nn\\Scripts\\python.exe" ^
      analysis\\filter_in_maze.py ^
      --input_dir D:\\simplermaze_output ^
      --calibration_dir D:\\simplermaze_output ^
      --save_overlays
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# Reuse the CSV writer from the exporter (same folder).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_export import export_csv  # noqa: E402

DEFAULT_PADDING = 50           # must match crop_and_align_maze.py --padding
DEFAULT_MARGIN_PX = 15.0       # hard boundary: drop anything beyond this (px)
                               # outside the maze, at any confidence. Keeps the
                               # ~13px "real edge" halo; see QuickGuide for tuning
DEFAULT_MIN_SCORE_OUTSIDE = 0.5  # an instance that is outside the maze at all
                                 # AND scores below this is treated as a
                                 # mouse-absent hallucination and dropped
DEFAULT_ANCHOR = "midbody"     # body-centre node used for the in/out test
CORE_NODES = ("headbase", "midbody", "tailbase")  # fallback anchor set
SIZE_TOLERANCE = 2             # px tolerance when matching cropped size -> rig
# Absent-block sweep: a kept detection sitting inside a sustained run of
# mostly-dropped frames is itself a mouse-absent hallucination. Look +/- this
# many frames; if the local fraction of "real" (kept) frames is below
# MIN_PRESENCE, drop the survivor too.
DEFAULT_BLOCK_WINDOW = 30
DEFAULT_BLOCK_MIN_PRESENCE = 0.34


# ---------------------------------------------------------------------------
# Maze polygon reconstruction (mirrors crop_and_align_maze.canonical_positions)
# ---------------------------------------------------------------------------

def canonical_positions(landmarks, padding):
    """Translate landmarks so their bbox sits at (padding, padding).
    Returns (canonical_pts (24,2) float32, (out_w, out_h)). Identical to the
    function in crop_and_align_maze.py so the polygon matches the warp."""
    pts = np.asarray(landmarks, dtype=np.float32).copy()
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pts[:, 0] -= xmin - padding
    pts[:, 1] -= ymin - padding
    out_w = int(round(xmax - xmin + 2 * padding))
    out_h = int(round(ymax - ymin + 2 * padding))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    return pts, (out_w, out_h)


def build_rig_polygons(calibration_dir, padding):
    """Map canonical (w, h) -> {"name", "poly" (Nx1x2 float32)} for every
    calibration*.json found in `calibration_dir`."""
    rigs = {}
    files = sorted(glob.glob(os.path.join(str(calibration_dir), "calibration*.json")))
    for f in files:
        try:
            cal = json.load(open(f))
        except Exception:
            continue
        if "landmarks" not in cal:
            continue
        cpts, size = canonical_positions(cal["landmarks"], padding)
        rigs[size] = {
            "name": Path(f).stem,
            "poly": cpts.reshape(-1, 1, 2).astype(np.float32),
        }
    return rigs


def match_rig(size_wh, rigs):
    """Find the rig whose canonical size matches (w, h) within tolerance."""
    if size_wh in rigs:
        return rigs[size_wh]
    w, h = size_wh
    for (rw, rh), rig in rigs.items():
        if abs(rw - w) <= SIZE_TOLERANCE and abs(rh - h) <= SIZE_TOLERANCE:
            return rig
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def signed_distance(poly, x, y):
    """+ inside, - outside, 0 on edge (pixels), via cv2.pointPolygonTest."""
    import cv2
    return cv2.pointPolygonTest(poly, (float(x), float(y)), True)


def instance_anchor(xy, name_to_idx, primary_node):
    """Body anchor for the in/out test, most-stable first:
      1. the `primary_node` (default 'midbody') if visible,
      2. else the median of the visible CORE_NODES (head/mid/tail),
      3. else the median of all visible nodes.
    Returns (x, y) or None. Using the body centre (not the median of *all*
    nodes) avoids the nose/tail dragging the anchor off-maze at edges."""
    pi = name_to_idx.get(primary_node)
    if pi is not None and not np.isnan(xy[pi]).any():
        return xy[pi]
    core_idx = [name_to_idx[n] for n in CORE_NODES if n in name_to_idx]
    core = xy[core_idx] if core_idx else np.empty((0, 2))
    core = core[~np.isnan(core).any(axis=1)]
    if len(core):
        return np.median(core, axis=0)
    vis = xy[~np.isnan(xy).any(axis=1)]
    if len(vis) == 0:
        return None
    return np.median(vis, axis=0)


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def video_size(labels, video_override):
    """Return (w, h) of the source video for this Labels object."""
    import cv2
    path = video_override or (labels.videos[0].filename if labels.videos else None)
    if path and Path(path).exists():
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return (w, h), path
    # fall back to the video object's recorded shape (frames, h, w, c)
    if labels.videos:
        shp = getattr(labels.videos[0], "shape", None)
        if shp is not None and len(shp) >= 3:
            return (int(shp[2]), int(shp[1])), path
    return None, path


def _classify(inst, poly, margin, min_score_outside, name_to_idx, anchor_node):
    """Base per-instance verdict: 'far', 'lowconf', or 'kept'."""
    xy = inst.numpy()
    anchor = instance_anchor(xy, name_to_idx, anchor_node)
    if anchor is None:
        return "far"
    dist = signed_distance(poly, anchor[0], anchor[1])
    if dist < -margin:
        return "far"               # clearly off-maze, any confidence
    score = getattr(inst, "score", None)
    if dist < 0 and score is not None and score < min_score_outside:
        return "lowconf"           # outside + low confidence -> hallucination
    return "kept"


def _clip_outside(inst, poly, margin):
    """NaN this instance's individual points that fall outside the maze."""
    xy = inst.numpy()
    n = 0
    for ni in range(len(xy)):
        x, y = xy[ni]
        if not np.isnan(x) and signed_distance(poly, x, y) < -margin:
            _nan_point(inst, ni)
            n += 1
    return n


def filter_labels(labels, poly, margin, min_score_outside, clip_points,
                  anchor_node, absent_block, block_window, block_min_presence):
    """Drop out-of-maze / hallucinated instances in place.

    Base rule -- an instance is dropped when its body anchor is:
      - clearly outside the maze (dist < -margin), at ANY confidence; or
      - outside the maze at all (dist < 0) AND scores below `min_score_outside`.
    A real edge pose (just outside, high score) is kept. The confidence
    populations are well separated: in-maze ~0.91, hallucination ~0.41.

    Absent-block sweep (if `absent_block`) -- the base rule plateaus at ~78%
    because some hallucinations are both confident and near the boundary. But
    the mouse is absent in *contiguous blocks*, so a surviving detection sitting
    in a sustained run of mostly-dropped frames is itself a hallucination: if the
    fraction of "kept" frames within +/-`block_window` is below
    `block_min_presence`, the survivor is dropped too. Real activity (dense kept
    frames) is untouched.

    Returns (n_instances, n_far, n_lowconf, n_block, n_points_clipped)."""
    name_to_idx = {n.name: i for i, n in enumerate(labels.skeleton.nodes)}

    # Phase 1: classify every instance, remember verdicts per labeled frame.
    frame_recs = []          # [(lf, [[inst, status], ...]), ...]
    max_idx = 0
    n_inst = 0
    for lf in labels.labeled_frames:
        recs = []
        for inst in lf.instances:
            n_inst += 1
            recs.append([inst, _classify(inst, poly, margin, min_score_outside,
                                         name_to_idx, anchor_node)])
        frame_recs.append((lf, recs))
        if lf.instances:
            max_idx = max(max_idx, lf.frame_idx)

    # Phase 2: absent-block sweep over the per-frame "presence" timeline.
    n_block = 0
    if absent_block and max_idx > 0:
        presence = np.zeros(max_idx + 1, dtype=np.float64)
        for lf, recs in frame_recs:
            if lf.frame_idx <= max_idx and any(s == "kept" for _, s in recs):
                presence[lf.frame_idx] = 1.0
        cumsum = np.concatenate([[0.0], np.cumsum(presence)])  # prefix sums
        for lf, recs in frame_recs:
            if not any(s == "kept" for _, s in recs):
                continue
            fi = lf.frame_idx
            lo = max(0, fi - block_window)
            hi = min(max_idx, fi + block_window)
            frac = (cumsum[hi + 1] - cumsum[lo]) / (hi - lo + 1)
            if frac < block_min_presence:
                for r in recs:
                    if r[1] == "kept":
                        r[1] = "block"
                        n_block += 1

    # Phase 3: tally and rebuild with only the survivors.
    n_far = sum(1 for _, recs in frame_recs for _, s in recs if s == "far")
    n_lowconf = sum(1 for _, recs in frame_recs for _, s in recs if s == "lowconf")
    n_clip = 0
    kept_frames = []
    for lf, recs in frame_recs:
        keep = []
        for inst, status in recs:
            if status != "kept":
                continue
            if clip_points:
                n_clip += _clip_outside(inst, poly, margin)
            keep.append(inst)
        if keep:
            lf.instances = keep
            kept_frames.append(lf)
    labels.labeled_frames = kept_frames
    return n_inst, n_far, n_lowconf, n_block, n_clip


def _nan_point(inst, ni):
    """Best-effort set node `ni` of a PredictedInstance to not-visible."""
    try:
        inst.points[ni]["visible"] = False
    except Exception:
        try:
            p = inst.points[ni]
            p.visible = False
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def process_one(slp_path, rigs, args):
    import sleap_io as sio
    slp_path = Path(slp_path)
    base = slp_path.name[:-len(".predictions.slp")] \
        if slp_path.name.endswith(".predictions.slp") else slp_path.stem

    labels = sio.load_slp(str(slp_path))
    size, video_path = video_size(labels, args.video)
    if size is None:
        print(f"  SKIP: could not determine video size for {slp_path.name}",
              file=sys.stderr)
        return
    rig = match_rig(size, rigs)
    if rig is None:
        print(f"  SKIP: no rig polygon matches cropped size {size[0]}x{size[1]} "
              f"(known: {sorted(rigs)})", file=sys.stderr)
        return

    if args.save_overlays and video_path:
        _save_overlay(labels, rig, video_path, slp_path.parent / f"{base}.maze_roi.png")

    n_inst, n_far, n_lowconf, n_block, n_clip = filter_labels(
        labels, rig["poly"], args.margin_px, args.min_score_outside,
        args.clip_points, args.anchor, not args.no_absent_block,
        args.absent_block_window, args.absent_block_min_presence)

    n_drop = n_far + n_lowconf + n_block
    pct = (100.0 * n_drop / n_inst) if n_inst else 0.0
    print(f"  rig={rig['name']} size={size[0]}x{size[1]} | anchor={args.anchor} "
          f"margin={args.margin_px:g}px min_score_outside={args.min_score_outside:g} | "
          f"detections={n_inst}  dropped={n_drop} ({pct:.1f}%): "
          f"far_off_maze={n_far} lowconf_off_maze={n_lowconf} absent_block={n_block}"
          + (f"  points_clipped={n_clip}" if args.clip_points else ""))

    if not args.no_slp:
        out_slp = slp_path.parent / f"{base}.predictions.filtered.slp"
        labels.save(str(out_slp), restore_original_videos=False)
        print(f"    SLP : {out_slp}")
    if not args.no_csv:
        out_csv = slp_path.parent / f"{base}.keypoints.filtered.csv"
        export_csv(labels, out_csv)
        print(f"    CSV : {out_csv}")


def _save_overlay(labels, rig, video_path, out_png):
    """Save a median-frame image with the maze polygon drawn, for verification."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(n - 1, 0), 80).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    cap.release()
    if not frames:
        return
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.polylines(med, [rig["poly"].astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_png), med)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_slps(input_dir):
    # Don't re-filter already-filtered files.
    return sorted(p for p in Path(input_dir).rglob("*.predictions.slp")
                  if not p.name.endswith(".filtered.slp"))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--slp", type=Path, help="A single *.predictions.slp file.")
    src.add_argument("--input_dir", type=Path,
                     help="Folder searched recursively for *.predictions.slp.")
    p.add_argument("--calibration_dir", type=Path, default=None,
                   help="Folder with calibration*.json (default: --input_dir, "
                        "or the .slp's folder in single-file mode).")
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING,
                   help="Must match crop_and_align_maze.py --padding (default %(default)s).")
    p.add_argument("--margin_px", type=float, default=DEFAULT_MARGIN_PX,
                   help="Outward slack on the maze boundary, px (default %(default)s). "
                        "Larger = keeps more edge/rim-walking poses.")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR,
                   help="Skeleton node used as the body centre for the in/out "
                        "test (default %(default)s). Falls back to head/mid/tail "
                        "median, then all-node median, if not visible.")
    p.add_argument("--min_score_outside", type=float,
                   default=DEFAULT_MIN_SCORE_OUTSIDE,
                   help="An instance outside the maze AND scoring below this is "
                        "dropped as a mouse-absent hallucination (default "
                        "%(default)s). Set to 0 to disable the confidence rule "
                        "and filter on geometry only.")
    p.add_argument("--no_absent_block", action="store_true",
                   help="Disable the temporal absent-block sweep (which removes "
                        "confident hallucinations sitting inside sustained "
                        "mouse-absent runs).")
    p.add_argument("--absent_block_window", type=int,
                   default=DEFAULT_BLOCK_WINDOW,
                   help="Half-width (frames) of the absent-block presence window "
                        "(default %(default)s, i.e. +/-1s at 30fps).")
    p.add_argument("--absent_block_min_presence", type=float,
                   default=DEFAULT_BLOCK_MIN_PRESENCE,
                   help="If fewer than this fraction of frames in the window are "
                        "kept ('real'), a survivor is treated as in an absent "
                        "block and dropped (default %(default)s).")
    p.add_argument("--clip_points", action="store_true",
                   help="Also NaN individual stray points outside the maze on "
                        "kept instances (off by default).")
    p.add_argument("--video", type=Path, default=None,
                   help="Override source video path (single-file mode).")
    p.add_argument("--save_overlays", action="store_true",
                   help="Write <name>.maze_roi.png (median frame + polygon) to verify.")
    p.add_argument("--no_slp", action="store_true", help="Don't write filtered .slp.")
    p.add_argument("--no_csv", action="store_true", help="Don't write filtered .csv.")
    return p.parse_args()


def main():
    args = parse_args()
    calib_dir = args.calibration_dir or (
        args.input_dir if args.input_dir else (args.slp.parent if args.slp else None))
    rigs = build_rig_polygons(calib_dir, args.padding)
    if not rigs:
        print(f"Error: no calibration*.json with landmarks found in {calib_dir}.",
              file=sys.stderr)
        return 1
    print(f"Loaded {len(rigs)} rig polygon(s) from {calib_dir}:")
    for size, rig in sorted(rigs.items()):
        print(f"  {rig['name']:22s} -> {size[0]}x{size[1]} px")

    targets = [args.slp] if args.slp else find_slps(args.input_dir)
    if args.input_dir:
        print(f"Found {len(targets)} prediction file(s) under {args.input_dir}.")
    for i, slp in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {Path(slp).name}")
        try:
            process_one(slp, rigs, args)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
