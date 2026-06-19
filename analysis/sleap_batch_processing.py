"""
sleap_batch_processing.py

Batch-predict mouse keypoints with a trained SLEAP-NN top-down model
(centroid + centered-instance) across every cropped video produced by
crop_and_align_maze.py.

IMPORTANT: the models in CROPPED_VIDEOS_FOR_SLEAP/models are *sleap-nn*
(the PyTorch rewrite) models -- each model folder has `best.ckpt` +
`training_config.yaml` with a `sleap_nn_version` field. They are NOT
classic TensorFlow SLEAP 1.x models, so this script uses the `sleap_nn`
package (`from sleap_nn.predict import Predictor`), not `import sleap`.

Workflow:
  1. Walk `--input_dir` recursively for files matching `--include_pattern`
     (default `*_cropped.mp4`).
  2. Load the two model folders ONCE (centroid + centered-instance) into a
     single Predictor so all videos share it -- avoids per-video model load
     and keeps the GPU/CUDA context warm.
  3. For each video, write predictions next to it as
     `<stem>.predictions.slp`, plus `<stem>.keypoints.csv` (per-frame
     keypoints) in the same pass unless `--no-write_csv` is given.
     Already-predicted videos are skipped by default; pass
     `--no-skip_existing` to force reprocessing.
  4. Write `sleap_inference_summary.csv` under `--input_dir` with one
     row per video (status, n_frames, runtime, error).

Model resolution (any of these works):
  - `--models_dir <dir>`: directory containing both model subfolders.
    The script picks the most recently modified `*centroid*` and
    `*centered_instance*` subfolders.
  - `--centroid_model <dir> --centered_instance_model <dir>`: pass each
    explicitly. Overrides `--models_dir`.

This script must run inside a Python environment that has `sleap-nn`
installed. On this machine it was installed as a uv tool, so run it with
that interpreter, e.g.:

  & "$env:APPDATA\\uv\\tools\\sleap-nn\\Scripts\\python.exe" ^
      analysis\\sleap_batch_processing.py ^
      --input_dir  D:\\simplermaze_output ^
      --models_dir "C:\\Users\\shahd\\OneDrive\\Desktop\\CROPPED_VIDEOS_FOR_SLEAP\\models"

(Install with: uv tool install --python 3.13 "sleap-nn[torch]" --torch-backend auto)
"""

import argparse
import csv
import fnmatch
import os
import sys
import time
from pathlib import Path

# Reuse the CSV writer from the exporter (same folder) so inference can emit the
# per-frame keypoints CSV in the same pass — no second script, no re-reading the
# .slp. slp_export only pulls in cv2/sleap_io lazily, so this import is cheap.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_export import export_csv  # noqa: E402


DEFAULT_INCLUDE_PATTERN = "*_cropped.mp4"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_INSTANCES = 1
DEFAULT_PEAK_THRESHOLD = 0.2
DEFAULT_DEVICE = "auto"
DEFAULT_PREDICTION_SUFFIX = ".predictions.slp"
DEFAULT_SUMMARY_BASENAME = "sleap_inference_summary.csv"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_videos(input_dir, include_pattern):
    """Walk `input_dir` recursively and return paths matching the pattern."""
    patterns = [p.strip() for p in include_pattern.split(",") if p.strip()]
    matches = []
    for root, _dirs, files in os.walk(input_dir):
        for fname in files:
            lower = fname.lower()
            for pat in patterns:
                if fnmatch.fnmatch(lower, pat.lower()):
                    matches.append(Path(root) / fname)
                    break
    return sorted(matches)


def find_model_pair(models_dir):
    """
    Find the most recently modified centroid + centered-instance model
    folders inside `models_dir`. Returns (centroid_path, centered_path)
    or (None, None) if either is missing.
    """
    centroids, instances = [], []
    for sub in Path(models_dir).iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if "centered_instance" in name or "centered-instance" in name:
            instances.append(sub)
        elif "centroid" in name:
            centroids.append(sub)
    if not centroids or not instances:
        return None, None
    centroids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    instances.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return centroids[0], instances[0]


# ---------------------------------------------------------------------------
# sleap-nn wrappers (imported lazily so --help works without sleap-nn installed)
# ---------------------------------------------------------------------------

def import_predictor():
    try:
        from sleap_nn.predict import Predictor
        return Predictor
    except ImportError:
        print(
            "Error: `sleap-nn` is not installed in this Python environment.\n"
            "These models are sleap-nn (PyTorch), not classic SLEAP. Run this\n"
            "script with the sleap-nn interpreter, e.g.\n"
            '  & "$env:APPDATA\\uv\\tools\\sleap-nn\\Scripts\\python.exe" '
            "analysis\\sleap_batch_processing.py ...\n"
            'Install with: uv tool install --python 3.13 "sleap-nn[torch]" '
            "--torch-backend auto",
            file=sys.stderr,
        )
        sys.exit(1)


def load_predictor(centroid_path, centered_instance_path, batch_size,
                   max_instances, device, peak_threshold):
    Predictor = import_predictor()
    from omegaconf import OmegaConf
    print("Loading sleap-nn models (one-time):")
    print(f"  centroid:          {centroid_path}")
    print(f"  centered_instance: {centered_instance_path}")
    print(f"  device={device}  batch_size={batch_size}  "
          f"max_instances={max_instances}  peak_threshold={peak_threshold}")
    # `from_model_paths` requires a preprocess_config mapping (not None). All
    # values None => fall back to each model's training_config. This mirrors
    # what sleap_nn.predict.run_inference passes internally.
    preprocess_config = OmegaConf.create({
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    })
    t0 = time.time()
    predictor = Predictor.from_model_paths(
        [str(centroid_path), str(centered_instance_path)],
        batch_size=batch_size,
        max_instances=max_instances,
        peak_threshold=peak_threshold,
        device=device,
        preprocess_config=preprocess_config,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")
    return predictor


def predict_one(predictor, video_path, output_path, csv_path=None):
    """Run prediction on one video, save the .slp, return n_frames.

    Reuses the already-loaded `predictor`; `make_pipeline` rebuilds the data
    provider for this video each call (safe to repeat across videos). If
    `csv_path` is given, also writes the per-frame keypoints CSV straight from
    the in-memory predictions. A CSV failure is logged but never fails the
    video — the .slp is the authoritative output.
    """
    predictor.make_pipeline(inference_object=str(video_path))
    labels = predictor.predict(make_labels=True)
    labels.save(str(output_path), restore_original_videos=False)
    if csv_path is not None:
        try:
            export_csv(labels, Path(csv_path))
        except Exception as e:
            print(f"      (CSV export failed, .slp is fine: {e})", flush=True)
    try:
        n_frames = len(labels.labeled_frames)
    except Exception:
        n_frames = 0
    return n_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True, type=Path,
                   help="Directory of cropped videos (searched recursively).")
    p.add_argument("--models_dir", type=Path, default=None,
                   help="Directory containing the centroid + centered-instance "
                        "model subfolders. The most recently modified of each "
                        "is chosen. Ignored if both --centroid_model and "
                        "--centered_instance_model are set.")
    p.add_argument("--centroid_model", type=Path, default=None,
                   help="Path to the centroid model folder (overrides "
                        "--models_dir).")
    p.add_argument("--centered_instance_model", type=Path, default=None,
                   help="Path to the centered-instance model folder (overrides "
                        "--models_dir).")
    p.add_argument("--include_pattern", default=DEFAULT_INCLUDE_PATTERN,
                   help=f"Comma-separated fnmatch globs for video filenames "
                        f"(default: \"{DEFAULT_INCLUDE_PATTERN}\").")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="GPU inference batch size (default %(default)s). Raise "
                        "if you have spare VRAM.")
    p.add_argument("--max_instances", type=int, default=DEFAULT_MAX_INSTANCES,
                   help="Maximum mouse instances per frame (default %(default)s "
                        "— the maze is single-animal).")
    p.add_argument("--device", default=DEFAULT_DEVICE,
                   help="Torch device: auto/cuda/cpu/mps or cuda:0 "
                        "(default %(default)s — picks cuda if available).")
    p.add_argument("--peak_threshold", type=float, default=DEFAULT_PEAK_THRESHOLD,
                   help="Confidence-map peak threshold (default %(default)s).")
    p.add_argument("--prediction_suffix", default=DEFAULT_PREDICTION_SUFFIX,
                   help="Suffix appended to each video stem to form the "
                        f"prediction filename (default \"{DEFAULT_PREDICTION_SUFFIX}\").")
    p.add_argument("--write_csv", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also write <stem>.keypoints.csv next to each .slp in "
                        "the same pass (default on). Use --no-write_csv to skip "
                        "(you can always run slp_export.py later).")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Skip videos whose prediction file already exists "
                        "(default on). Use --no-skip_existing to force "
                        "re-prediction.")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Process at most N videos (for testing).")
    p.add_argument("--summary_csv", type=Path, default=None,
                   help=f"Where to write the run summary. "
                        f"Defaults to <input_dir>/{DEFAULT_SUMMARY_BASENAME}.")
    p.add_argument("--dry_run", action="store_true",
                   help="List the videos that would be predicted, then exit "
                        "without loading models or running inference.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_model_paths(args):
    if args.centroid_model and args.centered_instance_model:
        return args.centroid_model, args.centered_instance_model
    if args.models_dir is not None:
        c, ci = find_model_pair(args.models_dir)
        if c is None or ci is None:
            print(f"Error: couldn't find both a *centroid* and a "
                  f"*centered_instance* subfolder under {args.models_dir}.",
                  file=sys.stderr)
            return None, None
        return c, ci
    print("Error: provide either --models_dir, or both --centroid_model "
          "and --centered_instance_model.", file=sys.stderr)
    return None, None


def main():
    args = parse_args()

    centroid_path, centered_path = resolve_model_paths(args)
    if centroid_path is None or centered_path is None:
        return 1
    for p in (centroid_path, centered_path):
        if not Path(p).exists():
            print(f"Error: model path does not exist: {p}", file=sys.stderr)
            return 1

    if not args.input_dir.exists():
        print(f"Error: input_dir does not exist: {args.input_dir}",
              file=sys.stderr)
        return 1

    videos = find_videos(args.input_dir, args.include_pattern)
    if not videos:
        print(f"No videos matching '{args.include_pattern}' found under "
              f"{args.input_dir}.")
        return 0
    if args.max_videos:
        videos = videos[: args.max_videos]
    total = len(videos)
    print(f"Found {total} video(s) under {args.input_dir} "
          f"(pattern: {args.include_pattern}).")

    summary_path = (
        args.summary_csv
        if args.summary_csv is not None
        else args.input_dir / DEFAULT_SUMMARY_BASENAME
    )

    if args.dry_run:
        print("\nDry run — would predict:")
        for i, v in enumerate(videos, start=1):
            out = v.parent / (v.stem + args.prediction_suffix)
            exists = out.exists()
            tag = "(skip)" if (exists and args.skip_existing) else "(predict)"
            print(f"  [{i}/{total}] {tag} {v.relative_to(args.input_dir)}")
        print(f"\nSummary would be written to: {summary_path}")
        return 0

    predictor = load_predictor(
        centroid_path, centered_path,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        device=args.device,
        peak_threshold=args.peak_threshold,
    )

    rows = []
    n_ok = n_skip = n_fail = 0
    overall_t0 = time.time()
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(args.input_dir)
        output_path = video_path.parent / (video_path.stem + args.prediction_suffix)
        csv_path = (video_path.parent / (video_path.stem + ".keypoints.csv")
                    if args.write_csv else None)

        if args.skip_existing and output_path.exists():
            n_skip += 1
            print(f"  [{i}/{total}] {video_path.name} — skipped (exists)")
            rows.append({
                "filename": str(rel).replace("\\", "/"),
                "output_path": str(output_path),
                "status": "skipped",
                "n_frames": "",
                "runtime_sec": "",
                "error": "",
            })
            continue

        print(f"  [{i}/{total}] {video_path.name} — predicting...", flush=True)
        t0 = time.time()
        try:
            n_frames = predict_one(predictor, video_path, output_path, csv_path)
            elapsed = time.time() - t0
            n_ok += 1
            print(f"      done in {elapsed:.1f}s "
                  f"({n_frames} labeled frame(s)) -> {output_path.name}"
                  + (f" + {csv_path.name}" if csv_path else ""))
            rows.append({
                "filename": str(rel).replace("\\", "/"),
                "output_path": str(output_path),
                "status": "ok",
                "n_frames": n_frames,
                "runtime_sec": f"{elapsed:.2f}",
                "error": "",
            })
        except Exception as e:
            elapsed = time.time() - t0
            n_fail += 1
            print(f"      FAILED in {elapsed:.1f}s: {e}")
            rows.append({
                "filename": str(rel).replace("\\", "/"),
                "output_path": str(output_path),
                "status": "failed",
                "n_frames": "",
                "runtime_sec": f"{elapsed:.2f}",
                "error": str(e),
            })

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "output_path", "status",
                        "n_frames", "runtime_sec", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.time() - overall_t0
    print(f"\nWrote summary: {summary_path}")
    print(f"ok={n_ok}, skipped={n_skip}, failed={n_fail} "
          f"(total {total} in {total_elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
