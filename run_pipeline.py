#!/usr/bin/env python3
r"""
run_pipeline.py -- full aMAZEing-maze pose pipeline in one command.

Runs Stage 2 (sleap-nn inference + per-frame CSV) then Stage 4 (in-maze filter).
Inference is the long part (~4-5 days for a few hundred ~100k-frame videos) and
is RESUMABLE -- re-run this script to continue (finished videos are skipped).
Stage 1 (crop & align) is assumed already done; see analysis/README.md.

This orchestrator uses ONLY the Python standard library, so you can launch it
with any Python on your machine:

    python run_pipeline.py

The actual inference/filtering runs in the sleap-nn environment (installed as a
uv tool); this script calls that interpreter for you. Override anything via
flags, e.g.:

    python run_pipeline.py --input_dir D:\simplermaze_output \
        --models_dir "C:\Users\shahd\OneDrive\Desktop\CROPPED_VIDEOS_FOR_SLEAP\models"

    python run_pipeline.py --skip_inference     # only (re)run the filter
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
INFER = REPO / "analysis" / "sleap_batch_processing.py"
FILT = REPO / "analysis" / "filter_in_maze.py"

# Defaults for this dataset (override with flags).
DEFAULT_PY = os.path.expandvars(r"%APPDATA%\uv\tools\sleap-nn\Scripts\python.exe")
DEFAULT_INPUT = r"D:\simplermaze_output"
DEFAULT_MODELS = r"C:\Users\shahd\OneDrive\Desktop\CROPPED_VIDEOS_FOR_SLEAP\models"
DEFAULT_CALIB = r"D:\simplermaze_output"


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def section(msg):
    print(f"\n========== {stamp()}  {msg} ==========", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=DEFAULT_INPUT)
    ap.add_argument("--models_dir", default=DEFAULT_MODELS)
    ap.add_argument("--calibration_dir", default=DEFAULT_CALIB)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=DEFAULT_PY,
                    help="Path to the sleap-nn interpreter (default: the uv tool).")
    ap.add_argument("--skip_inference", action="store_true",
                    help="Run only the filter stage.")
    ap.add_argument("--skip_filter", action="store_true",
                    help="Run only the inference stage.")
    a = ap.parse_args()

    # --- preflight ---------------------------------------------------------
    if not Path(a.python).exists():
        print(f"ERROR: sleap-nn Python not found at:\n  {a.python}\n"
              'Install it with:\n'
              '  uv tool install --python 3.13 "sleap-nn[torch]" --torch-backend auto',
              file=sys.stderr)
        return 1
    for p in (INFER, FILT):
        if not p.exists():
            print(f"ERROR: missing script {p}", file=sys.stderr)
            return 1
    for d in (a.input_dir, a.models_dir, a.calibration_dir):
        if not Path(d).exists():
            print(f"ERROR: path does not exist: {d}", file=sys.stderr)
            return 1

    print(f"sleap-nn python : {a.python}")
    print(f"input_dir       : {a.input_dir}")
    print(f"models_dir      : {a.models_dir}")
    print(f"calibration_dir : {a.calibration_dir}")
    print(f"device          : {a.device}")

    # --- Stage 2: inference (+ CSV) ---------------------------------------
    if not a.skip_inference:
        section("Stage 2/4  inference (+ keypoints CSV)  -- this is the multi-day step")
        rc = subprocess.run([a.python, str(INFER),
                             "--input_dir", a.input_dir,
                             "--models_dir", a.models_dir,
                             "--device", a.device]).returncode
        if rc != 0:
            print(f"\nInference exited with code {rc} -- stopping before the filter.\n"
                  "Re-run this script to resume (finished videos are skipped).",
                  file=sys.stderr)
            return rc
    else:
        section("Stage 2/4  inference  -- SKIPPED (--skip_inference)")

    # --- Stage 4: in-maze filter ------------------------------------------
    if not a.skip_filter:
        section("Stage 4/4  in-maze / hallucination filter")
        rc = subprocess.run([a.python, str(FILT),
                             "--input_dir", a.input_dir,
                             "--calibration_dir", a.calibration_dir]).returncode
        if rc != 0:
            print(f"\nFilter exited with code {rc}.\n"
                  "Re-run with --skip_inference to retry just the filter.",
                  file=sys.stderr)
            return rc
    else:
        section("Stage 4/4  filter  -- SKIPPED (--skip_filter)")

    section("Pipeline complete")
    print("Per video you now have:")
    print("  <name>.predictions.slp / .keypoints.csv            (raw)")
    print("  <name>.predictions.filtered.slp / .keypoints.filtered.csv  (in-maze only)")
    print(f"Run summary: {Path(a.input_dir) / 'sleap_inference_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
