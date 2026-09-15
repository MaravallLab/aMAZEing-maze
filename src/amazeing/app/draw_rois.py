"""Draw the ROI rectangles for a session config (child process of the app).

Uses exactly the interactive OpenCV routine the session itself uses when the
ROI file is missing, so the saved ``rois1.csv`` is identical either way.

    python -m amazeing.app.draw_rois --config session.yaml
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Interactively draw the ROIs for a session config.")
    p.add_argument("--config", required=True, help="Session config YAML")
    args = p.parse_args(argv)

    from amazeing.auditory.session_config import load_config
    from amazeing.auditory.vision import define_rois

    cfg = load_config(args.config)
    names = list(cfg.entrance_rois) + [str(i + 1) for i in range(cfg.rois_number)]
    os.makedirs(os.path.dirname(cfg.roi_csv_path) or ".", exist_ok=True)
    if os.path.exists(cfg.roi_csv_path):
        print(f"Replacing existing ROI file {cfg.roi_csv_path}")
    print(f"Camera {cfg.video_input}; draw {len(names)} ROIs in this order: {', '.join(names)}")
    define_rois(cfg.video_input, cfg.roi_csv_path, names)
    print(f"Saved {cfg.roi_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
