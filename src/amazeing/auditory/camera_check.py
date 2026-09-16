"""Live camera view for setting detection up before a session records anything.

Detection is the part of a session most likely to be wrong in a way that is
invisible until it is too late: an arm box drawn where the animal does not
go, a binary threshold suited to yesterday's lighting, a sensitivity that
never trips or trips on a shadow. All three are cheap to fix before a session
and impossible to fix after one.

This opens the camera, measures the empty-maze baselines exactly as a session
does, and then shows, live and before anything is recorded:

* the camera picture with the arm boxes drawn on it, red where an arm reads
  as occupied and blue where it does not
* the binary picture, which is what detection actually works from
* for every arm, how far its current reading sits from the line that counts
  as occupied

Binary threshold and detection sensitivity are sliders, so the effect of
moving them is visible in the same instant. Walk a hand down the maze and
watch which boxes turn red. Press ``s`` to write the values you settled on
back into the config file, ``d`` to redraw the boxes, ``c`` to measure the
baselines again.

    amaze-camera-check --config session.yaml

A note on the numbers. The baseline is the sum of the grey picture over an
arm while the maze is empty, and the reading is the sum of the black and
white picture over the same arm now. Those are different scales, so an empty
arm usually reads well above 1.00 rather than at it. What matters is the gap:
set the sensitivity line between where an arm sits empty and where it sits
with the animal in it. This is how the session has always measured, and the
panel shows it rather than changing it.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np

VIEW_WINDOW = "Camera and detection check"
BINARY_WINDOW = "Binary view: what detection sees"

PANEL_WIDTH = 380
BAR_LEFT = 150
BAR_RIGHT = PANEL_WIDTH - 20
RATIO_SCALE = 2.0          # the bar axis runs from 0 to this
MARGIN = 0.15              # closer than this to the line counts as marginal
FONT = cv.FONT_HERSHEY_SIMPLEX

OCCUPIED = (60, 60, 255)   # BGR
MARGINAL = (0, 180, 255)
CLEAR = (90, 210, 90)
INK = (225, 225, 225)
FAINT = (140, 140, 140)
GROUND = (38, 38, 38)


@dataclass
class RoiReading:
    """One arm's current state, as the panel reports it."""

    name: str
    baseline: float        # sum over the arm while the maze was empty
    total: float           # sum over the arm now
    ratio: float           # total / baseline
    occupied: bool         # the session's own debounced answer

    def status(self, sensitivity: float) -> str:
        if self.occupied:
            return "occupied"
        if self.ratio < sensitivity + MARGIN:
            return "marginal"
        return "clear"

    def colour(self, sensitivity: float):
        return {"occupied": OCCUPIED, "marginal": MARGINAL, "clear": CLEAR}[self.status(sensitivity)]


def read_rois(monitor, binary_frame: np.ndarray) -> List[RoiReading]:
    """Current reading for every arm the monitor has a baseline for."""
    readings: List[RoiReading] = []
    for name in monitor.roiNames:
        if name not in monitor.thresholds:
            continue
        baseline = float(monitor.thresholds[name])
        total = float(np.sum(monitor._crop_roi(binary_frame, name)))
        ratio = total / baseline if baseline > 0 else 0.0
        readings.append(RoiReading(name, baseline, total, ratio, bool(monitor.is_occupied[name])))
    return readings


def render_panel(readings: List[RoiReading], sensitivity: float, threshold: int,
                 height: int, width: int = PANEL_WIDTH) -> np.ndarray:
    """The readout drawn beside the picture: one bar per arm against the line."""
    panel = np.full((max(height, 240), width, 3), GROUND, dtype=np.uint8)
    line_x = int(BAR_LEFT + (min(sensitivity, RATIO_SCALE) / RATIO_SCALE) * (BAR_RIGHT - BAR_LEFT))

    y = 26
    cv.putText(panel, f"Binary threshold   {threshold}", (14, y), FONT, 0.48, INK, 1)
    y += 20
    cv.putText(panel, f"Sensitivity        {sensitivity:.2f}", (14, y), FONT, 0.48, INK, 1)
    y += 22
    cv.putText(panel, "An arm is occupied once its bar", (14, y), FONT, 0.4, FAINT, 1)
    y += 15
    cv.putText(panel, "falls to the left of the line.", (14, y), FONT, 0.4, FAINT, 1)
    y += 24

    top_of_bars = y
    for r in readings:
        if y > height - 76:
            cv.putText(panel, "...", (14, y + 6), FONT, 0.45, FAINT, 1)
            break
        colour = r.colour(sensitivity)
        cv.putText(panel, r.name[:9], (14, y + 11), FONT, 0.45, INK, 1)
        cv.putText(panel, f"{r.ratio:4.2f}", (95, y + 11), FONT, 0.45, colour, 1)
        cv.rectangle(panel, (BAR_LEFT, y), (BAR_RIGHT, y + 14), (58, 58, 58), -1)
        filled = int(BAR_LEFT + (min(r.ratio, RATIO_SCALE) / RATIO_SCALE) * (BAR_RIGHT - BAR_LEFT))
        if filled > BAR_LEFT:
            cv.rectangle(panel, (BAR_LEFT, y), (filled, y + 14), colour, -1)
        y += 20
    bottom_of_bars = y

    # the sensitivity line, drawn over every bar so the comparison is direct
    if readings:
        cv.line(panel, (line_x, top_of_bars - 4), (line_x, bottom_of_bars - 2), INK, 2)

    footer = height - 58
    for i, line in enumerate(("d  redraw the arm boxes",
                              "c  measure the baselines again",
                              "s  save these values to the config",
                              "q  close")):
        cv.putText(panel, line, (14, footer + i * 14), FONT, 0.38, FAINT, 1)
    return panel


def _nothing(_value) -> None:
    """Trackbars need a callback; the positions are read in the loop."""


def _calibrate(camera, monitor) -> bool:
    """Measure the empty-maze baselines, the way a session does."""
    print("Measuring the empty-maze baselines. Keep the maze empty.")
    frames = []
    for i in range(40):
        valid, frame = camera.get_frame()
        if valid and i >= 30:
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame)
    if len(frames) < 5:
        print("Could not read enough frames from the camera to measure baselines.")
        return False
    monitor.calibrate(frames)
    return True


def _live_view(cfg, names: List[str], config_path: str) -> str:
    """Show the live view. Returns 'redraw' or 'quit'."""
    from amazeing.auditory.hardware import Camera
    from amazeing.auditory.session_config import save_config
    from amazeing.auditory.vision import ROIMonitor

    camera = Camera(device_id=cfg.video_input)
    outcome = "quit"
    try:
        monitor = ROIMonitor(
            roi_csv_path=cfg.roi_csv_path,
            roiNames=names,
            video_input=cfg.video_input,
            detection_sensitivity=cfg.detection_sensitivity,
            debug_roi=cfg.debug_roi,
        )
        if not _calibrate(camera, monitor):
            return "quit"

        cv.namedWindow(VIEW_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(VIEW_WINDOW, 1280, 720)
        cv.namedWindow(BINARY_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(BINARY_WINDOW, 640, 480)
        cv.createTrackbar("Binary threshold", VIEW_WINDOW, int(cfg.binary_threshold), 255, _nothing)
        cv.createTrackbar("Sensitivity /100", VIEW_WINDOW, int(round(cfg.detection_sensitivity * 100)),
                          99, _nothing)

        print("\nLive view. Nothing is being recorded.")
        print("  Move the sliders and watch the bars. Walk a hand down the maze.")
        print("  d redraw boxes, c re-measure baselines, s save values, q close.")

        while True:
            valid, frame = camera.get_frame()
            if not valid:
                print("Lost the camera picture.")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            display = frame.copy() if frame.ndim == 3 else cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

            threshold = cv.getTrackbarPos("Binary threshold", VIEW_WINDOW)
            sensitivity = max(1, cv.getTrackbarPos("Sensitivity /100", VIEW_WINDOW)) / 100.0
            monitor.detection_sensitivity = sensitivity

            _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
            monitor.update(binary)
            readings = read_rois(monitor, binary)

            monitor.draw_feedback(display)
            panel = render_panel(readings, sensitivity, threshold, display.shape[0])
            if panel.shape[0] != display.shape[0]:
                panel = panel[:display.shape[0]] if panel.shape[0] > display.shape[0] else \
                    np.vstack([panel, np.full((display.shape[0] - panel.shape[0], panel.shape[1], 3),
                                              GROUND, dtype=np.uint8)])
            cv.imshow(VIEW_WINDOW, np.hstack([display, panel]))
            cv.imshow(BINARY_WINDOW, binary)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                outcome = "redraw"
                break
            if key == ord("c"):
                _calibrate(camera, monitor)
            if key == ord("s"):
                cfg.binary_threshold = threshold
                cfg.detection_sensitivity = sensitivity
                save_config(cfg, config_path)
                print(f"Saved threshold {threshold} and sensitivity {sensitivity:.2f} "
                      f"to {config_path}")
    finally:
        camera.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
    return outcome


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Live camera and detection check for a session config. Records nothing.")
    p.add_argument("--config", required=True, help="Session config YAML")
    args = p.parse_args(argv)

    from amazeing.auditory.session_config import load_config
    from amazeing.auditory.vision import define_rois

    cfg = load_config(args.config)
    names = list(cfg.entrance_rois) + [str(i + 1) for i in range(cfg.rois_number)]
    os.makedirs(os.path.dirname(cfg.roi_csv_path) or ".", exist_ok=True)

    if not os.path.exists(cfg.roi_csv_path):
        print(f"No arm boxes yet. Draw {len(names)} in this order: {', '.join(names)}")
        define_rois(cfg.video_input, cfg.roi_csv_path, names)
        print(f"Saved {cfg.roi_csv_path}")

    while True:
        if _live_view(cfg, names, args.config) != "redraw":
            break
        print(f"Redrawing. Draw {len(names)} boxes in this order: {', '.join(names)}")
        define_rois(cfg.video_input, cfg.roi_csv_path, names)
        print(f"Saved {cfg.roi_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
