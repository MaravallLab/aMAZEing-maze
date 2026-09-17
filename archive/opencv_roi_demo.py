"""
opencv_roi_demo.py
==================

Live OpenCV demo: select ROIs on a webcam feed, detect when something
enters them, and time how long it stays. Built as a stand-alone teaching
script (no project dependencies beyond opencv/numpy/pandas) so it can be
run cold in front of an audience.

How it works
------------
1. Camera starts and warms up (lets auto-exposure settle).
2. You draw one or more ROIs with the mouse.
3. We grab a "background" frame of the *empty* scene.
4. For every new frame we take abs-diff(frame, background), threshold it,
   and ask "what fraction of this ROI changed?". Above a fraction => occupied.
5. A debounce (enter/exit frame streaks, same idea as the lab's ROIMonitor)
   stops a flickering pixel from spamming false enter/exit events.
6. While an ROI is occupied we accumulate dwell time and draw it on screen.
7. On quit we dump a visitation log (one row per visit) + a per-ROI summary.

Why static background subtraction instead of MOG2?
--------------------------------------------------
MOG2 (the usual cv.createBackgroundSubtractor*) slowly *learns* a stationary
object into the background, so a hand/object that stops moving inside the ROI
would fade out and the dwell timer would stop. A fixed reference of the empty
scene keeps stationary objects detected, which is what we want for "time in
ROI". Press 'b' any time to re-grab the background if the lighting drifts.

Hotkeys (while the live window is focused)
------------------------------------------
    q / ESC : quit and write the logs
    b       : re-capture the empty-scene background
    r       : re-select the ROIs (resets timers + log)
    SPACE   : pause / resume the dwell timers
    m       : toggle the foreground-mask preview window

Run
---
    python archive/opencv_roi_demo.py                 # camera 0, sensible defaults
    python archive/opencv_roi_demo.py --camera 1
    python archive/opencv_roi_demo.py --roi-names door nest feeder
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, List

import cv2 as cv
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# ROI selection
# --------------------------------------------------------------------------- #
def select_rois(frame: np.ndarray, roi_names: List[str]) -> Dict[str, List[int]]:
    """Let the user drag boxes on a still frame. Returns {name: [x, y, w, h]}.

    We loop single-box cv.selectROI calls (one box each) rather than the
    plural cv.selectROIs, so we can print a running count after every
    confirmation and keep each confirmed box drawn on screen. You decide how
    many ROIs you want simply by drawing that many - press ESC to finish.
    """
    window_name = "Select ROIs  -  drag a box, SPACE/ENTER to confirm, ESC to finish"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)  # arbitrary, just so it's not tiny

    print("\n--- ROI SELECTION ---")
    print("  Drag a box, then SPACE or ENTER to confirm it.")
    print("  Each confirmed box is added and the running count is printed.")
    print("  Press ESC (or confirm an empty box) when you've drawn them all.\n")

    display = frame.copy()       # confirmed boxes get drawn here so they persist
    rois: Dict[str, List[int]] = {}

    while True:
        # selectROI blocks until you confirm (SPACE/ENTER) or cancel (ESC/'c').
        # On cancel it returns an all-zero rect, which is our "finished" signal.
        rect = cv.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        x, y, w, h = (int(v) for v in rect)

        if w == 0 or h == 0:
            break  # ESC / 'c' / stray click with no drag => done selecting

        # Use the supplied name if we have one for this slot, else auto-number.
        idx = len(rois)
        name = roi_names[idx] if idx < len(roi_names) else f"ROI{idx + 1}"
        rois[name] = [x, y, w, h]

        # Burn the confirmed box into the display frame so it stays visible
        # while you draw the next one.
        cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(display, name, (x, y - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"  Confirmed '{name}'  ->  {len(rois)} ROI(s) so far. "
              f"Draw another, or press ESC to finish.")

    cv.destroyWindow(window_name)

    if not rois:
        raise RuntimeError("No ROIs were selected - nothing to monitor.")

    print(f"\nDone: {len(rois)} ROI(s) selected -> {', '.join(rois)}")
    return rois


# --------------------------------------------------------------------------- #
# Background handling
# --------------------------------------------------------------------------- #
def grab_background(cap: cv.VideoCapture, n_frames: int = 30) -> np.ndarray:
    """Average several grayscale frames of the (ideally empty) scene.

    Averaging beats a single grab because it smooths out sensor noise, giving
    a cleaner reference to diff against.
    """
    print(f"Capturing background from {n_frames} frames - keep the scene empty...")
    acc = None
    grabbed = 0
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
        acc = gray if acc is None else acc + gray
        grabbed += 1
    if acc is None:
        raise RuntimeError("Could not read any frames to build the background.")
    background = (acc / grabbed).astype(np.uint8)
    print("Background captured.")
    return background


# --------------------------------------------------------------------------- #
# Detection + dwell-time tracking
# --------------------------------------------------------------------------- #
class ROITracker:
    """Detects occupancy per ROI (debounced) and accumulates dwell time."""

    def __init__(self,
                 rois: Dict[str, List[int]],
                 diff_threshold: int = 30,
                 min_area_fraction: float = 0.05,
                 enter_frames: int = 2,
                 exit_frames: int = 5):
        self.rois = rois
        self.diff_threshold = diff_threshold      # pixel intensity change to count as "changed"
        self.min_area_fraction = min_area_fraction  # fraction of ROI that must change to be "occupied"
        self.enter_frames = enter_frames          # debounce: frames of presence before ENTER fires
        self.exit_frames = exit_frames            # debounce: frames of absence before EXIT fires

        names = list(rois)
        self.is_occupied: Dict[str, bool] = {n: False for n in names}
        self.present_streak: Dict[str, int] = {n: 0 for n in names}
        self.absent_streak: Dict[str, int] = {n: 0 for n in names}

        # Timers (seconds), accumulated only while occupied.
        self.total_time: Dict[str, float] = {n: 0.0 for n in names}
        self.visit_start: Dict[str, float] = {n: 0.0 for n in names}
        self.visit_count: Dict[str, int] = {n: 0 for n in names}

        # One row per completed visit -> becomes the visitation log.
        self.visit_log: List[dict] = []

    def _roi_occupied_raw(self, mask: np.ndarray, name: str) -> bool:
        """Is more than `min_area_fraction` of this ROI lit up in the mask?"""
        x, y, w, h = self.rois[name]
        crop = mask[y:y + h, x:x + w]
        if crop.size == 0:
            return False
        # mask is 0/255, so mean/255 is the fraction of "changed" pixels.
        return (np.count_nonzero(crop) / crop.size) > self.min_area_fraction

    def update(self, mask: np.ndarray, now: float, paused: bool) -> None:
        """Advance the state machine for one frame.

        `mask`  : binary foreground mask (0/255) for the whole frame.
        `now`   : current timestamp (time.time()).
        `paused`: if True, freeze the dwell timers (still detect, just don't count).
        """
        for name in self.rois:
            raw_present = self._roi_occupied_raw(mask, name)

            # Update debounce streaks.
            if raw_present:
                self.present_streak[name] += 1
                self.absent_streak[name] = 0
            else:
                self.absent_streak[name] += 1
                self.present_streak[name] = 0

            was_occupied = self.is_occupied[name]

            # --- ENTER ---
            if not was_occupied and self.present_streak[name] >= self.enter_frames:
                self.is_occupied[name] = True
                self.visit_start[name] = now
                self.visit_count[name] += 1
                print(f"  [ROI] ENTER  {name}  (visit #{self.visit_count[name]})")

            # --- EXIT ---
            elif was_occupied and self.absent_streak[name] >= self.exit_frames:
                self.is_occupied[name] = False
                duration = now - self.visit_start[name]
                self.visit_log.append({
                    "roi": name,
                    "visit_number": self.visit_count[name],
                    "enter_time_s": round(self.visit_start[name], 3),
                    "exit_time_s": round(now, 3),
                    "duration_s": round(duration, 3),
                })
                print(f"  [ROI] EXIT   {name}  (visit lasted {duration:.2f}s)")

            # --- accumulate dwell time while occupied ---
            if self.is_occupied[name] and not paused:
                # Add the time since the visit's last accounting point.
                self.total_time[name] += now - max(self.visit_start[name], self._last_tick.get(name, now))

        # Remember when we last counted, so the next frame's delta is correct.
        self._last_tick = {n: now for n in self.rois}

    # _last_tick is created lazily on first update()
    _last_tick: Dict[str, float] = {}

    def close_open_visits(self, now: float) -> None:
        """At quit time, flush any ROI that's still occupied into the log."""
        for name in self.rois:
            if self.is_occupied[name]:
                duration = now - self.visit_start[name]
                self.visit_log.append({
                    "roi": name,
                    "visit_number": self.visit_count[name],
                    "enter_time_s": round(self.visit_start[name], 3),
                    "exit_time_s": round(now, 3),
                    "duration_s": round(duration, 3),
                })


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_overlay(frame: np.ndarray, tracker: ROITracker, fps: float, paused: bool) -> None:
    """Draw ROI boxes (green=empty, red=occupied) + live dwell timers."""
    for name, (x, y, w, h) in tracker.rois.items():
        occupied = tracker.is_occupied[name]
        colour = (0, 0, 255) if occupied else (0, 200, 0)  # BGR: red / green

        cv.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = f"{name}: {tracker.total_time[name]:.1f}s"
        # Background strip behind the text so it stays readable over the video.
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
        cv.putText(frame, label, (x + 3, y - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Top banner: FPS, pause state, and the hotkey reminder.
    banner = f"FPS:{fps:4.1f}   {'PAUSED' if paused else 'RUNNING'}   " \
             f"[q]uit  [b]ackground  [r]eselect  [space]pause  [m]ask"
    cv.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv.putText(frame, banner, (8, 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_logs(tracker: ROITracker, output_dir: str) -> None:
    """Write the visitation log (per visit) and a per-ROI summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Visitation log: one row per visit.
    log_path = os.path.join(output_dir, f"roi_visitation_log_{stamp}.csv")
    log_df = pd.DataFrame(
        tracker.visit_log,
        columns=["roi", "visit_number", "enter_time_s", "exit_time_s", "duration_s"],
    )
    log_df.to_csv(log_path, index=False)

    # 2) Summary: total time and visit count per ROI.
    summary = [{
        "roi": name,
        "total_time_s": round(tracker.total_time[name], 3),
        "n_visits": tracker.visit_count[name],
    } for name in tracker.rois]
    summary_path = os.path.join(output_dir, f"roi_summary_{stamp}.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("\n--- SESSION SUMMARY ---")
    for row in summary:
        print(f"  {row['roi']:<12} total={row['total_time_s']:.2f}s  visits={row['n_visits']}")
    print(f"\nVisitation log : {log_path}")
    print(f"Summary        : {summary_path}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Live OpenCV ROI dwell-time demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--output-dir", default="archive/roi_demo_logs",
                        help="Where to write the CSV logs.")
    parser.add_argument("--roi-names", nargs="*", default=[],
                        help="Optional names for the ROIs, in selection order.")
    parser.add_argument("--diff-threshold", type=int, default=30,
                        help="Pixel intensity change counted as motion (0-255).")
    parser.add_argument("--min-area", type=float, default=0.05,
                        help="Fraction of an ROI that must change to be 'occupied'.")
    parser.add_argument("--enter-frames", type=int, default=2,
                        help="Frames of presence before an ENTER is registered.")
    parser.add_argument("--exit-frames", type=int, default=5,
                        help="Frames of absence before an EXIT is registered.")
    args = parser.parse_args()

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}.")

    # Warm up so auto-exposure/white-balance settle before we grab anything.
    for _ in range(15):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read a frame from the camera.")

    # Step 1: pick ROIs on a still frame.
    rois = select_rois(frame, args.roi_names)

    # Step 2: reference background of the empty scene.
    background = grab_background(cap)

    # Step 3: tracker.
    tracker = ROITracker(
        rois,
        diff_threshold=args.diff_threshold,
        min_area_fraction=args.min_area,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
    )

    window_name = "ROI dwell-time demo"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)

    paused = False
    show_mask = False
    fps = 0.0
    prev_t = time.time()

    print("\nMonitoring... (focus the video window for hotkeys)\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost the camera feed - stopping.")
            break

        now = time.time()
        # Smoothed FPS for the banner.
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # --- background subtraction ---
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(gray, background)
        _, mask = cv.threshold(diff, args.diff_threshold, 255, cv.THRESH_BINARY)
        # Morphological opening kills speckle noise so single hot pixels don't
        # trip the detector.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # --- update state + timers ---
        tracker.update(mask, now, paused)

        # --- draw ---
        draw_overlay(frame, tracker, fps, paused)
        cv.imshow(window_name, frame)
        if show_mask:
            cv.imshow("foreground mask", mask)
        elif cv.getWindowProperty("foreground mask", cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow("foreground mask")

        # --- hotkeys ---
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):              # q or ESC
            break
        elif key == ord("b"):
            background = grab_background(cap)
        elif key == ord("r"):
            ret, still = cap.read()
            if ret:
                rois = select_rois(still, args.roi_names)
                tracker = ROITracker(
                    rois,
                    diff_threshold=args.diff_threshold,
                    min_area_fraction=args.min_area,
                    enter_frames=args.enter_frames,
                    exit_frames=args.exit_frames,
                )
                background = grab_background(cap)
        elif key == ord(" "):
            paused = not paused
            print(f"  [timers {'PAUSED' if paused else 'RESUMED'}]")
        elif key == ord("m"):
            show_mask = not show_mask

    # Flush any visit still open, write logs, clean up.
    tracker.close_open_visits(time.time())
    write_logs(tracker, args.output_dir)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
