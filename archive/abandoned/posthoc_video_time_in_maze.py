import os, sys, argparse, logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import cv2 as cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

class MazeTimeAnalyzer:
    def __init__(self, rois_csv_path: str, threshold_factor: float = 0.5, manual_offset_secs: float = 0.0):
        self.threshold_factor = float(threshold_factor)
        self.rois_csv_path = rois_csv_path
        self.rois: Dict[str, Dict[str, int]] = {}  # name -> {xstart,ystart,xlen,ylen}
        self.manual_offset_secs = float(manual_offset_secs)

    def load_rois(self) -> None:
        if not os.path.exists(self.rois_csv_path):
            raise FileNotFoundError(f"ROIs file not found: {self.rois_csv_path}")

        df0 = pd.read_csv(self.rois_csv_path)
        df0.columns = df0.columns.map(str.lower)  # <— normalize column labels

        cols = set(df0.columns)
        if {"name","x","y","w","h"}.issubset(cols):
            # long form
            for _, r in df0.iterrows():
                name = str(r["name"]).lower()  # <— normalize ROI names too
                self.rois[name] = dict(xstart=int(r["x"]), ystart=int(r["y"]),
                                    xlen=int(r["w"]), ylen=int(r["h"]))
        else:
            # matrix form
            df = pd.read_csv(self.rois_csv_path, index_col=0)
            df.index = df.index.map(str.lower)     # <— normalize index row labels
            required = {"xstart","ystart","xlen","ylen"}
            if not required.issubset(set(df.index)):
                raise ValueError("Unknown ROIs CSV schema ...")
            for name in df.columns:
                key = str(name).lower()            # <— normalize ROI names
                self.rois[key] = dict(
                    xstart=int(df.loc["xstart", name]),
                    ystart=int(df.loc["ystart", name]),
                    xlen=int(df.loc["xlen", name]),
                    ylen=int(df.loc["ylen", name]),
                )

        # Now your entrance checks will always match:
        if "entrance1" not in self.rois or "entrance2" not in self.rois:
            log.warning("Expected 'entrance1' and 'entrance2' in ROIs.")


    @staticmethod
    def grab(frame, r):  # r: dict with xstart,ystart,xlen,ylen
        return frame[r["ystart"]:r["ystart"]+r["ylen"], r["xstart"]:r["xstart"]+r["xlen"]]

    def compute_thresholds(self, cap, num_frames=20, thresh_value=160) -> Dict[str, float]:
        thresholds = {k: 0.0 for k in self.rois}
        counts = 0
        pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok: break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
            for name, rect in self.rois.items():
                area = self.grab(bw, rect)
                thresholds[name] += float(np.sum(area))
            counts += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        if counts == 0:
            raise RuntimeError("Could not read frames to compute thresholds.")
        for k in thresholds: thresholds[k] /= counts
        return thresholds

    def analyze_video_for_maze_time(self, video_path: str, trials_df: pd.DataFrame,
                                    thresh_value: int = 160) -> Dict[int, float]:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv.CAP_PROP_FPS) or 0.0
        if fps <= 0: fps = 30.0
        dt_ms = 1000.0 / fps

        thresholds = self.compute_thresholds(cap, num_frames=20, thresh_value=thresh_value)
        log.info("Baselines computed.")

        # Build per-trial windows aligned to first start
        meta = (trials_df.groupby("trial_ID")[["trial_start_time","end_trial_time"]].first()).sort_index()
        if meta.isna().any().any():
            raise ValueError("Trials CSV missing trial_start_time/end_trial_time values.")
        first_start = float(meta["trial_start_time"].min())
        # rel seconds since first_start + manual offset
        trial_windows = {int(tid): (float(r["trial_start_time"] - first_start - self.manual_offset_secs),
                                    float(r["end_trial_time"]  - first_start - self.manual_offset_secs))
                         for tid, r in meta.iterrows()}

        # Entrance logic (match main.py semantics)
        ent1_hist = [False, False]
        ent2_hist = [False, False]
        hasLeft1 = False
        hasLeft2 = False
        e2Aftere1 = False
        e1Aftere2 = False
        enteredMaze = False

        inside_ms = {int(t): 0.0 for t in meta.index}
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok: break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)

            # Per-ROI presence
            mouse_present = {}
            for name, rect in self.rois.items():
                area = self.grab(bw, rect)
                factor = self.threshold_factor
                mouse_present[name] = (np.sum(area) < thresholds[name] * factor)

            # Update entrance histories (2-sample like main)
            ent1_hist.insert(0, mouse_present.get("entrance1", False)); ent1_hist.pop()
            ent2_hist.insert(0, mouse_present.get("entrance2", False)); ent2_hist.pop()

            # edge transitions
            if (not ent1_hist[0]) and ent1_hist[1]:
                hasLeft1 = True
                if hasLeft2:
                    e1Aftere2 = True; e2Aftere1 = False
            if (not ent2_hist[0]) and ent2_hist[1]:
                hasLeft2 = True
                if hasLeft1:
                    e1Aftere2 = False; e2Aftere1 = True

            if e2Aftere1 and not enteredMaze:
                enteredMaze = True
            if e1Aftere2 and enteredMaze:
                enteredMaze = False

            # attribute dt to current trial window if inside
            t_rel = frame_idx / fps
            if enteredMaze:
                for tid, (s_rel, e_rel) in trial_windows.items():
                    if s_rel <= t_rel < e_rel:
                        inside_ms[tid] += dt_ms
                        break

            # reset per-trial edge-state when leaving a window boundary
            # (optional: if your main.py carries state across trials, you can remove this)
            # if any(t_rel >= e for (_, e) in trial_windows.values()):
            #     hasLeft1 = hasLeft2 = False
            #     e2Aftere1 = e1Aftere2 = False
            #     enteredMaze = False

            frame_idx += 1

        cap.release()
        return inside_ms

    def process_session(self, session_dir: str) -> bool:
        sp = Path(session_dir)
        vids = list(sp.glob("*.mp4")) + list(sp.glob("*.avi")) + list(sp.glob("*.mov"))
        if not vids:
            log.error(f"No video in {sp}"); return False
        trials_files = list(sp.glob("trials_time_*.csv"))
        if not trials_files:
            log.error(f"No trials CSV in {sp}"); return False

        video_path = str(vids[0])
        trials_csv = str(trials_files[0])
        trials_df = pd.read_csv(trials_csv)

        inside_ms = self.analyze_video_for_maze_time(video_path, trials_df)
        # write back as 'time_in_maze_ms'
        trials_df["time_in_maze_ms"] = trials_df["trial_ID"].map(inside_ms)

        backup = trials_csv + ".bak"
        os.replace(trials_csv, backup)
        log.info(f"Backup created: {backup}")
        trials_df.to_csv(trials_csv, index=False)
        log.info(f"Wrote updated CSV: {trials_csv}")
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("--rois", required=True, help="Path to ROIs CSV (name,x,y,w,h or matrix schema)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--manual-offset-secs", type=float, default=0.0)
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        log.error(f"Base dir not found: {base}"); sys.exit(1)

    analyzer = MazeTimeAnalyzer(args.rois, threshold_factor=args.threshold,
                                manual_offset_secs=args.manual_offset_secs)
    analyzer.load_rois()
    log.info(f"Loaded ROIs from: {args.rois}")

    def looks_like_session_dir(p: Path) -> bool:
        # A session dir must have at least one video and one trials_time_*.csv in the SAME folder
        has_video = any(p.glob("*.mp4")) or any(p.glob("*.avi")) or any(p.glob("*.mov"))
        has_trials = any(p.glob("trials_time_*.csv"))
        return has_video and has_trials

    # -------- build session list (robust across layouts) --------
    sessions: list[str] = []

    # Case A: user pointed directly at a session folder
    if base.is_dir() and looks_like_session_dir(base):
        sessions.append(str(base))

    # Case B: scan one and two levels down (e.g., w1_d1 -> time_*mouse*)
    if not sessions:
        for level1 in base.iterdir():
            if level1.is_dir() and looks_like_session_dir(level1):
                sessions.append(str(level1))
            elif level1.is_dir():
                for level2 in level1.iterdir():
                    if level2.is_dir() and looks_like_session_dir(level2):
                        sessions.append(str(level2))

    # Case C: fallback recursive (thorough)
    if not sessions:
        for d in base.rglob("*"):
            if d.is_dir() and looks_like_session_dir(d):
                sessions.append(str(d))

    if not sessions:
        log.error(f"No session directories found under: {base}")
        sys.exit(1)

    log.info(f"Found {len(sessions)} session(s).")
    for s in sessions:
        log.info(f"  - {s}")

    # -------- process sessions --------
    ok, fail = 0, 0
    total = len(sessions)
    for i, s in enumerate(sessions, 1):
        log.info(f"[{i}/{total}] Processing: {s}")
        try:
            if analyzer.process_session(s):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            log.exception(f"Failed: {s} -> {e}")
            fail += 1

    log.info(f"Done. Success: {ok}, Failed: {fail}")



if __name__ == "__main__":
    main()
