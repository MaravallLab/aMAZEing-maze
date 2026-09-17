import os, sys, argparse, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import cv2 as cv

# run as python automatic_roi_selection_posthoc_time_in_maze.py "C:\Users\shahd\Box\Awake Project\Maze data\Auditory experiments\8_arms_w_voc (1)" --save-overlays --no-review-rois


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("posthoc_maze")

# define rois names
ROI_NAMES = ["entrance1", "entrance2"]


def pick_best_video(sp: Path) -> Optional[str]:
    vids = [*sp.glob("*.mp4"), *sp.glob("*.avi"), *sp.glob("*.mov")]
    if not vids:
        return None
    vids.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(vids[0])

def open_video_any(path: str):
    for backend in (cv.CAP_MSMF, cv.CAP_FFMPEG, cv.CAP_DSHOW, cv.CAP_ANY):
        cap = cv.VideoCapture(path, backend)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            return cap
        try:
            cap.release()
        except Exception:
            pass
    return None

def grab_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = open_video_any(video_path)
    if cap is None:
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

# save the rois.csv in parent directory
def save_rois_csv(dest_csv: Path, rows: List[Tuple[str,int,int,int,int]]) -> None:
    df = pd.DataFrame(rows, columns=["name","x","y","w","h"])
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False)

# draw the rois on the first frame, but scale the frame x2
def draw_rois_on_first_frame(video_path: str, scale: float = 2.0) -> List[Tuple[str,int,int,int,int]]:
    """Draw ordered ROIs with labels, zoomed view; returns list of (name,x,y,w,h) in original coords."""
    cap = open_video_any(video_path)
    if cap is None:
        raise RuntimeError(f"Could not open video for ROI drawing: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame for ROI drawing: {video_path}")

    disp = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    win = "Draw ROIs - ENTER to confirm, ESC to skip"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, disp.shape[1], disp.shape[0])
    try:
        cv.setWindowProperty(win, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    selections: List[Tuple[str,int,int,int,int]] = []
    overlays: List[Tuple[str,int,int,int,int]] = []  # scaled copies for context

    for name in ROI_NAMES:
        frame_show = disp.copy()
        # show previously drawn
        for nm, sx, sy, sw, sh in overlays:
            cv.rectangle(frame_show, (sx, sy), (sx+sw, sy+sh), (255,0,0), 2)
            cv.putText(frame_show, nm, (sx, max(20, sy-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        # instruction
        cv.putText(frame_show, f"Draw {name} then ENTER (ESC to skip)", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv.LINE_AA)

        r = cv.selectROI(win, frame_show, fromCenter=False, showCrosshair=True)
        if r == (0,0,0,0):
            log.warning(f"Skipped ROI: {name}")
            continue
        x = int(r[0] / scale); y = int(r[1] / scale)
        w = int(r[2] / scale); h = int(r[3] / scale)
        selections.append((name.lower(), x, y, w, h))
        overlays.append((name, r[0], r[1], r[2], r[3]))

    cv.destroyAllWindows()
    if not selections:
        raise RuntimeError("No ROIs were drawn.")
    return selections

def load_ref_rois_long(csv_path: str) -> List[Tuple[str,int,int,int,int]]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.map(str.lower)
    need = {"name","x","y","w","h"}
    if not need.issubset(df.columns):
        raise ValueError(f"ROI CSV must have columns {need}")
    rows = []
    for name, x, y, w, h in df[["name","x","y","w","h"]].itertuples(index=False, name=None):
        rows.append((str(name).lower(), int(x), int(y), int(w), int(h)))
    return rows

def overlay_rois(image: np.ndarray, rows: List[Tuple[str,int,int,int,int]]) -> np.ndarray:
    out = image.copy()
    for name, x, y, w, h in rows:
        cv.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)
        cv.putText(out, name, (x, max(20, y-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)
    return out

# -------- Homography ROI transfer --------

def detect_homography(ref_img: np.ndarray, cur_img: np.ndarray) -> Optional[np.ndarray]:
    g1 = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
    g2 = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY) if cur_img.ndim == 3 else cur_img
    orb = cv.ORB_create(nfeatures=5000, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 20:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0)
    if H is None:
        return None
    if mask is not None and int(mask.sum()) < 20:  # require inliers
        return None
    return H

def project_rect(x: int, y: int, w: int, h: int, H: np.ndarray) -> Tuple[int,int,int,int]:
    pts = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=np.float32).reshape(-1,1,2)
    proj = cv.perspectiveTransform(pts, H).reshape(-1,2)
    xs, ys = proj[:,0], proj[:,1]
    x0 = max(0, int(np.floor(xs.min()))); y0 = max(0, int(np.floor(ys.min())))
    x1 = int(np.ceil(xs.max())); y1 = int(np.ceil(ys.max()))
    return x0, y0, max(1, x1-x0), max(1, y1-y0)

def auto_rois_from_reference(cur_img: np.ndarray,
                             ref_img: np.ndarray,
                             ref_rows: List[Tuple[str,int,int,int,int]],
                             scale: float = 1.0) -> Optional[List[Tuple[str,int,int,int,int]]]:
    # Optionally downscale for speed, then unscale H
    if scale != 1.0:
        def S(a): return np.array([[a,0,0],[0,a,0],[0,0,1]], dtype=np.float32)
        ref_s = cv.resize(ref_img, None, fx=scale, fy=scale) if scale != 1.0 else ref_img
        cur_s = cv.resize(cur_img, None, fx=scale, fy=scale) if scale != 1.0 else cur_img
        Hs = detect_homography(ref_s, cur_s)
        if Hs is None:
            return None
        H = np.linalg.inv(S(scale)) @ Hs @ S(scale)  # map ref(orig) -> cur(orig)
    else:
        H = detect_homography(ref_img, cur_img)
        if H is None:
            return None
    out = []
    for name, x, y, w, h in ref_rows:
        x2, y2, w2, h2 = project_rect(x, y, w, h, H)
        out.append((name, x2, y2, w2, h2))
    return out

# ---------- Homography sanity checks + interactive review ----------

def _bbox_out_of_bounds_ratio(x:int, y:int, w:int, h:int, W:int, H:int) -> float:
    # fraction of the bbox area that lies outside image bounds
    x0, y0, x1, y1 = x, y, x + w, y + h
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(W, x1), min(H, y1)
    inside_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    area = max(1, w * h)
    return 1.0 - (inside_area / area)

def sanity_check_projected_rois(
    rows: List[Tuple[str,int,int,int,int]],
    img_shape: Tuple[int, int, int],
    min_area: int = 100,                     # pixel^2
    max_area_ratio: float = 0.40,            # <= 40% of frame
    max_oob_ratio: float = 0.25              # <= 25% outside image
) -> Tuple[bool, List[str]]:
    """Basic geometry checks to catch obviously-bad homographies."""
    H, W = img_shape[:2]
    frame_area = float(H * W)
    issues = []
    ok = True
    for name, x, y, w, h in rows:
        area = max(0, w) * max(0, h)
        if w <= 0 or h <= 0:
            ok = False; issues.append(f"{name}: non-positive size ({w}x{h})")
            continue
        if area < min_area:
            ok = False; issues.append(f"{name}: area too small ({area}px^2)")
        if area > max_area_ratio * frame_area:
            ok = False; issues.append(f"{name}: area too large ({area}px^2)")
        oob = _bbox_out_of_bounds_ratio(x, y, w, h, W, H)
        if oob > max_oob_ratio:
            ok = False; issues.append(f"{name}: {oob*100:.1f}% outside frame")
    return ok, issues

def draw_rois_on_image(image: np.ndarray, scale: float = 2.0,
                       names: Optional[List[str]] = None) -> List[Tuple[str,int,int,int,int]]:
    """Draw ordered ROIs on a still image; returns list (name,x,y,w,h) in original coords."""
    if names is None:
        names = ROI_NAMES
    disp = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    win = "Draw ROIs (image) - ENTER to confirm each, ESC to skip"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, disp.shape[1], disp.shape[0])
    try:
        cv.setWindowProperty(win, cv.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    selections: List[Tuple[str,int,int,int,int]] = []
    overlays: List[Tuple[str,int,int,int,int]] = []
    for name in names:
        frame_show = disp.copy()
        for nm, sx, sy, sw, sh in overlays:
            cv.rectangle(frame_show, (sx, sy), (sx+sw, sy+sh), (255,0,0), 2)
            cv.putText(frame_show, nm, (sx, max(20, sy-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
        cv.putText(frame_show, f"Draw {name} then ENTER (ESC to skip)",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv.LINE_AA)

        r = cv.selectROI(win, frame_show, fromCenter=False, showCrosshair=True)
        if r == (0,0,0,0):
            log.warning(f"Skipped ROI: {name}")
            continue
        x = int(r[0] / scale); y = int(r[1] / scale)
        w = int(r[2] / scale); h = int(r[3] / scale)
        selections.append((name.lower(), x, y, w, h))
        overlays.append((name, r[0], r[1], r[2], r[3]))

    cv.destroyWindow(win)
    if not selections:
        raise RuntimeError("No ROIs were drawn on image.")
    return selections

def _overlay_rois_with_indices(image: np.ndarray, rows: List[Tuple[str,int,int,int,int]]) -> np.ndarray:
    # Show "index:name" to make single-ROI edits easy
    out = image.copy()
    for i, (name, x, y, w, h) in enumerate(rows, start=1):
        label = f"{i}:{name}"
        cv.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)
        cv.putText(out, label, (x, max(20, y-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)
    return out

def review_rois_interactive(cur_img: np.ndarray,
                            rows: List[Tuple[str,int,int,int,int]],
                            draw_scale: float = 2.0,
                            issues: Optional[List[str]] = None) -> Optional[List[Tuple[str,int,int,int,int]]]:
    """
    Preview auto-ROIs; keys:
      ENTER = accept
      R     = redraw ALL on image
      E     = edit a single ROI by its index key (1..9, 0=10)
      ESC   = cancel (return None)
    """
    rows = [(str(n), int(x), int(y), int(w), int(h)) for (n,x,y,w,h) in rows]  # copy/normalize
    win = "ROI review - ENTER accept | R redraw all | E edit one | ESC cancel"
    while True:
        preview = _overlay_rois_with_indices(cur_img, rows)
        if issues:
            y0 = 30
            for msg in issues[:8]:
                cv.putText(preview, f"⚠ {msg}", (20, y0), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv.LINE_AA)
                y0 += 28
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.imshow(win, preview)
        k = cv.waitKey(0) & 0xFF

        if k in (13, 10):  # ENTER
            cv.destroyWindow(win)
            return rows
        elif k in (27, ):   # ESC
            cv.destroyWindow(win)
            return None
        elif k in (ord('r'), ord('R')):
            try:
                rows = draw_rois_on_image(cur_img, scale=draw_scale, names=[n for n,_,_,_,_ in rows])
                issues = None
            except Exception as e:
                log.exception(f"Redraw-all failed: {e}")
        elif k in (ord('e'), ord('E')):
            cv.putText(preview, "Press index to edit (1..9, 0=10). Press any other key to exit edit.",
                       (20, preview.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
            cv.imshow(win, preview)
            k2 = cv.waitKey(0) & 0xFF
            idx = None
            if ord('1') <= k2 <= ord('9'):
                idx = k2 - ord('1')  # 0..8
            elif k2 == ord('0'):
                idx = 9
            if idx is not None and 0 <= idx < len(rows):
                name = rows[idx][0]
                try:
                    new_roi = draw_rois_on_image(cur_img, scale=draw_scale, names=[name])
                    for j, (nm, *_rest) in enumerate(rows):
                        if nm == name:
                            rows[j] = new_roi[0]
                            break
                    ok_after, issues = sanity_check_projected_rois(rows, cur_img.shape)
                except Exception as e:
                    log.exception(f"Edit for {name} failed: {e}")
        else:
            pass

# ===================== Analyzer (time in maze) =====================

class MazeTimeAnalyzer:
    def __init__(self, threshold_factor: float = 0.5, manual_offset_secs: float = 0.0):
        self.threshold_factor = float(threshold_factor)
        self.manual_offset_secs = float(manual_offset_secs)
        self.rois: Dict[str, Dict[str,int]] = {}

    def set_rois_from_long(self, rows: List[Tuple[str,int,int,int,int]]):
        self.rois = {}
        for name, x, y, w, h in rows:
            key = str(name).lower()
            self.rois[key] = dict(xstart=int(x), ystart=int(y), xlen=int(w), ylen=int(h))
        if "entrance1" not in self.rois or "entrance2" not in self.rois:
            log.warning("Expected 'entrance1' and 'entrance2' in ROIs.")

    @staticmethod
    def grab(frame, r):
        return frame[r["ystart"]:r["ystart"]+r["ylen"], r["xstart"]:r["xstart"]+r["xlen"]]

    def compute_thresholds(self, cap, num_frames=10, thresh_value=160) -> Dict[str,float]:
        thresholds = {k: 0.0 for k in self.rois}; n = 0
        pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
            for name, rect in self.rois.items():
                thresholds[name] += float(np.sum(self.grab(bw, rect)))
            n += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, pos)
        if n == 0:
            raise RuntimeError("Could not read frames to compute thresholds.")
        for k in thresholds:
            thresholds[k] /= n
        return thresholds

    def analyze_video_for_maze_time(self, video_path: str, trials_df: pd.DataFrame, thresh_value: int = 160) -> Dict[int,float]:
        cap = open_video_any(video_path)
        if cap is None:
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        dt_ms = 1000.0 / fps

        thresholds = self.compute_thresholds(cap, num_frames=10, thresh_value=thresh_value)
        log.info("Baselines computed.")
        meta = (trials_df.groupby("trial_ID")[["trial_start_time","end_trial_time"]].first()).sort_index()
        if meta.isna().any().any():
            raise ValueError("Trials CSV missing trial_start_time/end_trial_time.")
        first_start = float(meta["trial_start_time"].min())
        trial_windows = {int(tid): (float(r["trial_start_time"] - first_start - self.manual_offset_secs),
                                    float(r["end_trial_time"] - first_start - self.manual_offset_secs))
                         for tid, r in meta.iterrows()}
        ent1_hist=[False,False]; ent2_hist=[False,False]
        hasLeft1=False; hasLeft2=False
        e2Aftere1=False; e1Aftere2=False; enteredMaze=False
        inside_ms = {int(t):0.0 for t in meta.index}
        frame_idx = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim==3 else frame
            _, bw = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
            mp = {}
            for name, rect in self.rois.items():
                mp[name] = (np.sum(self.grab(bw, rect)) < thresholds[name]*self.threshold_factor)
            ent1_hist.insert(0, mp.get("entrance1", False)); ent1_hist.pop()
            ent2_hist.insert(0, mp.get("entrance2", False)); ent2_hist.pop()
            if (not ent1_hist[0]) and ent1_hist[1]:
                hasLeft1 = True
                if hasLeft2: e1Aftere2=True; e2Aftere1=False
            if (not ent2_hist[0]) and ent2_hist[1]:
                hasLeft2 = True
                if hasLeft1: e1Aftere2=False; e2Aftere1=True
            if e2Aftere1 and not enteredMaze: enteredMaze=True
            if e1Aftere2 and enteredMaze: enteredMaze=False
            t_rel = frame_idx/fps
            if enteredMaze:
                for tid,(s_rel,e_rel) in trial_windows.items():
                    if s_rel <= t_rel < e_rel:
                        inside_ms[tid] += dt_ms
                        break
            frame_idx += 1
        cap.release()
        return inside_ms

# ===================== Session discovery =====================

def looks_like_session_dir(p: Path) -> bool:
    has_video = any(p.glob("*.mp4")) or any(p.glob("*.avi")) or any(p.glob("*.mov"))
    has_trials = any(p.glob("trials_time_*.csv"))
    return has_video and has_trials

def collect_sessions(base: Path) -> List[str]:
    sessions: List[str] = []
    if base.is_dir() and looks_like_session_dir(base):
        sessions.append(str(base))
        return sessions
    for level1 in base.iterdir():
        if level1.is_dir() and looks_like_session_dir(level1):
            sessions.append(str(level1))
        elif level1.is_dir():
            for level2 in level1.iterdir():
                if level2.is_dir() and looks_like_session_dir(level2):
                    sessions.append(str(level2))
    if not sessions:
        for d in base.rglob("*"):
            if d.is_dir() and looks_like_session_dir(d):
                sessions.append(str(d))
    return sessions

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lower, strip, replace spaces/dashes with underscores
    mapping = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    return df.rename(columns=mapping)

def _maybe_ms_to_s(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # if values look like milliseconds, convert to seconds
    if s.dropna().max() and s.dropna().max() > 1e6:
        s = s / 1000.0
    return s

def _coerce_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has: trial_ID, trial_start_time, end_trial_time.
    If start/end are missing but a per-trial duration exists, fabricate a
    consecutive schedule: start(t) = cumulative sum of prior durations, end = start + duration.
    """
    df = _norm_cols(df)

    # --- find id column ---
    id_candidates = "trial_ID"

    # --- find existing start/end columns, if any ---
    start_candidates = ["trial_start_time"]
    end_candidates   = ["end_trial_time", "trial_end_time"]

    startc = next((c for c in start_candidates if c in df.columns), None)
    endc   = next((c for c in end_candidates   if c in df.columns), None)

    # --- duration candidates (per-trial) ---
    dur_candidates = [
        "total_time_trial_s", "trial_duration", "trial_duration_s",
        "duration", "dur_sec", "len_sec", "trial_len", "trial_length"
    ]
    durc = next((c for c in dur_candidates if c in df.columns), None)

    out = df.copy()

    # we’ll create a per-trial table to avoid duplicates across ROI rows
    # 1) get a per-trial duration if we have it
    if durc is not None:
        # pick first occurrence per trial (or you could use max/mean if that’s more appropriate)
        trial_dur = out.groupby("trial_ID")[durc].first()
        trial_dur = _maybe_ms_to_s(trial_dur).fillna(0.0)
    else:
        trial_dur = None

    # 2) decide how to build start/end
    have_start = startc is not None
    have_end   = endc is not None

    trial_meta = pd.DataFrame(index=out["trial_ID"].dropna().unique())
    trial_meta.index.name = "trial_ID"

    # attach duration if present
    if trial_dur is not None:
        trial_meta["duration_s"] = trial_dur
    else:
        trial_meta["duration_s"] = np.nan  # may stay NaN if we really can't infer

    if have_start:
        s = out.groupby("trial_ID")[startc].first()
        trial_meta["trial_start_time"] = _maybe_ms_to_s(s)
    if have_end:
        e = out.groupby("trial_ID")[endc].first()
        trial_meta["end_trial_time"] = _maybe_ms_to_s(e)

    # 3) fabricate missing values if we can
    # case A: no start & no end, but we DO have durations -> fabricate consecutive schedule
    if (not have_start) and (not have_end) and trial_meta["duration_s"].notna().any():
        # sort trials by numeric trial_ID if possible, else by natural order
        try:
            order = sorted(trial_meta.index, key=lambda x: float(x))
        except Exception:
            order = sorted(trial_meta.index, key=lambda x: str(x))

        starts = []
        t = 0.0
        for tid in order:
            starts.append((tid, t))
            d = float(trial_meta.loc[tid, "duration_s"]) if pd.notna(trial_meta.loc[tid, "duration_s"]) else 0.0
            t += max(0.0, d)
        start_series = pd.Series(dict(starts), name="trial_start_time")
        trial_meta["trial_start_time"] = start_series
        trial_meta["end_trial_time"]   = trial_meta["trial_start_time"] + trial_meta["duration_s"].fillna(0.0)

    # case B: have start & duration but no end -> end = start + duration
    if have_start and (not have_end) and trial_meta["duration_s"].notna().any():
        trial_meta["end_trial_time"] = trial_meta["trial_start_time"] + trial_meta["duration_s"].fillna(0.0)

    # case C: have end & duration but no start -> start = end - duration
    if have_end and (not have_start) and trial_meta["duration_s"].notna().any():
        trial_meta["trial_start_time"] = trial_meta["end_trial_time"] - trial_meta["duration_s"].fillna(0.0)

    # final check: must have both columns now
    if "trial_start_time" not in trial_meta.columns or "end_trial_time" not in trial_meta.columns:
        # As a last resort, create zero-length placeholders (not ideal, but unblocks the pipeline)
        trial_meta["trial_start_time"] = trial_meta.get("trial_start_time", pd.Series(0.0, index=trial_meta.index))
        trial_meta["end_trial_time"]   = trial_meta.get("end_trial_time",   pd.Series(0.0, index=trial_meta.index))

    # 4) map back to all rows
    out = out.merge(trial_meta[["trial_start_time", "end_trial_time"]], left_on="trial_ID", right_index=True, how="left")

    # safety: ensure numeric
    out["trial_start_time"] = pd.to_numeric(out["trial_start_time"], errors="coerce").fillna(0.0)
    out["end_trial_time"]   = pd.to_numeric(out["end_trial_time"],   errors="coerce").fillna(out["trial_start_time"])

    return out



# ===================== Main workflow =====================

def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc time-in-maze with reference-drawn ROIs auto-adapted via homography."
    )
    ap.add_argument("base_dir", help="Top-level folder, a day folder, or a single session folder.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mouse-present factor vs baseline (default 0.5).")
    ap.add_argument("--manual-offset-secs", type=float, default=0.0, help="Shift video time relative to trial epoch seconds.")
    ap.add_argument("--draw-scale", type=float, default=2.0, help="Zoom factor for ROI drawing window.")

    # Mutually exclusive review toggles
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--review-rois", dest="review_rois", action="store_true",
                     help="Show an interactive overlay to accept/edit/redraw ROIs after homography.")
    grp.add_argument("--no-review-rois", dest="review_rois", action="store_false",
                     help="Disable interactive ROI review.")
    ap.set_defaults(review_rois=True)  # default = on

    ap.add_argument("--auto-scale", type=float, default=1.0, help="Scale images before matching (speed). 1.0 = native.")
    ap.add_argument("--no-manual-fallback", action="store_true",
                    help="If set, sessions where auto-registration fails will be marked failed (no manual drawing).")
    ap.add_argument("--save-overlays", action="store_true", help="Save PNG overlays of ROIs per session for QC.")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        log.error(f"Base dir not found: {base}")
        sys.exit(1)

    sessions = collect_sessions(base)
    if not sessions:
        log.error(f"No session directories found under: {base}")
        sys.exit(1)

    sessions.sort()
    log.info(f"Found {len(sessions)} session(s).")
    for s in sessions:
        log.info(f"  - {s}")

    analyzer = MazeTimeAnalyzer(threshold_factor=args.threshold, manual_offset_secs=args.manual_offset_secs)
    failed: List[Tuple[str,str]] = []
    total = len(sessions)

    # ---------- 1) Reference session: draw ROIs once ----------
    ref_session = Path(sessions[0])
    log.info(f"[1/{total}] Reference session (draw ROIs): {ref_session}")

    ref_video = pick_best_video(ref_session)
    if not ref_video:
        log.error("No video found for reference session.")
        failed.append((str(ref_session), "no_video_found"))
        print("\n=== FAILED SESSIONS ===")
        for s, why in failed: print(f"- {s}\n    reason: {why}")
        sys.exit(1)

    # 1) Draw ROIs on the reference video
    try:
        ref_rows = draw_rois_on_first_frame(ref_video, scale=args.draw_scale)
    except Exception as e:
        log.exception(f"ROI drawing failed for reference: {e}")
        failed.append((str(ref_session), "ref_roi_drawing_failed"))
        print("\n=== FAILED SESSIONS ===")
        for s, why in failed: print(f"- {s}\n    reason: {why}")
        sys.exit(1)

    # 2) Grab the first frame NOW (for overlays/review/homography)
    ref_img = grab_first_frame(ref_video)
    if ref_img is None:
        log.error("Could not read reference frame for homography.")
        print("\n=== FAILED SESSIONS ===")
        for s, why in failed: print(f"- {s}\n    reason: {why}")
        sys.exit(1)

    # 3) Optional overlay (QC)
    if args.save_overlays:
        ov = overlay_rois(ref_img, ref_rows)
        cv.imwrite(str(ref_session / "_ref_rois_overlay.png"), ov)

    # 4) Optional interactive review
    if args.review_rois:
        ok_ref, ref_issues = sanity_check_projected_rois(ref_rows, ref_img.shape)
        reviewed = review_rois_interactive(ref_img, ref_rows, draw_scale=args.draw_scale,
                                           issues=(None if ok_ref else ref_issues))
        if reviewed is not None:
            ref_rows = reviewed

    # 5) Save the reference ROIs
    ref_rois_csv = ref_session / "rois1.csv"
    save_rois_csv(ref_rois_csv, ref_rows)
    log.info(f"Saved reference ROIs to: {ref_rois_csv}")

    # Analyze reference session
    ref_trials_files = list(ref_session.glob("trials_time_*.csv"))
    if not ref_trials_files:
        log.error("No trials CSV found in reference session.")
        failed.append((str(ref_session), "no_trials_csv"))
    else:
        ref_trials_csv = str(ref_trials_files[0])
        ref_df_raw = pd.read_csv(ref_trials_csv)
        ref_df = _coerce_trial_columns(ref_df_raw)

        analyzer.set_rois_from_long(ref_rows)
        try:
            inside_ms = analyzer.analyze_video_for_maze_time(ref_video, ref_df)
            ref_df["time_in_maze_ms"] = ref_df["trial_ID"].map(inside_ms)
            backup = ref_trials_csv + ".bak"
            os.replace(ref_trials_csv, backup)
            log.info(f"Backup created: {backup}")
            ref_df.to_csv(ref_trials_csv, index=False)
            log.info(f"Wrote updated CSV: {ref_trials_csv}")
        except Exception as e:
            log.exception(f"Reference analysis failed: {e}")
            failed.append((str(ref_session), f"analysis_failed: {e}"))

    # ---------- 2) Process remaining sessions with auto-ROIs ----------
    for idx, sdir in enumerate(sessions[1:], start=2):
        log.info(f"[{idx}/{total}] Processing: {sdir}")
        sp = Path(sdir)

        video = pick_best_video(sp)
        if not video:
            log.error("No video found.")
            failed.append((sdir, "no_video_found"))
            continue

        trials_files = list(sp.glob("trials_time_*.csv"))
        if not trials_files:
            log.error("No trials CSV found.")
            failed.append((sdir, "no_trials_csv"))
            continue
        trials_csv = str(trials_files[0])
        df_raw = pd.read_csv(trials_csv)
        df = _coerce_trial_columns(df_raw) 

        cur_img = grab_first_frame(video)
        if cur_img is None:
            log.error("Could not read first frame (video open failed).")
            failed.append((sdir, "cannot_open_video"))
            continue

        # Auto-generate ROIs via homography from reference
        rows = auto_rois_from_reference(cur_img, ref_img, ref_rows, scale=args.auto_scale)
        if rows is None:
            reason = "auto_homography_failed"
            if args.no_manual_fallback:
                log.error("Auto-ROI failed; skipping (no manual fallback).")
                failed.append((sdir, reason))
                continue
            # Manual fallback: draw all on the video frame interactively
            try:
                log.info("Auto-ROI failed; launching manual ROI drawer...")
                rows = draw_rois_on_first_frame(video, scale=args.draw_scale)
            except Exception as e:
                log.exception(f"Manual ROI drawing failed: {e}")
                failed.append((sdir, f"{reason}+manual_failed"))
                continue

        # Sanity check & interactive review before committing
        ok_auto, auto_issues = sanity_check_projected_rois(rows, cur_img.shape)
        if args.review_rois:
            reviewed = review_rois_interactive(cur_img, rows, draw_scale=args.draw_scale,
                                               issues=(None if ok_auto else auto_issues))
            if reviewed is None:
                if not args.no_manual_fallback:
                    try:
                        log.info("User cancelled review; opening manual redraw...")
                        rows = draw_rois_on_first_frame(video, scale=args.draw_scale)
                    except Exception as e:
                        log.exception(f"Manual redraw after cancel failed: {e}")
                        failed.append((sdir, "user_cancelled_review"))
                        continue
                else:
                    log.warning("User cancelled review and manual fallback disabled; skipping session.")
                    failed.append((sdir, "user_cancelled_review"))
                    continue
            else:
                rows = reviewed
        else:
            if not ok_auto and not args.no_manual_fallback:
                log.warning(f"Auto-ROIs failed sanity checks: {auto_issues}. Opening manual redraw...")
                try:
                    rows = draw_rois_on_first_frame(video, scale=args.draw_scale)
                except Exception as e:
                    log.exception(f"Manual redraw after failed checks failed: {e}")
                    failed.append((sdir, "bad_auto_rois_and_manual_failed"))
                    continue

        # Save session ROIs
        rois_csv = sp / "rois1.csv"
        save_rois_csv(rois_csv, rows)
        analyzer.set_rois_from_long(rows)

        # Optional overlay
        if args.save_overlays:
            ov = overlay_rois(cur_img, rows)
            cv.imwrite(str(sp / "_auto_rois_overlay.png"), ov)

        # Analyze & write
        try:
            inside_ms = analyzer.analyze_video_for_maze_time(video, df)
            df["time_in_maze_ms"] = df["trial_ID"].map(inside_ms)
            backup = trials_csv + ".bak"
            os.replace(trials_csv, backup)
            log.info(f"Backup created: {backup}")
            df.to_csv(trials_csv, index=False)
            log.info(f"Wrote updated CSV: {trials_csv}")
        except Exception as e:
            log.exception(f"Analysis failed: {e}")
            failed.append((sdir, f"analysis_failed: {e}"))
            continue

    # ---------- Summary ----------
    ok = len(sessions) - len(failed)
    log.info(f"Done. Success: {ok}, Failed: {len(failed)}")
    if failed:
        print("\n=== FAILED SESSIONS ===")
        for sess, why in failed:
            print(f"- {sess}\n    reason: {why}")

if __name__ == "__main__":
    main()
