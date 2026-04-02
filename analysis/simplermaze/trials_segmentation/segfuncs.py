import os
import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

#  CONFIGURATION 
# UPDATE THIS PATH TO YOUR DATA DIRECTORY
BASE_DIR = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/"

REF_ROIS_FILE = "rois_reference.csv"
ROI_NAMES = ["entrance1", "entrance2"]

# Homography scale (smaller = faster feature matching)
PYR_SCALE = 0.75

#  FILE FINDING LOGIC 
def get_video_csv_pairs(main_dir: str) -> dict:
    """Scans directory for specific mouse sessions as defined in the original notebook."""
    csvs = []
    videos = []
    
    if not os.path.exists(main_dir):
        print(f"Error: Directory not found: {main_dir}")
        return {}

    for i in os.listdir(main_dir):
        if "6357" in i or "6359" in i:
            mouse_dir = os.path.join(main_dir, i)
            if not os.path.isdir(mouse_dir): continue
            
            for session in os.listdir(mouse_dir):
                if any(x in session for x in ['habituation', '1.1', '3.5', '3.6', '3.7', '3.8']):
                    session_dir = os.path.join(mouse_dir, session)
                    if not os.path.isdir(session_dir): continue

                    for file in os.listdir(session_dir):
                        full_path = os.path.join(session_dir, file)
                        if "trial_info.csv" in file and "clean" not in file:
                            csvs.append(full_path)
                        elif file.endswith(".mp4") and "trial_" not in file: 
                            # Basic filtering to avoid grabbing the segments themselves if they exist
                            videos.append(full_path)

    # Dictionary mapping CSV path -> Video path
    # Note: This assumes 1 video per folder or matching order. 
    # The original notebook zipped lists; we assume lists detect in sync order.
    # A more robust match might be needed if filenames differ vastly.
    return dict(zip(csvs, videos))

#  VIDEO UTILS 
def open_video_any(path: str):
    for backend in (cv.CAP_MSMF, cv.CAP_FFMPEG, cv.CAP_DSHOW, cv.CAP_ANY):
        cap = cv.VideoCapture(path, backend)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            return cap
        try: cap.release()
        except Exception: pass
    return None

def grab_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = open_video_any(video_path)
    if cap is None: return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

#  ROI FUNCTIONS
def load_rois(csv_path: str) -> List[Tuple[str,int,int,int,int]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    out = []
    for name, x, y, w, h in df[["name","x","y","w","h"]].itertuples(index=False, name=None):
        out.append((str(name).lower(), int(x), int(y), int(w), int(h)))
    return out

def save_rois_csv(dest_csv: Path, rows: List[Tuple[str,int,int,int,int]]) -> None:
    df = pd.DataFrame(rows, columns=["name","x","y","w","h"])
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False)

def overlay_rois(image: np.ndarray, rows: List[Tuple], color=(0,255,0)) -> np.ndarray:
    out = image.copy()
    for name, x, y, w, h in rows:
        cv.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cv.putText(out, name, (x, max(20, y-10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
    return out

#  HOMOGRAPHY 
def detect_homography(ref_img: np.ndarray, cur_img: np.ndarray) -> Optional[np.ndarray]:
    g1 = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
    g2 = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY) if cur_img.ndim == 3 else cur_img
    orb = cv.ORB_create(nfeatures=5000, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20: return None
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 20: return None
    
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    H, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0)
    if H is None: return None
    if mask is not None and int(mask.sum()) < 20: return None
    return H

def project_rect(x, y, w, h, H):
    pts = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], dtype=np.float32).reshape(-1,1,2)
    proj = cv.perspectiveTransform(pts, H).reshape(-1,2)
    xs, ys = proj[:,0], proj[:,1]
    x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
    x1, y1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    return x0, y0, max(1, x1-x0), max(1, y1-y0)

def auto_rois_from_reference(cur_img, ref_img, ref_rows, pyr_scale=0.75):
    if pyr_scale != 1.0:
        def S(a): return np.array([[a,0,0],[0,a,0],[0,0,1]], dtype=np.float32)
        ref_s = cv.resize(ref_img, None, fx=pyr_scale, fy=pyr_scale)
        cur_s = cv.resize(cur_img, None, fx=pyr_scale, fy=pyr_scale)
        Hs = detect_homography(ref_s, cur_s)
        if Hs is None: return None
        H = np.linalg.inv(S(pyr_scale)) @ Hs @ S(pyr_scale)
    else:
        H = detect_homography(ref_img, cur_img)
        if H is None: return None

    out = []
    for name, x, y, w, h in ref_rows:
        x2, y2, w2, h2 = project_rect(x, y, w, h, H)
        out.append((name, x2, y2, w2, h2))
    return out