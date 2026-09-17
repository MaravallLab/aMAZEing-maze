import os
import shutil
import subprocess
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import imageio_ffmpeg
from segfuncs import BASE_DIR, get_video_csv_pairs, load_rois

#  Parameters 
THRESH_VALUE = 160
THRESH_FACTOR = 0.5
MIN_DURATION_S = 0.4
MERGE_GAP_S = 0.2
PADDING_S = 0.10

# Ensure FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

def cut_segment_ffmpeg(video, start_s, end_s, out_path, reencode=True):
    ffmpeg_bin = os.environ["IMAGEIO_FFMPEG_EXE"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate duration
    duration = max(0.0, end_s - start_s)
    
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(video)
    ]
    
    if reencode:
        # Re-encoding allows precise cuts
        cmd += ["-map", "0:v:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-movflags", "+faststart", "-reset_timestamps", "1"]
    else:
        # Copy is faster but less precise on keyframes
        cmd += ["-map", "0:v:0?", "-c", "copy", "-movflags", "+faststart", "-reset_timestamps", "1"]
        
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)

def detect_bouts(video_path, rois_csv):
    """
    Logic: 
    - Leave Entrance 1 then Entrance 2 = START
    - Leave Entrance 2 then Entrance 1 = END
    
    Returns: List of (start_frame, end_frame, fps, total_frames)
    """
    df = pd.read_csv(rois_csv)
    df.columns = [c.lower() for c in df.columns]
    
    rois = {}
    for row in df.itertuples():
        rois[str(row.name).lower()] = {
            "x": int(row.x), "y": int(row.y), 
            "w": int(row.w), "h": int(row.h)
        }

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened(): return [], 30.0, 0
    
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Compute Baselines (first 10 frames)
    baselines = {k: 0.0 for k in rois}
    count = 0
    for _ in range(10):
        ok, frame = cap.read()
        if not ok: break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, THRESH_VALUE, 255, cv.THRESH_BINARY)
        
        for name, r in rois.items():
            crop = bw[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']]
            baselines[name] += float(np.sum(crop))
        count += 1
    
    for k in baselines: baselines[k] /= max(1, count)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Detection Loop
    bouts = []
    e1_prev, e2_prev = False, False
    hasLeft1, hasLeft2 = False, False
    entered = False
    cur_start_frame = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, THRESH_VALUE, 255, cv.THRESH_BINARY)
        
        is_occupied = {}
        for name, r in rois.items():
            crop = bw[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']]
            is_occupied[name] = np.sum(crop) < (baselines[name] * THRESH_FACTOR)

        e1_now = is_occupied.get("entrance1", False)
        e2_now = is_occupied.get("entrance2", False)

        left1 = (not e1_now) and e1_prev
        left2 = (not e2_now) and e2_prev

        if left1:
            hasLeft1 = True
            # EXIT (2->1)
            if hasLeft2 and entered:
                if cur_start_frame is not None:
                    bouts.append((cur_start_frame, frame_idx))
                cur_start_frame = None
                entered = False
                hasLeft1, hasLeft2 = False, False
        
        if left2:
            hasLeft2 = True
            # ENTER (1->2)
            if hasLeft1 and not entered:
                cur_start_frame = frame_idx
                entered = True
        
        e1_prev, e2_prev = e1_now, e2_now
        frame_idx += 1

    # Cleanup open bout
    if entered and cur_start_frame is not None:
        bouts.append((cur_start_frame, total_frames))
    
    cap.release()
    
    # Merge small gaps (convert frames to seconds for threshold logic)
    if not bouts: return [], fps, total_frames
    
    merged = []
    if bouts:
        curr_s, curr_e = bouts[0] # frames
        for next_s, next_e in bouts[1:]:
            # Check gap in seconds
            gap_sec = (next_s - curr_e) / fps
            
            if gap_sec <= MERGE_GAP_S:
                curr_e = next_e # Extend
            else:
                # Check duration in seconds
                dur_sec = (curr_e - curr_s) / fps
                if dur_sec >= MIN_DURATION_S:
                    merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        
        dur_sec = (curr_e - curr_s) / fps
        if dur_sec >= MIN_DURATION_S:
            merged.append((curr_s, curr_e))
            
    return merged, fps, total_frames

def main():
    pairs = get_video_csv_pairs(BASE_DIR)
    print(f"Found {len(pairs)} sessions to process.")

    for trials_csv, video_path in pairs.items():
        vp = Path(video_path)
        tp = Path(trials_csv)
        
        rois_csv = vp.with_suffix("").as_posix() + "_rois.csv"
        if not os.path.exists(rois_csv):
            print(f"Skipping {vp.name}: No ROI file found.")
            continue

        print(f"Segmenting: {vp.name}")
        
        # 1. Clean old segments
        out_dir = vp.parent / "segments_from_dlc"
        if out_dir.exists():
            for f in out_dir.glob(f"{vp.stem}_trial_*.mp4"):
                try: f.unlink()
                except: pass
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. Detect (returns frames)
        bouts, fps, total_frames = detect_bouts(video_path, rois_csv)
        
        # 3. Process segments
        segment_paths = []
        start_frames = []
        end_frames = []
        
        padding_frames = int(PADDING_S * fps)

        for i, (f_start, f_end) in enumerate(bouts):
            # Apply padding to frames
            f_start_pad = max(0, f_start - padding_frames)
            f_end_pad = min(total_frames, f_end + padding_frames)
            
            # Convert to seconds for FFmpeg
            s_start = f_start_pad / fps
            s_end = f_end_pad / fps
            
            out_name = f"{vp.stem}_trial_{i:03d}.mp4"
            out_path = out_dir / out_name
            
            cut_segment_ffmpeg(str(vp), s_start, s_end, str(out_path))
            
            segment_paths.append(str(out_path))
            start_frames.append(f_start_pad)
            end_frames.append(f_end_pad)

        # 4. Update CSV
        df = pd.read_csv(tp)
        
        # Prepare columns
        cols_to_add = ["video_segment_path", "start_frame", "end_frame"]
        for c in cols_to_add:
            if c not in df.columns: df[c] = pd.NA
        
        # Fill rows
        n = min(len(df), len(segment_paths))
        
        # Pandas allows assigning lists to slices if indices align
        df.loc[:n-1, "video_segment_path"] = segment_paths[:n]
        df.loc[:n-1, "start_frame"] = start_frames[:n]
        df.loc[:n-1, "end_frame"] = end_frames[:n]
        
        # Save
        df.to_csv(tp, index=False)
        print(f"  -> Created {len(segment_paths)} clips. Updated CSV with frames.")

if __name__ == "__main__":
    main()