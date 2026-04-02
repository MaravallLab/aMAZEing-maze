import pandas as pd
import os
import subprocess
import imageio_ffmpeg
from tkinter import filedialog as fd
from tkinter import Tk

# Setup FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

def cut_clip(input_video, output_path, start_s, end_s):
    ffmpeg_bin = os.environ["IMAGEIO_FFMPEG_EXE"]
    duration = end_s - start_s
    
    # FFmpeg command: Fast seek (-ss) then Copy codec (-c copy) for speed
    # Note: -c copy is instant but snaps to keyframes. 
    # For frame-perfect accuracy re-encoding (-c:v libx264) is better but slower.
    # We will use re-encoding here to guarantee exact frames.
    
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.4f}",
        "-t", f"{duration:.4f}",
        "-i", input_video,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",  # Re-encode for accuracy
        "-movflags", "+faststart",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    print("Select the SESSION_DATA csv file:")
    root = Tk()
    root.withdraw() # Hide empty window
    csv_path = fd.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    
    if not csv_path: return

    # Infer video path (assuming standard naming convention from your script)
    # CSV: .../session_data_DATE.csv -> Video: .../ANIMAL_DATE.mp4
    # Or just look for the only .mp4 in that folder
    session_dir = os.path.dirname(csv_path)
    video_files = [f for f in os.listdir(session_dir) if f.endswith(".mp4") and "trial" not in f]
    
    if not video_files:
        print("Error: No master video found in the session folder.")
        return
    
    master_video_path = os.path.join(session_dir, video_files[0])
    print(f"Processing Master Video: {video_files[0]}")

    # Load Data
    df = pd.read_csv(csv_path)
    
    # Filter valid trials (must have start/end frames)
    valid_trials = df.dropna(subset=["start_frame", "end_frame"])
    
    # We need the FPS to convert Frames -> Seconds
    import cv2
    cap = cv2.VideoCapture(master_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps == 0:
        print("Error: Could not determine FPS.")
        return

    print(f"FPS: {fps}. Cutting {len(valid_trials)} segments...")

    for i, row in valid_trials.iterrows():
        # Get frame numbers
        f_start = int(row["start_frame"])
        f_end = int(row["end_frame"])
        target_path = row["video_segment_path"]
        
        # Convert to seconds
        # Adding a tiny buffer (0.1s) is sometimes nice, but for exact matching:
        t_start = f_start / fps
        t_end = f_end / fps
        
        # Ensure directory exists (in case you moved the csv)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            cut_clip(master_video_path, target_path, t_start, t_end)
            print(f"  [OK] Trial {i}: Frame {f_start}-{f_end}")
        except Exception as e:
            print(f"  [FAIL] Trial {i}: {e}")

    print("Done! All segments created.")

if __name__ == "__main__":
    main()