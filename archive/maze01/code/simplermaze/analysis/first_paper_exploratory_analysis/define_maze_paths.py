import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import json
import os
import cv2
from analysisfunc_config import Paths
import random

# --- Configuration ---
PX_PER_CM = 7.5
SESSION_PATH = Paths.session_path
TRIAL_CSV = Paths.TRIAL_INFO_PATH


def calculate_path_distance(points):
    """Calculates total Euclidean distance of a segmented path."""
    dist_px = 0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        dist_px += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist_px / PX_PER_CM
def get_background_frame():
    """Extracts a random frame from a randomly selected valid video in the trials CSV."""
    df_trials = pd.read_csv(os.path.join(SESSION_PATH, TRIAL_CSV))
    
    # 1. Gather all videos that actually exist on your hard drive
    valid_videos = []
    for idx, row in df_trials.iterrows():
        video_path = str(row['video_path'])
        if pd.notna(video_path) and os.path.exists(video_path):
            valid_videos.append(video_path)
            
    if not valid_videos:
        print("Warning: Could not find any existing videos. Check your paths.")
        return None
        
    # 2. Pick a random video
    random_video_path = random.choice(valid_videos)
    print(f"Selected random video: {random_video_path}")
    
    cap = cv2.VideoCapture(random_video_path)
    if not cap.isOpened():
        print("Warning: Could not open the selected video.")
        return None
        
    # 3. Find out how many frames are in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fallback just in case video metadata is corrupted
    if total_frames <= 0:
        total_frames = 100 
        
    # 4. Pick a random frame index
    random_frame_idx = random.randint(0, total_frames - 1)
    print(f"Extracting frame {random_frame_idx} out of {total_frames}...")
    
    # 5. Jump to that specific frame and capture it
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert from OpenCV's default BGR to standard RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        print("Warning: Failed to capture the specific frame.")
        return None
def main():

    print("Loading Maze Architecture...")
    
    # Load your existing structural files
    df_r = pd.read_csv(os.path.join(SESSION_PATH, "rois1.csv"), index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
    
    boundary_pts = pd.read_csv(os.path.join(SESSION_PATH, 'maze_boundary.csv')).values.tolist()

    # Grab the reference image
    bg_frame = get_background_frame()
    
    path_distances_cm = {}

    for roi_name, (rx, ry, rw, rh) in rois.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # --- MODIFIED: Updated instructions for keyboard users ---
        ax.set_title(f"Draw true path for {roi_name}\n"
                     f"Left Click: Add point | Backspace: Undo | Enter: Finish", 
                     fontsize=12, fontweight='bold')
        # ---------------------------------------------------------
        
        # 1. Plot the Video Frame as the Background
        if bg_frame is not None:
            ax.imshow(bg_frame)
        else:
            ax.set_aspect('equal')
            ax.invert_yaxis() 
            
        # 2. Draw the Maze Boundary
        maze_patch = Polygon(boundary_pts, closed=True, fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(maze_patch)
        
        # 3. Draw all ROIs
        for name, (x, y, w, h) in rois.items():
            color = 'red' if name == roi_name else 'yellow'
            rect = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y-5, name, color=color, fontsize=12, fontweight='bold', backgroundcolor='black')

        plt.axis('off')
        plt.tight_layout()
        
        # 4. Interactive Point Collection
        print(f"\n--- Defining path for {roi_name} ---")
        print("Click the entrance first, then click along the corridors, ending inside the ROI.")
        print("Press the ENTER key when you are done.") # Updated print statement
        
        # ginput handles the clicking interface natively with Backspace/Enter
        points = plt.ginput(n=-1, timeout=0, show_clicks=True) 
        plt.close(fig)

        # 5. Math & Storage
        if len(points) > 1:
            dist_cm = calculate_path_distance(points)
            path_distances_cm[roi_name] = dist_cm
            print(f"Success! True path distance for {roi_name}: {dist_cm:.2f} cm")
        else:
            print(f"Warning: Not enough points clicked for {roi_name}. Defaulting to 0.")
            path_distances_cm[roi_name] = 0.0

    # 6. Save the data to be used by the main script
    output_file = os.path.join(SESSION_PATH, "true_path_distances.json")
    with open(output_file, 'w') as f:
        json.dump(path_distances_cm, f, indent=4)
    
    print(f"\nAll distances calibrated and saved to {output_file}!")
if __name__ == "__main__":
    main()