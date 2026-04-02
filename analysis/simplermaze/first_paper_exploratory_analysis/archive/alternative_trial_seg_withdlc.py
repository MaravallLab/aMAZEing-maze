import cv2
import pandas as pd
import numpy as np
import random
import os

base_path = r"C:\Users\shahd\Box\Awake Project\Maze data\simplermaze\mouse 6357\2024-08-30_10_07_556357session3.8"
VIDEO_FILE = base_path + r"\6357_2024-08-30_10_07_55s3.8.mp4"
DLC_FILE = r"C:\Users\shahd\Box\Awake Project\Maze data\simplermaze\mouse 6357\deeplabcut\mouse6357\mouse6357-shahd-2025-09-08\videos\6357_2024-08-30_10_07_55s3.8DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
TRIALS_FILE = base_path + r"\mouse6357_session3.8_trial_info.csv"

OUTPUT_TRIALS_FILE = base_path + "\experimental_trials_with_frames.csv"
ROI_OUTPUT_FILE = base_path + "\maze_rois.csv"

def get_random_frame_rois(video_path, roi_names):
    """Opens a random frame, allows resizing, and keeps previous ROIs visible."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get total frames and pick a random one
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read the random frame.")

    rois = {}
    window_name = "ROI Selection - Drag corners to resize"
    
    print("\n--- ROI Selection ---")
    print("Draw the bounding box, then press SPACE or ENTER to confirm.")
    print("Press 'c' to cancel the current box and redraw.")
    
    # 1. Create the single resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 
    
    # 2. Make a working copy of the frame to draw on
    display_frame = frame.copy() 

    for name in roi_names:
        print(f"Please select: {name}")
        
        # 3. Call selectROI on the display_frame (which will have previous boxes drawn on it)
        roi = cv2.selectROI(window_name, display_frame, fromCenter=False, showCrosshair=True)
        rois[name] = roi
        print(f"Saved '{name}': {roi}")
        
        # 4. Draw the confirmed ROI onto the display_frame so it stays visible
        x, y, w, h = roi
        # Let's make entrance_1 green and entrance_2 blue to match the visualizer
        color = (0, 255, 0) if name == "entrance_1" else (255, 0, 0)
        
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Clean up the window after all ROIs are selected
    cv2.destroyWindow(window_name)
    
    return rois

def is_in_roi(x, y, roi):
    """Checks if a given (x, y) coordinate is inside the (x, y, w, h) ROI."""
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def extract_trial_videos(video_path, starts, ends, output_dir= base_path+ "\trial_videos"):
    """Extracts individual video clips for each trial based on start and end frames."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 'mp4v' is a standard, widely compatible codec for OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    print(f"\n--- Extracting Trial Videos to '{output_dir}/' ---")
    
    # Pair up the starts and ends
    for i, start_frame in enumerate(starts):
        # Ensure we have a matching end frame that isn't NaN
        if i < len(ends) and not pd.isna(ends[i]) and not pd.isna(start_frame):
            start_frame = int(start_frame)
            end_frame = int(ends[i])
            
            # Sanity check
            if start_frame >= end_frame:
                print(f"Skipping Trial {i+1}: Start frame ({start_frame}) is after end frame ({end_frame}).")
                continue
                
            output_path = os.path.join(output_dir, f"trial_{i+1}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Jump straight to the start frame of the trial
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break  # Video ended unexpectedly
                out.write(frame)
                current_frame += 1
                
            out.release()
            print(f"Saved Trial {i+1}: Frames {start_frame} to {end_frame} -> {output_path}")
        else:
            print(f"Skipping Trial {i+1}: Missing a valid end frame.")
            
    cap.release()
    print("Video extraction complete!")

def visualize_tracking(video_path, rois, x_coords, y_coords, starts, ends):
    """Plays back the video with ROIs, tracking point, and trial status overlaid."""
    print("\n--- Starting Visualization ---")
    print("Press 'q' to quit the playback early.")
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    trial_active = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in starts:
            trial_active = True
        elif frame_idx in ends:
            trial_active = False

        rx, ry, rw, rh = rois["entrance_1"]
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(frame, "Ent 1", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        rx, ry, rw, rh = rois["entrance_2"]
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        cv2.putText(frame, "Ent 2", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if frame_idx < len(x_coords):
            x, y = x_coords[frame_idx], y_coords[frame_idx]
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        status_text = "TRIAL ACTIVE" if trial_active else "OUTSIDE/WAITING"
        color = (0, 255, 0) if trial_active else (0, 0, 255)
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if frame_idx in starts:
            cv2.putText(frame, "TRIAL STARTED!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        if frame_idx in ends:
            cv2.putText(frame, "TRIAL ENDED!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

        cv2.imshow("Tracking Verification", frame)
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Visualization closed.")

def process_maze_trials(video_path, dlc_csv_path, trials_csv_path, output_trials_path, roi_output_path):
    roi_names = ["entrance_1", "entrance_2"]

    
    rois = get_random_frame_rois(video_path, roi_names)
    
    roi_df = pd.DataFrame.from_dict(rois, orient='index', columns=['x', 'y', 'width', 'height'])
    roi_df.to_csv(roi_output_path, index_label="roi_name")
    
    print("Loading DLC tracking data...")
    dlc_df = pd.read_csv(dlc_csv_path, header=[0, 1, 2], index_col=0)
    
    try:
        mid_body_data = dlc_df.xs('mid', level='bodyparts', axis=1)
        x_coords = mid_body_data.xs('x', level='coords', axis=1).iloc[:, 0].values
        y_coords = mid_body_data.xs('y', level='coords', axis=1).iloc[:, 0].values
    except KeyError:
        raise KeyError("Could not find 'mid' in the DLC CSV. Check exact naming.")

    starts = []
    ends = []
    state = "OUTSIDE"
    last_seen = None 
    
    print("Processing tracking data to find transitions...")
    for frame_idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        if np.isnan(x) or np.isnan(y):
            continue
            
        in_ent_1 = is_in_roi(x, y, rois["entrance_1"])
        in_ent_2 = is_in_roi(x, y, rois["entrance_2"])
        
        if in_ent_1:
            if last_seen == "entrance_2" and state == "INSIDE":
                ends.append(frame_idx)
                state = "OUTSIDE"
            last_seen = "entrance_1"
            
        elif in_ent_2:
            if last_seen == "entrance_1" and state == "OUTSIDE":
                starts.append(frame_idx)
                state = "INSIDE"
            last_seen = "entrance_2"

    print(f"Found {len(starts)} trial starts and {len(ends)} trial ends.")

    trials_df = pd.read_csv(trials_csv_path)
    num_csv_trials = len(trials_df)
    
    starts_padded = (starts + [np.nan] * num_csv_trials)[:num_csv_trials]
    ends_padded = (ends + [np.nan] * num_csv_trials)[:num_csv_trials]

    trials_df['video_start_frame'] = starts_padded
    trials_df['video_end_frame'] = ends_padded
    
    trials_df.to_csv(output_trials_path, index=False)
    print(f"\nSuccessfully saved updated experimental trials to {output_trials_path}")

    # 1. Ask the user if they want to chop up the videos
    extract_choice = input("\nDo you want to extract individual trial videos now? (y/n): ").strip().lower()
    if extract_choice == 'y':
        extract_trial_videos(video_path, starts, ends, output_dir=base_path + "\trial_videos_dlc")
    
    # 2. Ask the user if they want to run the visualizer
    vis_choice = input("\nDo you want to run the visual tracking verification? (y/n): ").strip().lower()
    if vis_choice == 'y':
        visualize_tracking(video_path, rois, x_coords, y_coords, starts, ends)


# --- EXECUTE THE SCRIPT ---
if __name__ == "__main__":


    
    process_maze_trials(
        video_path=VIDEO_FILE,
        dlc_csv_path=DLC_FILE,
        trials_csv_path=TRIALS_FILE,
        output_trials_path=OUTPUT_TRIALS_FILE,
        roi_output_path=ROI_OUTPUT_FILE
    )