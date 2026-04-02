import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

base_path = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/"
DLC_DATA_PATH = base_path + r"deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
VIDEO_PATH = base_path + r'2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_trial_006.mp4'
OUTPUT_FILENAME = base_path +r'filtered_dlc_data.csv'
OUTPUT_IMAGE = base_path +r'filtering_verification.png'


# draw the maze 

def select_maze_boundaries(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video.")
        return None

    # Resize for easier viewing 
    scale = 1.0
    if frame.shape[1] > 1920:
        scale = 0.5
        frame = cv.resize(frame, (0,0), fx=scale, fy=scale)

    points = []
    
    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw point
            cv.circle(frame, (x, y), 4, (0, 0, 255), -1)
            # Draw lines connecting points
            if len(points) > 1:
                cv.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            cv.imshow("Define Maze Walls", frame)


    print("1. Click the corners of the maze floor (create a polygon).")
    print("2. Press 'c' to close the shape (connect last point to first).")
    print("3. Press SPACE or ENTER to confirm and finish.")
    print("4. Press 'r' to reset if you make a mistake.")


    cv.imshow("Define Maze Walls", frame)
    cv.setMouseCallback("Define Maze Walls", click_event)

    while True:
        key = cv.waitKey(1) & 0xFF
        
        # 'c' to close polygon visually
        if key == ord('c') and len(points) > 2:
            cv.line(frame, points[-1], points[0], (0, 255, 0), 2)
            cv.imshow("Define Maze Walls", frame)
            
        # 'r' to reset
        if key == ord('r'):
            frame = cap.read()[1] # Reload fresh frame
            if scale != 1.0: frame = cv.resize(frame, (0,0), fx=scale, fy=scale)
            points = []
            cv.imshow("Define Maze Walls", frame)
            
        # Space or Enter to confirm
        if key == 32 or key == 13:
            break

    cv.destroyAllWindows()
    
    # Scale points back up to original resolution if we resized
    final_points = [(int(p[0]/scale), int(p[1]/scale)) for p in points]
    return final_points


# 2. FILTERING LOGIC

def filter_data_by_boundary(df, polygon_points):
    """
    Sets x,y coordinates to NaN if they fall outside the polygon.
    """
    # Create a matplotlib Path object for fast "contains_point" checking
    poly_path = Path(polygon_points)
    
    # Get the scorer name (top level of multi-index)
    scorer = df.columns[0][0]
    bodyparts = df.columns.get_level_values(1).unique()
    
    print(f"Filtering {len(df)} frames for {len(bodyparts)} bodyparts...")
    
    points_removed = 0
    total_points = 0
    
    for bp in bodyparts:
        # Extract X and Y columns for this bodypart
        x_col = (scorer, bp, 'x')
        y_col = (scorer, bp, 'y')
        
        # Combine into (N, 2) array
        # fillna with -1 temporarily so they don't crash the geometric test
        coords = df.loc[:, [x_col, y_col]].fillna(-1000).values
        
        # Check which points are inside the polygon
        # returns boolean array
        is_inside = poly_path.contains_points(coords)
        
        # Identify valid data (not NaNs) that is OUTSIDE the maze
        # (We only care about points that were originally valid DLC detections)
        originally_valid = df[x_col].notna()
        mask_to_remove = (~is_inside) & originally_valid
        
        points_removed += mask_to_remove.sum()
        total_points += originally_valid.sum()
        
        # Set outside points to NaN
        df.loc[mask_to_remove, (scorer, bp, 'x')] = np.nan
        df.loc[mask_to_remove, (scorer, bp, 'y')] = np.nan
        
        # Optional: Also nuke likelihood for consistency
        df.loc[mask_to_remove, (scorer, bp, 'likelihood')] = 0.0

    print(f"Done! Removed {points_removed} points out of {total_points} ({points_removed/total_points:.1%})")
    return df


# main

# Load DLC Data
print("Loading DLC data...")
df = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)

# Get Boundaries
boundary_points = select_maze_boundaries(VIDEO_PATH)

if len(boundary_points) < 3:
    print("Error: Need at least 3 points to form a shape.")
else:
    # Filter
    df_clean = filter_data_by_boundary(df.copy(), boundary_points)
    
    # Save
    df_clean.to_csv(OUTPUT_FILENAME)
    print(f"Filtered data saved to: {OUTPUT_FILENAME}")


    # We'll plot the 'mid' point trajectory: Raw vs Clean
    scorer = df.columns[0][0]
    bp = 'mid' # change if you want to check 'nose' etc.
    
    plt.figure(figsize=(10, 8))
    
    # Plot Filtered (Clean) points in Blue
    plt.scatter(df_clean[scorer][bp]['x'], df_clean[scorer][bp]['y'], 
                alpha=0.5, s=1, c='blue', label='Kept (Inside)')
    
    # Plot Removed points in Red (Difference between raw and clean)
    # We find indices where Raw is valid but Clean is NaN
    raw_valid = df[scorer][bp]['x'].notna()
    clean_nan = df_clean[scorer][bp]['x'].isna()
    removed_mask = raw_valid & clean_nan
    
    if removed_mask.sum() > 0:
        plt.scatter(df[scorer][bp]['x'][removed_mask], df[scorer][bp]['y'][removed_mask], 
                    alpha=0.8, s=5, c='red', label='Removed (Outside)')
    
    # Draw the boundary polygon
    poly_closed = boundary_points + [boundary_points[0]]
    px, py = zip(*poly_closed)
    plt.plot(px, py, 'g-', linewidth=2, label='Boundary')
    
    plt.title(f"Boundary Filtering Result ({bp})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.savefig(OUTPUT_IMAGE)
    plt.show()