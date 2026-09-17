import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import cv2 as cv
from analysisfunc_config import Paths
import analysisfunct as af



# CONFIGURATION

# FILE PATHS

base_path = Paths.base_path

session_path = Paths.session_path
DLC_DATA_PATH = Paths.DLC_DATA_PATH
BOUNDARY_CSV = session_path+ 'maze_boundary.csv'
ROIS_CSV = session_path+"rois1.csv"
VIDEO_PATH = Paths.VIDEO_PATH
DRAW_BOUNDARIES = False
DRAW_ROIS = False

# SETTINGS
BODYPART = 'mid'          # Body part to check
LIKELIHOOD_THRESH = 0.7

# 2. Boundary Filter
if DRAW_BOUNDARIES:
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(session_path+ 'maze_boundary.csv', index=False)
elif os.path.exists(session_path+'maze_boundary.csv'):
    boundary_pts = pd.read_csv(session_path+'maze_boundary.csv').values.tolist()
else:
    print("Set DRAW_BOUNDARIES=True"); exit()

# 4. ROIs
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROIS:
    rois = af.define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv(session_path+"rois1.csv")
elif os.path.exists(session_path+"rois1.csv"):
    df_r = pd.read_csv(session_path+"rois1.csv", index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
else:
    print("Set DRAW_ROIS=True"); exit()



def load_and_check_data():
    #  CHECK BOUNDARY FILE 
    boundary_points = []
    if not os.path.exists(BOUNDARY_CSV):
        print(f"MISSING: {BOUNDARY_CSV} not found.")

    else:
        try:
            # Try loading with headers
            df_b = pd.read_csv(BOUNDARY_CSV)
            if 'x' in df_b.columns and 'y' in df_b.columns:
                boundary_points = df_b[['x', 'y']].values.tolist()
            else:
                # Fallback: Try loading without headers
                print(f"WARNING: {BOUNDARY_CSV} missing headers. Attempting to fix")
                df_b = pd.read_csv(BOUNDARY_CSV, header=None)
                # Assume columns 0 and 1 are x and y
                boundary_points = df_b[[0, 1]].values.tolist()
                # Re-save correctly to fix it for next time
                pd.DataFrame(boundary_points, columns=['x', 'y']).to_csv(BOUNDARY_CSV, index=False)
                print("File fixed and saved.")
        except Exception as e:
            print(f"ERROR reading boundary file: {e}")


    if len(boundary_points) < 3:
        print("Invalid boundary (less than 3 points). Please re-run.")
        return

    #  LOAD DLC DATA 
    print(f"\nLoading DLC Data: {DLC_DATA_PATH}")
    try:
        df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
        scorer = df_dlc.columns[0][0]
    except FileNotFoundError:
        print("Error: DLC File not found. Check path.")
        return

    # Extract Bodypart
    data = df_dlc[scorer][BODYPART].copy()
    total_frames = len(data)
    
    print(f"Total Frames: {total_frames}")
    print("-" * 40)

    #  STEP 1: LIKELIHOOD FILTER 
    low_conf_mask = data['likelihood'] < LIKELIHOOD_THRESH
    num_low_conf = low_conf_mask.sum()
    pct_low_conf = (num_low_conf / total_frames) * 100
    
    print(f"STEP 1: LIKELIHOOD FILTER (Threshold < {LIKELIHOOD_THRESH})")
    print(f"  - Low confidence (dropped): {num_low_conf} ({pct_low_conf:.2f}%)")
    
    # Valid data subset
    valid_data = data[~low_conf_mask].copy()

    #  STEP 2: BOUNDARY CHECK 
    print("-" * 40)
    print(f"STEP 2: BOUNDARY CHECK (Applied to {len(valid_data)} valid points)")
    
    boundary_poly = Path(boundary_points)
    
    # Check containment
    points = valid_data[['x', 'y']].values
    is_inside = boundary_poly.contains_points(points)
    
    num_inside = is_inside.sum()
    num_outside = (~is_inside).sum()
    pct_outside = (num_outside / len(valid_data)) * 100 if len(valid_data) > 0 else 0
    
    print(f"  - Points INSIDE maze:  {num_inside}")
    print(f"  - Points OUTSIDE maze: {num_outside} ({pct_outside:.2f}%)")
    
    if pct_outside > 10:
        print("\n WARNING: >10% of high-confidence data is OUTSIDE.")
        print("   Possibilities: Boundary too small? Reflections tracked?")
    else:
        print("\nData looks healthy.")

    #  VISUALIZATION 
    plt.figure(figsize=(10, 10))
    
    # Plot Boundary
    bx, by = zip(*boundary_points)
    bx = list(bx) + [bx[0]]
    by = list(by) + [by[0]]
    plt.plot(bx, by, 'k-', linewidth=2, label='Boundary')
    
    # Plot ROIs (if available)
    if os.path.exists(ROIS_CSV):
        try:
            rois_df = pd.read_csv(ROIS_CSV, index_col=0)
            if 'x' in rois_df.columns:
                for name, row in rois_df.iterrows():
                    rect = plt.Rectangle((row['x'], row['y']), row['w'], row['h'], 
                                         linewidth=1, edgecolor='green', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(row['x'], row['y']-5, name, color='green', fontsize=8)
        except: pass

    # Plot Points
    outside = valid_data[~is_inside]
    inside = valid_data[is_inside]
    
    # Downsample inside points for speed
    if len(inside) > 10000: inside = inside.sample(10000)
        
    plt.scatter(outside['x'], outside['y'], s=10, c='red', alpha=0.7, label=f'Outside ({num_outside})')
    plt.scatter(inside['x'], inside['y'], s=1, c='blue', alpha=0.3, label='Inside (Sample)')

    plt.title(f"Diagnostic: Threshold={LIKELIHOOD_THRESH} | Removed Outside={num_outside}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    load_and_check_data()