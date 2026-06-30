import pandas as pd
import numpy as np
import os
import analysisfunct as af
from analysisfunc_config import Paths

# Variables from config
session_path = Paths.session_path
TRIAL_INFO_PATH = Paths.TRIAL_INFO_PATH
DLC_DATA_PATH = Paths.DLC_DATA_PATH
FPS = Paths.FPS
BODYPART = Paths.BODYPART
LIKELIHOOD_THRESH = Paths.LIKELIHOOD_THRESH
OUTPUT_DIR = os.path.join(session_path, "master_analysis")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load Data
print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH) #
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0) #
scorer = df_dlc.columns[0][0] #
tracking = df_dlc[scorer][BODYPART].copy() #

# Cleaning and Kinematics
mask = tracking['likelihood'] < LIKELIHOOD_THRESH #
tracking.loc[mask, ['x', 'y']] = np.nan #
tracking = tracking.interpolate().ffill().bfill() #
tracking = af.process_kinematics(tracking, FPS) #

# Load ROIs
rois_file = os.path.join(session_path, "rois1.csv")
df_r = pd.read_csv(rois_file, index_col=0)
rois = {name: tuple(row) for name, row in df_r.iterrows()}

master_data = []

print("Consolidating Metrics...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame']) #
        target = row['first_reward_area_visited'] #
        status = "Hit" if (row['hit'] == 1) else "Miss" #
        reward_loc = row['rew_location'] #
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    trial_data = tracking.iloc[f_start:f_end].copy() #
    if len(trial_data) < 5: continue

    # Identify split point
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue 

    p1 = trial_data.iloc[:split_idx] #
    p2 = trial_data.iloc[split_idx:] #
    
    # Calculate all consolidated metrics
    master_data.append({
        'trial_id': idx,
        'status': status,
        'target_visited': target,
        'actual_reward_loc': reward_loc,
        # Phase 1 Metrics
        'p1_duration_s': len(p1) / FPS,
        'p1_mean_speed_cm_s': p1['speed'].mean(),
        'p1_max_speed_cm_s': p1['speed'].max(),
        'p1_entropy': af.get_entropy(p1['x_smooth'], p1['y_smooth']), #
        # Phase 2 Metrics
        'p2_duration_s': len(p2) / FPS,
        'p2_mean_speed_cm_s': p2['speed'].mean(),
        'p2_max_speed_cm_s': p2['speed'].max(),
        'p2_entropy': af.get_entropy(p2['x_smooth'], p2['y_smooth']), #
        # Total trial info
        'total_duration_s': len(trial_data) / FPS,
        'total_dist_px': np.sqrt(trial_data['x'].diff()**2 + trial_data['y'].diff()**2).sum()
    })

# Save Master File
df_master = pd.DataFrame(master_data)
df_master.to_csv(f"{OUTPUT_DIR}/master_behavioral_data.csv", index=False)
print(f"Master CSV saved to: {OUTPUT_DIR}/master_behavioral_data.csv")