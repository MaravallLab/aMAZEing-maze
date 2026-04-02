import pandas as pd
import numpy as np
import os
import analysisfunct as af
from analysisfunc_config import Paths

# Setup paths from your config
TRIAL_INFO_PATH = Paths.TRIAL_INFO_PATH
DLC_DATA_PATH = Paths.DLC_DATA_PATH
FPS = Paths.FPS
BODYPART = Paths.BODYPART
OUTPUT_DIR = os.path.join(Paths.session_path, "duration_analysis")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Basic Cleaning (Interpolate to handle low likelihood)
mask = tracking['likelihood'] < Paths.LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 2. Load ROIs
rois_file = os.path.join(Paths.session_path, "rois1.csv")
df_r = pd.read_csv(rois_file, index_col=0)
rois = {name: tuple(row) for name, row in df_r.iterrows()}

results = []

print("Analyzing Phase Durations...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        status = "Hit" if row['hit'] == 1 else "Miss"
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    # Slice the trial tracking data
    trial_data = tracking.iloc[f_start:f_end]
    if len(trial_data) < 2: continue

    # Identify split point (Entry into ROI)
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    
    # Using 'x' and 'y' directly (or you can use smoothed if preferred)
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if (rx <= pos['x'] <= rx+rw) and (ry <= pos['y'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue 

    # Split into Phases
    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    # Calculate Durations in Seconds
    t1 = len(p1) / FPS
    t2 = len(p2) / FPS
    total_t = len(trial_data) / FPS
    
    results.append({
        'trial': idx,
        'status': status,
        'target': target,
        'P1_duration_sec': t1,
        'P2_duration_sec': t2,
        'total_duration_sec': total_t
    })

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/trial_durations.csv", index=False)

# Quick Summary Statistics
summary = df_results.groupby('status')[['P1_duration_sec', 'P2_duration_sec']].agg(['mean', 'sem'])
summary.to_csv(f"{OUTPUT_DIR}/summary_trial_durations.csv", index=False)
print("\n--- Summary Statistics (Time in Seconds) ---")
print(summary)
print(f"\nResults saved to: {OUTPUT_DIR}/trial_durations.csv")