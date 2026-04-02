import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.stats import entropy
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import analysisfunct as af
from analysisfunc_config import Paths 


# variables (change analysisfunc_config.py)
base_path = Paths.base_path

session_path = Paths.session_path

TRIAL_INFO_PATH = Paths.TRIAL_INFO_PATH

DLC_DATA_PATH = Paths.DLC_DATA_PATH

VIDEO_PATH = Paths.VIDEO_PATH
FPS = Paths.FPS
BODYPART = Paths.BODYPART
DRAW_ROIS = False
DRAW_BOUNDARIES = False
LIKELIHOOD_THRESH = Paths.LIKELIHOOD_THRESH
OUTPUT_DIR = session_path + r"trial_analysis_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Likelihood Filter
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 2. Boundary Filter
if DRAW_BOUNDARIES:
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(session_path+ 'maze_boundary.csv', index=False)
elif os.path.exists(session_path+'maze_boundary.csv'):
    boundary_pts = pd.read_csv(session_path+'maze_boundary.csv').values.tolist()
else:
    print("Set DRAW_BOUNDARIES=True"); exit()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = af.filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics
tracking = af.process_kinematics(tracking, FPS)

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

#  ANALYSIS LOOP 

results = []
p1_traces = [] 
p2_traces = []

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    # Extract Trial
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    
    # Skip if trial is too short
    if len(trial_data) < 10: continue

    # Identify Phase Split (Entry -> First ROI)
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    
    # We use the SMOOTHED data to find the split
    # Handle NaNs in search: Check if point is valid AND in ROI
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if np.isnan(pos['x_smooth']) or np.isnan(pos['y_smooth']):
            continue
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: 
        # print(f"Trial {idx}: Target {target} never reached.")
        continue

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    # Calculate Entropy (using the NEW safe function)
    ent1 = af.get_entropy(p1['x_smooth'], p1['y_smooth'])
    ent2 = af.get_entropy(p2['x_smooth'], p2['y_smooth'])
    
    # Calculate Mean Speed (ignoring NaNs automatically)
    s1 = p1['speed'].mean() if len(p1) > 0 else 0
    s2 = p2['speed'].mean() if len(p2) > 0 else 0
    
    results.append({
        'trial': idx, 'target': target,
        'P1_speed': s1, 'P1_entropy': ent1,
        'P2_speed': s2, 'P2_entropy': ent2
    })
    
    # STORE DATA FOR PLOTLY SUMMARY
    # Only store if we have valid data (drop NaNs for plotting)
    if len(p1) > 0:
        clean_p1 = p1.dropna(subset=['speed'])
        if not clean_p1.empty:
            time_p1 = np.arange(len(clean_p1)) / FPS
            p1_traces.append(pd.DataFrame({'Time': time_p1, 'Speed': clean_p1['speed'].values, 'Trial': f"Trial {idx}"}))
            
    if len(p2) > 0:
        clean_p2 = p2.dropna(subset=['speed'])
        if not clean_p2.empty:
            time_p2 = np.arange(len(clean_p2)) / FPS
            p2_traces.append(pd.DataFrame({'Time': time_p2, 'Speed': clean_p2['speed'].values, 'Trial': f"Trial {idx}"}))

    # HEATMAP GENERATION (Handle NaNs)
    plt.figure(figsize=(6, 6))
    # Filter NaNs for plotting
    x_plot = trial_data['x_smooth'].dropna()
    y_plot = trial_data['y_smooth'].dropna()
    
    if len(x_plot) > 1:
        # Align lengths (dropping NaNs from one series might mismatch the other if not aligned)
        valid_indices = trial_data['x_smooth'].notna() & trial_data['y_smooth'].notna()
        x_plot = trial_data.loc[valid_indices, 'x_smooth']
        y_plot = trial_data.loc[valid_indices, 'y_smooth']
        
        plt.hist2d(x_plot, y_plot, bins=30, cmap='inferno', norm=LogNorm())
    
    # Overlay Boundary
    b_x, b_y = zip(*boundary_pts) 
    b_x = list(b_x) + [b_x[0]] 
    b_y = list(b_y) + [b_y[0]]
    plt.plot(b_x, b_y, 'w-', lw=2)
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title(f"Trial {idx} Heatmap")
    plt.colorbar(label='Density')
    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_heatmap.png")
    plt.close()

#  SAVE RESULTS 
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)


# PLOTLY SUMMARY PLOTS

print("Generating Plotly Summaries...")

def plot_summary_plotly(trace_data, title, filename):
    if not trace_data: return
    
    # Combine all traces into one DataFrame
    df_plot = pd.concat(trace_data)
    
    # Create interactive line plot
    fig = px.line(df_plot, x='Time', y='Speed', color='Trial', 
                  title=title, labels={'Time': 'Time (s)', 'Speed': 'Speed (cm/s)'})
    
    # Improve styling
    fig.update_layout(hovermode="x unified", template="plotly_white")
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

plot_summary_plotly(p1_traces, "Phase 1 Speed (Entry -> First ROI)", "summary_speed_P1.html")
plot_summary_plotly(p2_traces, "Phase 2 Speed (First ROI -> Exit)", "summary_speed_P2.html")

print(f"Analysis Complete! Open {OUTPUT_DIR} to see:")
print("1. summary_speed_P1.html (Interactive)")
print("2. summary_speed_P2.html (Interactive)")
print("3. Individual trial_XX_heatmap.png files")
print("4. results.csv")