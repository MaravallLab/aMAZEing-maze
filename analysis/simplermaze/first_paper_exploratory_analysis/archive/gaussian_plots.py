import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.signal import savgol_filter
from scipy.stats import norm
import plotly.graph_objects as go
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
DRAW_ROIS = False #Paths.DRAW_ROIS
DRAW_BOUNDARIES = False #Paths.DRAW_BOUNDARIES
LIKELIHOOD_THRESH = Paths.LIKELIHOOD_THRESH
OUTPUT_DIR = session_path + r"trial_analysis_gaussian"
PX_PER_CM = Paths.PX_PER_CM

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)




# MAIN EXECUTION

print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Likelihood Filter
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 2. Boundary
boundary_file = session_path + 'maze_boundary.csv'
if not os.path.exists(boundary_file):
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(boundary_file, index=False)
else:
    boundary_pts = pd.read_csv(boundary_file).values.tolist()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = af.filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics
tracking = af.process_kinematics(tracking, FPS, PX_PER_CM)

# 4. ROIs
rois_file = session_path + "rois1.csv"
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if not os.path.exists(rois_file):
    rois = af.define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv(rois_file)
else:
    df_r = pd.read_csv(rois_file, index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}

# DATA COLLECTION
data_points = []

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        is_hit = (row['hit'] == 1)
        status = "Hit" if is_hit else "Miss"
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if np.isnan(pos['x_smooth']) or np.isnan(pos['y_smooth']): continue
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    if len(p1) > 0:
        data_points.append({'Phase': 'P1', 'Status': status, 'MeanSpeed': p1['speed'].mean()})
    if len(p2) > 0:
        data_points.append({'Phase': 'P2', 'Status': status, 'MeanSpeed': p2['speed'].mean()})

df_res = pd.DataFrame(data_points)
df_res.to_csv(f"{OUTPUT_DIR}/gaussian_source_data.csv", index=False)


# COMBINED GAUSSIAN PLOTTING FUNCTION

def plot_combined_gaussian(df, filename):
    if df.empty: return

    # Create subplots: 1 Row, 2 Columns
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Phase 1: Entry -> ROI", "Phase 2: ROI -> Exit"),
        horizontal_spacing=0.15 # Gap between plots
    )
    
    # Styles
    # Line color (Opaque) | Fill color (Transparent)
    styles = {
        'Hit':  {'line': 'rgba(0, 128, 0, 1)',   'fill': 'rgba(0, 128, 0, 0.1)'},
        'Miss': {'line': 'rgba(255, 0, 0, 1)',   'fill': 'rgba(255, 0, 0, 0.1)'}
    }
    
    phases = ['P1', 'P2']
    
    # Global Max Speed for shared X-axis
    global_max_speed = df['MeanSpeed'].max() * 1.3
    x_range = np.linspace(0, global_max_speed, 1000)

    for i, phase in enumerate(phases):
        col_idx = i + 1 # Plotly uses 1-based indexing
        subset = df[df['Phase'] == phase]
        
        if subset.empty: continue
        
        stats = subset.groupby('Status')['MeanSpeed'].agg(['mean', 'std']).reset_index()
        
        for idx, row in stats.iterrows():
            status = row['Status']
            mu = row['mean']
            sigma = row['std']
            
            if pd.isna(sigma) or sigma == 0: sigma = 0.1
            
            # PDF Calculation
            y = norm.pdf(x_range, mu, sigma)
            
            # Add Trace
            fig.add_trace(go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f"{status} (μ={mu:.1f})",
                line=dict(color=styles[status]['line'], width=4), # Thick Line
                fill='tozeroy', 
                fillcolor=styles[status]['fill'], # Transparent Fill
                legendgroup=status, # Click one legend to hide both P1/P2 traces
                showlegend=(col_idx==1) # Only show legend once
            ), row=1, col=col_idx)
            
            # Add Vertical Line for Mean
            fig.add_vline(x=mu, line_width=1, line_dash="dash", 
                          line_color=styles[status]['line'], row=1, col=col_idx)

    # Add a visual separator line (Draw a shape on the layout)
    fig.add_shape(type="line",
        x0=0.5, y0=0, x1=0.5, y1=1,
        xref="paper", yref="paper",
        line=dict(color="Black", width=2)
    )

    fig.update_layout(
        title="Speed Distributions: Hits vs Misses (Gaussian Fit)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        width=1200
    )
    
    # Set shared X-axis labels
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

print("Generating Combined Gaussian Plot...")
plot_combined_gaussian(df_res, 'summary_gaussian_combined.html')

print(f"Done! Open {OUTPUT_DIR}/summary_gaussian_combined.html")