import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import analysisfunct as af
from analysisfunc_config import Paths 

# 1. SETUP PATHS AND CONFIG
session_path = Paths.session_path
TRIAL_INFO_PATH = Paths.TRIAL_INFO_PATH
DLC_DATA_PATH = Paths.DLC_DATA_PATH
FPS = Paths.FPS
BODYPART = Paths.BODYPART
LIKELIHOOD_THRESH = Paths.LIKELIHOOD_THRESH
OUTPUT_DIR = os.path.join(session_path, "trial_duration_analysis")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. LOAD DATA
print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 3. CLEAN TRACKING DATA (Interpolation for low likelihood)
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

# 4. LOAD ROIs
rois_file = os.path.join(session_path, "rois1.csv")
if not os.path.exists(rois_file):
    print("ROI file not found. Ensure you have defined ROIs in previous steps.")
else:
    df_r = pd.read_csv(rois_file, index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}

# 5. DATA COLLECTION LOOP
data_points = []
print("Processing Trials for Duration Analysis...")

for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        status = "Hit" if (row['hit'] == 1) else "Miss"
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    trial_data = tracking.iloc[f_start:f_end].copy()
    if len(trial_data) < 5: continue

    # Identify split point (Entry into Target ROI)
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    # We use x and y to find when the mouse first enters the ROI bounding box
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if (rx <= pos['x'] <= rx+rw) and (ry <= pos['y'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue # ROI never reached

    # Split into Phase 1 (Entry to ROI) and Phase 2 (ROI to Exit)
    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    # Calculate Durations in Seconds: (Frame Count / FPS)
    p1_duration = len(p1) / FPS
    p2_duration = len(p2) / FPS
    
    data_points.append({'Phase': 'P1', 'Status': status, 'Duration': p1_duration})
    data_points.append({'Phase': 'P2', 'Status': status, 'Duration': p2_duration})

df_res = pd.DataFrame(data_points)
df_res.to_csv(f"{OUTPUT_DIR}/duration_source_data.csv", index=False)

# 6. GAUSSIAN PLOTTING FUNCTION
def plot_duration_gaussian(df, filename):
    if df.empty: return

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Phase 1: Entry -> ROI (Duration)", "Phase 2: ROI -> Exit (Duration)"),
        horizontal_spacing=0.15,
        shared_yaxes=True
    )
    
    styles = {
        'Hit':  {'line': 'rgba(0, 128, 0, 1)',   'fill': 'rgba(0, 128, 0, 0.1)'},
        'Miss': {'line': 'rgba(255, 0, 0, 1)',   'fill': 'rgba(255, 0, 0, 0.1)'}
    }
    
    phases = ['P1', 'P2']
    global_max_dur = df['Duration'].max() * 1.3
    x_range = np.linspace(0, global_max_dur, 1000)
    global_max_density = 0

    for i, phase in enumerate(phases):
        col_idx = i + 1
        subset = df[df['Phase'] == phase]
        if subset.empty: continue
        
        # Calculate Mean and Std for the Gaussian Fit
        stats = subset.groupby('Status')['Duration'].agg(['mean', 'std', 'var']).reset_index()
        
        for idx, row in stats.iterrows():
            status = row['Status']
            mu = row['mean']
            sigma = row['std']
            variance = row['var']
            
            if pd.isna(sigma) or sigma == 0: sigma = 0.05
            
            # Probability Density Function
            y = norm.pdf(x_range, mu, sigma)
            global_max_density = max(global_max_density, max(y))
            
            # Add Trace (with Variance in label)
            fig.add_trace(go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f"{status} (μ={mu:.2f}s, σ²={variance:.2f})",
                line=dict(color=styles[status]['line'], width=4),
                fill='tozeroy', 
                fillcolor=styles[status]['fill'],
                legendgroup=status,
                showlegend=(col_idx==1)
            ), row=1, col=col_idx)
            
            # Add Mean Marker
            fig.add_vline(x=mu, line_width=1, line_dash="dash", 
                          line_color=styles[status]['line'], row=1, col=col_idx)

    # Styling
    fig.update_yaxes(range=[0, global_max_density * 1.1])
    fig.update_layout(
        title="Duration Distributions: Hits vs Misses (Gaussian Fit)",
        template="plotly_white",
        height=600, width=1200
    )
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

# 7. EXECUTION
print("Generating Gaussian Duration Plot...")
plot_duration_gaussian(df_res, 'summary_duration_gaussian.html')
print(f"Analysis Complete! Results in {OUTPUT_DIR}")