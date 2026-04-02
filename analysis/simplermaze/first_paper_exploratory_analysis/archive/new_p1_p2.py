import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.stats import entropy
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px
import analysisfunct as af                                                                                                                                                                                                  
import os
from analysisfunc_config import Paths

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
# [-33, 200]
OUTPUT_DIR = session_path + r"trial_analysis_cm_aligned"



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
boundary_file = session_path + 'maze_boundary.csv'
if DRAW_BOUNDARIES:
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(boundary_file, index=False)
elif os.path.exists(boundary_file):
    try:
        df_b = pd.read_csv(boundary_file)
        if 'x' in df_b.columns:
            boundary_pts = df_b[['x', 'y']].values.tolist()
        else:
            boundary_pts = pd.read_csv(boundary_file, header=None).values.tolist()
    except:
        print("Error reading boundary. Set DRAW_BOUNDARIES=True"); exit()
else:
    print("Set DRAW_BOUNDARIES=True"); exit()

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = af.filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]

# 3. Kinematics (WITH CM CONVERSION)
tracking = af.process_kinematics(tracking, FPS)

# 4. ROIs
rois_file = session_path + "rois1.csv"
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROIS:
    rois = af.define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv(rois_file)
elif os.path.exists(rois_file):
    df_r = pd.read_csv(rois_file, index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
else:
    print("Set DRAW_ROIS=True"); exit()

# ANALYSIS LOOP 
results = []
aligned_traces = [] # For the "Gaussian" plot

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        is_hit = (row['hit'] == 1)
        status_label = "Hit" if is_hit else "Miss"
        rew_letter = str(row['rew_location']) 
        reward_roi_name = "rew" + rew_letter
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    # Identify Phase Split (Event Alignment Point)
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if np.isnan(pos['x_smooth']) or np.isnan(pos['y_smooth']): continue
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
            
    if split_idx == -1: continue # Never reached target, can't align

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    ent1 = af.get_entropy(p1['x_smooth'], p1['y_smooth'])
    ent2 = af.get_entropy(p2['x_smooth'], p2['y_smooth'])
    
    results.append({
        'trial': idx, 
        'target_visited': target,
        'reward_location': reward_roi_name,
        'status': status_label,
        'P1_speed_cm': p1['speed'].mean(), 'P1_entropy': ent1,
        'P2_speed_cm': p2['speed'].mean(), 'P2_entropy': ent2
    })
    
    #  EVENT ALIGNMENT FOR PLOT 
    # We want Time 0 to be the split_idx
    # Create relative time array for the WHOLE trial (P1 + P2)
    # If trial has 100 frames and split is at 40:
    # Times: -40, -39, ... 0 ... 59, 60
    
    total_len = len(trial_data)
    frames_relative = np.arange(total_len) - split_idx
    time_relative = frames_relative / FPS
    
    # Store full trace (P1 and P2 combined)
    aligned_traces.append(pd.DataFrame({
        'Relative_Time': time_relative,
        'Speed': trial_data['speed_cm_s'].values,
        'Trial': f"Trial {idx}",
        'Status': status_label
    }))


    # STATIC HEATMAP (With CM info in title?)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black') 
    
    valid = trial_data['x_smooth'].notna() & trial_data['y_smooth'].notna()
    x_plot = trial_data.loc[valid, 'x_smooth']
    y_plot = trial_data.loc[valid, 'y_smooth']
    
    if len(x_plot) > 1:
        counts, xedges, yedges, image = ax.hist2d(x_plot, y_plot, bins=30, cmap='inferno', norm=LogNorm())
        clip_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='none') 
        ax.add_patch(clip_patch)
        image.set_clip_path(clip_patch)
        draw_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='white', linewidth=2)
        ax.add_patch(draw_patch)
        cbar = plt.colorbar(image, ax=ax, label='Density')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    rx, ry, rw, rh = rois[target]
    rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.text(rx, ry-10, f"Visited: {target}", color='lime', fontsize=8, fontweight='bold')

    if reward_roi_name in rois and reward_roi_name != target:
        rrx, rry, rrw, rrh = rois[reward_roi_name]
        rect_rew = patches.Rectangle((rrx, rry), rrw, rrh, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax.add_patch(rect_rew)
        ax.text(rrx, rry-25, f"Reward: {reward_roi_name}", color='cyan', fontsize=8, fontweight='bold')
    elif reward_roi_name == target:
        ax.text(rx, ry-25, "(CORRECT)", color='cyan', fontsize=8, fontweight='bold')

    bx = [p[0] for p in boundary_pts]
    by = [p[1] for p in boundary_pts]
    ax.set_xlim(min(bx)-10, max(bx)+10)
    ax.set_ylim(min(by)-10, max(by)+10)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Trial {idx} ({status_label})", color='white')

    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_heatmap.png", bbox_inches='tight', facecolor='black')
    plt.close()

# SAVE RESULTS 
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)


# PLOTLY SUMMARY: EVENT ALIGNED (Mean +/- Std)

print("Making Aligned Summary Plot...")

def plot_aligned_distribution(trace_data, title, filename):
    if not trace_data: return
    df_all = pd.concat(trace_data)
    
    fig = go.Figure()

    color_map = {'Hit': 'rgba(0, 128, 0, 1)', 'Miss': 'rgba(255, 0, 0, 1)'}
    fill_map = {'Hit': 'rgba(0, 128, 0, 0.2)', 'Miss': 'rgba(255, 0, 0, 0.2)'}
    
    statuses = ['Hit', 'Miss']
    
    for status in statuses:
        df_status = df_all[df_all['Status'] == status]
        if df_status.empty: continue
            
        # Group by RELATIVE TIME (Binning might be needed if frame rates jitter, 
        # but with integer frames mapped to floats, direct grouping usually works well)
        # Rounding time to 2 decimals helps grouping
        df_status['Time_Rounded'] = df_status['Relative_Time'].round(2)
        
        stats = df_status.groupby('Time_Rounded')['Speed'].agg(['mean', 'std']).reset_index()
        
        # Sort by time
        stats = stats.sort_values('Time_Rounded')
        
        x = stats['Time_Rounded']
        y_mean = stats['mean']
        y_std = stats['std'].fillna(0)
        
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        y_lower = y_lower.clip(lower=0)
        
        # Draw Shaded Area
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=fill_map[status],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f"{status} Std Dev"
        ))
        
        # Draw Mean Line
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            mode='lines',
            line=dict(color=color_map[status], width=3),
            name=f"{status} Mean"
        ))

    # Add Vertical Line at 0 (ROI Entry)
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black", annotation_text="ROI Entry")

    fig.update_layout(
        title=title,
        xaxis_title="Time from ROI Entry (s)",
        yaxis_title="Speed (cm/s)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(range=[-33, 200]) # Optional: Zoom in on +/- 5 seconds around entry by default
    )
    
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

plot_aligned_distribution(aligned_traces, "Speed vs Time Aligned to First ROI Entry", "summary_speed_aligned_cm.html")

print(f"Analysis Complete! Results in {OUTPUT_DIR}")