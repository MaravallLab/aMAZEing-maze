import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from scipy.stats import entropy
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px  # This import was getting overwritten!
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
OUTPUT_DIR = session_path + r"maze_plotly_analysis"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



print("Loading Data")
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

# 3. Kinematics
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
p1_traces = [] 
p2_traces = []

print("Processing Trials")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        
        # Hit Status
        is_hit = (row['hit'] == 1)
        status_label = "Hit" if is_hit else "Miss"
        
        # Reward Location
        rew_letter = str(row['rew_location']) 
        reward_roi_name = "rew" + rew_letter
        
    except: continue
        
    if f_start >= f_end or target not in rois: continue
    
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    # Identify Phase Split
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
    
    ent1 = af.get_entropy(p1['x_smooth'], p1['y_smooth'])
    ent2 = af.get_entropy(p2['x_smooth'], p2['y_smooth'])
    
    results.append({
        'trial': idx, 
        'target_visited': target,
        'reward_location': reward_roi_name,
        'status': status_label,
        'P1_speed': p1['speed'].mean(), 'P1_entropy': ent1,
        'P2_speed': p2['speed'].mean(), 'P2_entropy': ent2
    })
    
    # Store Plotly Data with integer Frame Index for easier alignment later
    if len(p1) > 0:
        clean_p1 = p1.dropna(subset=['speed'])
        if not clean_p1.empty:
            frames_p1 = np.arange(len(clean_p1))
            p1_traces.append(pd.DataFrame({
                'Frame': frames_p1, 'Time': frames_p1/FPS, 
                'Speed': clean_p1['speed'].values, 
                'Trial': f"Trial {idx}", 'Status': status_label
            }))

    if len(p2) > 0:
        clean_p2 = p2.dropna(subset=['speed'])
        if not clean_p2.empty:
            frames_p2 = np.arange(len(clean_p2))
            p2_traces.append(pd.DataFrame({
                'Frame': frames_p2, 'Time': frames_p2/FPS, 
                'Speed': clean_p2['speed'].values, 
                'Trial': f"Trial {idx}", 'Status': status_label
            }))

   
    # STATIC HEATMAP (With Hit/Miss + Rew Loc)
   
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
        cbar.set_label('Density', color='white')
    
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
    ax.set_title(f"Trial {idx} ({status_label}) Heatmap", color='white')
    ax.axis('off')

    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_heatmap.png", bbox_inches='tight', facecolor='black')
    plt.close()

# SAVE RESULTS 
pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)


# PLOTLY SUMMARY WITH DISTRIBUTIONS

print("Generating Plotly Summaries (Mean +/- Std)")

def plot_summary_with_distribution(trace_data, title, filename):
    if not trace_data: return
    df_all = pd.concat(trace_data)
    
    # 1. Create Base Figure
    fig = go.Figure()

    # Colors
    color_map = {'Hit': 'rgba(0, 128, 0, 1)', 'Miss': 'rgba(255, 0, 0, 1)'} # Solid
    fill_map = {'Hit': 'rgba(0, 128, 0, 0.2)', 'Miss': 'rgba(255, 0, 0, 0.2)'} # Transparent
    line_map = {'Hit': 'rgba(0, 128, 0, 0.1)', 'Miss': 'rgba(255, 0, 0, 0.1)'} # Faint for individuals
    
    statuses = ['Hit', 'Miss']
    
    # 2. Add Individual Traces (Faint Background)
    # We group by trial to draw lines
    for status in statuses:
        df_status = df_all[df_all['Status'] == status]
        trials = df_status['Trial'].unique()
        
        # Add a dummy trace just for legend group toggle (optional)
        # It's better to just plot all lines. To avoid legend spam, we don't show these in legend.
        for trial in trials:
            t_data = df_status[df_status['Trial'] == trial]
            fig.add_trace(go.Scatter(
                x=t_data['Time'], y=t_data['Speed'],
                mode='lines',
                line=dict(color=line_map[status], width=1),
                showlegend=False,
                hoverinfo='skip', # Don't hover on background noise
                name=f"{status} (Individual)"
            ))

    # 3. Add Mean + Error Bands (The "Gaussian" Distribution)
    for status in statuses:
        df_status = df_all[df_all['Status'] == status]
        if df_status.empty: continue
            
        # Group by Frame (Time) to calculate Mean/Std across trials
        # We use Frame because Time is float and might have tiny diffs
        stats = df_status.groupby('Frame')['Speed'].agg(['mean', 'std', 'count']).reset_index()
        stats['Time'] = stats['Frame'] / FPS
        
        # Filter out times with too few trials if needed, but plotting all is usually fine
        
        x = stats['Time']
        y_mean = stats['mean']
        y_std = stats['std'].fillna(0) # If only 1 trial, std is NaN -> 0
        
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        # Clip lower bound to 0 (speed can't be negative)
        y_lower = y_lower.clip(lower=0)
        
        # Draw the Shaded Band (Upper then Lower, filled)
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]), # Go forward then backward
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=fill_map[status],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f"{status} Std Dev"
        ))
        
        # Draw the Mean Line
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            mode='lines',
            line=dict(color=color_map[status], width=3),
            name=f"{status} Mean"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Speed (cm/s)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    fig.write_html(f"{OUTPUT_DIR}/{filename}")

plot_summary_with_distribution(p1_traces, "Phase 1 Speed Distribution (Entry -> First ROI)", "summary_speed_P1_dist.html")
plot_summary_with_distribution(p2_traces, "Phase 2 Speed Distribution (First ROI -> Exit)", "summary_speed_P2_dist.html")

print(f"Analysis Complete! Results in {OUTPUT_DIR}")


# PLOTLY SUMMARY PLOTS
# print("Generating Plotly Summaries")

# def plot_summary_plotly(trace_data, title, filename):
#     if not trace_data: return
#     df_plot = pd.concat(trace_data)
#     color_map = {"Hit": "green", "Miss/Incorrect": "red"}
    
#     fig = px.line(df_plot, x='Time', y='Speed', 
#                   color='Status', 
#                   line_group='Trial', 
#                   color_discrete_map=color_map,
#                   title=title, 
#                   labels={'Time': 'Time (s)', 'Speed': 'Speed (px/s)'})
    
#     fig.update_layout(hovermode="x unified", template="plotly_white")
#     fig.write_html(f"{OUTPUT_DIR}/{filename}")

# plot_summary_plotly(p1_traces, "Phase 1 Speed (Entry -> First ROI) [Green=Hit, Red=Miss]", "summary_speed_P1_colored.html")
# plot_summary_plotly(p2_traces, "Phase 2 Speed (First ROI -> Exit) [Green=Hit, Red=Miss]", "summary_speed_P2_colored.html")

# print("Analysis Complete!")

