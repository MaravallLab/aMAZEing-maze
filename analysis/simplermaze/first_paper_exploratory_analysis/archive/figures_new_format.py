import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy, norm, sem
from scipy.signal import savgol_filter
import os
import analysisfunct as af
from analysisfunc_config import Paths 



# 0. PUBLICATION STYLE CONFIGURATION

# Set global font/style settings for "Nature" aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 7        # Tick labels
plt.rcParams['axes.labelsize'] = 8   # Axis titles
plt.rcParams['axes.titlesize'] = 9   # Subplot titles
plt.rcParams['legend.fontsize'] = 7  # Legend text
plt.rcParams['svg.fonttype'] = 'none' # Text as text, not paths (editable)
plt.rcParams['axes.linewidth'] = 0.5  # Thin axis lines
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# Colors (Colorblind safe / Professional)
COLOR_HIT = '#2CA02C'  # Professional Green
COLOR_MISS = '#D62728' # Professional Red
COLOR_NEUTRAL = 'black'


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

OUTPUT_DIR = session_path + r"publication_figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



# MAIN EXECUTION

print("Loading Data...")
df_trials = pd.read_csv(TRIAL_INFO_PATH)
df_dlc = pd.read_csv(DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][BODYPART].copy()

# 1. Filters
mask = tracking['likelihood'] < LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()

boundary_file = session_path + 'maze_boundary.csv'
if DRAW_BOUNDARIES:
    boundary_pts = af.select_maze_boundary(VIDEO_PATH)
    pd.DataFrame(boundary_pts).to_csv(boundary_file, index=False)
elif os.path.exists(boundary_file):
    try:
        df_b = pd.read_csv(boundary_file)
        if 'x' in df_b.columns: boundary_pts = df_b[['x', 'y']].values.tolist()
        else: boundary_pts = pd.read_csv(boundary_file, header=None).values.tolist()
    except: exit("Boundary Error")
else: exit("Set DRAW_BOUNDARIES=True")

temp_df = pd.DataFrame({(scorer, BODYPART, 'x'): tracking['x'], (scorer, BODYPART, 'y'): tracking['y']})
temp_df = af.filter_by_boundary(temp_df, boundary_pts)
tracking['x'] = temp_df[(scorer, BODYPART, 'x')]
tracking['y'] = temp_df[(scorer, BODYPART, 'y')]
tracking = af.process_kinematics(tracking, FPS, PX_PER_CM)

rois_file = session_path + "rois1.csv"
roi_names = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"]
if DRAW_ROIS:
    rois = af.define_rois(VIDEO_PATH, roi_names)
    pd.DataFrame(rois).T.to_csv(rois_file)
elif os.path.exists(rois_file):
    df_r = pd.read_csv(rois_file, index_col=0)
    rois = {name: tuple(row) for name, row in df_r.iterrows()}
else: exit("Set DRAW_ROIS=True")

# DATA CONTAINERS
results = []
aligned_traces = [] 
gaussian_data = []

print("Processing Trials...")
for idx, row in df_trials.iterrows():
    try:
        f_start, f_end = int(row['start_frame']), int(row['end_frame'])
        target = row['first_reward_area_visited']
        is_hit = (row['hit'] == 1)
        status = "Hit" if is_hit else "Miss"
        reward_roi_name = "rew" + str(row['rew_location'])
    except: continue
    
    if f_start >= f_end or target not in rois: continue
    try: trial_data = tracking.iloc[f_start:f_end].copy()
    except: continue
    if len(trial_data) < 10: continue

    # Split Phase
    rx, ry, rw, rh = rois[target]
    split_idx = -1
    for i, (f, pos) in enumerate(trial_data.iterrows()):
        if np.isnan(pos['x_smooth']): continue
        if (rx <= pos['x_smooth'] <= rx+rw) and (ry <= pos['y_smooth'] <= ry+rh):
            split_idx = i
            break
    if split_idx == -1: continue

    p1 = trial_data.iloc[:split_idx]
    p2 = trial_data.iloc[split_idx:]
    
    # Store Gaussian Data
    if len(p1) > 0: gaussian_data.append({'Phase': 'P1', 'Status': status, 'MeanSpeed': p1['speed_cm_s'].mean()})
    if len(p2) > 0: gaussian_data.append({'Phase': 'P2', 'Status': status, 'MeanSpeed': p2['speed_cm_s'].mean()})

    # Store Aligned Traces
    frames_relative = np.arange(len(trial_data)) - split_idx
    time_relative = frames_relative / FPS
    aligned_traces.append(pd.DataFrame({
        'Relative_Time': time_relative,
        'Speed': trial_data['speed_cm_s'].values,
        'Status': status
    }))


    # FIGURE 1: HEATMAP (Vector PDF)

    # 89mm width (single column) is standard ~3.5 inches
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor='black')
    ax.set_facecolor('black')
    
    valid = trial_data['x_smooth'].notna() & trial_data['y_smooth'].notna()
    x, y = trial_data.loc[valid, 'x_smooth'], trial_data.loc[valid, 'y_smooth']
    
    if len(x) > 1:
        # Heatmap
        counts, xedges, yedges, image = ax.hist2d(x, y, bins=30, cmap='inferno', norm=LogNorm())
        
        # Masking
        clip_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(clip_patch)
        image.set_clip_path(clip_patch)
        
        # Outline (White, thick)
        draw_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='white', linewidth=1.5)
        ax.add_patch(draw_patch)
        
        # Scale Bar
        af.add_scale_bar(ax, PX_PER_CM, length_cm=10)

    # ROIs
    rect = patches.Rectangle((rx, ry), rw, rh, linewidth=1.5, edgecolor=COLOR_HIT, facecolor='none')
    ax.add_patch(rect)
    
    if reward_roi_name in rois and reward_roi_name != target:
        rrx, rry, rrw, rrh = rois[reward_roi_name]
        rect_rew = patches.Rectangle((rrx, rry), rrw, rrh, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
        ax.add_patch(rect_rew)

    # Clean up axes
    bx, by = zip(*boundary_pts)
    ax.set_xlim(min(bx)-20, max(bx)+20)
    ax.set_ylim(min(by)-20, max(by)+20)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save Vector PDF
    plt.savefig(f"{OUTPUT_DIR}/trial_{idx}_heatmap.pdf", format='pdf', bbox_inches='tight', facecolor='black')
    plt.close()


# FIGURE 2: SPEED TRACES (Aligned Mean +/- SEM)

print("Generating Speed Trace Figure...")
df_aligned = pd.concat(aligned_traces)
df_aligned['Time_Rounded'] = df_aligned['Relative_Time'].round(1)

# Group statistics
stats = df_aligned.groupby(['Status', 'Time_Rounded'])['Speed'].agg(['mean', 'sem']).reset_index()

# Setup Plot (Single Column Width ~89mm = 3.5 inches)
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

for status, color in [('Hit', COLOR_HIT), ('Miss', COLOR_MISS)]:
    subset = stats[stats['Status'] == status].sort_values('Time_Rounded')
    
    # Smooth the curve slightly for display if needed, or plot raw mean
    x = subset['Time_Rounded']
    y = subset['mean']
    err = subset['sem'].fillna(0)
    
    # Shaded Error
    ax.fill_between(x, y-err, y+err, color=color, alpha=0.2, linewidth=0)
    # Mean Line
    ax.plot(x, y, color=color, linewidth=1.5, label=status)

# Formatting
ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time from ROI Entry (s)')
ax.set_ylabel('Speed (cm/s)')
ax.set_xlim(-4, 4) # Zoom in to relevant window
af.despine(ax)
ax.legend(frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Summary_Speed_Trace.pdf", format='pdf')
plt.close()


# FIGURE 3: GAUSSIAN DISTRIBUTIONS (Side-by-Side)

print("Generating Gaussian Figure...")
df_gauss = pd.DataFrame(gaussian_data)

# Setup Plot (Double Column Width ~180mm = 7 inches)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=300, sharey=True)
phases = [('P1', 'Phase 1\n(Entry → ROI)'), ('P2', 'Phase 2\n(ROI → Exit)')]

# Determine global max density for Y-limit scaling
global_max_density = 0
x_range = np.linspace(0, df_gauss['MeanSpeed'].max() * 1.2, 500)

for ax, (phase_code, phase_name) in zip(axes, phases):
    subset = df_gauss[df_gauss['Phase'] == phase_code]
    
    for status, color in [('Hit', COLOR_HIT), ('Miss', COLOR_MISS)]:
        grp = subset[subset['Status'] == status]
        if grp.empty: continue
        
        mu, sigma = grp['MeanSpeed'].mean(), grp['MeanSpeed'].std()
        if pd.isna(sigma) or sigma == 0: sigma = 0.1
        
        y = norm.pdf(x_range, mu, sigma)
        global_max_density = max(global_max_density, y.max())
        
        # Plot
        ax.plot(x_range, y, color=color, linewidth=2, label=status)
        ax.fill_between(x_range, y, color=color, alpha=0.1)
        ax.axvline(mu, color=color, linestyle=':', linewidth=1)

    ax.set_title(phase_name)
    ax.set_xlabel('Mean Speed (cm/s)')
    af.despine(ax)

axes[0].set_ylabel('Probability Density')
axes[0].set_ylim(0, global_max_density * 1.1)
axes[0].legend(frameon=False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Summary_Gaussian_Dist.pdf", format='pdf')
plt.close()

print(f"Publication-ready figures saved to: {OUTPUT_DIR}")