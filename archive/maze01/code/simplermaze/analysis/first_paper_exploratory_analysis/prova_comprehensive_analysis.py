import pandas as pd
import numpy as np
import os
import json # <-- Added to load the calibration file
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.stats import norm, mannwhitneyu
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import analysisfunct as af
from analysisfunc_config import Paths


# 1. ANALYSIS CONTROL PANEL (TOGGLES)

RUN_GAUSSIAN_SPEED    = True   # Plot speed distribution fits
RUN_GAUSSIAN_DURATION = True   # Plot duration distribution fits
RUN_ALIGNED_SPEED     = True   # Speed vs Time (T=0 at ROI Entry)
MAKE_MASTER_CSV       = True   # Consolidate all metrics to one CSV
RUN_STATISTICAL_TESTS = True   # Stats + Annotated Boxplots
GENERATE_HEATMAPS     = True   # Save individual trial density maps


# 2. SETUP & DATA PREPARATION


print("Initializing Consolidated Master Pipeline...")
FPS = Paths.FPS
OUTPUT_DIR = os.path.join(Paths.session_path, "total_analysis_output")
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Load data
df_trials = pd.read_csv(Paths.TRIAL_INFO_PATH)
df_dlc = pd.read_csv(Paths.DLC_DATA_PATH, header=[0, 1, 2], index_col=0)
scorer = df_dlc.columns[0][0]
tracking = df_dlc[scorer][Paths.BODYPART].copy()

# Global Data Cleaning
mask = tracking['likelihood'] < Paths.LIKELIHOOD_THRESH
tracking.loc[mask, ['x', 'y']] = np.nan
tracking = tracking.interpolate().ffill().bfill()
tracking = af.process_kinematics(tracking, FPS, Paths.PX_PER_CM)

# Load ROIs
df_r = pd.read_csv(os.path.join(Paths.session_path, "rois1.csv"), index_col=0)
rois = {name: tuple(row) for name, row in df_r.iterrows()}

# --- NEW: Load Calibrated Path Distances ---
try:
    path_json = os.path.join(Paths.session_path, "true_path_distances.json")
    with open(path_json, 'r') as f:
        true_distances_cm = json.load(f)
    print("Successfully loaded calibrated path distances.")
except FileNotFoundError:
    print("WARNING: true_path_distances.json not found. Falling back to 0 for missing data.")
    true_distances_cm = {}
# ------------------------------------------

# 3. LENIENT UNIFIED PROCESSING LOOP (STRING-SLICE TRIAL IDS)
master_results = []
aligned_traces = []

print("Processing trials. Extracting Trial IDs from video_segment_path...")

for idx, row in df_trials.iterrows():
    try:
        # 1. Extract Trial ID
        path_str = str(row['video_path'])
        current_trial_id = path_str[-7:-4] 
        
        f_start, f_end = int(row['start_trial_frame']), int(row['end_trial_frame'])
        
        # Type safety for targets
        target_val = row['first_reward_area_visited']
        target = str(target_val).strip() if pd.notna(target_val) else "Unknown"
        status = "Hit" if (row['hit'] == 1) else "Miss"
        
        # Case-insensitive ROI matching
        roi_match = next((name for name in rois if str(name).lower() == target.lower()), None)
        
        if f_start >= f_end or roi_match is None:
            continue
        
        # 2. Precise slicing using .loc
        trial_data = tracking.loc[f_start:f_end].copy()

        # Find frames where the AI actually sees the animal
        is_visible = trial_data['likelihood'] >= Paths.LIKELIHOOD_THRESH
        if is_visible.any():
            first_visible_idx = is_visible.idxmax() # Gets the first 'True' index
            trial_data = trial_data.loc[first_visible_idx:]
        
        if len(trial_data) < 1: 
            continue

        # 3. Split Point Detection (20-pixel buffer)
        rx, ry, rw, rh = rois[roi_match]
        buffer = 20  
        split_idx = -1
        
        for i, (f_idx, pos) in enumerate(trial_data.iterrows()):
            if (rx-buffer <= pos['x_smooth'] <= rx+rw+buffer) and \
               (ry-buffer <= pos['y_smooth'] <= ry+rh+buffer):
                split_idx = i
                break
        
        # Fallback for P2
        if split_idx == -1:
            split_idx = int(len(trial_data) * 0.8)
        
        p1 = trial_data.iloc[:split_idx]
        p2 = trial_data.iloc[split_idx:]
        
        # Calculate Initial P1 Metrics
        p1_duration_s = len(p1) / FPS
        p1_mean_speed = p1['speed'].mean() if not p1.empty else 0
        p1_entropy = af.get_entropy(p1['x_smooth'], p1['y_smooth']) if not p1.empty else 0
        
        # --- MODIFICATION 1: Skip 'Miss' trials ONLY if speed is 0 ---
        if status == "Miss" and p1_mean_speed == 0:
            continue
        # 

        # --- MODIFICATION 2: Recalculate Speed using Calibrated Distances ---
        if status == "Hit" and p1_mean_speed == 0:
            time_to_reward_ms = row.get('time_to_reward', pd.NA) 
            
            # Ensure we have a valid time value greater than 0
            if pd.notna(time_to_reward_ms) and float(time_to_reward_ms) > 0:
                time_s = float(time_to_reward_ms) / 1000.0  # Convert ms to seconds
                
                # Fetch the exact physical distance you drew for this specific target
                dist_cm = true_distances_cm.get(target, 0)
                
                if dist_cm > 0:
                    # Overwrite the missing p1 metrics
                    p1_mean_speed = dist_cm / time_s
                    p1_duration_s = time_s  

                    # set entropy to NaN so that it doesn't bias the results
                    p1_entropy = np.nan
                else:
                    print(f"  Warning: No true distance found for {target} in trial {current_trial_id}")
        # -------

        # 4. Append Results
        master_results.append({
            'trial_id': current_trial_id, 
            'status': status, 
            'target': target,
            'p1_duration_s': p1_duration_s, 
            'p1_mean_speed': p1_mean_speed,
            'p1_entropy': p1_entropy,
            'p2_duration_s': len(p2) / FPS, 
            'p2_mean_speed': p2['speed'].mean() if not p2.empty else 0,
            'p2_entropy': af.get_entropy(p2['x_smooth'], p2['y_smooth']) if not p2.empty else 0
        })
        
        # Aligned traces
        time_rel = (np.arange(len(trial_data)) - split_idx) / FPS
        aligned_traces.append(pd.DataFrame({
            'Relative_Time': time_rel, 
            'Speed': trial_data['speed'].values,
            'Status': status, 
            'Trial': current_trial_id
        }))
        
    except Exception as e:
        print(f"  Error at row {idx}: {str(e)}")
        continue

df_master = pd.DataFrame(master_results)
if not df_master.empty:
    print(f"Successfully processed {len(df_master)} trials.")



# 4. CONDITIONAL OUTPUTS & PLOTTING


if MAKE_MASTER_CSV:
    df_master.to_csv(os.path.join(OUTPUT_DIR, "master_behavioural_data.csv"), index=False)

if RUN_GAUSSIAN_SPEED or RUN_GAUSSIAN_DURATION:
    def plot_gaussians(df, metric_suffix, unit, title, filename):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"P1 {title}", f"P2 {title}"), shared_yaxes=True)
        for i, phase in enumerate(['p1', 'p2']):
            col = f"{phase}_{metric_suffix}"
            for status, color in zip(['Hit', 'Miss'], ['rgba(0,128,0,1)', 'rgba(255,0,0,1)']):
                subset = df[df['status'] == status][col].dropna()
                if len(subset) < 2: continue
                
                mu, std = subset.mean(), subset.std()
                if std == 0 or np.isnan(std): std = 0.1
                
                x = np.linspace(max(0, mu - 4*std), mu + 4*std, 200)
                y = norm.pdf(x, mu, std)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y, name=f"{status} (μ={mu:.1f} {unit})",
                    line=dict(color=color, width=3), fill='tozeroy'
                ), row=1, col=i+1)
        
        fig.update_layout(template="plotly_white", title_text=f"Combined {title} Distributions")
        fig.write_html(os.path.join(OUTPUT_DIR, filename))

    if RUN_GAUSSIAN_SPEED: plot_gaussians(df_master, 'mean_speed', 'cm/s', 'Speed', 'speed_gaussians.html')
    if RUN_GAUSSIAN_DURATION: plot_gaussians(df_master, 'duration_s', 's', 'Duration', 'duration_gaussians.html')

if RUN_STATISTICAL_TESTS:
    print("Running Global Statistics...")
    metrics = ['p1_duration_s', 'p1_mean_speed', 'p1_entropy', 'p2_duration_s', 'p2_mean_speed', 'p2_entropy']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        sns.boxplot(data=df_master, x='status', y=m, ax=axes[i], hue='status', palette={'Hit': 'green', 'Miss': 'red'}, legend=False)
        sns.stripplot(data=df_master, x='status', y=m, ax=axes[i], color='black', alpha=0.3)
        h, ms = df_master[df_master['status']=='Hit'][m].dropna(), df_master[df_master['status']=='Miss'][m].dropna()
        if len(h) > 0 and len(ms) > 0:
            _, p = mannwhitneyu(h, ms)
            axes[i].set_title(f"{m}\np={p:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_boxplots.png"))

if GENERATE_HEATMAPS:
    print("Generating High-Resolution Heatmaps...")
    heatmap_dir = os.path.join(OUTPUT_DIR, "trial_heatmaps")
    if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
    
    # Load boundaries for clipping
    boundary_pts = pd.read_csv(os.path.join(Paths.session_path, 'maze_boundary.csv')).values.tolist()
    bx, by = zip(*boundary_pts)
    xmin, xmax = min(bx), max(bx)
    ymin, ymax = min(by), max(by)

    for idx, row in df_trials.iterrows():
        try:
            # --- 1. PREPARE DATA ---
            trial_data = tracking.loc[int(row["start_trial_frame"]):int(row["end_trial_frame"])].copy()
            
            # Re-calculate these for the label
            target_val = row['first_reward_area_visited']
            target = str(target_val).strip() if pd.notna(target_val) else "Unknown"
            status_label = "Hit" if (row['hit'] == 1) else "Miss"
            
            # Coordinate check (CM vs Pixel fix)
            x_plot = trial_data['x_smooth'].values
            y_plot = trial_data['y_smooth'].values
            
            if len(x_plot) > 0 and x_plot.max() < (xmax / 10): 
                x_plot *= Paths.PX_PER_CM
                y_plot *= Paths.PX_PER_CM

            # Filter NaNs
            valid_mask = ~np.isnan(x_plot) & ~np.isnan(y_plot)
            x_plot, y_plot = x_plot[valid_mask], y_plot[valid_mask]

            if len(x_plot) < 5: 
                print(f"Skipping heatmap {idx}: No valid tracking points.")
                continue

            # --- 2. ADVANCED PLOTTING ---
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
            ax.set_facecolor('black') 
            
            # A. The Clipped Heatmap
            if len(x_plot) > 1:
                # Create the histogram
                counts, xedges, yedges, image = ax.hist2d(x_plot, y_plot, bins=30, cmap='inferno', norm=LogNorm())
                
                # Clip it to the maze shape
                clip_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='none') 
                ax.add_patch(clip_patch)
                image.set_clip_path(clip_patch)
                
                # Draw the white outline of the maze
                draw_patch = patches.Polygon(boundary_pts, closed=True, transform=ax.transData, facecolor='none', edgecolor='white', linewidth=2)
                ax.add_patch(draw_patch)
                
                # Custom Colorbar
                cbar = plt.colorbar(image, ax=ax, label='Density')
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.outline.set_edgecolor('white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                plt.setp(cbar.ax.yaxis.label, color='white')
            
                # B. ROI Visualization
                # Green box = Where they actually visited
            if 'reward_area' in row and pd.notna(row['reward_area']):
                    correct_target = str(row['reward_area']).strip()
                    
                    if correct_target in rois:
                        crx, cry, crw, crh = rois[correct_target]
                        # Draw dashed cyan box
                        rect_correct = patches.Rectangle((crx, cry), crw, crh, linewidth=2, 
                                                        edgecolor='cyan', facecolor='none', linestyle='--')
                        ax.add_patch(rect_correct)
                        # Label it
                        ax.text(crx, cry-25, f"GOAL: {correct_target}", color='cyan', fontsize=8, fontweight='bold')

                # 2. Plot the "VISITED" (Lime Solid Box)
            if target in rois:
                rx, ry, rw, rh = rois[target]
                
                # Logic: If Hit, the boxes overlap. If Miss, they are separate.
                if status_label == "Hit":
                    # Just emphasize the hit
                    rect_visit = patches.Rectangle((rx, ry), rw, rh, linewidth=2, 
                                                edgecolor='lime', facecolor='none')
                    ax.add_patch(rect_visit)
                    ax.text(rx, ry-10, "HIT!", color='lime', fontsize=10, fontweight='bold')
                
                else:
                    # It's a Miss - Draw the error box
                    rect_visit = patches.Rectangle((rx, ry), rw, rh, linewidth=2, 
                                                edgecolor='red', facecolor='none') # Red for error? or Lime?
                    ax.add_patch(rect_visit)
                    ax.text(rx, ry-10, f"Visited: {target}", color='red', fontsize=8, fontweight='bold')
        
            # C. Final Polish
            ax.set_xlim(xmin-10, xmax+10)
            ax.set_ylim(ymin-10, ymax+10)
            ax.invert_yaxis() # Fix upside-down issue
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"Trial {idx} ({status_label})", color='white')

            fig.savefig(os.path.join(heatmap_dir, f"trial_{idx}_heatmap_{status_label}.png"), bbox_inches='tight', facecolor='black', dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Heatmap Error at trial {idx}: {e}")
            plt.close()
            continue

print(f"Analysis Complete. Results in: {OUTPUT_DIR}")