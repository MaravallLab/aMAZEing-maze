"""
Batch-process DeepLabCut data to generate per-trial 3D interactive trajectory plots.
python make_3d_dlc_plot_trajectories.py --dlc "C:/Users/shahd/OneDrive - University of Sussex/DLC_MOSEQ/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv" 
--segments "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_segments_manifest.csv" 
--trials-types "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/mouse6357_session3.6_trial_info.csv" 
--keypoint back_mid 
--output-dir "C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/trajectories_maze_video_back_mid_v2" 
--min-likelihood 0.8 
--px-per-cm 7.5
"""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import medfilt
import cv2 




class Config:
    #  File Paths 
    
    DLC_FILE = r"C:/Users/shahd/OneDrive - University of Sussex/DLC_MOSEQ/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
    SEGMENTS_FILE = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_segments_manifest.csv" 
    TRIALS_TYPES_FILE = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/mouse6357_session3.6_trial_info.csv" 
    OUTPUT_DIR = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/trajectories_maze_video_back_mid_v2" 
    
    # Optional: Path to maze geometry JSON (set to None if not used)
    MAZE_GEOMETRY_FILE = None  # e.g., r"C:/path/to/maze.json"

    #  Analysis Parameters 
    KEYPOINT = "backmid"        # Bodypart to track (nose, mid, backmid,  tailbase)
    MIN_LIKELIHOOD = 0.8     # Drop points with confidence below this score
    FPS = 30.0               # Frame rate of the video
    SMOOTHING_WINDOW = 5     # Window size for median filter smoothing (must be odd)
    PX_PER_CM = 7.5         # Set to float to convert pixels to cm. If you don't know it, run code\simplermaze\measure_pixel_per_cm.py


def read_dlc_csv_multiindex(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    except Exception as e:
        print(f"Error reading DLC file: {e}")
        df = pd.read_csv(path, header=[0, 1, 2])
        
    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) != 3:
        raise ValueError("DLC CSV does not appear to have a 3-level header.")
    return df

def get_scorer_and_bodyparts(df: pd.DataFrame) -> Tuple[str, List[str]]:
    level0 = sorted(set([lvl0 for (lvl0, _, _) in df.columns]))
    scorer_candidates = [s for s in level0 if s.lower() != "scorer"]
    if not scorer_candidates:
        raise ValueError("Could not find DLC 'scorer' level.")
    scorer = scorer_candidates[0]
    bodyparts = sorted(set([bp for (_, bp, _) in df.columns if bp.lower() != "bodyparts"]))
    return scorer, bodyparts

def extract_frame_series(df: pd.DataFrame) -> pd.Series:
    if ('scorer', 'bodyparts', 'coords') in df.columns:
        frames = df[('scorer', 'bodyparts', 'coords')].astype(int)
        frames.name = 'frame'
        return frames
    return pd.Series(np.arange(len(df)), name='frame')

def pick_keypoint(bodyparts: List[str], desired: Optional[str]) -> str:
    if desired and desired in bodyparts:
        return desired
    for candidate in ["nose", "mid", "center", "centroid", "tailbase"]:
        if candidate in bodyparts:
            print(f"Warning: Keypoint '{desired}' not found. Defaulting to '{candidate}'.")
            return candidate
    return bodyparts[0]

def load_maze_geometry(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            maze = json.load(f)
        for comp in maze.get('compartments', []):
            pts = np.array(comp['contour']).reshape(-1, 1, 2).astype(np.int32)
            comp['_cv2_contour'] = pts
        print(f" Loaded maze structure with {len(maze.get('compartments', []))} compartments")
        return maze
    except Exception as e:
        print(f"Error loading maze geometry: {e}")
        return None

def get_compartment(x: float, y: float, maze_structure: Dict) -> str:
    for comp in maze_structure.get('compartments', []):
        contour = comp.get('_cv2_contour')
        if contour is None:
            continue
        result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
        if result >= 0:
            return comp.get('id', 'Unknown')
    return 'Outside'

def decode_state_sequence(value: Any) -> str:
    if pd.isna(value): return "Unknown"
    mapping = {'1': 'Explore', '2': 'Exploit', '3': 'Nest', '4': 'Groom'}
    s_val = str(value).replace('"', '').replace("'", "")
    parts = [p.strip() for p in s_val.split(',')]
    decoded_parts = []
    for p in parts:
        if p.replace('.','',1).isdigit():
            p = str(int(float(p)))
        decoded_parts.append(mapping.get(p, p))
    return "-".join(decoded_parts)

def extract_trial_id_from_path(path_str: Any) -> str:
    if pd.isna(path_str): return ""
    path_str = str(path_str).strip()
    if len(path_str) < 13: return ""
    return path_str[-13:-4]

def process_trial_data(df_full: pd.DataFrame, frames: pd.Series, 
                       start_frame: int, end_frame: int, trial_id: str,
                       scorer: str, keypoint: str, min_likelihood: float, 
                       fps: float, smooth_window: int, maze_structure: Optional[Dict],
                       px_per_cm: Optional[float]) -> Optional[pd.DataFrame]:
    
    mask = (frames >= start_frame) & (frames <= end_frame)
    if mask.sum() == 0:
        print(f"Warning: No frames found for {trial_id}. Skipping.")
        return None
        
    df_trial = df_full.loc[mask].copy()
    xcol = (scorer, keypoint, 'x')
    ycol = (scorer, keypoint, 'y')
    lkcol = (scorer, keypoint, 'likelihood')
    
    if lkcol not in df_trial.columns:
        min_likelihood = 0.0
    
    if min_likelihood > 0:
        low_like_mask = df_trial[lkcol] < min_likelihood
        df_trial.loc[low_like_mask, [xcol, ycol]] = np.nan
        
    df_out = pd.DataFrame({
        'frame': df_trial.index if isinstance(df_trial.index, pd.RangeIndex) else frames[mask],
        'x': df_trial[xcol],
        'y': df_trial[ycol],
        'likelihood': df_trial[lkcol] if min_likelihood > 0 else 1.0
    })
    
    df_out['x'] = df_out['x'].interpolate(method='linear').bfill().ffill()
    df_out['y'] = df_out['y'].interpolate(method='linear').bfill().ffill()
    
    if df_out.isnull().values.any():
        return None

    if smooth_window > 1 and len(df_out) > smooth_window:
        df_out['x'] = medfilt(df_out['x'], kernel_size=smooth_window)
        df_out['y'] = medfilt(df_out['y'], kernel_size=smooth_window)
        
    #  Time and Speed Calculation 
    df_out['time_seconds'] = (df_out['frame'] - start_frame) / float(fps)
    
    # Speed in Pixels per Frame
    speed_px = np.sqrt(np.diff(df_out['x'])**2 + np.diff(df_out['y'])**2)
    df_out['speed_px_per_frame'] = np.concatenate([[0], speed_px])
    
    # Speed in CM per Second (if conversion factor provided)
    if px_per_cm and px_per_cm > 0:
        df_out['speed_cm_per_sec'] = (df_out['speed_px_per_frame'] * fps) / px_per_cm
    
    if maze_structure:
        df_out['compartment'] = df_out.apply(lambda row: get_compartment(row['x'], row['y'], maze_structure), axis=1)
        
    return df_out


#  PLOTTING FUNCTIONS


def plot_trial_3d(df_trial: pd.DataFrame, trial_id: str, 
                  maze_structure: Optional[Dict], plot_type: str = 'time',
                  subtitle: str = "") -> go.Figure:
    
    fig = go.Figure()
    
    #  Determine colouring 
    if plot_type == 'speed':
        if 'speed_cm_per_sec' in df_trial.columns:
            colour_data = df_trial['speed_cm_per_sec']
            colourbar_title = "Speed (cm/s)"
            hover_text = "Speed: %{line.color:.2f} cm/s<br>"
        elif 'speed_px_per_frame' in df_trial.columns:
            colour_data = df_trial['speed_px_per_frame']
            colourbar_title = "Speed (px/frame)"
            hover_text = "Speed: %{line.color:.2f} px/f<br>"
        else:
            colour_data = df_trial['time_seconds']
            colourbar_title = "Time"
            hover_text = ""
        colorscale = 'Jet'
        
    elif plot_type == 'compartment' and 'compartment' in df_trial.columns:
        unique_comps = df_trial['compartment'].unique()
        colour_map = {comp: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                     for i, comp in enumerate(unique_comps)}
        df_trial['colour'] = df_trial['compartment'].map(colour_map)
        
        for comp_name, comp_data in df_trial.groupby('compartment'):
            hover_str = (
                f"<b>{comp_name}</b><br>"
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}<br>"
                "Time: %{z:.2f}s<br>"
                "<extra></extra>"
            )
            
            fig.add_trace(go.Scatter3d(
                x=comp_data['x'],
                y=comp_data['y'],
                z=comp_data['time_seconds'],
                mode='lines+markers',
                line=dict(color=colour_map[comp_name], width=4),
                marker=dict(size=2),
                name=comp_name,
                hovertemplate=hover_str
            ))
        
        title_text = f'{trial_id} - {subtitle}' if subtitle else f'{trial_id}'
        fig.update_layout(title=title_text)
        return fig
        
    else: # Default to 'time'
        u = df_trial['time_seconds']
        u = 'Viridis'
        u = "Time (s)"
        hover_text = ""

    #  Add Main Trajectory Line 
    fig.add_trace(go.Scatter3d(
        x=df_trial['x'],
        y=df_trial['y'],
        z=df_trial['time_seconds'],
        mode='lines',
        line=dict(
            color=u,
            u=u,
            width=4,
            colorbar=dict(title=u, x=1.1)
        ),
        name='Trajectory',
        hovertemplate='<b>Position</b><br>' +
                      'X: %{x:.1f}<br>Y: %{y:.1f}<br>Time: %{z:.2f}s<br>' +
                      hover_text + '<extra></extra>'
    ))

    #  Add Start/End Markers 
    fig.add_trace(go.Scatter3d(
        x=[df_trial['x'].iloc[0]], y=[df_trial['y'].iloc[0]], z=[df_trial['time_seconds'].iloc[0]],
        mode='markers',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[df_trial['x'].iloc[-1]], y=[df_trial['y'].iloc[-1]], z=[df_trial['time_seconds'].iloc[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='diamond'),
        name='End'
    ))

    #  Add Maze Structure 
    if maze_structure is not None:
        for comp in maze_structure.get('compartments', []):
            contour = np.array(comp['contour'])
            contour_closed = np.vstack([contour, contour[0]]) 
            fig.add_trace(go.Scatter3d(
                x=contour_closed[:, 0],
                y=contour_closed[:, 1],
                z=np.zeros(len(contour_closed)),
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    #  Add Vertical Projection Lines 
    sample_rate = max(1, len(df_trial) // 50)
    for i in range(0, len(df_trial), sample_rate):
        row = df_trial.iloc[i]
        fig.add_trace(go.Scatter3d(
            x=[row['x'], row['x']], y=[row['y'], row['y']], z=[0, row['time_seconds']],
            mode='lines',
            line=dict(color='lightblue', width=1, dash='dot'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    title_text = f'{trial_id} - {subtitle}' if subtitle else f'{trial_id}'
    fig.update_layout(title=f'{title_text} (Coloured by {plot_type.capitalize()})')
    return fig

def set_common_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (px)', showbackground=True, backgroundcolor="rgb(230,230,230)"),
            yaxis=dict(title='Y (px)', showbackground=True, backgroundcolor="rgb(230,230,230)", autorange='reversed'),
            zaxis=dict(title='Time (s)', showbackground=True, backgroundcolor="rgb(230,230,230)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=1000, height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest'
    )
    return fig


#  MAIN 


def main():
    # Grab settings from the Global Config class
    outdir = Path(Config.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Fix window size for median filter (must be odd)
    smooth_win = Config.SMOOTHING_WINDOW
    if smooth_win % 2 == 0: 
        smooth_win += 1
        
    print("Starting batch analysis")
    if Config.PX_PER_CM:
        print(f" Converting Pixels to CM using ratio: {Config.PX_PER_CM} px/cm")

    try:
        df_full = read_dlc_csv_multiindex(Config.DLC_FILE)
        seg_df = pd.read_csv(Config.SEGMENTS_FILE)
        types_df = pd.read_csv(Config.TRIALS_TYPES_FILE)
    except Exception as e:
        print(f"Fatal Error reading files: {e}")
        print("Please check the paths in the 'USER CONFIGURATION' section at the top of the script.")
        return

    scorer, bodyparts = get_scorer_and_bodyparts(df_full)
    frames = extract_frame_series(df_full)
    keypoint = pick_keypoint(bodyparts, Config.KEYPOINT)
    maze = load_maze_geometry(Config.MAZE_GEOMETRY_FILE)

    # Clean Segments columns
    seg_df.columns = seg_df.columns.str.lower().str.strip()
    
    #  Prepare Segments Lookup Table 
    print("Building segment lookup table")
    
    seg_path_col = None
    possible_names = ['name', 'video', 'file', 'path', 'segment_path', 'video_segment_path']
    for c in seg_df.columns:
        if any(x in c for x in possible_names):
            seg_path_col = c
            break
    if not seg_path_col:
        seg_path_col = seg_df.columns[0]

    segments_lookup = {}
    for _, row in seg_df.iterrows():
        path_val = row[seg_path_col]
        
        # Segments Logic: [-7, -4] -> "001"
        if pd.notna(path_val):
            s_path = str(path_val).strip()
            if len(s_path) >= 7:
                seg_id = s_path[-7:-4] 
                segments_lookup[seg_id] = (row['start_frame'], row['end_frame'])

    print(f" Indexed {len(segments_lookup)} segments.")
    
    calculated_metrics = {} 

    # Process Each Trial in TRIALS TYPES
    print(f"\nProcessing {len(types_df)} trials")
    
    for idx, row in types_df.iterrows():
        # Trials Logic: [-13, -4] -> "trial_001"
        vid_path = row.get('video_segment_path', '')
        full_trial_id = extract_trial_id_from_path(vid_path)
        
        if not full_trial_id:
            continue
            
        if "_" in full_trial_id:
            trial_index = full_trial_id.split('_')[-1]
        else:
            trial_index = full_trial_id

        if trial_index not in segments_lookup:
            print(f"  {full_trial_id} -> Not found in Segments. Skipping.")
            continue
            
        print(f"  Processing {full_trial_id}", end=" ")
        
        start_f, end_f = segments_lookup[trial_index]
        state_label = decode_state_sequence(row.get('state_sequence', ''))
        
        df_trial = process_trial_data(df_full, frames, int(start_f), int(end_f), full_trial_id,
                                      scorer, keypoint, Config.MIN_LIKELIHOOD, Config.FPS, 
                                      smooth_win, maze, Config.PX_PER_CM)
                                      
        if df_trial is None or df_trial.empty:
            print(f"No data.")
            continue
            
        #Store Stats AND Frames
        stats = {
            'matched_trial_id': full_trial_id,
            'start_frame': start_f,
            'end_frame': end_f
        }
        
        if 'speed_cm_per_sec' in df_trial.columns:
            stats['avg_speed_cm_per_sec'] = df_trial['speed_cm_per_sec'].mean()
            stats['max_speed_cm_per_sec'] = df_trial['speed_cm_per_sec'].max()
            stats['min_speed_cm_per_sec'] = df_trial['speed_cm_per_sec'].min()
        else:
            stats['avg_speed_px_per_frame'] = df_trial['speed_px_per_frame'].mean()
            stats['max_speed_px_per_frame'] = df_trial['speed_px_per_frame'].max()
            stats['min_speed_px_per_frame'] = df_trial['speed_px_per_frame'].min()

        calculated_metrics[idx] = stats
        
        # Generate Plots
        plot_types = ['time', 'speed']
        if 'compartment' in df_trial.columns:
            plot_types.append('compartment')
            
        for plot_type in plot_types:
            fig = plot_trial_3d(df_trial, full_trial_id, maze, plot_type=plot_type, subtitle=state_label)
            fig = set_common_layout(fig)
            sanitized_state = state_label.replace(" ", "_").replace(",", "-")
            plot_path = outdir / f"{full_trial_id}_{sanitized_state}_{plot_type}.html"
            fig.write_html(str(plot_path))
            
        data_path = outdir / f"{full_trial_id}_data.csv"
        df_trial.to_csv(data_path, index=False)
        
        print(f" Done")

    # Export Summary
    print("\nGenerating summary CSV")
    metrics_df = pd.DataFrame.from_dict(calculated_metrics, orient='index')
    final_df = types_df.merge(metrics_df, left_index=True, right_index=True, how='left')
    
    summary_path = outdir / "trials_with_metrics_matched.csv"
    final_df.to_csv(summary_path, index=False)
    print(f"Saved matched summary to: {summary_path}")

if __name__ == "__main__":
    main()