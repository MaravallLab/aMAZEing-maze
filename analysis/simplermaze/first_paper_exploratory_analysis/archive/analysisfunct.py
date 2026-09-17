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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from analysisfunc_config import Paths



def resize_for_display(img, max_height=800):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv.resize(img, (int(w*scale), int(h*scale))), scale
    return img.copy(), 1.0

def click(event, x, y, flags, param, display_frame, points):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv.circle(display_frame, (x, y), 4, (0, 0, 255), -1)
        if len(points) > 1: 
            cv.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
        cv.imshow("Draw Boundary", display_frame)

def select_maze_boundary(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return []

    display_frame, scale = resize_for_display(frame)
    points = []

    cv.namedWindow("Draw Boundary", cv.WINDOW_NORMAL)
    cv.imshow("Draw Boundary", display_frame)
    cv.setMouseCallback("Draw Boundary", lambda event, x, y, flags, param: click(event, x, y, flags, param, display_frame, points))
    print("Draw Boundary: Click corners -> 'c' to close -> Space to confirm")
    while True:
        k = cv.waitKey(1) & 0xFF
        if k == ord('c') and len(points) > 2:
            cv.line(display_frame, points[-1], points[0], (0, 255, 0), 2)
            cv.imshow("Draw Boundary", display_frame)
        if k == 32 or k == 13: break
    cv.destroyAllWindows()
    return [(int(p[0]/scale), int(p[1]/scale)) for p in points]

def define_rois(video_path, roi_names):
    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()

    display_frame, scale = resize_for_display(frame)
    cv.namedWindow("Select ROIs", cv.WINDOW_NORMAL)
    rois = {}

    print("Draw ROIs: Select box -> Space to confirm")
    for name in roi_names:
        print(f"Select: {name}")
        r = cv.selectROI("Select ROIs", display_frame, fromCenter=False)
        real_r = (int(r[0]/scale), int(r[1]/scale), int(r[2]/scale), int(r[3]/scale))
        rois[name] = real_r

        # showing rois selected
        cv.rectangle(display_frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0,255,0), 2)
        cv.putText(display_frame, name, (r[0], r[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv.imshow("Select ROIs", display_frame)
    cv.destroyAllWindows()
    return rois

def filter_by_boundary(df, boundary_points):
    poly = Path(boundary_points)
    scorer = df.columns[0][0]
    bp = df.columns[0][1]
    coords = df[scorer][bp][['x', 'y']].fillna(-100).values
    is_inside = poly.contains_points(coords)
    df.loc[~is_inside, (scorer, bp, 'x')] = np.nan
    df.loc[~is_inside, (scorer, bp, 'y')] = np.nan
    return df

def process_kinematics(df, fps, px_per_cm = Paths.PX_PER_CM):
    # smooth coordinates
    df['x_smooth'] = savgol_filter(df['x'], 7, 3)
    df['y_smooth'] = savgol_filter(df['y'], 7, 3)

    #calculate pixel distance
    dist_px = np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2)
    
    # Convert to speed in cm/s
    # Distance (cm) = Distance (px) / px_per_cm
    # Speed (cm/s) = Distance (cm) * FPS  

    df['speed'] = (dist_px / px_per_cm) * fps
    return df


def get_entropy(x, y, grid_size=20):
    # Filter out NaNs (e.g., points outside the maze boundary)
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # If we have less than 2 valid points, we can't calculate entropy
    if len(x_valid) < 2: 
        return 0
    hist, _, _ = np.histogram2d(x_valid, y_valid, bins=grid_size, density=True)

    # Calculate entropy only on non-zero bins to avoid errors
    probs = hist.flatten()
    return entropy(probs[probs > 0])

def calculate_duration(df_segment, fps):
    """Returns duration in seconds for a given dataframe segment."""
    return len(df_segment) / fps


# COMBINED GAUSSIAN PLOTTING FUNCTION (SHARED Y)

def plot_combined_gaussian(df, filename):
    if df.empty: return

    # Create subplots: 1 Row, 2 Columns
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Phase 1: Entry -> ROI", "Phase 2: ROI -> Exit"),
        horizontal_spacing=0.1,
        shared_yaxes=True # This aligns the Y ticks
    )
    
    # Styles
    styles = {
        'Hit':  {'line': 'rgba(0, 128, 0, 1)',   'fill': 'rgba(0, 128, 0, 0.1)'},
        'Miss': {'line': 'rgba(255, 0, 0, 1)',   'fill': 'rgba(255, 0, 0, 0.1)'}
    }
    
    phases = ['P1', 'P2']
    
    global_max_speed = df['MeanSpeed'].max() * 1.3
    x_range = np.linspace(0, global_max_speed, 1000)
    
    # Store max height found to manually set Y limit later
    global_max_density = 0

    for i, phase in enumerate(phases):
        col_idx = i + 1
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
            
            # Track max Y for scaling
            current_max_y = max(y)
            if current_max_y > global_max_density:
                global_max_density = current_max_y
            
            # Add Trace
            fig.add_trace(go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f"{status} (μ={mu:.1f})",
                line=dict(color=styles[status]['line'], width=4),
                fill='tozeroy', 
                fillcolor=styles[status]['fill'],
                legendgroup=status,
                showlegend=(col_idx==1)
            ), row=1, col=col_idx)
            
            # Add Vertical Line
            fig.add_vline(x=mu, line_width=1, line_dash="dash", 
                          line_color=styles[status]['line'], row=1, col=col_idx)

    # Add separator line
    fig.add_shape(type="line",
        x0=0.5, y0=0, x1=0.5, y1=1,
        xref="paper", yref="paper",
        line=dict(color="Black", width=2)
    )

    # Force both Y-axes to the same global maximum
    y_limit = global_max_density * 1.1 # Add 10% headroom
    fig.update_yaxes(range=[0, y_limit], row=1, col=1)
    fig.update_yaxes(range=[0, y_limit], row=1, col=2)

    fig.update_layout(
        title="Speed Distributions: Hits vs Misses (Gaussian Fit)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        width=1200
    )
    
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Speed (cm/s)", row=1, col=2)
    # Only need Y label on left plot
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.write_html(f"{filename}")




## for the new format
def despine(ax):
    """Remove top and right spines for a clean look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

def add_scale_bar(ax, px_per_cm = Paths.PX_PER_CM, length_cm=10, location=(0.9, 0.05)):
    """Draws a physical scale bar on the heatmap."""
    bar_len_px = length_cm * px_per_cm
    # Get current limits to position relatively
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    
    x_start = xlim[0] + w * location[0]
    y_start = ylim[0] + h * location[1]
    
    # Draw line
    ax.plot([x_start - bar_len_px, x_start], [y_start, y_start], 
            color='white', linewidth=3)
    # Add Label
    ax.text(x_start - bar_len_px/2, y_start - (h*0.05), f"{length_cm} cm", 
            color='white', ha='center', fontsize=7, fontweight='bold')
