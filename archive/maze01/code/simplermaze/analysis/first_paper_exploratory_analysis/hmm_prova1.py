import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
DLC_FILE = "C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/deeplabcut/mouse6357/mouse6357-shahd-2025-09-08/videos/6357_2024-08-28_11_58_14s3.6DLC_Resnet50_mouse6357Sep8shuffle1_snapshot_200.csv"
FPS = 30
SMOOTH_WINDOW = 15
VTE_WINDOW = 30

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_dlc_data(filename):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]
    
    data = pd.DataFrame()
    for bp in ['nose', 'mid', 'tailbase']:
        data[f'{bp}_x'] = df[scorer][bp]['x']
        data[f'{bp}_y'] = df[scorer][bp]['y']
        data[f'{bp}_p'] = df[scorer][bp]['likelihood']
    return data

df_tracking = load_dlc_data(DLC_FILE)
print("Data loaded.")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# Calculate Velocity
dx = np.diff(df_tracking['mid_x'], prepend=df_tracking['mid_x'][0])
dy = np.diff(df_tracking['mid_y'], prepend=df_tracking['mid_y'][0])
raw_vel = np.sqrt(dx**2 + dy**2) * FPS
df_tracking['velocity'] = savgol_filter(raw_vel, window_length=SMOOTH_WINDOW, polyorder=3)
df_tracking['log_velocity'] = np.log(df_tracking['velocity'] + 1e-6)

# Calculate IdPhi (Heading Changes)
nose_dx = np.diff(df_tracking['nose_x'], prepend=df_tracking['nose_x'][0])
nose_dy = np.diff(df_tracking['nose_y'], prepend=df_tracking['nose_y'][0])
heading = np.degrees(np.arctan2(nose_dy, nose_dx))
dphi = np.diff(heading, prepend=heading[0])
dphi = (dphi + 180) % 360 - 180
df_tracking['idphi'] = pd.Series(np.abs(dphi)).rolling(window=VTE_WINDOW, center=True).sum().fillna(0)

# Calculate Body Length
df_tracking['body_length'] = np.sqrt(
    (df_tracking['nose_x'] - df_tracking['tailbase_x'])**2 + 
    (df_tracking['nose_y'] - df_tracking['tailbase_y'])**2
)

# ==========================================
# 4. HMM SEGMENTATION
# ==========================================
print("Running HMM...")
features = ['log_velocity', 'idphi', 'body_length']
X = np.nan_to_num(df_tracking[features].values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model.fit(X_scaled)
df_tracking['state'] = model.predict(X_scaled)

# Map States
state_means = pd.DataFrame(model.means_, columns=features)
nest_id = state_means['log_velocity'].idxmin()
exploit_id = state_means['log_velocity'].idxmax()
explore_id = list(set([0, 1, 2]) - {nest_id, exploit_id})[0]
state_map = {nest_id: 'Nest', explore_id: 'Explore', exploit_id: 'Exploit'}
df_tracking['state_name'] = df_tracking['state'].map(state_map)

# Color mapping for Plotly
color_map = {'Nest': '#1f77b4', 'Explore': '#ff7f0e', 'Exploit': '#2ca02c'}

# ==========================================
# 5. INTERACTIVE PLOTLY VISUALIZATIONS
# ==========================================

# --- A. Interactive Digital Ethogram ---
# We use a Scattergl plot for performance with large datasets
print("Generating Ethogram...")
fig_ethogram = px.scatter(
    df_tracking.reset_index(), 
    x='index', 
    y='state_name', 
    color='state_name',
    color_discrete_map=color_map,
    title='Interactive Digital Ethogram (Zoom/Pan enabled)',
    labels={'index': 'Frame', 'state_name': 'Behavioral State'},
    hover_data=['velocity', 'idphi'],
    height=400,
    render_mode='webgl' # Critical for speed with large video files
)

# Customize layout to look like a strip chart
fig_ethogram.update_traces(marker=dict(size=5, symbol='square'))
fig_ethogram.update_layout(
    xaxis_title="Frame Number",
    yaxis_title="State",
    showlegend=False
)
fig_ethogram.show()

# --- B. Interactive Spatial Map ---
# Plot trajectory colored by state
print("Generating Spatial Map...")
fig_spatial = px.scatter(
    df_tracking, 
    x='mid_x', 
    y='mid_y', 
    color='state_name',
    color_discrete_map=color_map,
    title='Spatial Distribution of Behavioral States',
    labels={'mid_x': 'X Position (pixels)', 'mid_y': 'Y Position (pixels)'},
    hover_data=['velocity'],
    height=700,
    width=800,
    render_mode='webgl'
)

# Fix aspect ratio and invert Y axis (standard for video coords)
fig_spatial.update_yaxes(autorange="reversed")
fig_spatial.update_layout(yaxis_scaleanchor="x")
fig_spatial.show()

# --- C. Feature Distributions (Optional) ---
# Check separation of states
print("Generating Feature Boxplots...")
fig_feats = make_subplots(rows=1, cols=3, subplot_titles=("Velocity", "IdPhi (VTE)", "Body Length"))

for i, feature in enumerate(features):
    for state in ['Nest', 'Explore', 'Exploit']:
        subset = df_tracking[df_tracking['state_name'] == state]
        fig_feats.add_trace(
            go.Box(y=subset[feature], name=state, marker_color=color_map[state], showlegend=False),
            row=1, col=i+1
        )

fig_feats.update_layout(title_text="Feature Distributions by State", height=500)
fig_feats.show()

# ==========================================
# 6. EXPORT
# ==========================================
# Save processed data
df_tracking.to_csv('processed_behavior_states_plotly.csv')

# Save figures as HTML (optional, can be opened in any browser)
# fig_ethogram.write_html("ethogram.html")
# fig_spatial.write_html("spatial_map.html")
print("Done!")