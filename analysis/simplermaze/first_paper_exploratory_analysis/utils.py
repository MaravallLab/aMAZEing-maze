"""
Shared utility functions for the analysis pipeline.

Provides:
- Trial CSV loading with column normalisation
- DLC tracking data loading
- ROI loading and point-in-ROI checks
- Kinematics (smoothing, speed)
- Spatial entropy
- Transition probability computation
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import entropy as shannon_entropy
from typing import Dict, List, Optional, Tuple


# ── trial data ──────────────────────────────────────────────────────

def load_trials(csv_path: str) -> pd.DataFrame:
    """Load a trial CSV and normalise columns across formats."""
    df = pd.read_csv(csv_path)

    # ensure numeric types for hit/miss/incorrect (may be float with NaN)
    for col in ["hit", "miss", "incorrect"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def classify_trial(row: pd.Series) -> str:
    """
    Classify a trial as 'correct', 'incorrect', or 'no_choice'.

    Uses first_reward_area_visited[-1] == rew_location as the ground truth,
    ignoring the hit/miss/incorrect columns (which may have misdetections).
    """
    first_visited = row.get("first_reward_area_visited")
    rew_loc = row.get("rew_location")

    if pd.isna(first_visited) or str(first_visited).strip() == "":
        # mouse never entered any reward arm
        return "no_choice"

    visited_char = str(first_visited).strip()[-1].upper()
    correct_char = str(rew_loc).strip().upper()

    if visited_char == correct_char:
        return "correct"
    else:
        return "incorrect"


def has_entered_any_arm(row: pd.Series) -> bool:
    """Check whether the mouse entered at least one reward arm."""
    for col in ["rewA", "rewB", "rewC", "rewD"]:
        val = row.get(col)
        if pd.notna(val) and val != "" and val != 0:
            return True
    return False


# ── DLC tracking ────────────────────────────────────────────────────

def load_dlc(csv_path: str, bodypart: str = "mid",
             likelihood_thresh: float = 0.8) -> pd.DataFrame:
    """
    Load DLC tracking CSV and return a clean (x, y, likelihood) DataFrame.

    Filters points below the likelihood threshold (sets to NaN).
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    # get scorer name (first level)
    scorer = df.columns.get_level_values(0)[0]

    x = df[(scorer, bodypart, "x")].values.astype(float)
    y = df[(scorer, bodypart, "y")].values.astype(float)
    lk = df[(scorer, bodypart, "likelihood")].values.astype(float)

    # mask low-likelihood points
    mask = lk < likelihood_thresh
    x[mask] = np.nan
    y[mask] = np.nan

    return pd.DataFrame({"x": x, "y": y, "likelihood": lk},
                        index=df.index)


# ── ROI loading ─────────────────────────────────────────────────────

def load_rois(csv_path: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Load ROI CSV.

    Returns dict: roi_name -> (x, y, w, h).
    Format: first column is roi name, columns 0-3 are x, y, w, h.
    """
    df = pd.read_csv(csv_path, index_col=0)
    rois = {}
    for name, row in df.iterrows():
        vals = row.values.astype(int)
        rois[str(name)] = (vals[0], vals[1], vals[2], vals[3])
    return rois


def point_in_roi(x: float, y: float, roi: Tuple[int, int, int, int],
                 buffer: int = 0) -> bool:
    """Check if (x, y) is inside the ROI bounding box (with optional buffer)."""
    rx, ry, rw, rh = roi
    return (rx - buffer <= x <= rx + rw + buffer and
            ry - buffer <= y <= ry + rh + buffer)


def assign_roi(x: float, y: float,
               rois: Dict[str, Tuple[int, int, int, int]],
               buffer: int = 5) -> str:
    """Assign a point to an ROI, or 'corridor' if outside all ROIs."""
    if np.isnan(x) or np.isnan(y):
        return "unknown"
    for name, bbox in rois.items():
        if point_in_roi(x, y, bbox, buffer=buffer):
            return name
    return "corridor"


# ── kinematics ──────────────────────────────────────────────────────

def compute_speed(x: np.ndarray, y: np.ndarray,
                  fps: int = 30, px_per_cm: float = 7.5,
                  smooth_window: int = 7, smooth_poly: int = 3) -> np.ndarray:
    """
    Compute instantaneous speed in cm/s from (x, y) pixel coordinates.

    Applies Savitzky-Golay smoothing before computing frame-to-frame distance.
    """
    # interpolate NaNs for smoothing
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < smooth_window:
        return np.full(len(x), np.nan)

    x_smooth = _interp_nans(x.copy())
    y_smooth = _interp_nans(y.copy())

    if len(x_smooth) >= smooth_window:
        x_smooth = savgol_filter(x_smooth, smooth_window, smooth_poly)
        y_smooth = savgol_filter(y_smooth, smooth_window, smooth_poly)

    dx = np.diff(x_smooth, prepend=x_smooth[0])
    dy = np.diff(y_smooth, prepend=y_smooth[0])
    dist_px = np.sqrt(dx**2 + dy**2)
    speed_cm_s = (dist_px / px_per_cm) * fps

    # restore NaN where original data was missing
    speed_cm_s[~valid] = np.nan

    return speed_cm_s


def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in an array."""
    nans = np.isnan(arr)
    if nans.all() or not nans.any():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


# ── spatial entropy ─────────────────────────────────────────────────

def spatial_entropy(x: np.ndarray, y: np.ndarray,
                    grid_size: int = 20) -> float:
    """
    Compute 2D spatial entropy from (x, y) coordinates.

    Higher entropy = more uniform spatial coverage.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 2:
        return 0.0

    xv, yv = x[valid], y[valid]
    hist, _, _ = np.histogram2d(xv, yv, bins=grid_size)
    probs = hist.flatten()
    probs = probs[probs > 0]
    probs = probs / probs.sum()

    return shannon_entropy(probs)


# ── transition probabilities ───────────────────────────────────────

def compute_state_sequence(tracking: pd.DataFrame,
                           rois: Dict[str, Tuple[int, int, int, int]],
                           buffer: int = 5) -> List[str]:
    """
    Assign each frame to an ROI or 'corridor'.

    Returns list of state labels, one per frame.
    """
    states = []
    for _, row in tracking.iterrows():
        states.append(assign_roi(row["x"], row["y"], rois, buffer=buffer))
    return states


def collapse_state_sequence(states: List[str]) -> List[str]:
    """
    Collapse consecutive identical states.

    [corridor, corridor, rewA, rewA, corridor, rewB]
    -> [corridor, rewA, corridor, rewB]
    """
    if not states:
        return []
    collapsed = [states[0]]
    for s in states[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    return collapsed


def transition_matrix(collapsed_states: List[str],
                      state_labels: Optional[List[str]] = None,
                      normalise: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Build a transition probability matrix from a collapsed state sequence.

    Returns (matrix, state_labels).
    If normalise=True, rows sum to 1 (probabilities).
    """
    if state_labels is None:
        state_labels = sorted(set(collapsed_states))

    n = len(state_labels)
    idx = {s: i for i, s in enumerate(state_labels)}
    mat = np.zeros((n, n), dtype=float)

    for a, b in zip(collapsed_states[:-1], collapsed_states[1:]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1

    if normalise:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat = mat / row_sums

    return mat, state_labels
