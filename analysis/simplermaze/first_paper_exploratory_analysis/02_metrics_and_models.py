"""
02 — P1/P2 behavioural metrics + statistical models.

For sessions with DLC data (3.6, 3.7, 3.8):
  - Splits each trial into Phase 1 (entry -> first ROI) and Phase 2 (ROI -> exit)
  - Computes: duration, mean speed, spatial entropy per phase
  - Violin plots: Hit vs Miss
  - LMM  (statsmodels): metric ~ status + (1 | session)
  - GLMM (rpy2/lme4):   Gamma family for speed/duration, logit for binary

Processing pipeline (aligned with the legacy analysis script):
  1. Load DLC data, filter by likelihood, interpolate + smooth
  2. Trim trial start to first visible (above-threshold) frame
  3. Find split point using smoothed coordinates + ROI buffer
  4. For Hit trials with zero DLC speed: use calibrated path distances
     (true_path_distances.json) and time_to_reward from trials CSV
  5. Skip Miss trials with zero speed (tracking failure)

Outputs:
  - master_behavioural_data.csv (pooled across sessions)
  - violin_plots.png / .pdf
  - stats_report.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf

from config import (MOUSE_ID, MOUSE_DIR, FPS, BODYPART,
                    LIKELIHOOD_THRESH, PX_PER_CM,
                    get_sessions_with_dlc)
from utils import (load_trials, classify_trial, load_dlc, load_rois,
                   point_in_roi, spatial_entropy)

OUTPUT_DIR = os.path.join(MOUSE_DIR, f"MOUSE_{MOUSE_ID}_TOTAL_ANALYSIS")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helper: process kinematics (matches legacy af.process_kinematics) ─

def process_kinematics(tracking, fps, px_per_cm,
                       smooth_window=7, smooth_poly=3):
    """Smooth coordinates and compute speed column.

    Matches the legacy analysisfunct.process_kinematics pipeline:
      1. Interpolate NaN gaps in x, y
      2. Forward-fill + back-fill remaining NaNs
      3. Savitzky-Golay smooth -> x_smooth, y_smooth
      4. Frame-to-frame distance / px_per_cm * fps -> speed (cm/s)
    """
    df = tracking.copy()

    # interpolate + fill
    df["x"] = df["x"].interpolate().ffill().bfill()
    df["y"] = df["y"].interpolate().ffill().bfill()

    # smooth
    if len(df) >= smooth_window:
        df["x_smooth"] = savgol_filter(df["x"].values, smooth_window, smooth_poly)
        df["y_smooth"] = savgol_filter(df["y"].values, smooth_window, smooth_poly)
    else:
        df["x_smooth"] = df["x"].values
        df["y_smooth"] = df["y"].values

    # speed from smoothed coordinates
    dx = np.diff(df["x_smooth"].values, prepend=df["x_smooth"].values[0])
    dy = np.diff(df["y_smooth"].values, prepend=df["y_smooth"].values[0])
    dist_px = np.sqrt(dx**2 + dy**2)
    df["speed"] = (dist_px / px_per_cm) * fps

    return df


# ── 1. compute per-trial metrics ────────────────────────────────────

sessions = get_sessions_with_dlc()
print(f"Processing {len(sessions)} sessions with DLC data")

all_metrics = []

for sess in sessions:
    print(f"\n--- Session {sess.session_id} ---")

    df_trials = load_trials(sess.trial_csv)
    rois = load_rois(sess.roi_csv) if sess.roi_csv else {}

    # Load and process tracking (same pipeline as legacy script)
    tracking_raw = load_dlc(sess.dlc_csv, bodypart=BODYPART,
                            likelihood_thresh=LIKELIHOOD_THRESH)
    tracking = process_kinematics(tracking_raw, FPS, PX_PER_CM)

    # Load calibrated path distances if available
    true_distances_cm = {}
    path_json = os.path.join(sess.session_dir, "true_path_distances.json")
    if os.path.exists(path_json):
        try:
            with open(path_json, "r") as f:
                true_distances_cm = json.load(f)
            print(f"  Loaded calibrated path distances: {list(true_distances_cm.keys())}")
        except Exception:
            pass

    # determine which frame columns exist
    if "start_trial_frame" in df_trials.columns:
        frame_start_col, frame_end_col = "start_trial_frame", "end_trial_frame"
    elif "start_frame" in df_trials.columns:
        frame_start_col, frame_end_col = "start_frame", "end_frame"
    else:
        print(f"  No frame columns found, skipping session {sess.session_id}")
        continue

    for idx, row in df_trials.iterrows():
        try:
            f_start = int(row[frame_start_col])
            f_end = int(row[frame_end_col])
        except (ValueError, TypeError):
            continue

        if f_start >= f_end or (f_end - f_start) < 10:
            continue

        outcome = classify_trial(row)
        status = "Hit" if outcome == "correct" else "Miss"

        target_str = row.get("first_reward_area_visited", "")
        if pd.isna(target_str) or str(target_str).strip() == "":
            continue
        target = str(target_str).strip()

        # ── get trial tracking data ──
        # Use .loc for label-based slicing (matches legacy script)
        trial_track = tracking.loc[f_start:f_end].copy()

        # ── trim to first visible frame (legacy behaviour) ──
        is_visible = trial_track["likelihood"] >= LIKELIHOOD_THRESH
        if is_visible.any():
            first_visible_idx = is_visible.idxmax()
            trial_track = trial_track.loc[first_visible_idx:]

        if len(trial_track) < 1:
            continue

        # ── find split point using smoothed coordinates + buffer ──
        # Case-insensitive ROI matching (legacy behaviour)
        roi_match = next(
            (name for name in rois if str(name).lower() == target.lower()),
            None,
        )

        split_idx = -1
        if roi_match is not None:
            rx, ry, rw, rh = rois[roi_match]
            buffer = 20
            for i, (f_idx, pos) in enumerate(trial_track.iterrows()):
                if (rx - buffer <= pos["x_smooth"] <= rx + rw + buffer and
                        ry - buffer <= pos["y_smooth"] <= ry + rh + buffer):
                    split_idx = i
                    break

        # fallback: 80% split if ROI never reached
        if split_idx == -1:
            split_idx = int(len(trial_track) * 0.8)

        p1 = trial_track.iloc[:split_idx]
        p2 = trial_track.iloc[split_idx:]

        # ── compute initial P1 metrics ──
        p1_duration_s = len(p1) / FPS
        p1_mean_speed = p1["speed"].mean() if not p1.empty else 0
        p1_entropy = spatial_entropy(
            p1["x_smooth"].values, p1["y_smooth"].values
        ) if not p1.empty else 0

        # ── skip Miss trials with zero speed (tracking failure) ──
        if status == "Miss" and p1_mean_speed == 0:
            continue

        # ── Hit trials with zero speed: use calibrated distances ──
        if status == "Hit" and p1_mean_speed == 0:
            time_to_reward_ms = row.get("time_to_reward", pd.NA)
            if pd.notna(time_to_reward_ms) and float(time_to_reward_ms) > 0:
                time_s = float(time_to_reward_ms) / 1000.0
                dist_cm = true_distances_cm.get(target, 0)
                if dist_cm > 0:
                    p1_mean_speed = dist_cm / time_s
                    p1_duration_s = time_s
                    p1_entropy = np.nan  # no valid DLC for entropy

        # ── P2 metrics ──
        p2_duration_s = len(p2) / FPS
        p2_mean_speed = p2["speed"].mean() if not p2.empty else 0
        p2_entropy = spatial_entropy(
            p2["x_smooth"].values, p2["y_smooth"].values
        ) if not p2.empty else 0

        all_metrics.append({
            "mouse_id": MOUSE_ID,
            "session_id": sess.session_id,
            "trial_id": row.get("trial_ID", idx),
            "status": status,
            "target": target,
            "p1_duration_s": p1_duration_s,
            "p1_mean_speed": p1_mean_speed,
            "p1_entropy": p1_entropy,
            "p2_duration_s": p2_duration_s,
            "p2_mean_speed": p2_mean_speed,
            "p2_entropy": p2_entropy,
        })

df_all = pd.DataFrame(all_metrics)
df_all.to_csv(os.path.join(OUTPUT_DIR, "master_behavioural_data.csv"), index=False)
print(f"\nComputed metrics for {len(df_all)} trials across {len(sessions)} sessions")
print(df_all.groupby("status").size())


# ── 2. violin plots ─────────────────────────────────────────────────

metrics = ["p1_duration_s", "p1_mean_speed", "p1_entropy",
           "p2_duration_s", "p2_mean_speed", "p2_entropy"]
metric_labels = ["P1 Duration (s)", "P1 Speed (cm/s)", "P1 Entropy",
                 "P2 Duration (s)", "P2 Speed (cm/s)", "P2 Entropy"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

palette = {"Hit": "#2CA02C", "Miss": "#D62728"}
stats_summary = []

for i, (m, label) in enumerate(zip(metrics, metric_labels)):
    if m not in df_all.columns:
        continue

    ax = axes[i]

    sns.violinplot(data=df_all, x="status", y=m, ax=ax,
                   palette=palette, hue="status",
                   inner="quart", alpha=0.4, legend=False)
    sns.stripplot(data=df_all, x="status", y=m, ax=ax,
                  color="black", alpha=0.3, size=4, legend=False)

    # Mann-Whitney U
    h = df_all[df_all["status"] == "Hit"][m].dropna()
    ms = df_all[df_all["status"] == "Miss"][m].dropna()
    p_mwu = np.nan
    if len(h) > 0 and len(ms) > 0:
        _, p_mwu = mannwhitneyu(h, ms, alternative="two-sided")

    # LMM: metric ~ status + (1 | session_id)
    p_lmm = np.nan
    clean = df_all[["status", "session_id", m]].dropna()
    if len(clean["session_id"].unique()) > 1 and len(clean) > 5:
        try:
            model = smf.mixedlm(f"{m} ~ status", clean,
                                groups=clean["session_id"])
            result = model.fit(reml=True)
            p_lmm = result.pvalues.get("status[T.Miss]", np.nan)
        except Exception as e:
            warnings.warn(f"LMM failed for {m}: {e}")

    sig_mwu = "*" if p_mwu < 0.05 else "NS"
    sig_lmm = "*" if p_lmm < 0.05 else "NS"

    ax.set_title(f"{label}\nMWU p={p_mwu:.3f} ({sig_mwu})"
                 f"  LMM p={p_lmm:.3f} ({sig_lmm})", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel(label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_summary.append({
        "metric": m, "MWU_p": p_mwu, "MWU_sig": sig_mwu,
        "LMM_p": p_lmm, "LMM_sig": sig_lmm,
    })

plt.suptitle(f"Mouse {MOUSE_ID} — Hit vs Miss (pooled sessions)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "violin_plots.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "violin_plots.pdf"), bbox_inches="tight")


# ── 3. Gamma GLMM via rpy2 ──────────────────────────────────────────

print("\n--- Gamma GLMM (speed & duration via lme4) ---")

HAS_RPY2 = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    lme4 = importr("lme4")
    HAS_RPY2 = True
except Exception as e:
    print(f"rpy2/R not available ({e}) — skipping Gamma GLMM")

if HAS_RPY2:
    for m in ["p1_mean_speed", "p2_mean_speed", "p1_duration_s", "p2_duration_s"]:
        clean = df_all[["status", "session_id", m]].dropna()
        clean = clean[clean[m] > 0]

        if len(clean) < 10:
            print(f"  {m}: not enough data, skipping")
            continue

        r_df = pandas2ri.py2rpy(clean)
        ro.globalenv["df"] = r_df

        n_sessions = len(clean["session_id"].unique())

        try:
            if n_sessions > 1:
                formula = f"{m} ~ status + (1 | session_id)"
                ro.r(f"""
                    library(lme4)
                    model <- glmer({formula},
                                   data = df, family = Gamma(link = "log"))
                    cat("\\n", "{m}:", "\\n")
                    print(summary(model))
                """)
            else:
                ro.r(f"""
                    model <- glm({m} ~ status,
                                 data = df, family = Gamma(link = "log"))
                    cat("\\n", "{m}:", "\\n")
                    print(summary(model))
                """)

            coefs = ro.r("coef(summary(model))")
            coef_df = pandas2ri.rpy2py(coefs)
            p_val = coef_df.iloc[1, -1] if len(coef_df) > 1 else np.nan
            sig = "*" if p_val < 0.05 else "NS"

            for row in stats_summary:
                if row["metric"] == m:
                    row["GLMM_Gamma_p"] = p_val
                    row["GLMM_Gamma_sig"] = sig

        except Exception as e:
            print(f"  Gamma GLMM failed for {m}: {e}")


# ── 4. save stats report ────────────────────────────────────────────

stats_df = pd.DataFrame(stats_summary)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "stats_report.csv"), index=False)
print(f"\nStats report saved to {OUTPUT_DIR}/stats_report.csv")
print(stats_df.to_string(index=False))

plt.show()
