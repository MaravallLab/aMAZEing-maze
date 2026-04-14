"""
01 — Auditory Preference Analysis: Batch PI, Complexity & Individual Differences
=================================================================================

Extends the preference index (PI) analysis from Parkes & Maravall to all
mice and all experiment days, adding:
  - Sensory complexity effects
  - Vocalisation vs non-vocalisation comparison
  - Mixed-effects variance decomposition (individual differences)
  - Roaming entropy from habituation trials

Data source: trials CSVs from 8_arms_w_voc experiment.
  - Each session has 9 trial blocks (odd = silent/habituation, even = sound)
  - Trial 1 = habituation (15 min, no sound)
  - Trials 2,4,6,8 = sound (15 min each, 8 ROIs with stimuli)
  - Trials 3,5,7,9 = silent (2 min each, shuffled locations)
  - time_spent (ms) and visitation_count per ROI per trial

Preference Index (PI):
  PI = (Avg_Sound_Visit_Duration - Avg_Silent_Visit_Duration)
     / (Avg_Sound_Visit_Duration + Avg_Silent_Visit_Duration)
  Ranges: -1 (prefer silence) to +1 (prefer sound)

Outputs (saved to BATCH_ANALYSIS/):
  - preference_data.csv           — per-mouse per-session PI + metrics
  - fig1_pi_trajectories.png/pdf  — individual mouse PI across days
  - fig2_pi_by_day.png/pdf        — mean PI per day with CI
  - fig3_pi_violins.png/pdf       — violin plots by day
  - fig4_complexity_heatmap.png   — stimulus-type visit duration heatmap
  - fig5_vocalisation_contrast.png — vocalisation vs other days
  - fig6_re_vs_pi.png/pdf         — roaming entropy predicts PI
  - fig7_icc_summary.png/pdf      — variance decomposition
  - stats_report.txt              — all statistical test results
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from scipy.stats import entropy as shannon_entropy
import statsmodels.formula.api as smf
import statsmodels.api as sm

from preference_analysis_config import (
    BASE_PATH, OUTPUT_DIR, EXPERIMENT_DAYS, PI_DAYS, DAY_ORDER, DAY_SHORT,
    discover_sessions, get_mice_with_min_sessions, compute_first_minute_re,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
stats_lines = []  # accumulate stats output


def log_stat(msg):
    print(msg)
    stats_lines.append(msg)


# ── helper functions ──────────────────────────────────────────────────


def classify_stimulus(row, day_key):
    """Classify a trial row's stimulus type based on the experiment mode."""
    mode = EXPERIMENT_DAYS[day_key]["mode"]

    if mode == "temporal_envelope_modulation":
        st = str(row.get("sound_type", "")).strip().lower()
        freq = str(row.get("frequency", "")).strip().lower()
        mod = str(row.get("temporal_modulation", "")).strip().lower()
        # silent arms
        if freq in ("0", "silent_arm", "") or st in ("silent", "silent_trial"):
            return "silent"
        # vocalisation control (sound_type=control, frequency=vocalisation)
        if freq == "vocalisation" or mod == "vocalisation":
            return "vocalisation"
        # no-stimulus control (sound_type=control, frequency=silent_arm)
        if st == "control" and mod == "no_stimulus":
            return "silent"
        # the three TEM categories
        if st in ("smooth", "rough", "rough_complex"):
            return st
        if st == "control":
            return "control"
        return st if st else "unknown"

    elif mode == "complex_intervals":
        it = str(row.get("interval_type", "")).strip().lower()
        freq = str(row.get("frequency", "")).strip().lower()
        if it == "silent_trial" or freq in ("0", ""):
            return "silent"
        if freq == "vocalisation":
            return "vocalisation"
        if it in ("consonant", "dissonant", "smooth", "rough"):
            return it
        if it == "control":
            return "control"
        return it if it else "unknown"

    elif mode == "sequences":
        pat = str(row.get("pattern", "")).strip()
        if pat in ("AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA"):
            return pat
        if pat in ("silence", "0"):
            return "silent"
        if "vocalisation" in pat.lower():
            return "vocalisation"
        freq = row.get("frequency", 0)
        if freq == 0 or str(freq) == "0":
            return "silent"
        return pat

    elif mode == "vocalisation":
        it = str(row.get("interval_type", "")).strip().lower()
        if it == "silent_trial":
            return "silent"
        freq = str(row.get("frequency", ""))
        if freq == "0" or freq == "":
            return "silent"
        # extract vocalisation filename
        if "/" in freq or "\\" in freq:
            return os.path.splitext(os.path.basename(freq))[0]
        return "vocalisation"

    return "unknown"


def compute_roaming_entropy(time_per_roi, n_rois=8):
    """Roaming entropy from time proportions across ROIs.

    RE = -sum(p_i * log2(p_i)) / log2(N)
    Normalised so RE in [0, 1]. High RE = even exploration.
    """
    total = np.nansum(time_per_roi)
    if total <= 0:
        return np.nan
    props = np.array(time_per_roi) / total
    props = props[props > 0]  # remove zeros for log
    re = shannon_entropy(props, base=2) / np.log2(n_rois)
    return re


def _is_file_locally_available(path):
    """Check if a Box Drive file is locally cached (not online-only).

    On Windows, Box Drive online-only files have the FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
    attribute (0x00400000). We check using ctypes to avoid hanging on read.
    Falls back to a quick byte-read test if ctypes fails.
    """
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False  # file doesn't exist
        # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
        # FILE_ATTRIBUTE_OFFLINE = 0x00001000
        if attrs & 0x00400000 or attrs & 0x00001000:
            return False  # online-only
        return True
    except Exception:
        # Non-Windows or ctypes unavailable: try a quick read
        try:
            with open(path, 'rb') as f:
                f.read(1)
            return True
        except Exception:
            return False


def safe_read_csv(path):
    """Read CSV, skipping Box cloud online-only files.

    Box Drive stores files as 'online-only' stubs. To get the full
    analysis: right-click the 8_arms_w_voc folder in File Explorer
    and select 'Make available offline' / 'Always keep on this device'.
    """
    if not _is_file_locally_available(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── 1. load all sessions and compute metrics ──────────────────────────

print("Discovering sessions...")
all_sessions = discover_sessions()
print(f"Found {len(all_sessions)} sessions across {len(EXPERIMENT_DAYS)} days")

records = []
stim_records = []  # per-stimulus-type records for complexity analysis
skipped = 0
loaded = 0

for sess in all_sessions:
    if not sess.trials_csv:
        skipped += 1
        continue

    df = safe_read_csv(sess.trials_csv)
    if df is None:
        skipped += 1
        continue

    if "trial_ID" not in df.columns or "ROIs" not in df.columns:
        skipped += 1
        continue

    loaded += 1

    # ensure numeric columns
    for col in ["time_spent", "visitation_count", "time_in_maze_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- habituation metrics (trial 1) ----
    hab = df[df["trial_ID"] == 1]
    hab_time_per_roi = []
    if len(hab) > 0 and "time_spent" in hab.columns:
        # get time in each ROI during habituation
        for roi_name in [f"ROI{i}" for i in range(1, 9)]:
            roi_rows = hab[hab["ROIs"] == roi_name]
            t = roi_rows["time_spent"].sum()
            hab_time_per_roi.append(t if not np.isnan(t) else 0)
    hab_total = sum(hab_time_per_roi)
    roaming_entropy = compute_roaming_entropy(hab_time_per_roi)
    hab_visits = hab["visitation_count"].sum() if "visitation_count" in hab.columns else 0

    # ---- sound vs silent trials ----
    # Sound trials: even trial_IDs (2, 4, 6, 8)
    # Silent trials: odd trial_IDs > 1 (3, 5, 7, 9)
    sound_trials = df[df["trial_ID"].isin([2, 4, 6, 8])]
    silent_trials = df[df["trial_ID"].isin([3, 5, 7, 9])]

    # Filter to ROI rows (not entrance)
    roi_names = [f"ROI{i}" for i in range(1, 9)]
    sound_rois = sound_trials[sound_trials["ROIs"].isin(roi_names)].copy()
    silent_rois = silent_trials[silent_trials["ROIs"].isin(roi_names)]

    # Classify each ROI within sound trials
    sound_rois["_stim_label"] = [classify_stimulus(row, sess.day)
                                  for _, row in sound_rois.iterrows()]

    # Exclude silent control arms from sound aggregate
    actual_sound = sound_rois[sound_rois["_stim_label"] != "silent"]

    def _safe_val(x):
        return 0 if (isinstance(x, float) and np.isnan(x)) else x

    silent_time = _safe_val(silent_rois["time_spent"].sum() if "time_spent" in silent_rois.columns else 0)
    silent_visits = _safe_val(silent_rois["visitation_count"].sum() if "visitation_count" in silent_rois.columns else 0)
    avg_silent_dur = silent_time / silent_visits if silent_visits > 0 else 0

    sound_time = _safe_val(actual_sound["time_spent"].sum() if "time_spent" in actual_sound.columns else 0)
    sound_visits = _safe_val(actual_sound["visitation_count"].sum() if "visitation_count" in actual_sound.columns else 0)
    avg_sound_dur = sound_time / sound_visits if sound_visits > 0 else 0

    # Preference Index
    denom = avg_sound_dur + avg_silent_dur
    pi = (avg_sound_dur - avg_silent_dur) / denom if denom > 0 else np.nan

    # Vocalisation-specific PI
    voc_subset = actual_sound[actual_sound["_stim_label"] == "vocalisation"]
    voc_time = _safe_val(voc_subset["time_spent"].sum())
    voc_visits_n = _safe_val(voc_subset["visitation_count"].sum())
    voc_pi = np.nan
    if voc_visits_n > 0:
        avg_voc = voc_time / voc_visits_n
        d = avg_voc + avg_silent_dur
        voc_pi = (avg_voc - avg_silent_dur) / d if d > 0 else np.nan
    elif sess.day == "w2_vocalisations" and len(actual_sound) > 0:
        # all non-silent stim types are vocalisations
        vt = _safe_val(actual_sound["time_spent"].sum())
        vv = _safe_val(actual_sound["visitation_count"].sum())
        if vv > 0:
            avg_voc = vt / vv
            d = avg_voc + avg_silent_dur
            voc_pi = (avg_voc - avg_silent_dur) / d if d > 0 else np.nan

    # Other-sounds PI (non-vocalisation, non-silent stimuli)
    other_subset = actual_sound[actual_sound["_stim_label"] != "vocalisation"]
    other_time = _safe_val(other_subset["time_spent"].sum())
    other_visits_n = _safe_val(other_subset["visitation_count"].sum())
    other_sounds_pi = np.nan
    if other_visits_n > 0:
        avg_other = other_time / other_visits_n
        d = avg_other + avg_silent_dur
        other_sounds_pi = (avg_other - avg_silent_dur) / d if d > 0 else np.nan

    # First-minute roaming entropy (from detailed_visits)
    re_first_min = compute_first_minute_re(sess)

    records.append({
        "mouse_id": sess.mouse_id,
        "day": sess.day,
        "day_label": EXPERIMENT_DAYS[sess.day]["short"],
        "mode": EXPERIMENT_DAYS[sess.day]["mode"],
        "sound_time_ms": sound_time,
        "sound_visits": sound_visits,
        "silent_time_ms": silent_time,
        "silent_visits": silent_visits,
        "avg_sound_dur_ms": avg_sound_dur,
        "avg_silent_dur_ms": avg_silent_dur,
        "preference_index": pi,
        "voc_pi": voc_pi,
        "other_sounds_pi": other_sounds_pi,
        "hab_time_ms": hab_total,
        "hab_visits": hab_visits,
        "roaming_entropy": roaming_entropy,
        "re_first_min": re_first_min,
    })

    # ---- per-stimulus-type breakdown (within sound trials only) ----
    for _, row in sound_rois.iterrows():
        stim_type = row["_stim_label"]
        if stim_type == "silent":
            continue
        t = row.get("time_spent", 0)
        v = row.get("visitation_count", 0)
        t = 0 if pd.isna(t) else t
        v = 0 if pd.isna(v) else v
        avg_dur = t / v if v > 0 else 0
        d = avg_dur + avg_silent_dur
        stim_pi = (avg_dur - avg_silent_dur) / d if d > 0 else np.nan
        stim_records.append({
            "mouse_id": sess.mouse_id,
            "day": sess.day,
            "trial_ID": row["trial_ID"],
            "ROI": row["ROIs"],
            "stim_type": stim_type,
            "time_spent_ms": t,
            "visitation_count": v,
            "avg_visit_dur_ms": avg_dur,
            "stim_pi": stim_pi,
        })

print(f"\nLoaded {loaded} sessions, skipped {skipped}")
if skipped > loaded:
    print(f"  NOTE: {skipped} files could not be read (Box cloud-sync).")
    print(f"  To fix: right-click the 8_arms_w_voc folder in File Explorer")
    print(f"  and select 'Make available offline' / 'Always keep on this device'.")

df_pref = pd.DataFrame(records)
df_stim = pd.DataFrame(stim_records)

# Save
df_pref.to_csv(os.path.join(OUTPUT_DIR, "preference_data.csv"), index=False)
df_stim.to_csv(os.path.join(OUTPUT_DIR, "stimulus_breakdown.csv"), index=False)
print(f"Saved preference_data.csv ({len(df_pref)} rows)")
print(f"Saved stimulus_breakdown.csv ({len(df_stim)} rows)")

# Voc PI vs other sounds PI per session
df_voc_other = df_pref[["mouse_id", "day", "day_label", "voc_pi", "other_sounds_pi"]].copy()
df_voc_other.to_csv(os.path.join(OUTPUT_DIR, "voc_vs_other_sounds_pi.csv"), index=False)
print(f"Saved voc_vs_other_sounds_pi.csv ({len(df_voc_other)} rows)")

# Filter to days with silent controls for PI analysis
if len(df_pref) == 0:
    print("\nERROR: No data loaded. Ensure Box files are available offline.")
    print("Right-click 8_arms_w_voc in File Explorer -> 'Make available offline'")
    print("Then re-run this script.")
    sys.exit(1)

df_pi = df_pref[df_pref["day"].isin(PI_DAYS)].copy()
print(f"\nPI analysis: {len(df_pi)} sessions across {PI_DAYS}")
print(f"Unique mice: {df_pi['mouse_id'].nunique()}")


# ── 2. Figure 1: Individual PI trajectories across days ──────────────

print("\nPlotting Figure 1: PI trajectories...")

# Only include mice with at least 3 sessions for trajectory plot
multi_mice = get_mice_with_min_sessions(all_sessions, 3)
df_traj = df_pi[df_pi["mouse_id"].isin(multi_mice)].copy()

fig1, ax1 = plt.subplots(figsize=(12, 6))

day_x = {d: i for i, d in enumerate(PI_DAYS)}
cmap = plt.cm.tab20(np.linspace(0, 1, len(multi_mice)))

for i, mouse in enumerate(sorted(multi_mice)):
    sub = df_traj[df_traj["mouse_id"] == mouse].copy()
    sub["x"] = sub["day"].map(day_x)
    sub = sub.sort_values("x")
    ax1.plot(sub["x"], sub["preference_index"], "o-",
             color=cmap[i % len(cmap)], alpha=0.5, linewidth=1,
             markersize=4, label=mouse if i < 20 else None)

# Grand mean
for d in PI_DAYS:
    vals = df_pi[df_pi["day"] == d]["preference_index"].dropna()
    x = day_x[d]
    mean = vals.mean()
    ax1.plot(x, mean, "D", color="black", markersize=10, zorder=5)

ax1.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax1.set_xticks(range(len(PI_DAYS)))
ax1.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], fontsize=10)
ax1.set_ylabel("Preference Index", fontsize=12)
ax1.set_xlabel("Experiment Day", fontsize=12)
ax1.set_title("Individual mouse preference trajectories across days\n"
              "(black diamonds = grand mean)", fontsize=13)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(-1.05, 1.05)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_pi_trajectories.png"),
             dpi=200, bbox_inches="tight")
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_pi_trajectories.pdf"),
             bbox_inches="tight")
print("  Saved fig1_pi_trajectories")


# ── 3. Figure 2: Mean PI per day with CI ──────────────────────────────

print("Plotting Figure 2: Mean PI per day...")

fig2, ax2 = plt.subplots(figsize=(10, 6))

means = []
cis = []
ns = []
for d in PI_DAYS:
    vals = df_pi[df_pi["day"] == d]["preference_index"].dropna()
    m = vals.mean()
    se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
    means.append(m)
    cis.append(1.96 * se)
    ns.append(len(vals))

x = np.arange(len(PI_DAYS))
colours = ["#457B9D" if d != "w2_vocalisations" else "#E63946" for d in PI_DAYS]

ax2.bar(x, means, yerr=cis, color=colours, alpha=0.8, capsize=5,
        edgecolor="white", linewidth=1.5)

for i, (m, n) in enumerate(zip(means, ns)):
    ax2.text(i, m + cis[i] + 0.02, f"n={n}", ha="center", fontsize=9)

ax2.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], fontsize=10)
ax2.set_ylabel("Mean Preference Index (±95% CI)", fontsize=12)
ax2.set_title("Sound preference by experiment day\n"
              "(red = vocalisations)", fontsize=13)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(-0.8, 0.8)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_pi_by_day.png"),
             dpi=200, bbox_inches="tight")
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_pi_by_day.pdf"),
             bbox_inches="tight")
print("  Saved fig2_pi_by_day")


# ── 4. Figure 3: Violin plots by day ─────────────────────────────────

print("Plotting Figure 3: PI violins...")

fig3, ax3 = plt.subplots(figsize=(12, 6))

df_pi["day_short"] = df_pi["day"].map(DAY_SHORT)
day_order_short = [DAY_SHORT[d] for d in PI_DAYS]

palette = {DAY_SHORT[d]: "#457B9D" for d in PI_DAYS}
palette[DAY_SHORT["w2_vocalisations"]] = "#E63946"

sns.violinplot(data=df_pi, x="day_short", y="preference_index",
               order=day_order_short, palette=palette,
               inner="quart", alpha=0.4, ax=ax3, hue="day_short",
               legend=False)
sns.stripplot(data=df_pi, x="day_short", y="preference_index",
              order=day_order_short, color="black", alpha=0.4,
              size=5, ax=ax3, jitter=0.15)

ax3.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax3.set_xlabel("Experiment Day", fontsize=12)
ax3.set_ylabel("Preference Index", fontsize=12)
ax3.set_title("Distribution of preference index by experiment day",
              fontsize=13)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_pi_violins.png"),
             dpi=200, bbox_inches="tight")
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_pi_violins.pdf"),
             bbox_inches="tight")
print("  Saved fig3_pi_violins")


# ── 5. Figure 4: Stimulus complexity heatmap ──────────────────────────

print("Plotting Figure 4: Complexity heatmap...")

# Aggregate: mean visit duration per stimulus type per day
if len(df_stim) > 0:
    stim_agg = df_stim.groupby(["day", "stim_type"]).agg(
        mean_dur=("avg_visit_dur_ms", "mean"),
        n_visits=("visitation_count", "sum"),
        n_obs=("time_spent_ms", "count"),
    ).reset_index()

    # Create heatmap: rows = days, columns = stimulus types
    # Get all unique stim types per day
    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    axes4 = axes4.flatten()

    for idx, day in enumerate(DAY_ORDER):
        ax = axes4[idx]
        day_data = stim_agg[stim_agg["day"] == day]

        if len(day_data) == 0:
            ax.set_title(f"{DAY_SHORT[day]}\n(no data)", fontsize=10)
            ax.set_visible(False)
            continue

        # per-mouse mean visit duration per stimulus type
        day_stim = df_stim[df_stim["day"] == day]
        pivot = day_stim.groupby(["mouse_id", "stim_type"])[
            "avg_visit_dur_ms"
        ].mean().reset_index()

        if len(pivot) == 0:
            ax.set_visible(False)
            continue

        # box plot per stimulus type
        stim_order = EXPERIMENT_DAYS[day].get("complexity_order", [])
        available = [s for s in stim_order if s in pivot["stim_type"].values]
        # add any not in the predefined order
        extras = [s for s in pivot["stim_type"].unique()
                  if s not in available and s != "silent"]
        plot_order = available + extras

        if len(plot_order) == 0:
            ax.set_visible(False)
            continue

        sns.boxplot(data=pivot[pivot["stim_type"].isin(plot_order)],
                    x="stim_type", y="avg_visit_dur_ms",
                    order=plot_order, ax=ax, palette="viridis",
                    hue="stim_type", legend=False)
        sns.stripplot(data=pivot[pivot["stim_type"].isin(plot_order)],
                      x="stim_type", y="avg_visit_dur_ms",
                      order=plot_order, ax=ax, color="black",
                      alpha=0.4, size=3)

        ax.set_title(DAY_SHORT[day], fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Avg visit duration (ms)" if idx % 3 == 0 else "")
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Visit duration by stimulus type within each experiment day\n"
                 "(stimulus types ordered by complexity where applicable)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_complexity_heatmap.png"),
                 dpi=200, bbox_inches="tight")
    fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_complexity_heatmap.pdf"),
                 bbox_inches="tight")
    print("  Saved fig4_complexity_heatmap")


# ── 6. Figure 5: Vocalisation vs other days ──────────────────────────

print("Plotting Figure 5: Vocalisation contrast...")

# Paired comparison: for mice that did vocalisations + at least one other day
voc_mice = set(df_pi[df_pi["day"] == "w2_vocalisations"]["mouse_id"])
non_voc_days = [d for d in PI_DAYS if d != "w2_vocalisations"]

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: paired PI — vocalisation day vs mean of other days
paired_data = []
for mouse in voc_mice:
    voc_pi = df_pi[(df_pi["mouse_id"] == mouse) &
                   (df_pi["day"] == "w2_vocalisations")]["preference_index"]
    other_pis = df_pi[(df_pi["mouse_id"] == mouse) &
                      (df_pi["day"].isin(non_voc_days))]["preference_index"]

    if len(voc_pi) > 0 and len(other_pis) > 0:
        paired_data.append({
            "mouse_id": mouse,
            "pi_voc": voc_pi.values[0],
            "pi_other": other_pis.mean(),
        })

if paired_data:
    df_paired = pd.DataFrame(paired_data)

    ax = axes5[0]
    for _, row in df_paired.iterrows():
        ax.plot([0, 1], [row["pi_other"], row["pi_voc"]],
                "o-", color="#888888", alpha=0.4, markersize=5)
    ax.plot(0, df_paired["pi_other"].mean(), "D", color="#457B9D",
            markersize=12, zorder=5)
    ax.plot(1, df_paired["pi_voc"].mean(), "D", color="#E63946",
            markersize=12, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Other days\n(mean)", "Vocalisations"], fontsize=11)
    ax.set_ylabel("Preference Index", fontsize=12)
    ax.set_title(f"Paired comparison (n={len(df_paired)} mice)", fontsize=12)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    if len(df_paired) >= 6:
        stat, p_wil = wilcoxon(df_paired["pi_voc"], df_paired["pi_other"])
        log_stat(f"\nVocalisation vs Other days (Wilcoxon signed-rank):")
        log_stat(f"  Voc PI mean: {df_paired['pi_voc'].mean():.3f}")
        log_stat(f"  Other PI mean: {df_paired['pi_other'].mean():.3f}")
        log_stat(f"  W={stat:.1f}, p={p_wil:.4f}")
        ax.set_title(f"Paired comparison (n={len(df_paired)} mice)\n"
                     f"Wilcoxon p={p_wil:.4f}", fontsize=11)

# Panel B: distribution comparison
ax = axes5[1]
df_pi["is_voc"] = df_pi["day"] == "w2_vocalisations"
df_pi["group"] = df_pi["is_voc"].map({True: "Vocalisations",
                                       False: "Other days"})
sns.violinplot(data=df_pi, x="group", y="preference_index",
               palette={"Other days": "#457B9D", "Vocalisations": "#E63946"},
               inner="quart", alpha=0.4, ax=ax, hue="group", legend=False)
sns.stripplot(data=df_pi, x="group", y="preference_index",
              color="black", alpha=0.4, size=4, ax=ax)
ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("")
ax.set_ylabel("Preference Index", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# MWU test
voc_vals = df_pi[df_pi["is_voc"]]["preference_index"].dropna()
other_vals = df_pi[~df_pi["is_voc"]]["preference_index"].dropna()
if len(voc_vals) > 0 and len(other_vals) > 0:
    _, p_mwu = mannwhitneyu(voc_vals, other_vals, alternative="two-sided")
    ax.set_title(f"All sessions (MWU p={p_mwu:.4f})", fontsize=11)
    log_stat(f"\nVocalisation vs Other days (Mann-Whitney U):")
    log_stat(f"  Voc: n={len(voc_vals)}, mean={voc_vals.mean():.3f}")
    log_stat(f"  Other: n={len(other_vals)}, mean={other_vals.mean():.3f}")
    log_stat(f"  p={p_mwu:.4f}")

plt.suptitle("Do vocalisations drive stronger preference than other stimuli?",
             fontsize=14, y=1.02)
plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_vocalisation_contrast.png"),
             dpi=200, bbox_inches="tight")
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_vocalisation_contrast.pdf"),
             bbox_inches="tight")
print("  Saved fig5_vocalisation_contrast")


# ── 7. Figure 6: Roaming entropy vs PI ────────────────────────────────

print("Plotting Figure 6: Roaming entropy vs PI...")

df_re = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).copy()

fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: within-mouse (session deviations from mouse mean)
ax = axes6[0]
if len(df_re) > 10:
    mouse_means = df_re.groupby("mouse_id").agg(
        mean_re=("roaming_entropy", "mean"),
        mean_pi=("preference_index", "mean"),
    )
    df_re = df_re.merge(mouse_means, on="mouse_id")
    df_re["dev_re"] = df_re["roaming_entropy"] - df_re["mean_re"]
    df_re["dev_pi"] = df_re["preference_index"] - df_re["mean_pi"]

    ax.scatter(df_re["dev_re"], df_re["dev_pi"], alpha=0.4, s=20,
               color="#457B9D")
    r, p = spearmanr(df_re["dev_re"], df_re["dev_pi"])
    ax.set_xlabel("Deviation from mouse mean RE", fontsize=11)
    ax.set_ylabel("Deviation from mouse mean PI", fontsize=11)
    ax.set_title(f"Within-mouse\nSpearman r={r:.3f}, p={p:.4f}", fontsize=11)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    log_stat(f"\nWithin-mouse RE vs PI:")
    log_stat(f"  Spearman r={r:.3f}, p={p:.4f}")

# Panel B: between-mouse (mouse averages)
ax = axes6[1]
mouse_avg = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).groupby(
    "mouse_id"
).agg(
    mean_re=("roaming_entropy", "mean"),
    mean_pi=("preference_index", "mean"),
).reset_index()

if len(mouse_avg) > 5:
    ax.scatter(mouse_avg["mean_re"], mouse_avg["mean_pi"], alpha=0.7, s=40,
               color="#E63946")
    r, p = spearmanr(mouse_avg["mean_re"], mouse_avg["mean_pi"])
    # regression line
    z = np.polyfit(mouse_avg["mean_re"], mouse_avg["mean_pi"], 1)
    x_line = np.linspace(mouse_avg["mean_re"].min(),
                         mouse_avg["mean_re"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#E63946", alpha=0.5)

    ax.set_xlabel("Mean Roaming Entropy", fontsize=11)
    ax.set_ylabel("Mean Preference Index", fontsize=11)
    ax.set_title(f"Between-mouse\nSpearman r={r:.3f}, p={p:.4f}",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    log_stat(f"\nBetween-mouse RE vs PI:")
    log_stat(f"  n={len(mouse_avg)} mice")
    log_stat(f"  Spearman r={r:.3f}, p={p:.4f}")

plt.suptitle("Roaming entropy (habituation) predicts sound preference",
             fontsize=14, y=1.02)
plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_re_vs_pi.png"),
             dpi=200, bbox_inches="tight")
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_re_vs_pi.pdf"),
             bbox_inches="tight")
print("  Saved fig6_re_vs_pi")


# ── 7b. Figure 6b: First-minute RE vs PI ────────────────────────────

print("Plotting Figure 6b: First-minute RE vs PI...")

df_re1 = df_pi.dropna(subset=["re_first_min", "preference_index"]).copy()
n_re1 = len(df_re1)
print(f"  Sessions with first-minute RE: {n_re1} / {len(df_pi)}")

if n_re1 > 10:
    fig6b, axes6b = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: within-mouse (session deviations from mouse mean)
    ax = axes6b[0]
    mouse_means_1m = df_re1.groupby("mouse_id").agg(
        mean_re1=("re_first_min", "mean"),
        mean_pi=("preference_index", "mean"),
    )
    df_re1 = df_re1.merge(mouse_means_1m, on="mouse_id")
    df_re1["dev_re1"] = df_re1["re_first_min"] - df_re1["mean_re1"]
    df_re1["dev_pi"] = df_re1["preference_index"] - df_re1["mean_pi"]

    ax.scatter(df_re1["dev_re1"], df_re1["dev_pi"], alpha=0.4, s=20,
               color="#2a9d8f")
    r, p = spearmanr(df_re1["dev_re1"], df_re1["dev_pi"])
    ax.set_xlabel("Deviation from mouse mean RE (first min)", fontsize=11)
    ax.set_ylabel("Deviation from mouse mean PI", fontsize=11)
    ax.set_title(f"Within-mouse\nSpearman r={r:.3f}, p={p:.4f}", fontsize=11)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    log_stat(f"\nWithin-mouse first-min RE vs PI:")
    log_stat(f"  Spearman r={r:.3f}, p={p:.4f}, n={n_re1}")

    # Panel B: between-mouse (mouse averages)
    ax = axes6b[1]
    mouse_avg_1m = df_pi.dropna(subset=["re_first_min", "preference_index"]).groupby(
        "mouse_id"
    ).agg(
        mean_re1=("re_first_min", "mean"),
        mean_pi=("preference_index", "mean"),
    ).reset_index()

    if len(mouse_avg_1m) > 5:
        ax.scatter(mouse_avg_1m["mean_re1"], mouse_avg_1m["mean_pi"],
                   alpha=0.7, s=40, color="#e76f51")
        r, p = spearmanr(mouse_avg_1m["mean_re1"], mouse_avg_1m["mean_pi"])
        z = np.polyfit(mouse_avg_1m["mean_re1"], mouse_avg_1m["mean_pi"], 1)
        x_line = np.linspace(mouse_avg_1m["mean_re1"].min(),
                             mouse_avg_1m["mean_re1"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#e76f51", alpha=0.5)

        ax.set_xlabel("Mean First-Minute RE", fontsize=11)
        ax.set_ylabel("Mean Preference Index", fontsize=11)
        ax.set_title(f"Between-mouse\nSpearman r={r:.3f}, p={p:.4f}",
                     fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        log_stat(f"\nBetween-mouse first-min RE vs PI:")
        log_stat(f"  n={len(mouse_avg_1m)} mice")
        log_stat(f"  Spearman r={r:.3f}, p={p:.4f}")

    # Compare full-hab RE vs first-min RE
    both = df_pi.dropna(subset=["roaming_entropy", "re_first_min"]).copy()
    if len(both) > 5:
        r_corr, p_corr = spearmanr(both["roaming_entropy"], both["re_first_min"])
        log_stat(f"\nFull-hab RE vs first-min RE correlation:")
        log_stat(f"  Spearman r={r_corr:.3f}, p={p_corr:.4f}, n={len(both)}")

    plt.suptitle("First-minute roaming entropy (first 60s of habituation) predicts PI",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig6b.savefig(os.path.join(OUTPUT_DIR, "fig6b_re_firstmin_vs_pi.png"),
                  dpi=200, bbox_inches="tight")
    fig6b.savefig(os.path.join(OUTPUT_DIR, "fig6b_re_firstmin_vs_pi.pdf"),
                  bbox_inches="tight")
    print("  Saved fig6b_re_firstmin_vs_pi")
else:
    print(f"  Skipping fig6b: only {n_re1} sessions with first-minute RE data")


# ── 7c. Figure: Voc PI vs Other Sounds PI ───────────────────────────

print("Plotting Figure: Voc PI vs Other Sounds PI...")

df_voc_vs = df_pref.dropna(subset=["voc_pi", "other_sounds_pi"]).copy()
# Exclude w2_vocalisations (all stim are voc, no "other" sounds)
df_voc_vs = df_voc_vs[df_voc_vs["day"] != "w2_vocalisations"]

if len(df_voc_vs) > 0:
    days_with_both = [d for d in DAY_ORDER
                      if d in df_voc_vs["day"].values and d != "w2_vocalisations"]
    n_panels = len(days_with_both) + 1  # +1 for pooled panel
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))

    fig_vo, axes_vo = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                                    squeeze=False)

    all_voc_vals = []
    all_other_vals = []

    for idx, day in enumerate(days_with_both):
        r, c = divmod(idx, ncols)
        ax = axes_vo[r][c]
        sub = df_voc_vs[df_voc_vs["day"] == day]

        ax.scatter(sub["other_sounds_pi"], sub["voc_pi"], s=40, color="#E63946",
                   edgecolors="white", linewidths=0.5, zorder=3)

        lo = min(sub["other_sounds_pi"].min(), sub["voc_pi"].min()) * 1.1
        hi = max(sub["other_sounds_pi"].max(), sub["voc_pi"].max()) * 1.1
        lo = min(lo, -0.05)
        hi = max(hi, 0.05)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.6, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Other Sounds PI", fontsize=10)
        ax.set_ylabel("Vocalisation PI", fontsize=10)

        n = len(sub)
        n_above = int((sub["voc_pi"] > sub["other_sounds_pi"]).sum())
        title = f"{DAY_SHORT[day]} (n={n})"

        vv = sub["voc_pi"].values
        ov = sub["other_sounds_pi"].values
        all_voc_vals.extend(vv)
        all_other_vals.extend(ov)

        if n >= 6:
            diff = vv - ov
            diff_nz = diff[diff != 0]
            if len(diff_nz) >= 6:
                from scipy.stats import wilcoxon as wil
                stat, p = wil(diff_nz)
                title += f"\nWilcoxon p={p:.4f}"

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.03, 0.97, f"{n_above} prefer voc",
                transform=ax.transAxes, fontsize=9, va="top", color="#E63946")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15)

    # Pooled panel
    pool_idx = len(days_with_both)
    r, c = divmod(pool_idx, ncols)
    ax = axes_vo[r][c]
    all_voc_vals = np.array(all_voc_vals)
    all_other_vals = np.array(all_other_vals)
    ax.scatter(all_other_vals, all_voc_vals, s=40, color="#333333",
               edgecolors="white", linewidths=0.5, zorder=3)
    lo = min(all_other_vals.min(), all_voc_vals.min()) * 1.1
    hi = max(all_other_vals.max(), all_voc_vals.max()) * 1.1
    lo = min(lo, -0.05)
    hi = max(hi, 0.05)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.6, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Other Sounds PI", fontsize=10)
    ax.set_ylabel("Vocalisation PI", fontsize=10)
    n_pool = len(all_voc_vals)
    n_above_pool = int((all_voc_vals > all_other_vals).sum())
    pool_title = f"All days pooled (n={n_pool})"
    if n_pool >= 6:
        diff_pool = all_voc_vals - all_other_vals
        diff_pool_nz = diff_pool[diff_pool != 0]
        if len(diff_pool_nz) >= 6:
            stat, p = wilcoxon_1s(diff_pool_nz)
            pool_title += f"\nWilcoxon p={p:.4f}"
    ax.set_title(pool_title, fontsize=11, fontweight="bold")
    ax.text(0.03, 0.97, f"{n_above_pool} prefer voc",
            transform=ax.transAxes, fontsize=9, va="top", color="#E63946")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

    # Hide unused axes
    for idx in range(pool_idx + 1, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_vo[r][c].set_visible(False)

    fig_vo.suptitle(
        "Vocalisation PI vs Other Sounds PI per session\n"
        "(each point = one mouse × day, above diagonal = prefers voc)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_vo.savefig(os.path.join(OUTPUT_DIR, "fig_voc_vs_other_sounds_pi.png"),
                   dpi=200, bbox_inches="tight")
    fig_vo.savefig(os.path.join(OUTPUT_DIR, "fig_voc_vs_other_sounds_pi.pdf"),
                   bbox_inches="tight")
    print("  Saved fig_voc_vs_other_sounds_pi")
else:
    print("  No sessions with both voc PI and other sounds PI — skipping figure")


# ── 8. Mixed models: variance decomposition ──────────────────────────

print("\nFitting mixed-effects models...")
log_stat("\n" + "=" * 60)
log_stat("MIXED-EFFECTS MODELS")
log_stat("=" * 60)

# Model 1: Unconditional means model
# PI ~ 1 + (1 | mouse_id)
df_model = df_pi.dropna(subset=["preference_index"]).copy()
icc_mouse = np.nan
icc_session = np.nan

if len(df_model) > 10 and df_model["mouse_id"].nunique() > 2:
    try:
        model1 = smf.mixedlm("preference_index ~ 1", df_model,
                              groups=df_model["mouse_id"])
        result1 = model1.fit(reml=True)

        # Extract variance components
        var_mouse = result1.cov_re.iloc[0, 0]  # random intercept variance
        var_resid = result1.scale  # residual variance
        icc_mouse = var_mouse / (var_mouse + var_resid)
        icc_session = var_resid / (var_mouse + var_resid)

        log_stat("\nModel 1: Unconditional means model")
        log_stat(f"  PI ~ 1 + (1 | mouse_id)")
        log_stat(f"  Intercept (grand mean PI): "
                 f"{result1.fe_params['Intercept']:.3f} "
                 f"(p={result1.pvalues['Intercept']:.4f})")
        log_stat(f"  Var(mouse):  {var_mouse:.4f}")
        log_stat(f"  Var(resid):  {var_resid:.4f}")
        log_stat(f"  ICC(mouse):  {icc_mouse:.3f}")
        log_stat(f"  ICC(session): {icc_session:.3f}")
        log_stat(f"  {icc_mouse*100:.1f}% of variance is between mice, "
                 f"{icc_session*100:.1f}% is within mice (between sessions)")

    except Exception as e:
        log_stat(f"\nModel 1 failed: {e}")
        icc_mouse, icc_session = np.nan, np.nan

    # Model 2: Day as fixed effect
    # PI ~ day + (1 | mouse_id)
    try:
        # use treatment coding with w1_d1 as reference
        df_model["day_cat"] = pd.Categorical(df_model["day"],
                                              categories=PI_DAYS)
        model2 = smf.mixedlm("preference_index ~ C(day_cat)",
                              df_model, groups=df_model["mouse_id"])
        result2 = model2.fit(reml=True)

        log_stat("\nModel 2: Day effect model")
        log_stat(f"  PI ~ day + (1 | mouse_id)")
        log_stat(f"\n{result2.summary().tables[1].to_string()}")

        # Is vocalisation day different?
        voc_coef_name = [c for c in result2.fe_params.index
                         if "w2_vocalisations" in c]
        if voc_coef_name:
            coef = result2.fe_params[voc_coef_name[0]]
            pval = result2.pvalues[voc_coef_name[0]]
            log_stat(f"\n  Vocalisation day coefficient: {coef:.3f} "
                     f"(p={pval:.4f})")

    except Exception as e:
        log_stat(f"\nModel 2 failed: {e}")

    # Model 3: RE predicts PI (full model from the report)
    # PI ~ RE_between + RE_within + (1 | mouse_id)
    df_re_model = df_model.dropna(subset=["roaming_entropy"]).copy()
    if len(df_re_model) > 10:
        try:
            # Centre RE at mouse mean (within) and grand mean (between)
            mouse_re = df_re_model.groupby("mouse_id")[
                "roaming_entropy"].mean()
            grand_re = mouse_re.mean()
            df_re_model["re_mouse_mean"] = df_re_model["mouse_id"].map(
                mouse_re)
            df_re_model["re_within"] = (df_re_model["roaming_entropy"]
                                        - df_re_model["re_mouse_mean"])
            df_re_model["re_between"] = (df_re_model["re_mouse_mean"]
                                         - grand_re)

            model3 = smf.mixedlm(
                "preference_index ~ re_within + re_between",
                df_re_model, groups=df_re_model["mouse_id"]
            )
            result3 = model3.fit(reml=True)

            log_stat("\nModel 3: Exploration predicts preference")
            log_stat("  PI ~ RE_within + RE_between + (1 | mouse_id)")
            log_stat(f"\n{result3.summary().tables[1].to_string()}")

        except Exception as e:
            log_stat(f"\nModel 3 failed: {e}")


# ── 9. Figure 7: Variance decomposition summary ──────────────────────

print("Plotting Figure 7: Variance decomposition...")

fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: ICC pie chart
if not np.isnan(icc_mouse):
    ax = axes7[0]
    sizes = [icc_mouse * 100, icc_session * 100]
    labels = [f"Between mice\n({sizes[0]:.1f}%)",
              f"Between sessions\n({sizes[1]:.1f}%)"]
    colours = ["#E63946", "#457B9D"]
    ax.pie(sizes, labels=labels, colors=colours, autopct="",
           startangle=90, textprops={"fontsize": 11})
    ax.set_title("Variance in preference index", fontsize=12)

# Panel B: Kruskal-Wallis across days
ax = axes7[1]
day_groups = [df_pi[df_pi["day"] == d]["preference_index"].dropna().values
              for d in PI_DAYS]
day_groups_nonempty = [g for g in day_groups if len(g) > 0]
if len(day_groups_nonempty) > 1:
    stat, p_kw = kruskal(*day_groups_nonempty)
    log_stat(f"\nKruskal-Wallis across days: H={stat:.2f}, p={p_kw:.4f}")

    # Median per day
    medians = [np.median(g) for g in day_groups]
    x = np.arange(len(PI_DAYS))
    ax.bar(x, medians, color="#457B9D", alpha=0.7, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], fontsize=9)
    ax.set_ylabel("Median PI", fontsize=11)
    ax.set_title(f"Kruskal-Wallis: H={stat:.1f}, p={p_kw:.4f}",
                 fontsize=11)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Individual differences in sound preference", fontsize=14,
             y=1.02)
plt.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_icc_summary.png"),
             dpi=200, bbox_inches="tight")
fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_icc_summary.pdf"),
             bbox_inches="tight")
print("  Saved fig7_icc_summary")


# ── 10. Complexity analysis: stats ────────────────────────────────────

log_stat("\n" + "=" * 60)
log_stat("SENSORY COMPLEXITY ANALYSIS")
log_stat("=" * 60)

# For each day with complexity ordering, test if visit duration differs
for day in DAY_ORDER:
    order = EXPERIMENT_DAYS[day].get("complexity_order", [])
    if len(order) < 2:
        continue

    day_data = df_stim[df_stim["day"] == day]
    if len(day_data) == 0:
        continue

    # Per-mouse mean visit duration per stimulus type
    mouse_means = day_data.groupby(["mouse_id", "stim_type"])[
        "avg_visit_dur_ms"
    ].mean().reset_index()

    available = [s for s in order if s in mouse_means["stim_type"].values]
    if len(available) < 2:
        continue

    log_stat(f"\n--- {DAY_SHORT[day]}: {' < '.join(available)} ---")

    groups = [mouse_means[mouse_means["stim_type"] == s][
        "avg_visit_dur_ms"].values for s in available]

    for s, g in zip(available, groups):
        log_stat(f"  {s}: mean={np.mean(g):.1f}ms, "
                 f"median={np.median(g):.1f}ms, n={len(g)}")

    if all(len(g) > 0 for g in groups):
        stat, p = kruskal(*groups)
        log_stat(f"  Kruskal-Wallis: H={stat:.2f}, p={p:.4f}")


# ── 11. Per-day one-sample tests: is PI different from 0? ─────────────

log_stat("\n" + "=" * 60)
log_stat("ONE-SAMPLE TESTS: PI vs 0")
log_stat("=" * 60)

from scipy.stats import wilcoxon as wilcoxon_1s

for d in PI_DAYS:
    vals = df_pi[df_pi["day"] == d]["preference_index"].dropna().values
    if len(vals) >= 6:
        stat, p = wilcoxon_1s(vals)
        log_stat(f"  {DAY_SHORT[d]}: median={np.median(vals):.3f}, "
                 f"mean={np.mean(vals):.3f}, Wilcoxon p={p:.4f} "
                 f"(n={len(vals)})")
    else:
        log_stat(f"  {DAY_SHORT[d]}: n={len(vals)} (too few for test)")


# ── 12. Vocalisation vs other sounds (within-session paired test) ────

log_stat("\n" + "=" * 60)
log_stat("VOCALISATION vs OTHER SOUNDS (within-session, paired)")
log_stat("=" * 60)
log_stat("For each mouse × day, compare vocalisation avg visit duration")
log_stat("against the mean of all other (non-vocalisation) stimulus types.")

for d in DAY_ORDER:
    day_stim = df_stim[df_stim["day"] == d]
    if len(day_stim) == 0:
        continue

    # Per-mouse mean visit duration per stimulus type
    mouse_stim = (
        day_stim.groupby(["mouse_id", "stim_type"])["avg_visit_dur_ms"]
        .mean()
        .reset_index()
    )

    # Identify mice that have both vocalisation AND at least one other type
    voc_labels = set()
    if EXPERIMENT_DAYS[d]["mode"] == "vocalisation":
        # w2_vocalisations: every non-silent stim is a vocalisation file
        # skip this day — there are no "other" sounds to compare against
        log_stat(f"\n  {DAY_SHORT[d]}: skipped (all stimuli are vocalisations)")
        continue
    else:
        voc_labels = {"vocalisation"}

    paired_voc = []
    paired_other = []
    for mouse in mouse_stim["mouse_id"].unique():
        msub = mouse_stim[mouse_stim["mouse_id"] == mouse]
        voc_rows = msub[msub["stim_type"].isin(voc_labels)]
        other_rows = msub[~msub["stim_type"].isin(voc_labels | {"silent", "unknown"})]
        if len(voc_rows) == 0 or len(other_rows) == 0:
            continue
        paired_voc.append(voc_rows["avg_visit_dur_ms"].mean())
        paired_other.append(other_rows["avg_visit_dur_ms"].mean())

    paired_voc = np.array(paired_voc)
    paired_other = np.array(paired_other)
    n = len(paired_voc)

    if n == 0:
        log_stat(f"\n  {DAY_SHORT[d]}: no mice with both vocalisation and other sounds")
        continue

    diff = paired_voc - paired_other
    log_stat(f"\n  {DAY_SHORT[d]} (n={n} mice):")
    log_stat(f"    Vocalisation avg: {np.mean(paired_voc):.0f}ms "
             f"(median {np.median(paired_voc):.0f}ms)")
    log_stat(f"    Other sounds avg: {np.mean(paired_other):.0f}ms "
             f"(median {np.median(paired_other):.0f}ms)")
    log_stat(f"    Difference (voc - other): mean={np.mean(diff):+.0f}ms, "
             f"median={np.median(diff):+.0f}ms")

    if n >= 6:
        diff_nz = diff[diff != 0]
        if len(diff_nz) >= 6:
            stat, p = wilcoxon_1s(diff_nz)
            log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}")
        else:
            log_stat(f"    Too few non-zero differences for Wilcoxon ({len(diff_nz)})")
    else:
        log_stat(f"    Too few mice for test (n={n})")


# ── 12b. Voc PI vs Other Sounds PI (per-day paired test) ─────────────

log_stat("\n" + "=" * 60)
log_stat("VOCALISATION PI vs OTHER SOUNDS PI (per-day, paired)")
log_stat("=" * 60)
log_stat("For each mouse × day, compare voc_pi against other_sounds_pi.")

df_vo_stats = df_pref.dropna(subset=["voc_pi", "other_sounds_pi"]).copy()
df_vo_stats = df_vo_stats[df_vo_stats["day"] != "w2_vocalisations"]

for d in DAY_ORDER:
    if d == "w2_vocalisations":
        log_stat(f"\n  {DAY_SHORT[d]}: skipped (all stimuli are vocalisations)")
        continue
    sub = df_vo_stats[df_vo_stats["day"] == d]
    n = len(sub)
    if n == 0:
        continue

    vv = sub["voc_pi"].values
    ov = sub["other_sounds_pi"].values
    diff = vv - ov

    log_stat(f"\n  {DAY_SHORT[d]} (n={n} mice):")
    log_stat(f"    Voc PI: mean={np.mean(vv):.3f}, median={np.median(vv):.3f}")
    log_stat(f"    Other PI: mean={np.mean(ov):.3f}, median={np.median(ov):.3f}")
    log_stat(f"    Diff (voc - other): mean={np.mean(diff):+.3f}, "
             f"median={np.median(diff):+.3f}")

    if n >= 6:
        diff_nz = diff[diff != 0]
        if len(diff_nz) >= 6:
            stat, p = wilcoxon_1s(diff_nz)
            log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}")
        else:
            log_stat(f"    Too few non-zero differences ({len(diff_nz)})")
    else:
        log_stat(f"    Too few mice for test (n={n})")

# Pooled across days
all_vo = df_vo_stats["voc_pi"].values
all_ot = df_vo_stats["other_sounds_pi"].values
n_pool = len(all_vo)
if n_pool >= 6:
    diff_pool = all_vo - all_ot
    diff_nz = diff_pool[diff_pool != 0]
    if len(diff_nz) >= 6:
        stat, p = wilcoxon_1s(diff_nz)
        log_stat(f"\n  Pooled across days (n={n_pool}):")
        log_stat(f"    Voc PI mean={np.mean(all_vo):.3f}, Other PI mean={np.mean(all_ot):.3f}")
        log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}")


# ── 12c. First-minute RE stats ──────────────────────────────────────

log_stat("\n" + "=" * 60)
log_stat("FIRST-MINUTE ROAMING ENTROPY")
log_stat("=" * 60)

df_re1_stats = df_pi.dropna(subset=["re_first_min"]).copy()
n_re1_total = len(df_re1_stats)
log_stat(f"  Sessions with first-minute RE: {n_re1_total} / {len(df_pi)}")

if n_re1_total > 5:
    log_stat(f"  Mean first-min RE: {df_re1_stats['re_first_min'].mean():.3f}")
    log_stat(f"  Median first-min RE: {df_re1_stats['re_first_min'].median():.3f}")

    # Correlation with full-hab RE
    both_re = df_pi.dropna(subset=["roaming_entropy", "re_first_min"])
    if len(both_re) > 5:
        r, p = spearmanr(both_re["roaming_entropy"], both_re["re_first_min"])
        log_stat(f"\n  Full-hab RE vs first-min RE:")
        log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={len(both_re)}")

    # Within-mouse first-min RE vs PI
    if n_re1_total > 10:
        r, p = spearmanr(df_re1_stats["re_first_min"],
                         df_re1_stats["preference_index"])
        log_stat(f"\n  First-min RE vs PI (all sessions):")
        log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={n_re1_total}")

    # Between-mouse
    mouse_avg_1m = df_re1_stats.groupby("mouse_id").agg(
        mean_re1=("re_first_min", "mean"),
        mean_pi=("preference_index", "mean"),
    ).reset_index()
    if len(mouse_avg_1m) > 5:
        r, p = spearmanr(mouse_avg_1m["mean_re1"], mouse_avg_1m["mean_pi"])
        log_stat(f"\n  Between-mouse first-min RE vs PI:")
        log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={len(mouse_avg_1m)}")


# ── 13. save stats report ─────────────────────────────────────────────

stats_path = os.path.join(OUTPUT_DIR, "stats_report.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stats_lines))
print(f"\nStats report saved to {stats_path}")

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
plt.show()
