"""
Fiber Photometry ↔ Auditory Maze Alignment & Plotting
======================================================

Aligns Tucker-Davis Technologies (TDT) fiber photometry recordings
with the auditory maze visit CSV using TTL pulse matching.

The TDT recording typically starts before the maze experiment.
This script:
  1. Reads TDT block data (405nm isosbestic, 465nm GCaMP)
  2. Extracts TTL onset times from the MTL_ epoc store
  3. Matches them to sound_on_time events in the visit CSV
  4. Computes the precise clock offset between systems
  5. Calculates delta F/F (motion-corrected via isosbestic)
  6. Plots delta F/F with colour-coded TTL markers by stimulus type

Usage:
    python fibpho_alignment.py

Outputs (saved to OUTPUT_DIR):
    - fibpho_aligned_overview.png/.pdf   — full-session delta F/F with TTLs
    - fibpho_trial_panels.png/.pdf       — per-trial zoomed panels
    - fibpho_peri_event.png/.pdf         — peri-event average (PETH)
    - alignment_report.csv               — matched TTL ↔ CSV events
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


def safe_savefig(fig, path, **kwargs):
    """Save figure, catching PermissionError for locked files (e.g. open PDFs)."""
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: Could not write {os.path.basename(path)} (file locked)")

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# ── configuration ─────────────────────────────────────────────────────

# TDT tank path
TANK_PATH = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "dopamine recordings",
    "dopaminergic recordings", "MickeyMouse-251205-113917"
)

# Visit CSV from auditory maze
VISIT_CSV = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "dopamine recordings",
    "vocalisations", "time_2025-12-05_11_41_09mouseK30_5",
    "mouseK30_5_vocalisations_detailed_visits.csv"
)

# Trials CSV
TRIALS_CSV = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "dopamine recordings",
    "vocalisations", "time_2025-12-05_11_41_09mouseK30_5",
    "trials_time_2025-12-05_11_41_09.csv"
)

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "dopamine recordings",
    "vocalisations", "time_2025-12-05_11_41_09mouseK30_5",
    "fibpho_analysis"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TDT store names
STREAM_465 = "_465A"   # GCaMP calcium-dependent
STREAM_405 = "_405A"   # Isosbestic reference
EPOC_TTL = "MTL_"      # TTL epoc store

# Signal processing
DOWNSAMPLE = 10        # downsample factor for plotting (1017 Hz -> ~102 Hz)
SMOOTH_WIN = 51        # Savitzky-Golay window for dF/F smoothing
BASELINE_PERCENTILE = 10  # percentile for F0 baseline
PERI_EVENT_WINDOW = (-3, 7)  # seconds around each event for PETH

# Stimulus colour mapping
STIM_COLOURS = {
    "dyadic_social_interaction":          "#E63946",  # red
    "male_male_interaction":              "#457B9D",  # blue
    "male_w_female_oestrus":              "#2A9D8F",  # teal
    "mousetube3BB89473_socialcomparison": "#E9C46A",  # gold
    "run1_day2_male_w_female_oestrus":    "#F4A261",  # orange
    "run3_day2-2_male_w_female_oestrus":  "#264653",  # dark teal
    "male-female aggression":             "#9B2226",  # dark red
    "female_oestrus_alone":               "#A8DADC",  # light blue
    "silent_trial":                       "#CCCCCC",  # grey
    "unknown":                            "#888888",  # dark grey
}

# ── helper functions ──────────────────────────────────────────────────


def extract_stim_name(stimulus_str):
    """Extract a short stimulus name from the CSV stimulus field.

    The stimulus field can be:
      - 'frequency:0 | interval_type:silent_trial | ...'
      - 'C:/Users/.../trimmed_vocalisations/dyadic_social_interaction.wav'
      - 'Unknown_Stimulus'
    """
    s = str(stimulus_str).strip()
    if "silent_trial" in s:
        return "silent_trial"
    if "Unknown" in s:
        return "unknown"
    # vocalisation path: extract filename stem
    if "/" in s or "\\" in s:
        fname = os.path.splitext(os.path.basename(s))[0]
        return fname
    return s


def get_stim_colour(stim_name):
    """Get colour for a stimulus name, with fallback."""
    for key, colour in STIM_COLOURS.items():
        if key in stim_name:
            return colour
    return "#888888"


def compute_delta_f_over_f(signal_465, signal_405, fs,
                           smooth_win=SMOOTH_WIN,
                           baseline_pct=BASELINE_PERCENTILE):
    """Compute motion-corrected delta F/F.

    Uses the isosbestic (405nm) channel to remove motion artefacts
    via linear regression, then computes dF/F relative to a
    rolling baseline.

    Parameters
    ----------
    signal_465 : array — GCaMP calcium-dependent signal
    signal_405 : array — isosbestic reference signal
    fs : float — sampling rate (Hz)
    smooth_win : int — Savitzky-Golay smoothing window
    baseline_pct : float — percentile for F0 estimation

    Returns
    -------
    dff : array — delta F/F (dimensionless)
    time : array — time vector (seconds)
    """
    # ensure same length
    n = min(len(signal_465), len(signal_405))
    sig = signal_465[:n].astype(np.float64)
    ref = signal_405[:n].astype(np.float64)

    # remove photobleaching with a long exponential fit
    # (or simply use a high-percentile sliding baseline)
    # Here we use the isosbestic for motion correction:
    # fit 405 to predict 465, residual = calcium signal

    # 1. smooth both channels
    if smooth_win > 3 and smooth_win < len(sig):
        win = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
        sig_s = savgol_filter(sig, win, polyorder=3)
        ref_s = savgol_filter(ref, win, polyorder=3)
    else:
        sig_s = sig
        ref_s = ref

    # 2. linear regression: sig = a * ref + b
    # (fit in chunks to handle photobleaching)
    coeffs = np.polyfit(ref_s, sig_s, deg=1)
    predicted_465 = np.polyval(coeffs, ref_s)

    # 3. delta F / F = (signal - predicted) / predicted
    dff = (sig_s - predicted_465) / predicted_465

    # 4. baseline normalisation using rolling percentile
    # Use a ~60s window for the rolling baseline
    baseline_win = int(60 * fs)
    if baseline_win > len(dff):
        baseline_win = len(dff)

    # rolling percentile via uniform filter on sorted-ish data
    # (fast approximation: use the overall baseline percentile)
    f0 = np.percentile(dff, baseline_pct)
    dff = dff - f0

    time = np.arange(n) / fs
    return dff, time


# ── 1. load TDT data ─────────────────────────────────────────────────

print("Loading TDT block...")
try:
    import tdt
except ImportError:
    print("ERROR: tdt package not installed. Install with: pip install tdt")
    sys.exit(1)

data = tdt.read_block(TANK_PATH)

# Extract photometry signals
sig_465 = data.streams[STREAM_465].data
sig_405 = data.streams[STREAM_405].data
fs = data.streams[STREAM_465].fs
start_time_tdt = data.streams[STREAM_465].start_time

print(f"  465nm: {len(sig_465)} samples at {fs:.1f} Hz "
      f"({len(sig_465)/fs:.1f}s)")
print(f"  405nm: {len(sig_405)} samples")
print(f"  Stream start_time: {start_time_tdt:.4f}s")

# Extract TTL events
ttl_onsets = data.epocs[EPOC_TTL].onset
ttl_offsets = data.epocs[EPOC_TTL].offset
print(f"  TTL events: {len(ttl_onsets)} onsets")


# ── 2. load visit CSV ────────────────────────────────────────────────

print("\nLoading visit CSV...")
df_visits = pd.read_csv(VISIT_CSV)
print(f"  {len(df_visits)} visit records, "
      f"trials {df_visits.trial_ID.min()}-{df_visits.trial_ID.max()}")

# Separate active sound events (non-silent, non-entrance)
active_mask = (
    ~df_visits.stimulus.str.contains("silent_trial", na=False) &
    ~df_visits.stimulus.str.contains("Unknown", na=False)
)
df_active = df_visits[active_mask].copy()
df_active["stim_name"] = df_active["stimulus"].apply(extract_stim_name)
print(f"  {len(df_active)} active sound events")

# Load trials CSV for trial boundaries
df_trials = pd.read_csv(TRIALS_CSV)
print(f"  {len(df_trials.trial_ID.unique())} trials from trials CSV")


# ── 3. align clocks ──────────────────────────────────────────────────

print("\nAligning TDT and CSV clocks...")

# The first TTL (onset[0]) is typically a test pulse before the
# experiment. The real experiment TTLs start from onset[1].
# We match inter-event intervals to confirm.

tdt_experiment_onsets = ttl_onsets[1:]  # skip test pulse
tdt_experiment_offsets = ttl_offsets[1:]
csv_sound_times = df_active["sound_on_time"].values

# Compute offset: TDT onset[1] ↔ CSV sound_on_time[0]
TIME_OFFSET = csv_sound_times[0] - tdt_experiment_onsets[0]
print(f"  Time offset (Unix - TDT): {TIME_OFFSET:.4f}s")

# Verify with IEI correlation
tdt_iei = np.diff(tdt_experiment_onsets)
csv_iei = np.diff(csv_sound_times[:len(tdt_iei) + 1])
min_len = min(len(tdt_iei), len(csv_iei))
corr = np.corrcoef(tdt_iei[:min_len], csv_iei[:min_len])[0, 1]
print(f"  IEI correlation: {corr:.6f}")

# Match each TDT TTL to nearest CSV event
alignment_records = []
for i, (ton, toff) in enumerate(zip(tdt_experiment_onsets,
                                     tdt_experiment_offsets)):
    tdt_unix = ton + TIME_OFFSET
    diffs = np.abs(csv_sound_times - tdt_unix)
    best_j = np.argmin(diffs)
    residual = csv_sound_times[best_j] - tdt_unix

    csv_row = df_active.iloc[best_j]
    alignment_records.append({
        "ttl_idx": i,
        "csv_idx": best_j,
        "tdt_time_s": ton,
        "tdt_unix": tdt_unix,
        "csv_sound_on": csv_sound_times[best_j],
        "residual_s": residual,
        "ttl_duration_s": toff - ton,
        "trial_ID": csv_row["trial_ID"],
        "ROI_visited": csv_row["ROI_visited"],
        "stim_name": csv_row["stim_name"],
        "stim_colour": get_stim_colour(csv_row["stim_name"]),
    })

df_alignment = pd.DataFrame(alignment_records)
df_alignment.to_csv(os.path.join(OUTPUT_DIR, "alignment_report.csv"),
                    index=False)

residuals = df_alignment["residual_s"].values
print(f"  Matched {len(df_alignment)} events")
print(f"  Residuals: mean={np.mean(residuals)*1000:.1f}ms, "
      f"std={np.std(residuals)*1000:.1f}ms, "
      f"max={np.max(np.abs(residuals))*1000:.1f}ms")


# ── 4. compute delta F/F ─────────────────────────────────────────────

print("\nComputing delta F/F...")
dff, time_dff = compute_delta_f_over_f(sig_465, sig_405, fs)

# Downsample for plotting
dff_ds = dff[::DOWNSAMPLE]
time_ds = time_dff[::DOWNSAMPLE]
fs_ds = fs / DOWNSAMPLE
print(f"  dF/F computed: {len(dff)} samples -> {len(dff_ds)} (downsampled {DOWNSAMPLE}x)")


# ── 5. convert trial boundaries to TDT time ──────────────────────────

# Get unique trial start/end times from the trials CSV
trial_boundaries = []
for tid in sorted(df_trials["trial_ID"].unique()):
    sub = df_trials[df_trials["trial_ID"] == tid]
    t_start_unix = sub["trial_start_time"].iloc[0]
    t_end_unix = sub["end_trial_time"].iloc[0]
    # Convert to TDT time
    t_start_tdt = t_start_unix - TIME_OFFSET
    t_end_tdt = t_end_unix - TIME_OFFSET
    is_silent = sub["interval_type"].iloc[0] == "silent_trial"
    trial_boundaries.append({
        "trial_ID": tid,
        "start_tdt": t_start_tdt,
        "end_tdt": t_end_tdt,
        "start_unix": t_start_unix,
        "end_unix": t_end_unix,
        "is_silent": is_silent,
    })

df_trial_bounds = pd.DataFrame(trial_boundaries)
print(f"\nTrial boundaries (TDT time):")
for _, tb in df_trial_bounds.iterrows():
    label = "SILENT" if tb["is_silent"] else "ACTIVE"
    print(f"  Trial {tb['trial_ID']}: {tb['start_tdt']:.1f}s - "
          f"{tb['end_tdt']:.1f}s ({label})")


# ── 6. plot 1: full-session overview ──────────────────────────────────

print("\nPlotting full-session overview...")

fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                         gridspec_kw={"height_ratios": [4, 0.5]})

# Panel 1: delta F/F with TTL stimulus bars overlaid
ax1 = axes[0]

# Draw TTL stimulus bars BEHIND the trace (zorder=1)
for _, row in df_alignment.iterrows():
    t_on = row["tdt_time_s"]
    dur = row["ttl_duration_s"]
    ax1.axvspan(t_on, t_on + dur, alpha=0.35,
                color=row["stim_colour"], zorder=1, linewidth=0)

# dF/F trace ON TOP of the stimulus bars (zorder=2)
ax1.plot(time_ds, dff_ds, color="#1a1a2e", linewidth=0.3, alpha=0.9,
         zorder=2)

# Clip y-axis to exclude the initial ramp-up outlier: use 0.5th-99.5th
# percentile of the signal AFTER the first 60 seconds
stable_mask = time_ds > 60
if stable_mask.any():
    y_lo = np.percentile(dff_ds[stable_mask], 0.5)
    y_hi = np.percentile(dff_ds[stable_mask], 99.5)
    y_margin = (y_hi - y_lo) * 0.05
    ax1.set_ylim(y_lo - y_margin, y_hi + y_margin)

ax1.set_ylabel("ΔF/F", fontsize=12)
ax1.set_title("Mouse K30_5 — Fiber Photometry (465nm GCaMP, "
              "motion-corrected via 405nm isosbestic)\n"
              "Coloured bars = stimulus presentations",
              fontsize=13)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel 2: trial structure
ax2 = axes[1]
for _, tb in df_trial_bounds.iterrows():
    colour = "#2CA02C" if not tb["is_silent"] else "#D62728"
    ax2.barh(0, tb["end_tdt"] - tb["start_tdt"],
             left=tb["start_tdt"], height=0.8,
             color=colour, alpha=0.6, edgecolor="white", linewidth=0.5)
    ax2.text((tb["start_tdt"] + tb["end_tdt"]) / 2, 0,
             f"T{int(tb['trial_ID'])}", ha="center", va="center",
             fontsize=8, fontweight="bold", color="white")
ax2.set_ylim(-0.5, 0.5)
ax2.set_yticks([])
ax2.set_ylabel("Trial", fontsize=10)
ax2.set_xlabel("Time (seconds from TDT recording start)", fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend for stimuli
unique_stims = df_alignment["stim_name"].unique()
legend_patches = [mpatches.Patch(color=get_stim_colour(s),
                                  label=s.replace("_", " "))
                  for s in unique_stims]
legend_patches.append(mpatches.Patch(color="#2CA02C", alpha=0.6,
                                      label="Active trial"))
legend_patches.append(mpatches.Patch(color="#D62728", alpha=0.6,
                                      label="Silent trial"))
ax1.legend(handles=legend_patches, loc="upper right", fontsize=7,
           ncol=2, framealpha=0.9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fibpho_aligned_overview.png"),
            dpi=200, bbox_inches="tight")
safe_savefig(fig, os.path.join(OUTPUT_DIR, "fibpho_aligned_overview.pdf"),
             bbox_inches="tight")
print("  Saved fibpho_aligned_overview.png/.pdf")


# ── 7. plot 2: per-trial zoomed panels ────────────────────────────────

print("Plotting per-trial panels...")

# Only plot active trials (even-numbered: 2, 4, 6, 8)
active_trials = df_trial_bounds[~df_trial_bounds["is_silent"]]
n_active = len(active_trials)

fig2, axes2 = plt.subplots(n_active, 1, figsize=(18, 4 * n_active),
                           sharex=False)
if n_active == 1:
    axes2 = [axes2]

for ax, (_, tb) in zip(axes2, active_trials.iterrows()):
    tid = int(tb["trial_ID"])
    t_start = tb["start_tdt"]
    t_end = tb["end_tdt"]

    # Add 30s padding
    t_plot_start = max(t_start - 30, 0)
    t_plot_end = min(t_end + 30, time_ds[-1])

    # 1) Plot dF/F trace FIRST so matplotlib sets the correct y-limits
    mask = (time_ds >= t_plot_start) & (time_ds <= t_plot_end)
    ax.plot(time_ds[mask], dff_ds[mask], color="#1a1a2e",
            linewidth=0.5, alpha=0.9, zorder=2)

    # 2) Trial boundary dashed lines
    ax.axvline(t_start, color="#888888", linestyle="--", alpha=0.4, lw=0.8)
    ax.axvline(t_end, color="#888888", linestyle="--", alpha=0.4, lw=0.8)

    # 3) TTL stimulus duration spans (behind the trace, like the overview)
    trial_events = df_alignment[df_alignment["trial_ID"] == tid]
    y_top = ax.get_ylim()[1]
    for _, ev in trial_events.iterrows():
        t_on = ev["tdt_time_s"]
        dur = ev["ttl_duration_s"]
        ax.axvspan(t_on, t_on + dur, alpha=0.30,
                   color=ev["stim_colour"], zorder=1, linewidth=0)
        # small label near the top
        ax.text(t_on, y_top * 0.95,
                ev["ROI_visited"], fontsize=5, rotation=90,
                ha="right", va="top", color=ev["stim_colour"],
                alpha=0.7, zorder=3)

    ax.set_ylabel("ΔF/F", fontsize=10)
    ax.set_title(f"Trial {tid} (Active) — "
                 f"{len(trial_events)} sound events", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add stimulus legend for this trial
    trial_stims = trial_events["stim_name"].unique()
    patches = [mpatches.Patch(color=get_stim_colour(s),
                               label=s.replace("_", " "))
               for s in trial_stims]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              framealpha=0.9)

axes2[-1].set_xlabel("Time (seconds from TDT recording start)",
                     fontsize=11)

plt.suptitle("Mouse K30_5 — Per-trial ΔF/F with stimulus events",
             fontsize=14, y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fibpho_trial_panels.png"),
             dpi=200, bbox_inches="tight")
safe_savefig(fig2, os.path.join(OUTPUT_DIR, "fibpho_trial_panels.pdf"),
             bbox_inches="tight")
print("  Saved fibpho_trial_panels.png/.pdf")


# ── 8. plot 3: peri-event time histogram (PETH) ──────────────────────

print("Plotting peri-event average (PETH)...")

pre_s, post_s = PERI_EVENT_WINDOW
pre_samples = int(abs(pre_s) * fs_ds)
post_samples = int(post_s * fs_ds)
total_samples = pre_samples + post_samples
peth_time = np.linspace(pre_s, post_s, total_samples)

# Group events by stimulus
stim_groups = df_alignment.groupby("stim_name")

fig3, ax3 = plt.subplots(figsize=(10, 6))

peth_data = {}
for stim_name, group in stim_groups:
    traces = []
    for _, ev in group.iterrows():
        t_event = ev["tdt_time_s"]
        # Find index in downsampled time
        idx = int(t_event * fs_ds)
        start_idx = idx - pre_samples
        end_idx = idx + post_samples

        if start_idx < 0 or end_idx > len(dff_ds):
            continue

        trace = dff_ds[start_idx:end_idx]
        if len(trace) == total_samples:
            traces.append(trace)

    if len(traces) < 3:
        continue

    traces = np.array(traces)
    mean_trace = np.mean(traces, axis=0)
    sem_trace = np.std(traces, axis=0) / np.sqrt(len(traces))

    colour = get_stim_colour(stim_name)
    label = stim_name.replace("_", " ")
    ax3.plot(peth_time, mean_trace, color=colour, linewidth=1.5,
             label=f"{label} (n={len(traces)})")
    ax3.fill_between(peth_time, mean_trace - sem_trace,
                     mean_trace + sem_trace, color=colour, alpha=0.15)

    peth_data[stim_name] = {
        "mean": mean_trace,
        "sem": sem_trace,
        "n": len(traces),
    }

ax3.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1,
            label="Sound onset")
ax3.axhline(0, color="grey", linestyle="-", alpha=0.3, linewidth=0.5)
ax3.set_xlabel("Time relative to sound onset (s)", fontsize=12)
ax3.set_ylabel("ΔF/F (mean ± SEM)", fontsize=12)
ax3.set_title("Mouse K30_5 — Peri-event ΔF/F by vocalisation type",
              fontsize=13)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlim(pre_s, post_s)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "fibpho_peri_event.png"),
             dpi=200, bbox_inches="tight")
safe_savefig(fig3, os.path.join(OUTPUT_DIR, "fibpho_peri_event.pdf"),
             bbox_inches="tight")
print("  Saved fibpho_peri_event.png/.pdf")


# ── 9. plot 4: combined PETH (all stimuli pooled) ────────────────────

print("Plotting pooled PETH...")

all_traces = []
for stim_name, info in peth_data.items():
    # We need the raw traces, not just mean — recompute
    pass

# Recompute pooled
pooled_traces = []
for _, ev in df_alignment.iterrows():
    t_event = ev["tdt_time_s"]
    idx = int(t_event * fs_ds)
    start_idx = idx - pre_samples
    end_idx = idx + post_samples
    if start_idx < 0 or end_idx > len(dff_ds):
        continue
    trace = dff_ds[start_idx:end_idx]
    if len(trace) == total_samples:
        pooled_traces.append(trace)

if pooled_traces:
    pooled_traces = np.array(pooled_traces)
    pooled_mean = np.mean(pooled_traces, axis=0)
    pooled_sem = np.std(pooled_traces, axis=0) / np.sqrt(len(pooled_traces))

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(peth_time, pooled_mean, color="#1a1a2e", linewidth=2)
    ax4.fill_between(peth_time, pooled_mean - pooled_sem,
                     pooled_mean + pooled_sem, color="#1a1a2e", alpha=0.15)
    ax4.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                label="Sound onset")
    ax4.axhline(0, color="grey", linestyle="-", alpha=0.3, linewidth=0.5)
    ax4.set_xlabel("Time relative to sound onset (s)", fontsize=12)
    ax4.set_ylabel("ΔF/F (mean ± SEM)", fontsize=12)
    ax4.set_title(f"Mouse K30_5 — Pooled peri-event ΔF/F "
                  f"(n={len(pooled_traces)} events)", fontsize=13)
    ax4.legend(fontsize=10, framealpha=0.9)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_xlim(pre_s, post_s)

    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "fibpho_peri_event_pooled.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig4, os.path.join(OUTPUT_DIR, "fibpho_peri_event_pooled.pdf"),
                 bbox_inches="tight")
    print("  Saved fibpho_peri_event_pooled.png/.pdf")


# ── 10. summary ───────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Alignment summary")
print(f"{'='*60}")
print(f"TDT recording duration:  {time_dff[-1]:.1f}s "
      f"({time_dff[-1]/60:.1f} min)")
print(f"TTL events (total):      {len(ttl_onsets)}")
print(f"TTL events (experiment): {len(tdt_experiment_onsets)}")
print(f"CSV active events:       {len(csv_sound_times)}")
print(f"Matched events:          {len(df_alignment)}")
print(f"Unmatched CSV events:    "
      f"{len(csv_sound_times) - len(set(df_alignment['csv_idx']))}")
print(f"Time offset:             {TIME_OFFSET:.4f}s")
print(f"Alignment quality:       "
      f"mean={np.mean(residuals)*1000:.1f}ms, "
      f"std={np.std(residuals)*1000:.1f}ms")
print(f"\nStimulus breakdown:")
print(df_alignment["stim_name"].value_counts().to_string())
print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")

plt.show()
