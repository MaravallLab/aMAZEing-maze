"""
02 — Within-Trial Preference: Sound vs Silent Visit Duration
=============================================================

For each mouse × experiment day, computes average visit duration
separately for sound-playing ROIs and silent-control ROIs **within
sound trials only** (trials 2, 4, 6, 8).  Trials 1, 3, 5, 7, 9
are excluded entirely.

This avoids the bias introduced by comparing 15-min sound trials
against 2-min silent trials in the original PI formula.

Output figure:
  One scatter panel per experiment day.
  x = mean silent-arm visit duration (ms)
  y = mean sound-arm visit duration (ms)
  Each dot = one mouse (averaged across trials 2, 4, 6, 8).
  Dashed diagonal = y = x (no preference).
  Points above the line → mouse prefers sound.

Outputs (saved to BATCH_ANALYSIS/):
  within_trial_preference.csv            per-mouse per-day metrics
  fig_within_trial_preference.png/pdf    scatter panels
"""

import os
import sys
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

from preference_analysis_config import (
    BASE_PATH, OUTPUT_DIR, EXPERIMENT_DAYS, DAY_ORDER, DAY_SHORT,
    discover_sessions,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── stimulus classifier (same as 01_preference_analysis.py) ──────────

def classify_stimulus(row, day_key):
    """Classify a trial-CSV row as 'silent' or a stimulus label."""
    mode = EXPERIMENT_DAYS[day_key]["mode"]

    if mode == "temporal_envelope_modulation":
        st = str(row.get("sound_type", "")).strip().lower()
        freq = str(row.get("frequency", "")).strip().lower()
        mod = str(row.get("temporal_modulation", "")).strip().lower()
        if freq in ("0", "silent_arm", "") or st in ("silent", "silent_trial"):
            return "silent"
        if freq == "vocalisation" or mod == "vocalisation":
            return "vocalisation"
        if st == "control" and mod == "no_stimulus":
            return "silent"
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
        return "silent" if freq == 0 or str(freq) == "0" else pat

    elif mode == "vocalisation":
        it = str(row.get("interval_type", "")).strip().lower()
        if it == "silent_trial":
            return "silent"
        freq = str(row.get("frequency", ""))
        if freq == "0" or freq == "":
            return "silent"
        if "/" in freq or "\\" in freq:
            return os.path.splitext(os.path.basename(freq))[0]
        return "vocalisation"

    return "unknown"


# ── helpers ──────────────────────────────────────────────────────────

def _is_file_locally_available(path):
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        if attrs & 0x00400000 or attrs & 0x00001000:
            return False
        return True
    except Exception:
        try:
            with open(path, "rb") as f:
                f.read(1)
            return True
        except Exception:
            return False


def safe_read_csv(path):
    if not _is_file_locally_available(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_savefig(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: could not write {os.path.basename(path)} (file locked)")


# ── main ─────────────────────────────────────────────────────────────

print("Discovering sessions...")
all_sessions = discover_sessions()
print(f"Found {len(all_sessions)} sessions across {len(EXPERIMENT_DAYS)} days")

records = []
skipped = loaded = 0

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

    for col in ["time_spent", "visitation_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only sound trials (2, 4, 6, 8)
    roi_names = [f"ROI{i}" for i in range(1, 9)]
    sound_trials = df[
        df["trial_ID"].isin([2, 4, 6, 8]) & df["ROIs"].isin(roi_names)
    ].copy()

    if len(sound_trials) == 0:
        skipped += 1
        continue

    # Classify each ROI
    sound_trials["_stim"] = [
        classify_stimulus(row, sess.day) for _, row in sound_trials.iterrows()
    ]

    # Split into sound-playing vs silent-control ROIs
    snd = sound_trials[sound_trials["_stim"] != "silent"]
    sil = sound_trials[sound_trials["_stim"] == "silent"]

    def _agg(subset):
        t = subset["time_spent"].sum()
        v = subset["visitation_count"].sum()
        t = 0 if (isinstance(t, float) and np.isnan(t)) else t
        v = 0 if (isinstance(v, float) and np.isnan(v)) else v
        return t, v

    snd_t, snd_v = _agg(snd)
    sil_t, sil_v = _agg(sil)

    avg_sound = snd_t / snd_v if snd_v > 0 else np.nan
    avg_silent = sil_t / sil_v if sil_v > 0 else np.nan

    records.append({
        "mouse_id": sess.mouse_id,
        "day": sess.day,
        "day_label": EXPERIMENT_DAYS[sess.day]["short"],
        "sound_time_ms": snd_t,
        "sound_visits": snd_v,
        "silent_ctrl_time_ms": sil_t,
        "silent_ctrl_visits": sil_v,
        "avg_sound_dur_ms": avg_sound,
        "avg_silent_dur_ms": avg_silent,
    })

print(f"\nLoaded {loaded} sessions, skipped {skipped}")

df_all = pd.DataFrame(records)

# Average across sessions per mouse × day
df_mouse = (
    df_all
    .groupby(["mouse_id", "day", "day_label"])
    .agg(
        avg_sound_dur_ms=("avg_sound_dur_ms", "mean"),
        avg_silent_dur_ms=("avg_silent_dur_ms", "mean"),
        n_sessions=("sound_time_ms", "size"),
    )
    .reset_index()
)

df_mouse.to_csv(os.path.join(OUTPUT_DIR, "within_trial_preference.csv"), index=False)
print(f"Saved within_trial_preference.csv ({len(df_mouse)} rows)")


# ── figure: one panel per day ────────────────────────────────────────

days_with_data = [d for d in DAY_ORDER if d in df_mouse["day"].values]
n_days = len(days_with_data)

if n_days == 0:
    print("No data to plot.")
    sys.exit(0)

ncols = min(n_days, 3)
nrows = int(np.ceil(n_days / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                         squeeze=False)

stats_lines = []

for idx, day in enumerate(days_with_data):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    sub = df_mouse[df_mouse["day"] == day].dropna(
        subset=["avg_sound_dur_ms", "avg_silent_dur_ms"]
    )

    if len(sub) == 0:
        ax.set_visible(False)
        continue

    x = sub["avg_silent_dur_ms"].values
    y = sub["avg_sound_dur_ms"].values

    ax.scatter(x, y, s=40, color="#333333", edgecolors="white", linewidths=0.5,
               zorder=3)

    # Diagonal y = x
    lo = min(x.min(), y.min()) * 0.85
    hi = max(x.max(), y.max()) * 1.10
    lo = max(lo, 0)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.6,
            zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Silent Visit Duration, ms", fontsize=11)
    ax.set_ylabel("Sound Visit Duration, ms", fontsize=11)

    # Count above / below diagonal
    n_above = int((y > x).sum())
    n_below = int((y < x).sum())
    n_on = int((y == x).sum())
    n_total = len(sub)

    title = f"{DAY_SHORT[day]}\n(n={n_total})"

    # Wilcoxon signed-rank test: sound vs silent
    if n_total >= 6:
        diff = y - x
        diff_nz = diff[diff != 0]
        if len(diff_nz) >= 6:
            stat, p = wilcoxon(diff_nz)
            title += f"\nWilcoxon p={p:.4f}"
            stats_lines.append(
                f"{DAY_SHORT[day]}: n={n_total}, above={n_above}, below={n_below}, "
                f"median_diff={np.median(diff):.0f}ms, Wilcoxon p={p:.4f}"
            )

    ax.set_title(title, fontsize=12, fontweight="bold")

    # Annotations
    ax.text(0.03, 0.97, f"{n_above} prefer sound",
            transform=ax.transAxes, fontsize=9, va="top", color="#2a9d8f")
    ax.text(0.97, 0.03, f"{n_below} prefer silence",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            color="#e76f51")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)

# Hide unused axes
for idx in range(n_days, nrows * ncols):
    r, c = divmod(idx, ncols)
    axes[r][c].set_visible(False)

fig.suptitle(
    "Within-trial preference: sound vs silent-arm visit duration\n"
    "(each point = one mouse, averaged across trials 2, 4, 6, 8)",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()

fig.savefig(os.path.join(OUTPUT_DIR, "fig_within_trial_preference.png"),
            dpi=200, bbox_inches="tight")
safe_savefig(fig, os.path.join(OUTPUT_DIR, "fig_within_trial_preference.pdf"),
             bbox_inches="tight")
print("Saved fig_within_trial_preference")

# Print stats
if stats_lines:
    print("\n=== Within-trial preference stats ===")
    for line in stats_lines:
        print(f"  {line}")

    # Append to existing stats report
    stats_path = os.path.join(OUTPUT_DIR, "stats_report.txt")
    with open(stats_path, "a") as f:
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("WITHIN-TRIAL PREFERENCE (sound vs silent-arm, trials 2/4/6/8)\n")
        f.write("=" * 60 + "\n")
        for line in stats_lines:
            f.write(line + "\n")
    print(f"  Appended to {stats_path}")

print("\nDone.")
