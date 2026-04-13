#!/usr/bin/env python3
"""
Batch Auditory Preference Analysis — Portable Runner
=====================================================

Self-contained script that computes preference metrics for all mice
across all experiment days in the 8-arm auditory maze.  Designed to
run on any machine where the data folder is available locally.

Quick start
-----------
    python run_batch_preference.py  "D:\data\8_arms_w_voc"

    # Or without arguments — it prompts for the path interactively:
    python run_batch_preference.py

Dependencies (pip install):
    numpy  pandas  matplotlib  seaborn  scipy  statsmodels

Outputs  → <DATA_ROOT>/BATCH_ANALYSIS/
    preference_data.csv           per-mouse per-session PI + voc_pi
    stimulus_breakdown.csv        per-stimulus-type visit duration + stim_pi
    fig1_pi_trajectories.png/pdf  individual mouse PI trajectories
    fig2_pi_by_day.png/pdf        mean PI per day ±95 % CI
    fig3_pi_violins.png/pdf       violin plots by day
    fig4_complexity_heatmap.png   visit duration by stimulus type
    fig5_vocalisation_contrast.png vocalisation PI analysis (3 panels)
    fig6_re_vs_pi.png/pdf         roaming entropy → PI
    fig7_icc_summary.png/pdf      ICC variance decomposition
    fig8_stimulus_pi.png/pdf      per-stimulus PI vs silent baseline
    stats_report.txt              all statistical test results
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import (entropy as shannon_entropy,
                          kruskal, mannwhitneyu, spearmanr, wilcoxon,
                          norm, rankdata)
from statsmodels.stats.multitest import multipletests

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── experiment structure ─────────────────────────────────────────────
EXPERIMENT_DAYS = {
    "w1_d1": {
        "folder": "w1_d1",
        "mode": "temporal_envelope_modulation",
        "label": "Day 1: Temporal Envelope",
        "short": "D1 (TEM)",
        "complexity_order": ["smooth", "rough", "rough_complex", "control", "vocalisation"],
    },
    "w1_d2": {
        "folder": "w1_d2",
        "mode": "complex_intervals",
        "label": "Day 2: Intervals (consonant/dissonant)",
        "short": "D2 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w1_d3": {
        "folder": "w1_d3",
        "mode": "complex_intervals",
        "label": "Day 3: Intervals (consonant/dissonant)",
        "short": "D3 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w1_d4": {
        "folder": "w1_d4",
        "mode": "complex_intervals",
        "label": "Day 4: Intervals (no silent control)",
        "short": "D4 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w2_sequences": {
        "folder": "w2_sequences",
        "mode": "sequences",
        "label": "Week 2: Sequences",
        "short": "W2 (Seq)",
        "complexity_order": ["AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA"],
    },
    "w2_vocalisations": {
        "folder": "w2_vocalisations",
        "mode": "vocalisation",
        "label": "Week 2: Vocalisations",
        "short": "W2 (Voc)",
        "complexity_order": [],
    },
}

PI_DAYS = ["w1_d1", "w1_d2", "w1_d3", "w2_sequences", "w2_vocalisations"]
DAY_ORDER = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
DAY_SHORT = {d: EXPERIMENT_DAYS[d]["short"] for d in DAY_ORDER}


# ── data classes ─────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    mouse_id: str
    day: str
    session_folder: str
    trials_csv: Optional[str] = None
    detailed_visits_csv: Optional[str] = None


# ── session discovery ────────────────────────────────────────────────

def _extract_mouse_id(folder_name: str) -> Optional[str]:
    m = re.search(r"mouse(\d{4,6})", folder_name, re.IGNORECASE)
    return f"mouse{m.group(1)}" if m else None


def discover_sessions(base_path: str) -> List[SessionInfo]:
    """Auto-discover all sessions across all experiment days."""
    sessions: list[SessionInfo] = []
    for day_key, day_info in EXPERIMENT_DAYS.items():
        day_dir = os.path.join(base_path, day_info["folder"])
        if not os.path.isdir(day_dir):
            continue
        for entry in os.listdir(day_dir):
            sess_path = os.path.join(day_dir, entry)
            if not os.path.isdir(sess_path):
                continue
            mouse_id = _extract_mouse_id(entry)
            if not mouse_id:
                continue
            trials = glob.glob(os.path.join(sess_path, "trials_*.csv"))
            dv = glob.glob(os.path.join(sess_path, "*detailed_visits*.csv"))
            fixed = os.path.join(sess_path, "fixed")
            if os.path.isdir(fixed):
                trials += glob.glob(os.path.join(fixed, "trials_*.csv"))
                dv += glob.glob(os.path.join(fixed, "*detailed_visits*.csv"))
            trials = [f for f in trials if not f.endswith(".bak")]
            dv = [f for f in dv if not f.endswith(".bak")]
            sessions.append(SessionInfo(
                mouse_id=mouse_id, day=day_key,
                session_folder=sess_path,
                trials_csv=trials[0] if trials else None,
                detailed_visits_csv=dv[0] if dv else None,
            ))
    return sessions


def get_mice_with_min_sessions(sessions, min_n=2):
    seen = set()
    counts: Counter = Counter()
    for s in sessions:
        key = (s.mouse_id, s.day)
        if key not in seen:
            seen.add(key)
            counts[s.mouse_id] += 1
    return {m for m, c in counts.items() if c >= min_n}


# ── helpers ──────────────────────────────────────────────────────────

def classify_stimulus(row, day_key):
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
        # silent / no-stimulus
        if it == "silent_trial" or freq in ("0", ""):
            return "silent"
        # vocalisation control
        if freq == "vocalisation":
            return "vocalisation"
        # main categories
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


def compute_roaming_entropy(time_per_roi, n_rois=8):
    total = np.nansum(time_per_roi)
    if total <= 0:
        return np.nan
    props = np.array(time_per_roi) / total
    props = props[props > 0]
    return shannon_entropy(props, base=2) / np.log2(n_rois)


def _is_file_locally_available(path):
    """Check if a file is locally available (not a cloud-only stub).

    On Windows, Box Drive / OneDrive online-only files have the
    FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS (0x00400000) or
    FILE_ATTRIBUTE_OFFLINE (0x00001000) attribute.  Reading such
    files blocks until the cloud provider downloads them, which
    can take minutes.  This check lets us skip them.
    """
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        if attrs & 0x00400000 or attrs & 0x00001000:
            return False
        return True
    except Exception:
        # Non-Windows: just try a quick byte read
        try:
            with open(path, "rb") as f:
                f.read(1)
            return True
        except Exception:
            return False


def safe_read_csv(path):
    """Read a CSV, skipping cloud-only files that would block."""
    if path is None or not _is_file_locally_available(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# Import the canonical visit-data loader from the shared config module.
# This ensures run_batch_preference.py uses the exact same DV-first loading,
# sanity caps, and per-visit clip as 02_within_trial_preference.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preference_analysis_config import load_session_visits, compute_first_minute_re


def safe_savefig(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: could not write {os.path.basename(path)} (file locked)")


def log_stat(msg, lines):
    print(msg)
    lines.append(msg)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch auditory preference analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_path", nargs="?", default=None,
        help="Path to the 8_arms_w_voc root folder (contains w1_d1/, w1_d2/, etc.)",
    )
    args = parser.parse_args()

    # resolve data path
    base_path = args.data_path
    if base_path is None:
        # try env var
        base_path = os.environ.get("MAZE_DATA_DIR")
    if base_path is None:
        # interactive prompt
        print("Enter the path to the 8_arms_w_voc data folder:")
        print("  e.g.  D:\\data\\8_arms_w_voc")
        print("  e.g.  /home/user/maze_data/8_arms_w_voc")
        base_path = input("> ").strip().strip('"').strip("'")

    if not os.path.isdir(base_path):
        print(f"ERROR: directory not found: {base_path}")
        sys.exit(1)

    # verify at least one day folder exists
    found_days = [d for d in DAY_ORDER
                  if os.path.isdir(os.path.join(base_path, EXPERIMENT_DAYS[d]["folder"]))]
    if not found_days:
        print(f"ERROR: no experiment day folders (w1_d1, w1_d2, ...) found in:\n  {base_path}")
        sys.exit(1)

    output_dir = os.path.join(base_path, "BATCH_ANALYSIS")
    os.makedirs(output_dir, exist_ok=True)
    stats_lines: list[str] = []

    # ── 1. discover & load ───────────────────────────────────────────
    print(f"\nData path:   {base_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Day folders: {', '.join(found_days)}\n")

    print("Discovering sessions...")
    all_sessions = discover_sessions(base_path)
    print(f"Found {len(all_sessions)} sessions across {len(found_days)} days")

    records = []
    stim_records = []
    skipped = loaded = 0
    source_counts = {"detailed_visits": 0, "trials_capped": 0, "trials_raw": 0}

    for sess in all_sessions:
        df, source = load_session_visits(sess)
        if df is None:
            skipped += 1
            continue
        loaded += 1
        source_counts[source] = source_counts.get(source, 0) + 1

        # habituation (trial 1)
        hab = df[df["trial_ID"] == 1]
        hab_time_per_roi = []
        if len(hab) > 0 and "time_spent" in hab.columns:
            for roi in [f"ROI{i}" for i in range(1, 9)]:
                t = hab.loc[hab["ROIs"] == roi, "time_spent"].sum()
                hab_time_per_roi.append(0 if np.isnan(t) else t)
        hab_total = sum(hab_time_per_roi)
        roaming_entropy = compute_roaming_entropy(hab_time_per_roi)
        hab_visits = hab["visitation_count"].sum() if "visitation_count" in hab.columns else 0

        # ---- within-trial comparison (sound trials 2,4,6,8 only) ----
        # Both the "sound" and "silent" baselines come from the SAME
        # trials, split by ROI classification: sound-playing vs silent-arm.
        # This avoids the bias from comparing 15-min sound trials against
        # 2-min silent trials.
        roi_names = [f"ROI{i}" for i in range(1, 9)]
        sound_rois = df[df["trial_ID"].isin([2, 4, 6, 8]) & df["ROIs"].isin(roi_names)]

        def _safe_sum(series):
            v = series.sum()
            return 0 if np.isnan(v) else v

        # Classify each ROI within sound trials
        sound_stim_labels = []
        for _, row in sound_rois.iterrows():
            sound_stim_labels.append(classify_stimulus(row, sess.day))
        sound_rois = sound_rois.copy()
        sound_rois["_stim_label"] = sound_stim_labels

        # Split: sound-playing ROIs vs silent-arm ROIs (within same trials)
        actual_sound = sound_rois[sound_rois["_stim_label"] != "silent"]
        silent_ctrl = sound_rois[sound_rois["_stim_label"] == "silent"]

        # Silent baseline: silent-arm ROIs within sound trials
        silent_time = _safe_sum(silent_ctrl["time_spent"]) if "time_spent" in silent_ctrl.columns else 0
        silent_visits = _safe_sum(silent_ctrl["visitation_count"]) if "visitation_count" in silent_ctrl.columns else 0
        avg_silent_dur = silent_time / silent_visits if silent_visits > 0 else 0

        # Sound aggregate: sound-playing ROIs within sound trials
        sound_time = _safe_sum(actual_sound["time_spent"]) if "time_spent" in actual_sound.columns else 0
        sound_visits_n = _safe_sum(actual_sound["visitation_count"]) if "visitation_count" in actual_sound.columns else 0
        avg_sound_dur = sound_time / sound_visits_n if sound_visits_n > 0 else 0

        denom = avg_sound_dur + avg_silent_dur
        pi = (avg_sound_dur - avg_silent_dur) / denom if denom > 0 else np.nan

        # ---- per-stimulus-type PI ----
        stim_pi_dict = {}
        for stim_type in actual_sound["_stim_label"].unique():
            stim_subset = actual_sound[actual_sound["_stim_label"] == stim_type]
            st_time = _safe_sum(stim_subset["time_spent"])
            st_visits = _safe_sum(stim_subset["visitation_count"])
            avg_st = st_time / st_visits if st_visits > 0 else 0
            d = avg_st + avg_silent_dur
            stim_pi_dict[stim_type] = (avg_st - avg_silent_dur) / d if d > 0 else np.nan

        # Vocalisation-specific PI
        voc_pi = stim_pi_dict.get("vocalisation", np.nan)
        # For w2_vocalisations, aggregate all individual vocalisation files
        if sess.day == "w2_vocalisations" and np.isnan(voc_pi):
            # all non-silent stim types are vocalisations
            voc_time = _safe_sum(actual_sound["time_spent"])
            voc_visits = _safe_sum(actual_sound["visitation_count"])
            avg_voc = voc_time / voc_visits if voc_visits > 0 else 0
            d = avg_voc + avg_silent_dur
            voc_pi = (avg_voc - avg_silent_dur) / d if d > 0 else np.nan

        # Other-sounds PI (non-vocalisation, non-silent stimuli)
        other_subset = actual_sound[actual_sound["_stim_label"] != "vocalisation"]
        other_time = _safe_sum(other_subset["time_spent"]) if "time_spent" in other_subset.columns else 0
        other_visits_n = _safe_sum(other_subset["visitation_count"]) if "visitation_count" in other_subset.columns else 0
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
            "sound_visits": sound_visits_n,
            "silent_ctrl_time_ms": silent_time,     # silent-arm ROIs in sound trials
            "silent_ctrl_visits": silent_visits,     # (NOT silent trials 3,5,7,9)
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

        # per-stimulus breakdown
        for _, row in sound_rois.iterrows():
            stim_type = row["_stim_label"]
            if stim_type == "silent":
                continue
            t = row.get("time_spent", 0)
            v = row.get("visitation_count", 0)
            t = 0 if pd.isna(t) else t
            v = 0 if pd.isna(v) else v
            avg_dur = t / v if v > 0 else 0
            # per-stimulus PI vs silent baseline
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
    print(f"Visit-data sources: {source_counts}")
    log_stat(f"\nVisit-data sources (after corruption fix): {source_counts}",
             stats_lines)

    df_pref = pd.DataFrame(records)
    df_stim = pd.DataFrame(stim_records)

    df_pref.to_csv(os.path.join(output_dir, "preference_data.csv"), index=False)
    df_stim.to_csv(os.path.join(output_dir, "stimulus_breakdown.csv"), index=False)
    print(f"Saved preference_data.csv ({len(df_pref)} rows)")
    print(f"Saved stimulus_breakdown.csv ({len(df_stim)} rows)")

    # Voc PI vs other sounds PI per session
    df_voc_other = df_pref[["mouse_id", "day", "day_label", "voc_pi", "other_sounds_pi"]].copy()
    df_voc_other.to_csv(os.path.join(output_dir, "voc_vs_other_sounds_pi.csv"), index=False)
    print(f"Saved voc_vs_other_sounds_pi.csv ({len(df_voc_other)} rows)")

    if len(df_pref) == 0:
        print("\nERROR: No data loaded. Check that the data path is correct")
        print(f"  and contains folders: {', '.join(d['folder'] for d in EXPERIMENT_DAYS.values())}")
        sys.exit(1)

    df_pi = df_pref[df_pref["day"].isin(PI_DAYS)].copy()
    print(f"\nPI analysis: {len(df_pi)} sessions across {len(PI_DAYS)} days")
    print(f"Unique mice: {df_pi['mouse_id'].nunique()}")

    # ── 2. Fig 1: PI trajectories ────────────────────────────────────
    print("\nPlotting Figure 1: PI trajectories...")
    multi_mice = get_mice_with_min_sessions(all_sessions, 3)
    df_traj = df_pi[df_pi["mouse_id"].isin(multi_mice)].copy()

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    day_x = {d: i for i, d in enumerate(PI_DAYS)}
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(multi_mice), 1)))

    for i, mouse in enumerate(sorted(multi_mice)):
        sub = df_traj[df_traj["mouse_id"] == mouse].copy()
        sub["x"] = sub["day"].map(day_x)
        sub = sub.sort_values("x")
        ax1.plot(sub["x"], sub["preference_index"], "o-",
                 color=cmap[i % len(cmap)], alpha=0.5, lw=1, ms=4,
                 label=mouse if i < 20 else None)
    for d in PI_DAYS:
        vals = df_pi[df_pi["day"] == d]["preference_index"].dropna()
        ax1.plot(day_x[d], vals.mean(), "D", color="black", ms=10, zorder=5)

    ax1.axhline(0, color="grey", ls="--", alpha=0.5)
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
    fig1.savefig(os.path.join(output_dir, "fig1_pi_trajectories.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig1, os.path.join(output_dir, "fig1_pi_trajectories.pdf"),
                 bbox_inches="tight")
    print("  Saved fig1")

    # ── 3. Fig 2: Mean PI per day ────────────────────────────────────
    print("Plotting Figure 2: Mean PI per day...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    means, cis, ns = [], [], []
    for d in PI_DAYS:
        vals = df_pi[df_pi["day"] == d]["preference_index"].dropna()
        m = vals.mean()
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
        means.append(m); cis.append(1.96 * se); ns.append(len(vals))
    x = np.arange(len(PI_DAYS))
    colours = ["#457B9D" if d != "w2_vocalisations" else "#E63946" for d in PI_DAYS]
    ax2.bar(x, means, yerr=cis, color=colours, alpha=0.8, capsize=5,
            edgecolor="white", lw=1.5)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax2.text(i, m + cis[i] + 0.02, f"n={n}", ha="center", fontsize=9)
    ax2.axhline(0, color="grey", ls="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], fontsize=10)
    ax2.set_ylabel("Mean Preference Index (±95% CI)", fontsize=12)
    ax2.set_title("Sound preference by experiment day (red = vocalisations)",
                  fontsize=13)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(-0.8, 0.8)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "fig2_pi_by_day.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig2, os.path.join(output_dir, "fig2_pi_by_day.pdf"),
                 bbox_inches="tight")
    print("  Saved fig2")

    # ── 4. Fig 3: Violin plots ───────────────────────────────────────
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
    ax3.axhline(0, color="grey", ls="--", alpha=0.5)
    ax3.set_xlabel("Experiment Day", fontsize=12)
    ax3.set_ylabel("Preference Index", fontsize=12)
    ax3.set_title("Distribution of preference index by experiment day",
                  fontsize=13)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_pi_violins.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig3, os.path.join(output_dir, "fig3_pi_violins.pdf"),
                 bbox_inches="tight")
    print("  Saved fig3")

    # ── 5. Fig 4: Complexity box plots ───────────────────────────────
    print("Plotting Figure 4: Complexity box plots...")
    if len(df_stim) > 0:
        fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
        axes4 = axes4.flatten()
        for idx, day in enumerate(DAY_ORDER):
            ax = axes4[idx]
            day_stim = df_stim[df_stim["day"] == day]
            if len(day_stim) == 0:
                ax.set_visible(False)
                continue
            pivot = day_stim.groupby(["mouse_id", "stim_type"])[
                "avg_visit_dur_ms"].mean().reset_index()
            if len(pivot) == 0:
                ax.set_visible(False)
                continue
            stim_order = EXPERIMENT_DAYS[day].get("complexity_order", [])
            available = [s for s in stim_order if s in pivot["stim_type"].values]
            extras = [s for s in pivot["stim_type"].unique()
                      if s not in available and s != "silent"]
            plot_order = available + extras
            if not plot_order:
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
        plt.suptitle("Visit duration by stimulus type within each experiment day",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, "fig4_complexity_heatmap.png"),
                     dpi=200, bbox_inches="tight")
        safe_savefig(fig4, os.path.join(output_dir, "fig4_complexity_heatmap.pdf"),
                     bbox_inches="tight")
        print("  Saved fig4")

    # ── 6. Fig 5: Vocalisation contrast ──────────────────────────────
    # Uses voc_pi (stimulus-specific vocalisation PI) for a fair comparison
    print("Plotting Figure 5: Vocalisation contrast...")

    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: voc_pi across ALL days (wherever vocalisations appeared)
    df_voc_pi = df_pref.dropna(subset=["voc_pi"]).copy()
    df_voc_pi["day_short"] = df_voc_pi["day"].map(DAY_SHORT)
    ax = axes5[0]
    if len(df_voc_pi) > 0:
        voc_day_order = [DAY_SHORT[d] for d in DAY_ORDER
                         if d in df_voc_pi["day"].values]
        sns.boxplot(data=df_voc_pi, x="day_short", y="voc_pi",
                    order=voc_day_order, ax=ax, palette="Reds_d",
                    hue="day_short", legend=False)
        sns.stripplot(data=df_voc_pi, x="day_short", y="voc_pi",
                      order=voc_day_order, ax=ax, color="black",
                      alpha=0.4, size=4)
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.set_ylabel("Vocalisation PI", fontsize=12)
        ax.set_xlabel("")
        ax.set_title("Vocalisation-specific PI by day\n(voc visit dur vs silent baseline)",
                     fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # One-sample tests
        for d in DAY_ORDER:
            vals = df_voc_pi[df_voc_pi["day"] == d]["voc_pi"].dropna().values
            if len(vals) >= 6:
                stat, p = wilcoxon(vals)
                log_stat(f"  Voc PI {DAY_SHORT[d]}: median={np.median(vals):.3f}, "
                         f"mean={np.mean(vals):.3f}, Wilcoxon p={p:.4f} (n={len(vals)})",
                         stats_lines)

    # Panel B: paired voc PI vs overall PI (same mice)
    ax = axes5[1]
    paired_data = []
    for _, row in df_pref.iterrows():
        if not np.isnan(row.get("voc_pi", np.nan)) and not np.isnan(row["preference_index"]):
            paired_data.append({
                "mouse_id": row["mouse_id"],
                "day": row["day"],
                "pi_overall": row["preference_index"],
                "pi_voc": row["voc_pi"],
            })
    if paired_data:
        df_paired = pd.DataFrame(paired_data)
        for _, row in df_paired.iterrows():
            ax.plot([0, 1], [row["pi_overall"], row["pi_voc"]],
                    "o-", color="#888888", alpha=0.3, ms=4)
        ax.plot(0, df_paired["pi_overall"].mean(), "D", color="#457B9D", ms=12, zorder=5)
        ax.plot(1, df_paired["pi_voc"].mean(), "D", color="#E63946", ms=12, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Overall PI\n(all stimuli)", "Vocalisation PI\n(voc only)"],
                           fontsize=11)
        ax.set_ylabel("Preference Index", fontsize=12)
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(df_paired) >= 6:
            stat, p_wil = wilcoxon(df_paired["pi_voc"], df_paired["pi_overall"])
            log_stat(f"\nVoc PI vs Overall PI (Wilcoxon): W={stat:.1f}, p={p_wil:.4f}",
                     stats_lines)
            ax.set_title(f"Paired (n={len(df_paired)}), Wilcoxon p={p_wil:.4f}",
                         fontsize=11)
        else:
            ax.set_title(f"Paired (n={len(df_paired)})", fontsize=11)

    # Panel C: per-stimulus PI box plots for w2_vocalisations
    ax = axes5[2]
    voc_day_stim = df_stim[df_stim["day"] == "w2_vocalisations"]
    if len(voc_day_stim) > 0 and "stim_pi" in voc_day_stim.columns:
        voc_means = voc_day_stim.groupby(["mouse_id", "stim_type"])[
            "stim_pi"].mean().reset_index()
        # keep top stimulus types by median PI
        stim_medians = voc_means.groupby("stim_type")["stim_pi"].median()
        top_stims = stim_medians.nlargest(8).index.tolist()
        voc_plot = voc_means[voc_means["stim_type"].isin(top_stims)]
        if len(voc_plot) > 0:
            # shorten names for display
            voc_plot = voc_plot.copy()
            voc_plot["stim_short"] = voc_plot["stim_type"].apply(
                lambda s: s[:25] + "..." if len(s) > 28 else s)
            sns.boxplot(data=voc_plot, x="stim_short", y="stim_pi",
                        ax=ax, palette="Reds_d", hue="stim_short", legend=False)
            ax.axhline(0, color="grey", ls="--", alpha=0.5)
            ax.set_ylabel("PI vs silent", fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            ax.set_title("W2 Voc: PI by recording", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    else:
        ax.set_visible(False)

    plt.suptitle("Do vocalisations drive stronger preference?", fontsize=14, y=1.02)
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, "fig5_vocalisation_contrast.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig5, os.path.join(output_dir, "fig5_vocalisation_contrast.pdf"),
                 bbox_inches="tight")
    print("  Saved fig5")

    # ── 7. Fig 6: Roaming entropy vs PI ──────────────────────────────
    print("Plotting Figure 6: Roaming entropy vs PI...")
    df_re = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).copy()
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))

    if len(df_re) > 10:
        mouse_means = df_re.groupby("mouse_id").agg(
            mean_re=("roaming_entropy", "mean"),
            mean_pi=("preference_index", "mean"))
        df_re = df_re.merge(mouse_means, on="mouse_id")
        df_re["dev_re"] = df_re["roaming_entropy"] - df_re["mean_re"]
        df_re["dev_pi"] = df_re["preference_index"] - df_re["mean_pi"]

        ax = axes6[0]
        ax.scatter(df_re["dev_re"], df_re["dev_pi"], alpha=0.4, s=20, color="#457B9D")
        r, p = spearmanr(df_re["dev_re"], df_re["dev_pi"])
        ax.set_xlabel("Deviation from mouse mean RE"); ax.set_ylabel("Deviation from mouse mean PI")
        ax.set_title(f"Within-mouse\nSpearman r={r:.3f}, p={p:.4f}")
        ax.axhline(0, color="grey", ls="--", alpha=0.3)
        ax.axvline(0, color="grey", ls="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        log_stat(f"\nWithin-mouse RE vs PI: r={r:.3f}, p={p:.4f}", stats_lines)

    mouse_avg = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).groupby(
        "mouse_id").agg(mean_re=("roaming_entropy", "mean"),
                        mean_pi=("preference_index", "mean")).reset_index()
    if len(mouse_avg) > 5:
        ax = axes6[1]
        ax.scatter(mouse_avg["mean_re"], mouse_avg["mean_pi"], alpha=0.7, s=40, color="#E63946")
        r, p = spearmanr(mouse_avg["mean_re"], mouse_avg["mean_pi"])
        z = np.polyfit(mouse_avg["mean_re"], mouse_avg["mean_pi"], 1)
        x_line = np.linspace(mouse_avg["mean_re"].min(), mouse_avg["mean_re"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#E63946", alpha=0.5)
        ax.set_xlabel("Mean Roaming Entropy"); ax.set_ylabel("Mean Preference Index")
        ax.set_title(f"Between-mouse\nSpearman r={r:.3f}, p={p:.4f}")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        log_stat(f"Between-mouse RE vs PI: n={len(mouse_avg)}, r={r:.3f}, p={p:.4f}",
                 stats_lines)

    plt.suptitle("Roaming entropy (habituation) predicts sound preference",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, "fig6_re_vs_pi.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig6, os.path.join(output_dir, "fig6_re_vs_pi.pdf"),
                 bbox_inches="tight")
    print("  Saved fig6")

    # ── 7b. Fig 6b: First-minute RE vs PI ────────────────────────────
    print("Plotting Figure 6b: First-minute RE vs PI...")
    df_re1 = df_pi.dropna(subset=["re_first_min", "preference_index"]).copy()
    n_re1 = len(df_re1)
    print(f"  Sessions with first-minute RE: {n_re1} / {len(df_pi)}")

    if n_re1 > 10:
        fig6b, axes6b = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes6b[0]
        mm1 = df_re1.groupby("mouse_id").agg(
            mean_re1=("re_first_min", "mean"),
            mean_pi=("preference_index", "mean"))
        df_re1 = df_re1.merge(mm1, on="mouse_id")
        df_re1["dev_re1"] = df_re1["re_first_min"] - df_re1["mean_re1"]
        df_re1["dev_pi"] = df_re1["preference_index"] - df_re1["mean_pi"]

        ax.scatter(df_re1["dev_re1"], df_re1["dev_pi"], alpha=0.4, s=20, color="#2a9d8f")
        r, p = spearmanr(df_re1["dev_re1"], df_re1["dev_pi"])
        ax.set_xlabel("Dev from mouse mean RE (first min)")
        ax.set_ylabel("Dev from mouse mean PI")
        ax.set_title(f"Within-mouse\nSpearman r={r:.3f}, p={p:.4f}")
        ax.axhline(0, color="grey", ls="--", alpha=0.3)
        ax.axvline(0, color="grey", ls="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        log_stat(f"\nWithin-mouse first-min RE vs PI: r={r:.3f}, p={p:.4f}, n={n_re1}",
                 stats_lines)

        ax = axes6b[1]
        ma1 = df_pi.dropna(subset=["re_first_min", "preference_index"]).groupby(
            "mouse_id").agg(mean_re1=("re_first_min", "mean"),
                            mean_pi=("preference_index", "mean")).reset_index()
        if len(ma1) > 5:
            ax.scatter(ma1["mean_re1"], ma1["mean_pi"], alpha=0.7, s=40, color="#e76f51")
            r, p = spearmanr(ma1["mean_re1"], ma1["mean_pi"])
            z = np.polyfit(ma1["mean_re1"], ma1["mean_pi"], 1)
            x_line = np.linspace(ma1["mean_re1"].min(), ma1["mean_re1"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="#e76f51", alpha=0.5)
            ax.set_xlabel("Mean First-Minute RE"); ax.set_ylabel("Mean PI")
            ax.set_title(f"Between-mouse\nSpearman r={r:.3f}, p={p:.4f}")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            log_stat(f"Between-mouse first-min RE vs PI: n={len(ma1)}, r={r:.3f}, p={p:.4f}",
                     stats_lines)

        both_re = df_pi.dropna(subset=["roaming_entropy", "re_first_min"])
        if len(both_re) > 5:
            r_c, p_c = spearmanr(both_re["roaming_entropy"], both_re["re_first_min"])
            log_stat(f"Full-hab RE vs first-min RE: r={r_c:.3f}, p={p_c:.4f}, n={len(both_re)}",
                     stats_lines)

        plt.suptitle("First-minute RE (first 60s of habituation) predicts PI",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        fig6b.savefig(os.path.join(output_dir, "fig6b_re_firstmin_vs_pi.png"),
                      dpi=200, bbox_inches="tight")
        safe_savefig(fig6b, os.path.join(output_dir, "fig6b_re_firstmin_vs_pi.pdf"),
                     bbox_inches="tight")
        print("  Saved fig6b")
    else:
        print(f"  Skipping fig6b: only {n_re1} sessions with first-minute RE")

    # ── 7c. Fig: Voc PI vs Other Sounds PI ───────────────────────────
    print("Plotting Figure: Voc PI vs Other Sounds PI...")
    df_voc_vs = df_pref.dropna(subset=["voc_pi", "other_sounds_pi"]).copy()
    df_voc_vs = df_voc_vs[df_voc_vs["day"] != "w2_vocalisations"]

    if len(df_voc_vs) > 0:
        days_vb = [d for d in DAY_ORDER
                   if d in df_voc_vs["day"].values and d != "w2_vocalisations"]
        n_panels = len(days_vb) + 1  # +1 for pooled
        nc = min(n_panels, 3)
        nr = int(np.ceil(n_panels / nc))

        fig_vo, axes_vo = plt.subplots(nr, nc, figsize=(6 * nc, 5.5 * nr), squeeze=False)

        all_vv, all_ov = [], []
        for idx, day in enumerate(days_vb):
            r, c = divmod(idx, nc)
            ax = axes_vo[r][c]
            sub = df_voc_vs[df_voc_vs["day"] == day]
            ax.scatter(sub["other_sounds_pi"], sub["voc_pi"], s=40, color="#E63946",
                       edgecolors="white", linewidths=0.5, zorder=3)
            lo = min(sub["other_sounds_pi"].min(), sub["voc_pi"].min()) * 1.1
            hi = max(sub["other_sounds_pi"].max(), sub["voc_pi"].max()) * 1.1
            lo, hi = min(lo, -0.05), max(hi, 0.05)
            ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.6, zorder=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Other Sounds PI", fontsize=10)
            ax.set_ylabel("Vocalisation PI", fontsize=10)
            n = len(sub)
            n_above = int((sub["voc_pi"] > sub["other_sounds_pi"]).sum())
            title = f"{DAY_SHORT[day]} (n={n})"
            vv = sub["voc_pi"].values; ov = sub["other_sounds_pi"].values
            all_vv.extend(vv); all_ov.extend(ov)
            if n >= 6:
                diff = vv - ov; diff_nz = diff[diff != 0]
                if len(diff_nz) >= 6:
                    stat, p = wilcoxon(diff_nz)
                    title += f"\nWilcoxon p={p:.4f}"
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.03, 0.97, f"{n_above} prefer voc",
                    transform=ax.transAxes, fontsize=9, va="top", color="#E63946")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15)

        # Pooled panel
        pi_idx = len(days_vb)
        r, c = divmod(pi_idx, nc)
        ax = axes_vo[r][c]
        all_vv, all_ov = np.array(all_vv), np.array(all_ov)
        ax.scatter(all_ov, all_vv, s=40, color="#333333",
                   edgecolors="white", linewidths=0.5, zorder=3)
        lo = min(all_ov.min(), all_vv.min()) * 1.1
        hi = max(all_ov.max(), all_vv.max()) * 1.1
        lo, hi = min(lo, -0.05), max(hi, 0.05)
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, alpha=0.6, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Other Sounds PI", fontsize=10)
        ax.set_ylabel("Vocalisation PI", fontsize=10)
        n_pool = len(all_vv)
        n_above_pool = int((all_vv > all_ov).sum())
        pool_title = f"All days pooled (n={n_pool})"
        if n_pool >= 6:
            diff_pool = all_vv - all_ov; diff_nz = diff_pool[diff_pool != 0]
            if len(diff_nz) >= 6:
                stat, p = wilcoxon(diff_nz)
                pool_title += f"\nWilcoxon p={p:.4f}"
        ax.set_title(pool_title, fontsize=11, fontweight="bold")
        ax.text(0.03, 0.97, f"{n_above_pool} prefer voc",
                transform=ax.transAxes, fontsize=9, va="top", color="#E63946")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15)
        for idx in range(pi_idx + 1, nr * nc):
            r, c = divmod(idx, nc)
            axes_vo[r][c].set_visible(False)

        fig_vo.suptitle(
            "Vocalisation PI vs Other Sounds PI per session\n"
            "(above diagonal = prefers vocalisation)",
            fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig_vo.savefig(os.path.join(output_dir, "fig_voc_vs_other_sounds_pi.png"),
                       dpi=200, bbox_inches="tight")
        safe_savefig(fig_vo, os.path.join(output_dir, "fig_voc_vs_other_sounds_pi.pdf"),
                     bbox_inches="tight")
        print("  Saved fig_voc_vs_other_sounds_pi")
    else:
        print("  No sessions with both voc PI and other sounds PI — skipping")

    # ── 8. Mixed models + Fig 7 ──────────────────────────────────────
    print("\nFitting mixed-effects models...")
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("MIXED-EFFECTS MODELS", stats_lines)
    log_stat("=" * 60, stats_lines)

    df_model = df_pi.dropna(subset=["preference_index"]).copy()
    icc_mouse = icc_session = np.nan

    if HAS_STATSMODELS and len(df_model) > 10 and df_model["mouse_id"].nunique() > 2:
        try:
            m1 = smf.mixedlm("preference_index ~ 1", df_model,
                              groups=df_model["mouse_id"]).fit(reml=True)
            var_mouse = m1.cov_re.iloc[0, 0]
            var_resid = m1.scale
            icc_mouse = var_mouse / (var_mouse + var_resid)
            icc_session = var_resid / (var_mouse + var_resid)
            log_stat(f"\nModel 1: PI ~ 1 + (1|mouse)", stats_lines)
            log_stat(f"  Grand mean PI: {m1.fe_params['Intercept']:.3f} "
                     f"(p={m1.pvalues['Intercept']:.4f})", stats_lines)
            log_stat(f"  ICC(mouse): {icc_mouse:.3f}  ({icc_mouse*100:.1f}%)",
                     stats_lines)
        except Exception as e:
            log_stat(f"\nModel 1 failed: {e}", stats_lines)

        try:
            df_model["day_cat"] = pd.Categorical(df_model["day"], categories=PI_DAYS)
            m2 = smf.mixedlm("preference_index ~ C(day_cat)", df_model,
                              groups=df_model["mouse_id"]).fit(reml=True)
            log_stat(f"\nModel 2: PI ~ day + (1|mouse)", stats_lines)
            log_stat(f"\n{m2.summary().tables[1].to_string()}", stats_lines)
        except Exception as e:
            log_stat(f"\nModel 2 failed: {e}", stats_lines)

        df_re_m = df_model.dropna(subset=["roaming_entropy"]).copy()
        if len(df_re_m) > 10:
            try:
                mouse_re = df_re_m.groupby("mouse_id")["roaming_entropy"].mean()
                grand_re = mouse_re.mean()
                df_re_m["re_mouse_mean"] = df_re_m["mouse_id"].map(mouse_re)
                df_re_m["re_within"] = df_re_m["roaming_entropy"] - df_re_m["re_mouse_mean"]
                df_re_m["re_between"] = df_re_m["re_mouse_mean"] - grand_re
                m3 = smf.mixedlm("preference_index ~ re_within + re_between",
                                  df_re_m, groups=df_re_m["mouse_id"]).fit(reml=True)
                log_stat(f"\nModel 3: PI ~ RE_within + RE_between + (1|mouse)",
                         stats_lines)
                log_stat(f"\n{m3.summary().tables[1].to_string()}", stats_lines)
            except Exception as e:
                log_stat(f"\nModel 3 failed: {e}", stats_lines)

    # Fig 7: variance decomposition
    print("Plotting Figure 7: Variance decomposition...")
    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))
    if not np.isnan(icc_mouse):
        ax = axes7[0]
        sizes = [icc_mouse * 100, icc_session * 100]
        labels = [f"Between mice\n({sizes[0]:.1f}%)",
                  f"Between sessions\n({sizes[1]:.1f}%)"]
        ax.pie(sizes, labels=labels, colors=["#E63946", "#457B9D"],
               autopct="", startangle=90, textprops={"fontsize": 11})
        ax.set_title("Variance in preference index", fontsize=12)

    ax = axes7[1]
    day_groups = [df_pi[df_pi["day"] == d]["preference_index"].dropna().values
                  for d in PI_DAYS]
    day_groups_ne = [g for g in day_groups if len(g) > 0]
    if len(day_groups_ne) > 1:
        stat, p_kw = kruskal(*day_groups_ne)
        log_stat(f"\nKruskal-Wallis across days: H={stat:.2f}, p={p_kw:.4f}",
                 stats_lines)
        medians = [np.median(g) for g in day_groups]
        x = np.arange(len(PI_DAYS))
        ax.bar(x, medians, color="#457B9D", alpha=0.7, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], fontsize=9)
        ax.set_ylabel("Median PI"); ax.set_title(f"Kruskal-Wallis: H={stat:.1f}, p={p_kw:.4f}")
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.suptitle("Individual differences in sound preference", fontsize=14, y=1.02)
    plt.tight_layout()
    fig7.savefig(os.path.join(output_dir, "fig7_icc_summary.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig7, os.path.join(output_dir, "fig7_icc_summary.pdf"),
                 bbox_inches="tight")
    print("  Saved fig7")

    # ── 9. Fig 8: Per-stimulus PI by day ────────────────────────────
    print("Plotting Figure 8: Per-stimulus PI by day...")
    if len(df_stim) > 0 and "stim_pi" in df_stim.columns:
        fig8, axes8 = plt.subplots(2, 3, figsize=(18, 10))
        axes8 = axes8.flatten()
        for idx, day in enumerate(DAY_ORDER):
            ax = axes8[idx]
            day_stim = df_stim[df_stim["day"] == day]
            if len(day_stim) == 0:
                ax.set_visible(False)
                continue
            pivot = day_stim.groupby(["mouse_id", "stim_type"])[
                "stim_pi"].mean().reset_index()
            if len(pivot) == 0:
                ax.set_visible(False)
                continue
            stim_order = EXPERIMENT_DAYS[day].get("complexity_order", [])
            available = [s for s in stim_order if s in pivot["stim_type"].values]
            extras = [s for s in pivot["stim_type"].unique()
                      if s not in available and s != "silent"]
            plot_order = available + sorted(extras)
            if not plot_order:
                ax.set_visible(False)
                continue
            plot_data = pivot[pivot["stim_type"].isin(plot_order)]
            sns.boxplot(data=plot_data, x="stim_type", y="stim_pi",
                        order=plot_order, ax=ax, palette="viridis",
                        hue="stim_type", legend=False)
            sns.stripplot(data=plot_data, x="stim_type", y="stim_pi",
                          order=plot_order, ax=ax, color="black",
                          alpha=0.4, size=3)
            ax.axhline(0, color="grey", ls="--", alpha=0.5)
            ax.set_title(DAY_SHORT[day], fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("PI vs silent" if idx % 3 == 0 else "")
            ax.tick_params(axis="x", rotation=45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.suptitle("Per-stimulus preference index (each stimulus vs silent baseline)",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        fig8.savefig(os.path.join(output_dir, "fig8_stimulus_pi.png"),
                     dpi=200, bbox_inches="tight")
        safe_savefig(fig8, os.path.join(output_dir, "fig8_stimulus_pi.pdf"),
                     bbox_inches="tight")
        print("  Saved fig8")

    # ── 9b. Interactive Plotly figures ─────────────────────────────────
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        print("  (Plotly not installed -- skipping interactive figures)")

    if HAS_PLOTLY:
        print("\nGenerating interactive Plotly figures...")

        # ---- Plotly Fig 1: PI trajectories ----
        try:
            fig1_ly = go.Figure()
            cmap_hex = [matplotlib.colors.to_hex(c)
                        for c in plt.cm.tab20(np.linspace(0, 1, max(len(multi_mice), 1)))]
            for i, mouse in enumerate(sorted(multi_mice)):
                sub = df_traj[df_traj["mouse_id"] == mouse].copy()
                sub["x"] = sub["day"].map(day_x)
                sub = sub.sort_values("x")
                fig1_ly.add_trace(go.Scatter(
                    x=[DAY_SHORT[d] for d in sub["day"]],
                    y=sub["preference_index"],
                    mode="lines+markers", name=mouse,
                    line=dict(color=cmap_hex[i % len(cmap_hex)], width=1.5),
                    marker=dict(size=6),
                    opacity=0.6,
                    hovertemplate=f"<b>{mouse}</b><br>%{{x}}<br>PI: %{{y:.3f}}<extra></extra>",
                ))
            # Grand means
            grand_x = [DAY_SHORT[d] for d in PI_DAYS]
            grand_y = [df_pi[df_pi["day"] == d]["preference_index"].dropna().mean() for d in PI_DAYS]
            fig1_ly.add_trace(go.Scatter(
                x=grand_x, y=grand_y, mode="markers", name="Grand mean",
                marker=dict(size=14, color="black", symbol="diamond"),
                hovertemplate="<b>Grand mean</b><br>%{x}<br>PI: %{y:.3f}<extra></extra>",
            ))
            fig1_ly.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig1_ly.update_layout(
                title="Fig 1: Individual mouse preference trajectories<br>"
                      "<sup>Click legend to show/hide mice, hover for details</sup>",
                xaxis_title="Experiment Day", yaxis_title="Preference Index",
                yaxis_range=[-1.05, 1.05], template="plotly_white",
                height=600, width=1000,
            )
            fig1_ly.write_html(os.path.join(output_dir, "fig1_pi_trajectories.html"))
            print("  Saved fig1 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig1 failed: {e}")

        # ---- Plotly Fig 3: PI violins with mouse IDs ----
        try:
            fig3_ly = go.Figure()
            for d in PI_DAYS:
                sub = df_pi[df_pi["day"] == d].dropna(subset=["preference_index"])
                fig3_ly.add_trace(go.Box(
                    y=sub["preference_index"],
                    x=[DAY_SHORT[d]] * len(sub),
                    name=DAY_SHORT[d],
                    boxpoints="all", jitter=0.3, pointpos=0,
                    marker=dict(
                        color="#E63946" if d == "w2_vocalisations" else "#457B9D",
                        size=6,
                    ),
                    text=sub["mouse_id"],
                    hovertemplate="<b>%{text}</b><br>PI: %{y:.3f}<extra></extra>",
                    showlegend=False,
                ))
            fig3_ly.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig3_ly.update_layout(
                title="Fig 3: PI distribution by day<br>"
                      "<sup>Hover over points to identify mice</sup>",
                yaxis_title="Preference Index", template="plotly_white",
                height=600, width=900,
            )
            fig3_ly.write_html(os.path.join(output_dir, "fig3_pi_violins.html"))
            print("  Saved fig3 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig3 failed: {e}")

        # ---- Plotly Fig 4: Complexity box plots ----
        try:
            if len(df_stim) > 0:
                fig4_ly = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=[DAY_SHORT[d] for d in DAY_ORDER],
                    horizontal_spacing=0.06, vertical_spacing=0.12,
                )
                for idx, day in enumerate(DAY_ORDER):
                    r, c = divmod(idx, 3)
                    day_stim = df_stim[df_stim["day"] == day]
                    if len(day_stim) == 0:
                        continue
                    pivot = day_stim.groupby(["mouse_id", "stim_type"])[
                        "avg_visit_dur_ms"].mean().reset_index()
                    stim_order = EXPERIMENT_DAYS[day].get("complexity_order", [])
                    available = [s for s in stim_order if s in pivot["stim_type"].values]
                    extras = [s for s in pivot["stim_type"].unique()
                              if s not in available and s != "silent"]
                    plot_order = available + extras
                    if not plot_order:
                        continue
                    for stim in plot_order:
                        stim_data = pivot[pivot["stim_type"] == stim]
                        fig4_ly.add_trace(go.Box(
                            y=stim_data["avg_visit_dur_ms"],
                            x=[stim] * len(stim_data),
                            boxpoints="all", jitter=0.3, pointpos=0,
                            marker=dict(size=5),
                            text=stim_data["mouse_id"],
                            hovertemplate="<b>%{text}</b><br>%{x}<br>Duration: %{y:.0f}ms<extra></extra>",
                            showlegend=False,
                        ), row=r + 1, col=c + 1)
                    fig4_ly.update_yaxes(title_text="Avg visit dur (ms)" if c == 0 else "",
                                         row=r + 1, col=c + 1)
                fig4_ly.update_layout(
                    title="Fig 4: Visit duration by stimulus type<br>"
                          "<sup>Hover over points to identify mice</sup>",
                    template="plotly_white", height=800, width=1200,
                )
                fig4_ly.write_html(os.path.join(output_dir, "fig4_complexity_heatmap.html"))
                print("  Saved fig4 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig4 failed: {e}")

        # ---- Plotly Fig 5: Vocalisation contrast ----
        try:
            fig5_ly = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Voc PI by day", "Overall vs Voc PI", "W2 Voc: PI by recording"],
                horizontal_spacing=0.08,
            )
            # Panel A
            df_voc_pi2 = df_pref.dropna(subset=["voc_pi"]).copy()
            if len(df_voc_pi2) > 0:
                for d in DAY_ORDER:
                    sub = df_voc_pi2[df_voc_pi2["day"] == d]
                    if len(sub) == 0:
                        continue
                    fig5_ly.add_trace(go.Box(
                        y=sub["voc_pi"], x=[DAY_SHORT[d]] * len(sub),
                        boxpoints="all", jitter=0.3, pointpos=0,
                        marker=dict(size=5, color="#E63946"),
                        text=sub["mouse_id"],
                        hovertemplate="<b>%{text}</b><br>Voc PI: %{y:.3f}<extra></extra>",
                        showlegend=False,
                    ), row=1, col=1)
            fig5_ly.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5,
                              row=1, col=1)

            # Panel B: paired lines
            if paired_data:
                df_p = pd.DataFrame(paired_data)
                for _, row in df_p.iterrows():
                    fig5_ly.add_trace(go.Scatter(
                        x=["Overall PI", "Voc PI"],
                        y=[row["pi_overall"], row["pi_voc"]],
                        mode="lines+markers",
                        line=dict(color="#888888", width=1), opacity=0.4,
                        marker=dict(size=5),
                        hovertemplate=f"<b>{row['mouse_id']}</b> ({DAY_SHORT.get(row['day'], row['day'])})<br>"
                                      "%{x}: %{y:.3f}<extra></extra>",
                        showlegend=False,
                    ), row=1, col=2)
            fig5_ly.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5,
                              row=1, col=2)

            # Panel C: per-recording for w2_voc
            voc_day_stim2 = df_stim[df_stim["day"] == "w2_vocalisations"]
            if len(voc_day_stim2) > 0 and "stim_pi" in voc_day_stim2.columns:
                voc_m = voc_day_stim2.groupby(["mouse_id", "stim_type"])[
                    "stim_pi"].mean().reset_index()
                stim_medians2 = voc_m.groupby("stim_type")["stim_pi"].median()
                top_s = stim_medians2.nlargest(8).index.tolist()
                voc_p = voc_m[voc_m["stim_type"].isin(top_s)]
                for stim in top_s:
                    sd = voc_p[voc_p["stim_type"] == stim]
                    short = stim[:25] + "..." if len(stim) > 28 else stim
                    fig5_ly.add_trace(go.Box(
                        y=sd["stim_pi"], x=[short] * len(sd),
                        boxpoints="all", jitter=0.3, pointpos=0,
                        marker=dict(size=5),
                        text=sd["mouse_id"],
                        hovertemplate="<b>%{text}</b><br>%{x}<br>PI: %{y:.3f}<extra></extra>",
                        showlegend=False,
                    ), row=1, col=3)
            fig5_ly.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5,
                              row=1, col=3)

            fig5_ly.update_layout(
                title="Fig 5: Vocalisation preference<br>"
                      "<sup>Hover to identify mice</sup>",
                template="plotly_white", height=550, width=1400,
            )
            fig5_ly.write_html(os.path.join(output_dir, "fig5_vocalisation_contrast.html"))
            print("  Saved fig5 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig5 failed: {e}")

        # ---- Plotly Fig 6: RE vs PI ----
        try:
            fig6_ly = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Within-mouse", "Between-mouse"],
                horizontal_spacing=0.1,
            )
            df_re2 = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).copy()
            if len(df_re2) > 10:
                mm = df_re2.groupby("mouse_id").agg(
                    mean_re=("roaming_entropy", "mean"),
                    mean_pi=("preference_index", "mean"))
                df_re2 = df_re2.merge(mm, on="mouse_id")
                df_re2["dev_re"] = df_re2["roaming_entropy"] - df_re2["mean_re"]
                df_re2["dev_pi"] = df_re2["preference_index"] - df_re2["mean_pi"]
                fig6_ly.add_trace(go.Scatter(
                    x=df_re2["dev_re"], y=df_re2["dev_pi"],
                    mode="markers",
                    marker=dict(size=7, color="#457B9D", opacity=0.5),
                    text=df_re2["mouse_id"],
                    customdata=np.stack([
                        df_re2["day"].map(DAY_SHORT),
                        df_re2["roaming_entropy"],
                        df_re2["preference_index"],
                    ], axis=-1),
                    hovertemplate="<b>%{text}</b> (%{customdata[0]})<br>"
                                  "RE: %{customdata[1]:.3f}<br>"
                                  "PI: %{customdata[2]:.3f}<extra></extra>",
                    showlegend=False,
                ), row=1, col=1)
            fig6_ly.update_xaxes(title_text="Dev from mouse mean RE", row=1, col=1)
            fig6_ly.update_yaxes(title_text="Dev from mouse mean PI", row=1, col=1)

            ma2 = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).groupby(
                "mouse_id").agg(mean_re=("roaming_entropy", "mean"),
                                mean_pi=("preference_index", "mean")).reset_index()
            if len(ma2) > 5:
                fig6_ly.add_trace(go.Scatter(
                    x=ma2["mean_re"], y=ma2["mean_pi"],
                    mode="markers",
                    marker=dict(size=9, color="#E63946", opacity=0.7),
                    text=ma2["mouse_id"],
                    hovertemplate="<b>%{text}</b><br>Mean RE: %{x:.3f}<br>Mean PI: %{y:.3f}<extra></extra>",
                    showlegend=False,
                ), row=1, col=2)
            fig6_ly.update_xaxes(title_text="Mean Roaming Entropy", row=1, col=2)
            fig6_ly.update_yaxes(title_text="Mean PI", row=1, col=2)

            fig6_ly.update_layout(
                title="Fig 6: Roaming entropy vs preference<br>"
                      "<sup>Hover to identify mice</sup>",
                template="plotly_white", height=500, width=1000,
            )
            fig6_ly.write_html(os.path.join(output_dir, "fig6_re_vs_pi.html"))
            print("  Saved fig6 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig6 failed: {e}")

        # ---- Plotly Fig 8: Per-stimulus PI ----
        try:
            if len(df_stim) > 0 and "stim_pi" in df_stim.columns:
                fig8_ly = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=[DAY_SHORT[d] for d in DAY_ORDER],
                    horizontal_spacing=0.06, vertical_spacing=0.12,
                )
                for idx, day in enumerate(DAY_ORDER):
                    r, c = divmod(idx, 3)
                    day_stim = df_stim[df_stim["day"] == day]
                    if len(day_stim) == 0:
                        continue
                    pivot = day_stim.groupby(["mouse_id", "stim_type"])[
                        "stim_pi"].mean().reset_index()
                    stim_order = EXPERIMENT_DAYS[day].get("complexity_order", [])
                    available = [s for s in stim_order if s in pivot["stim_type"].values]
                    extras = [s for s in pivot["stim_type"].unique()
                              if s not in available and s != "silent"]
                    plot_order = available + sorted(extras)
                    if not plot_order:
                        continue
                    for stim in plot_order:
                        sd = pivot[pivot["stim_type"] == stim]
                        fig8_ly.add_trace(go.Box(
                            y=sd["stim_pi"], x=[stim] * len(sd),
                            boxpoints="all", jitter=0.3, pointpos=0,
                            marker=dict(size=5),
                            text=sd["mouse_id"],
                            hovertemplate="<b>%{text}</b><br>%{x}<br>PI: %{y:.3f}<extra></extra>",
                            showlegend=False,
                        ), row=r + 1, col=c + 1)
                    fig8_ly.add_hline(y=0, line_dash="dash", line_color="grey",
                                      opacity=0.5, row=r + 1, col=c + 1)
                    fig8_ly.update_yaxes(title_text="PI vs silent" if c == 0 else "",
                                         row=r + 1, col=c + 1)
                fig8_ly.update_layout(
                    title="Fig 8: Per-stimulus PI vs silent baseline<br>"
                          "<sup>Hover over points to identify mice</sup>",
                    template="plotly_white", height=800, width=1200,
                )
                fig8_ly.write_html(os.path.join(output_dir, "fig8_stimulus_pi.html"))
                print("  Saved fig8 interactive")
        except Exception as e:
            print(f"  WARNING: Plotly fig8 failed: {e}")

    # ── 10. Complexity stats ─────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("SENSORY COMPLEXITY ANALYSIS", stats_lines)
    log_stat("=" * 60, stats_lines)

    for day in DAY_ORDER:
        order = EXPERIMENT_DAYS[day].get("complexity_order", [])
        if len(order) < 2:
            continue
        day_data = df_stim[df_stim["day"] == day]
        if len(day_data) == 0:
            continue
        mouse_means = day_data.groupby(["mouse_id", "stim_type"])[
            "avg_visit_dur_ms"].mean().reset_index()
        available = [s for s in order if s in mouse_means["stim_type"].values]
        if len(available) < 2:
            continue
        log_stat(f"\n--- {DAY_SHORT[day]}: {' < '.join(available)} ---", stats_lines)
        groups = [mouse_means[mouse_means["stim_type"] == s][
            "avg_visit_dur_ms"].values for s in available]
        for s, g in zip(available, groups):
            log_stat(f"  {s}: mean={np.mean(g):.1f}ms, median={np.median(g):.1f}ms, n={len(g)}",
                     stats_lines)
        if all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            log_stat(f"  Kruskal-Wallis: H={stat:.2f}, p={p:.4f}", stats_lines)

    # ── 11. One-sample PI tests ─────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("ONE-SAMPLE TESTS: PI vs 0", stats_lines)
    log_stat("=" * 60, stats_lines)
    for d in PI_DAYS:
        vals = df_pi[df_pi["day"] == d]["preference_index"].dropna().values
        if len(vals) >= 6:
            stat, p = wilcoxon(vals)
            log_stat(f"  {DAY_SHORT[d]}: median={np.median(vals):.3f}, "
                     f"mean={np.mean(vals):.3f}, Wilcoxon p={p:.4f} (n={len(vals)})",
                     stats_lines)
        else:
            log_stat(f"  {DAY_SHORT[d]}: n={len(vals)} (too few)", stats_lines)

    # ── 11b. Vocalisation-specific PI tests ──────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("VOCALISATION-SPECIFIC PI TESTS", stats_lines)
    log_stat("=" * 60, stats_lines)
    voc_all = df_pref.dropna(subset=["voc_pi"])
    if len(voc_all) >= 6:
        vals = voc_all["voc_pi"].values
        stat, p = wilcoxon(vals)
        log_stat(f"  All days combined: median={np.median(vals):.3f}, "
                 f"mean={np.mean(vals):.3f}, Wilcoxon p={p:.4f} (n={len(vals)})",
                 stats_lines)
    for d in DAY_ORDER:
        vals = voc_all[voc_all["day"] == d]["voc_pi"].dropna().values
        if len(vals) >= 6:
            stat, p = wilcoxon(vals)
            log_stat(f"  {DAY_SHORT[d]}: median={np.median(vals):.3f}, "
                     f"mean={np.mean(vals):.3f}, Wilcoxon p={p:.4f} (n={len(vals)})",
                     stats_lines)
        elif len(vals) > 0:
            log_stat(f"  {DAY_SHORT[d]}: median={np.median(vals):.3f}, "
                     f"mean={np.mean(vals):.3f}, n={len(vals)} (too few for test)",
                     stats_lines)

    # ══════════════════════════════════════════════════════════════════
    #  ENHANCED ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    log_stat("\n\n" + "#" * 70, stats_lines)
    log_stat("#  ENHANCED ANALYSIS", stats_lines)
    log_stat("#  (post-hoc tests, effect sizes, trend tests, corrections)", stats_lines)
    log_stat("#" * 70, stats_lines)

    # ── helper: Wilcoxon rank-biserial effect size ──────────────────
    def wilcoxon_effect_size(x, y=None):
        """Rank-biserial r = Z / sqrt(N).
        If y is None, one-sample test of x vs 0."""
        if y is not None:
            diff = np.array(x) - np.array(y)
        else:
            diff = np.asarray(x)
        diff_nz = diff[diff != 0]
        n = len(diff_nz)
        if n < 6:
            return np.nan, np.nan, np.nan
        stat, p = wilcoxon(diff_nz)
        # z approximation: W ~ N(mu, sigma) where mu = n(n+1)/4
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (stat - mu) / sigma if sigma > 0 else 0.0
        r = z / np.sqrt(n)
        return stat, p, r

    # ── helper: Kruskal-Wallis epsilon-squared ──────────────────────
    def kw_epsilon_squared(H, groups):
        """Epsilon-squared = (H - k + 1) / (N - k)."""
        k = len(groups)
        N = sum(len(g) for g in groups)
        if N <= k:
            return np.nan
        return (H - k + 1) / (N - k)

    # ── helper: Dunn's post-hoc with Holm correction ───────────────
    def dunns_posthoc(groups, labels):
        """Dunn's test with Holm-Bonferroni correction.
        Returns list of (label_a, label_b, z, p_raw, p_adj)."""
        # Pool all observations and rank
        all_data = np.concatenate(groups)
        all_ranks = rankdata(all_data)
        N = len(all_data)

        # Assign ranks back to groups
        idx = 0
        group_ranks = []
        for g in groups:
            group_ranks.append(all_ranks[idx:idx + len(g)])
            idx += len(g)

        mean_ranks = [np.mean(r) for r in group_ranks]
        ns = [len(g) for g in groups]

        # Tied-ranks correction for sigma
        _, counts = np.unique(all_ranks, return_counts=True)
        tie_corr = np.sum(counts ** 3 - counts)
        sigma_base = (N * (N + 1) / 12.0) - tie_corr / (12.0 * (N - 1)) if N > 1 else 1.0

        pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                diff = mean_ranks[i] - mean_ranks[j]
                se = np.sqrt(sigma_base * (1.0 / ns[i] + 1.0 / ns[j]))
                z = diff / se if se > 0 else 0.0
                p = 2.0 * norm.sf(abs(z))
                pairs.append((labels[i], labels[j], z, p))

        if not pairs:
            return []
        raw_ps = [p for _, _, _, p in pairs]
        _, adj_ps, _, _ = multipletests(raw_ps, method="holm")
        return [(a, b, z, rp, ap)
                for (a, b, z, rp), ap in zip(pairs, adj_ps)]

    # ── helper: Jonckheere-Terpstra trend test ──────────────────────
    def jonckheere_terpstra_test(groups):
        """JT test for ordered alternatives.
        groups: list of arrays in hypothesised order (low->high).
        Returns (JT statistic, z, p_one_sided)."""
        k = len(groups)
        if k < 2:
            return np.nan, np.nan, np.nan
        # Count concordant pairs across ordered groups
        J = 0
        for i in range(k):
            for j in range(i + 1, k):
                for xi in groups[i]:
                    for xj in groups[j]:
                        if xj > xi:
                            J += 1
                        elif xj == xi:
                            J += 0.5
        # Expected value and variance under H0
        N = sum(len(g) for g in groups)
        ns = [len(g) for g in groups]
        E_J = (N ** 2 - sum(n ** 2 for n in ns)) / 4.0
        # Variance (no-tie formula)
        sum_ni2 = sum(n ** 2 for n in ns)
        sum_ni3 = sum(n ** 3 for n in ns)
        var_num = (N ** 2 * (2 * N + 3) - sum_ni2 * (2 * sum_ni2 + 3))  # approximation
        # More precise: use the standard JT variance formula
        var_J = (1.0 / 72.0) * (
            N ** 2 * (2 * N + 3)
            - sum(n ** 2 * (2 * n + 3) for n in ns)
        )
        if var_J <= 0:
            return J, 0.0, 0.5
        z = (J - E_J) / np.sqrt(var_J)
        p = norm.sf(z)  # one-sided (testing increasing trend)
        return J, z, p

    # ── helper: format table row ───────────────────────────────────
    def fmt_test(test_name, stat_name, stat_val, df_or_n, p_raw,
                 p_adj=None, effect_name=None, effect_val=None):
        """Consistent output: stat | df/N | p_raw | p_adj | effect."""
        parts = [f"  {test_name}: {stat_name}={stat_val:.3f}"]
        parts.append(f"N={df_or_n}" if isinstance(df_or_n, int) else f"df={df_or_n}")
        parts.append(f"p={p_raw:.4f}")
        if p_adj is not None:
            parts.append(f"p_adj={p_adj:.4f}")
        if effect_name and effect_val is not None and not np.isnan(effect_val):
            parts.append(f"{effect_name}={effect_val:.3f}")
        return " | ".join(parts)

    # ────────────────────────────────────────────────────────────────
    # EA-1. ONE-SAMPLE PI TESTS WITH EFFECT SIZES + HOLM CORRECTION
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-1. ONE-SAMPLE PI TESTS (effect sizes + Holm correction)",
             stats_lines)
    log_stat("=" * 60, stats_lines)

    onesamp_results = []
    for d in PI_DAYS:
        vals = df_pi[df_pi["day"] == d]["preference_index"].dropna().values
        if len(vals) >= 6:
            W, p_raw, r_rb = wilcoxon_effect_size(vals)
            onesamp_results.append((d, len(vals), W, p_raw, r_rb))

    if onesamp_results:
        raw_ps = [r[3] for r in onesamp_results]
        _, adj_ps, _, _ = multipletests(raw_ps, method="holm")
        for (d, n, W, p_raw, r_rb), p_adj in zip(onesamp_results, adj_ps):
            log_stat(
                fmt_test(DAY_SHORT[d], "W", W, n, p_raw,
                         p_adj=p_adj, effect_name="r_rb", effect_val=r_rb),
                stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-2. KRUSKAL-WALLIS ACROSS DAYS + DUNN'S POST-HOC + EFFECT SIZE
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-2. KRUSKAL-WALLIS ACROSS DAYS (post-hoc + effect size)",
             stats_lines)
    log_stat("=" * 60, stats_lines)

    day_groups_kw = []
    day_labels_kw = []
    for d in PI_DAYS:
        g = df_pi[df_pi["day"] == d]["preference_index"].dropna().values
        if len(g) > 0:
            day_groups_kw.append(g)
            day_labels_kw.append(DAY_SHORT[d])

    if len(day_groups_kw) > 1:
        H, p_kw = kruskal(*day_groups_kw)
        eps2 = kw_epsilon_squared(H, day_groups_kw)
        log_stat(
            fmt_test("Across days", "H", H, len(day_labels_kw),
                     p_kw, effect_name="eps2", effect_val=eps2),
            stats_lines)
        if p_kw < 0.05:
            log_stat("\n  Dunn's pairwise post-hoc (Holm-corrected):", stats_lines)
            dunn_results = dunns_posthoc(day_groups_kw, day_labels_kw)
            log_stat(f"  {'Pair':<28s} {'z':>8s} {'p_raw':>8s} {'p_adj':>8s} {'sig':>5s}",
                     stats_lines)
            log_stat(f"  {'-'*61}", stats_lines)
            for a, b, z, rp, ap in dunn_results:
                sig = "***" if ap < 0.001 else "**" if ap < 0.01 else "*" if ap < 0.05 else ""
                log_stat(f"  {a+' vs '+b:<28s} {z:>8.3f} {rp:>8.4f} {ap:>8.4f} {sig:>5s}",
                         stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-3. SENSORY COMPLEXITY: JT TREND + KW POST-HOC + EFFECT SIZES
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-3. SENSORY COMPLEXITY (JT trend + KW post-hoc + effect sizes)",
             stats_lines)
    log_stat("=" * 60, stats_lines)

    for day in DAY_ORDER:
        order = EXPERIMENT_DAYS[day].get("complexity_order", [])
        if len(order) < 2:
            continue
        day_data = df_stim[df_stim["day"] == day]
        if len(day_data) == 0:
            continue
        mouse_means = day_data.groupby(["mouse_id", "stim_type"])[
            "avg_visit_dur_ms"].mean().reset_index()
        available = [s for s in order if s in mouse_means["stim_type"].values]
        if len(available) < 2:
            continue

        log_stat(f"\n--- {DAY_SHORT[day]}: {' < '.join(available)} ---", stats_lines)
        groups = [mouse_means[mouse_means["stim_type"] == s][
            "avg_visit_dur_ms"].values for s in available]
        for s, g in zip(available, groups):
            log_stat(f"  {s}: mean={np.mean(g):.1f}ms, median={np.median(g):.1f}ms, n={len(g)}",
                     stats_lines)

        # Kruskal-Wallis + effect size
        if all(len(g) > 0 for g in groups):
            H, p_kw = kruskal(*groups)
            eps2 = kw_epsilon_squared(H, groups)
            log_stat(
                fmt_test("KW", "H", H, len(available), p_kw,
                         effect_name="eps2", effect_val=eps2),
                stats_lines)
            if p_kw < 0.05:
                log_stat("  Dunn's pairwise post-hoc (Holm-corrected):", stats_lines)
                dunn_results = dunns_posthoc(groups, available)
                log_stat(f"    {'Pair':<32s} {'z':>8s} {'p_raw':>8s} {'p_adj':>8s} {'sig':>5s}",
                         stats_lines)
                log_stat(f"    {'-'*65}", stats_lines)
                for a, b, z, rp, ap in dunn_results:
                    sig = "***" if ap < 0.001 else "**" if ap < 0.01 else "*" if ap < 0.05 else ""
                    log_stat(f"    {a+' vs '+b:<32s} {z:>8.3f} {rp:>8.4f} {ap:>8.4f} {sig:>5s}",
                             stats_lines)

        # Jonckheere-Terpstra trend test
        if len(available) >= 3 and all(len(g) >= 3 for g in groups):
            J, z_jt, p_jt = jonckheere_terpstra_test(groups)
            log_stat(
                fmt_test("JT trend", "J", J, sum(len(g) for g in groups),
                         p_jt, effect_name="z", effect_val=z_jt),
                stats_lines)
            if p_jt < 0.05:
                log_stat("    --> Significant increasing trend in visit duration "
                         "with stimulus complexity", stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-4. VOC PI TESTS WITH EFFECT SIZES + HOLM CORRECTION
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-4. VOC PI TESTS (effect sizes + Holm correction)", stats_lines)
    log_stat("=" * 60, stats_lines)

    voc_results = []
    for d in DAY_ORDER:
        vals = voc_all[voc_all["day"] == d]["voc_pi"].dropna().values
        if len(vals) >= 6:
            W, p_raw, r_rb = wilcoxon_effect_size(vals)
            voc_results.append((d, len(vals), W, p_raw, r_rb))

    if voc_results:
        raw_ps = [r[3] for r in voc_results]
        _, adj_ps, _, _ = multipletests(raw_ps, method="holm")
        for (d, n, W, p_raw, r_rb), p_adj in zip(voc_results, adj_ps):
            log_stat(
                fmt_test(DAY_SHORT[d], "W", W, n, p_raw,
                         p_adj=p_adj, effect_name="r_rb", effect_val=r_rb),
                stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-5. WITHIN-MOUSE RE-PI: MIXED MODEL (replacing Pearson r)
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-5. WITHIN-MOUSE RE-PI ASSOCIATION (mixed model)", stats_lines)
    log_stat("=" * 60, stats_lines)

    if HAS_STATSMODELS:
        df_re_ea = df_pi.dropna(subset=["roaming_entropy", "preference_index"]).copy()
        if len(df_re_ea) > 10 and df_re_ea["mouse_id"].nunique() > 2:
            mouse_re = df_re_ea.groupby("mouse_id")["roaming_entropy"].mean()
            df_re_ea["re_mouse_mean"] = df_re_ea["mouse_id"].map(mouse_re)
            df_re_ea["re_within"] = df_re_ea["roaming_entropy"] - df_re_ea["re_mouse_mean"]
            try:
                m_within = smf.mixedlm(
                    "preference_index ~ re_within",
                    df_re_ea, groups=df_re_ea["mouse_id"]
                ).fit(reml=True)
                coef = m_within.fe_params["re_within"]
                se = m_within.bse_fe["re_within"]
                z_val = m_within.tvalues["re_within"]
                p_val = m_within.pvalues["re_within"]
                log_stat("  Mixed model: PI ~ re_within + (1|mouse)", stats_lines)
                log_stat(f"    re_within: coef={coef:.4f}, SE={se:.4f}, "
                         f"z={z_val:.3f}, p={p_val:.4f}", stats_lines)
                log_stat("  (This supersedes the flat within-mouse Pearson r)", stats_lines)
            except Exception as e:
                log_stat(f"  Within-mouse mixed model failed: {e}", stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-6. MIXED MODEL COMPARISON: AIC, BIC, LRT
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-6. MODEL COMPARISON (AIC, BIC, LRT)", stats_lines)
    log_stat("=" * 60, stats_lines)

    if HAS_STATSMODELS:
        df_model_ea = df_pi.dropna(subset=["preference_index"]).copy()
        if len(df_model_ea) > 10 and df_model_ea["mouse_id"].nunique() > 2:
            try:
                # Refit with ML (not REML) for valid LRT comparison
                m1_ml = smf.mixedlm(
                    "preference_index ~ 1", df_model_ea,
                    groups=df_model_ea["mouse_id"]
                ).fit(reml=False)

                df_model_ea["day_cat"] = pd.Categorical(
                    df_model_ea["day"], categories=PI_DAYS)
                m2_ml = smf.mixedlm(
                    "preference_index ~ C(day_cat)", df_model_ea,
                    groups=df_model_ea["mouse_id"]
                ).fit(reml=False)

                models = [("M1: PI ~ 1 + (1|mouse)", m1_ml),
                          ("M2: PI ~ day + (1|mouse)", m2_ml)]

                # Model 3: RE predictors
                df_re_m_ea = df_model_ea.dropna(subset=["roaming_entropy"]).copy()
                m3_ml = None
                if len(df_re_m_ea) > 10:
                    mouse_re3 = df_re_m_ea.groupby("mouse_id")["roaming_entropy"].mean()
                    grand_re3 = mouse_re3.mean()
                    df_re_m_ea["re_mouse_mean"] = df_re_m_ea["mouse_id"].map(mouse_re3)
                    df_re_m_ea["re_within"] = df_re_m_ea["roaming_entropy"] - df_re_m_ea["re_mouse_mean"]
                    df_re_m_ea["re_between"] = df_re_m_ea["re_mouse_mean"] - grand_re3
                    try:
                        m3_ml = smf.mixedlm(
                            "preference_index ~ re_within + re_between",
                            df_re_m_ea, groups=df_re_m_ea["mouse_id"]
                        ).fit(reml=False)
                        models.append(("M3: PI ~ RE_w + RE_b + (1|mouse)", m3_ml))
                    except Exception:
                        pass

                # Table
                log_stat(f"\n  {'Model':<40s} {'LL':>10s} {'AIC':>10s} {'BIC':>10s} {'k':>4s}",
                         stats_lines)
                log_stat(f"  {'-'*76}", stats_lines)
                for name, m in models:
                    ll = m.llf
                    aic_val = -2 * ll + 2 * m.df_modelwc
                    bic_val = -2 * ll + np.log(m.nobs) * m.df_modelwc
                    log_stat(
                        f"  {name:<40s} {ll:>10.2f} {aic_val:>10.2f} {bic_val:>10.2f} {m.df_modelwc:>4d}",
                        stats_lines)

                # LRT: M1 vs M2
                from scipy.stats import chi2
                ll1 = m1_ml.llf
                ll2 = m2_ml.llf
                lr_stat = 2 * (ll2 - ll1)
                lr_df = m2_ml.df_modelwc - m1_ml.df_modelwc
                lr_p = chi2.sf(lr_stat, lr_df) if lr_df > 0 else np.nan
                log_stat(f"\n  LRT M1 vs M2: chi2={lr_stat:.3f}, df={lr_df}, p={lr_p:.4f}",
                         stats_lines)

                # LRT: M1 vs M3
                if m3_ml is not None:
                    # M3 may be on a subset — refit M1 on same subset
                    try:
                        m1_sub = smf.mixedlm(
                            "preference_index ~ 1", df_re_m_ea,
                            groups=df_re_m_ea["mouse_id"]
                        ).fit(reml=False)
                        lr3 = 2 * (m3_ml.llf - m1_sub.llf)
                        lr3_df = m3_ml.df_modelwc - m1_sub.df_modelwc
                        lr3_p = chi2.sf(lr3, lr3_df) if lr3_df > 0 else np.nan
                        log_stat(f"  LRT M1 vs M3: chi2={lr3:.3f}, df={lr3_df}, p={lr3_p:.4f} "
                                 f"(on RE subset, n={len(df_re_m_ea)})", stats_lines)
                    except Exception:
                        pass

                # Marginal & conditional R-squared (Nakagawa & Schielzeth)
                log_stat("\n  Marginal & conditional R-squared (Nakagawa & Schielzeth):",
                         stats_lines)
                for name, m in models:
                    try:
                        var_fe = np.var(m.fittedvalues - m.random_effects_cov  # not right
                                        if hasattr(m, 'random_effects_cov') else 0)
                    except Exception:
                        pass
                    # Approximate: var_fixed, var_random, var_resid
                    try:
                        # Fixed-effects predicted values (population level)
                        Xb = np.array(m.fittedvalues) - np.array(
                            [m.random_effects[g]["Group"] if "Group" in m.random_effects.get(g, {})
                             else 0.0 for g in df_model_ea["mouse_id"]]
                            if name != models[-1][0] or m3_ml is None
                            else [m.random_effects[g]["Group"] if "Group" in m.random_effects.get(g, {})
                                  else 0.0 for g in df_re_m_ea["mouse_id"]]
                        )
                        var_f = np.var(Xb)
                        var_u = m.cov_re.iloc[0, 0] if hasattr(m.cov_re, 'iloc') else float(m.cov_re)
                        var_e = m.scale
                        r2_marginal = var_f / (var_f + var_u + var_e)
                        r2_conditional = (var_f + var_u) / (var_f + var_u + var_e)
                        log_stat(f"    {name}: R2m={r2_marginal:.4f}, R2c={r2_conditional:.4f}",
                                 stats_lines)
                    except Exception as e:
                        log_stat(f"    {name}: R2 computation failed ({e})", stats_lines)

            except Exception as e:
                log_stat(f"  Model comparison failed: {e}", stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-7. BETA REGRESSION SENSITIVITY CHECK
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-7. BETA REGRESSION SENSITIVITY CHECK", stats_lines)
    log_stat("  PI_scaled = (PI + 1) / 2, mapped to (0, 1)", stats_lines)
    log_stat("=" * 60, stats_lines)

    if HAS_STATSMODELS:
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.families import Tweedie
        except ImportError:
            pass

        df_beta = df_pi.dropna(subset=["preference_index"]).copy()
        # Transform PI [-1,1] -> (0,1), squeezing away exact 0/1
        df_beta["pi_scaled"] = (df_beta["preference_index"] + 1.0) / 2.0
        # Squeeze to open interval (epsilon, 1-epsilon) for beta regression
        eps = 1e-6
        df_beta["pi_scaled"] = df_beta["pi_scaled"].clip(eps, 1.0 - eps)

        try:
            # Try statsmodels GLM with Binomial(logit) as beta-like approximation
            # (true Beta family not in statsmodels; use quasi-likelihood with logit link)
            import statsmodels.api as sm

            # Model 1: intercept-only
            X1 = sm.add_constant(np.ones(len(df_beta)))
            glm1 = sm.GLM(
                df_beta["pi_scaled"], X1,
                family=sm.families.Binomial(link=sm.families.links.Logit())
            ).fit()

            # Model 2: with day dummies
            day_dummies = pd.get_dummies(
                df_beta["day"].astype(pd.CategoricalDtype(categories=PI_DAYS)),
                drop_first=True, dtype=float
            )
            X2 = sm.add_constant(day_dummies.values)
            glm2 = sm.GLM(
                df_beta["pi_scaled"], X2,
                family=sm.families.Binomial(link=sm.families.links.Logit())
            ).fit()

            log_stat("\n  Quasi-beta (Binomial/logit GLM) sensitivity check:", stats_lines)
            log_stat(f"  Note: uses logit-link Binomial as beta-like approximation", stats_lines)

            # Compare intercept
            mu_logit = glm1.params[0]
            mu_prob = 1.0 / (1.0 + np.exp(-mu_logit))
            mu_pi = mu_prob * 2.0 - 1.0  # back-transform to PI scale
            log_stat(f"\n  M1 (intercept-only):", stats_lines)
            log_stat(f"    logit(PI_scaled) intercept = {mu_logit:.4f} (p={glm1.pvalues[0]:.4f})",
                     stats_lines)
            log_stat(f"    Back-transformed grand mean PI = {mu_pi:.3f}", stats_lines)

            # Day effects
            log_stat(f"\n  M2 (with day effects):", stats_lines)
            day_names = ["Intercept (D1)"] + [
                f"vs {DAY_SHORT[d]}" for d in PI_DAYS[1:]]
            for i, (nm, coef, p) in enumerate(
                    zip(day_names, glm2.params, glm2.pvalues)):
                log_stat(f"    {nm}: coef={coef:.4f}, p={p:.4f}", stats_lines)

            # Compare conclusions with linear mixed model
            log_stat("\n  Comparison with linear mixed model:", stats_lines)
            linear_sign = m1_ml.fe_params["Intercept"] < 0  # negative PI?
            beta_sign = mu_pi < 0
            if linear_sign == beta_sign:
                log_stat("    --> Conclusions AGREE: same direction of grand mean PI",
                         stats_lines)
            else:
                log_stat("    --> WARNING: Conclusions DIFFER in direction of grand mean PI",
                         stats_lines)

        except Exception as e:
            log_stat(f"  Beta regression failed: {e}", stats_lines)

    # ────────────────────────────────────────────────────────────────
    # EA-8. PAIRED VOC-PI vs OVERALL-PI WITH EFFECT SIZE
    # ────────────────────────────────────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("EA-8. PAIRED VOC PI vs OVERALL PI (with effect size)", stats_lines)
    log_stat("=" * 60, stats_lines)

    if paired_data:
        df_paired_ea = pd.DataFrame(paired_data)
        if len(df_paired_ea) >= 6:
            W, p_raw, r_rb = wilcoxon_effect_size(
                df_paired_ea["pi_voc"].values,
                df_paired_ea["pi_overall"].values)
            log_stat(
                fmt_test("Voc PI vs Overall PI", "W", W,
                         len(df_paired_ea), p_raw,
                         effect_name="r_rb", effect_val=r_rb),
                stats_lines)

    log_stat("\n" + "#" * 70, stats_lines)
    log_stat("#  END OF ENHANCED ANALYSIS", stats_lines)
    log_stat("#" * 70, stats_lines)

    # ── 12. Vocalisation vs other sounds (within-session paired) ────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("VOCALISATION vs OTHER SOUNDS (within-session, paired)", stats_lines)
    log_stat("=" * 60, stats_lines)
    log_stat("For each mouse × day, compare vocalisation avg visit duration", stats_lines)
    log_stat("against the mean of all other (non-vocalisation) stimulus types.", stats_lines)

    for d in DAY_ORDER:
        day_stim = df_stim[df_stim["day"] == d]
        if len(day_stim) == 0:
            continue

        mouse_stim = (
            day_stim.groupby(["mouse_id", "stim_type"])["avg_visit_dur_ms"]
            .mean()
            .reset_index()
        )

        if EXPERIMENT_DAYS[d]["mode"] == "vocalisation":
            log_stat(f"\n  {DAY_SHORT[d]}: skipped (all stimuli are vocalisations)",
                     stats_lines)
            continue

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
            log_stat(f"\n  {DAY_SHORT[d]}: no mice with both voc and other sounds",
                     stats_lines)
            continue

        diff = paired_voc - paired_other
        log_stat(f"\n  {DAY_SHORT[d]} (n={n} mice):", stats_lines)
        log_stat(f"    Vocalisation avg: {np.mean(paired_voc):.0f}ms "
                 f"(median {np.median(paired_voc):.0f}ms)", stats_lines)
        log_stat(f"    Other sounds avg: {np.mean(paired_other):.0f}ms "
                 f"(median {np.median(paired_other):.0f}ms)", stats_lines)
        log_stat(f"    Difference (voc − other): mean={np.mean(diff):+.0f}ms, "
                 f"median={np.median(diff):+.0f}ms", stats_lines)

        if n >= 6:
            diff_nz = diff[diff != 0]
            if len(diff_nz) >= 6:
                stat, p = wilcoxon(diff_nz)
                log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}",
                         stats_lines)
            else:
                log_stat(f"    Too few non-zero differences for Wilcoxon ({len(diff_nz)})",
                         stats_lines)
        else:
            log_stat(f"    Too few mice for test (n={n})", stats_lines)

    # ── 12b. Voc PI vs Other Sounds PI (per-day paired test) ────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("VOCALISATION PI vs OTHER SOUNDS PI (per-day, paired)", stats_lines)
    log_stat("=" * 60, stats_lines)
    log_stat("For each mouse × day, compare voc_pi against other_sounds_pi.",
             stats_lines)

    df_vo_stats = df_pref.dropna(subset=["voc_pi", "other_sounds_pi"]).copy()
    df_vo_stats = df_vo_stats[df_vo_stats["day"] != "w2_vocalisations"]

    for d in DAY_ORDER:
        if d == "w2_vocalisations":
            log_stat(f"\n  {DAY_SHORT[d]}: skipped (all stimuli are vocalisations)",
                     stats_lines)
            continue
        sub = df_vo_stats[df_vo_stats["day"] == d]
        n = len(sub)
        if n == 0:
            continue

        vv = sub["voc_pi"].values
        ov = sub["other_sounds_pi"].values
        diff = vv - ov

        log_stat(f"\n  {DAY_SHORT[d]} (n={n} mice):", stats_lines)
        log_stat(f"    Voc PI: mean={np.mean(vv):.3f}, median={np.median(vv):.3f}",
                 stats_lines)
        log_stat(f"    Other PI: mean={np.mean(ov):.3f}, median={np.median(ov):.3f}",
                 stats_lines)
        log_stat(f"    Diff (voc − other): mean={np.mean(diff):+.3f}, "
                 f"median={np.median(diff):+.3f}", stats_lines)

        if n >= 6:
            diff_nz = diff[diff != 0]
            if len(diff_nz) >= 6:
                stat, p = wilcoxon(diff_nz)
                log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}",
                         stats_lines)
            else:
                log_stat(f"    Too few non-zero differences ({len(diff_nz)})",
                         stats_lines)
        else:
            log_stat(f"    Too few mice for test (n={n})", stats_lines)

    # Pooled across days
    all_vo = df_vo_stats["voc_pi"].values
    all_ot = df_vo_stats["other_sounds_pi"].values
    n_pool = len(all_vo)
    if n_pool >= 6:
        diff_pool = all_vo - all_ot
        diff_nz = diff_pool[diff_pool != 0]
        if len(diff_nz) >= 6:
            stat, p = wilcoxon(diff_nz)
            log_stat(f"\n  Pooled across days (n={n_pool}):", stats_lines)
            log_stat(f"    Voc PI mean={np.mean(all_vo):.3f}, "
                     f"Other PI mean={np.mean(all_ot):.3f}", stats_lines)
            log_stat(f"    Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}",
                     stats_lines)

    # ── 12c. First-minute RE stats ───────────────────────────────────
    log_stat("\n" + "=" * 60, stats_lines)
    log_stat("FIRST-MINUTE ROAMING ENTROPY", stats_lines)
    log_stat("=" * 60, stats_lines)

    df_re1_stats = df_pi.dropna(subset=["re_first_min"]).copy()
    n_re1_total = len(df_re1_stats)
    log_stat(f"  Sessions with first-minute RE: {n_re1_total} / {len(df_pi)}",
             stats_lines)

    if n_re1_total > 5:
        log_stat(f"  Mean first-min RE: {df_re1_stats['re_first_min'].mean():.3f}",
                 stats_lines)
        log_stat(f"  Median first-min RE: {df_re1_stats['re_first_min'].median():.3f}",
                 stats_lines)

        both_re = df_pi.dropna(subset=["roaming_entropy", "re_first_min"])
        if len(both_re) > 5:
            r, p = spearmanr(both_re["roaming_entropy"], both_re["re_first_min"])
            log_stat(f"\n  Full-hab RE vs first-min RE:", stats_lines)
            log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={len(both_re)}",
                     stats_lines)

        if n_re1_total > 10:
            r, p = spearmanr(df_re1_stats["re_first_min"],
                             df_re1_stats["preference_index"])
            log_stat(f"\n  First-min RE vs PI (all sessions):", stats_lines)
            log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={n_re1_total}",
                     stats_lines)

        mouse_avg_1m = df_re1_stats.groupby("mouse_id").agg(
            mean_re1=("re_first_min", "mean"),
            mean_pi=("preference_index", "mean")).reset_index()
        if len(mouse_avg_1m) > 5:
            r, p = spearmanr(mouse_avg_1m["mean_re1"], mouse_avg_1m["mean_pi"])
            log_stat(f"\n  Between-mouse first-min RE vs PI:", stats_lines)
            log_stat(f"    Spearman r={r:.3f}, p={p:.4f}, n={len(mouse_avg_1m)}",
                     stats_lines)

    # ── 13. Save stats ───────────────────────────────────────────────
    stats_path = os.path.join(output_dir, "stats_report.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines))
    print(f"\nStats report saved to {stats_path}")

    # ── 14. Run within-trial preference analysis (02_*) ──────────────
    # 02_within_trial_preference.py uses preference_analysis_config.py,
    # which respects the MAZE_DATA_DIR env var.  We pass base_path that way.
    print(f"\n{'='*60}")
    print("Running 02_within_trial_preference.py ...")
    print(f"{'='*60}")
    try:
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_02 = os.path.join(script_dir, "02_within_trial_preference.py")
        if os.path.isfile(script_02):
            env = os.environ.copy()
            env["MAZE_DATA_DIR"] = base_path
            # Inherit VISIT_CLIP_MS if set so both scripts use same setting
            result = subprocess.run(
                [sys.executable, script_02],
                cwd=script_dir, env=env,
            )
            if result.returncode != 0:
                print(f"  WARNING: 02_within_trial_preference.py exited with code {result.returncode}")
        else:
            print(f"  Skipping: {script_02} not found")
    except Exception as e:
        print(f"  WARNING: failed to run 02_within_trial_preference.py: {e}")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
