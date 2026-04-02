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
    preference_data.csv           per-mouse per-session PI + metrics
    stimulus_breakdown.csv        per-stimulus-type visit duration
    fig1_pi_trajectories.png/pdf  individual mouse PI trajectories
    fig2_pi_by_day.png/pdf        mean PI per day ±95 % CI
    fig3_pi_violins.png/pdf       violin plots by day
    fig4_complexity_heatmap.png   visit duration by stimulus type
    fig5_vocalisation_contrast.png vocalisation vs other days
    fig6_re_vs_pi.png/pdf         roaming entropy → PI
    fig7_icc_summary.png/pdf      ICC variance decomposition
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
                          kruskal, mannwhitneyu, spearmanr, wilcoxon)

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
        "complexity_order": ["smooth", "constant_rough", "complex_rough"],
    },
    "w1_d2": {
        "folder": "w1_d2",
        "mode": "complex_intervals",
        "label": "Day 2: Intervals (consonant/dissonant)",
        "short": "D2 (Int)",
        "complexity_order": ["consonant", "dissonant"],
    },
    "w1_d3": {
        "folder": "w1_d3",
        "mode": "complex_intervals",
        "label": "Day 3: Intervals (consonant/dissonant)",
        "short": "D3 (Int)",
        "complexity_order": ["consonant", "dissonant"],
    },
    "w1_d4": {
        "folder": "w1_d4",
        "mode": "complex_intervals",
        "label": "Day 4: Intervals (no silent control)",
        "short": "D4 (Int)",
        "complexity_order": ["consonant", "dissonant"],
    },
    "w2_sequences": {
        "folder": "w2_sequences",
        "mode": "sequences",
        "label": "Week 2: Sequences",
        "short": "W2 (Seq)",
        "complexity_order": ["AAAAA", "AoAo", "ABAB", "ABCABC"],
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
        if st in ("smooth", "constant_rough", "complex_rough"):
            return st
        if st in ("silent", "silent_trial"):
            return "silent"
        freq = row.get("frequency", 0)
        return "silent" if freq == 0 or str(freq) == "0" else "smooth"
    elif mode == "complex_intervals":
        it = str(row.get("interval_type", "")).strip().lower()
        if it in ("consonant", "dissonant", "control"):
            return it
        if it == "silent_trial":
            return "silent"
        freq = row.get("frequency", 0)
        return "silent" if freq == 0 or str(freq) == "0" else "consonant"
    elif mode == "sequences":
        pat = str(row.get("pattern", "")).strip()
        if pat in ("AAAAA", "AoAo", "ABAB", "ABCABC"):
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

    for sess in all_sessions:
        if not sess.trials_csv:
            skipped += 1
            continue
        try:
            df = pd.read_csv(sess.trials_csv)
        except Exception:
            skipped += 1
            continue
        if "trial_ID" not in df.columns or "ROIs" not in df.columns:
            skipped += 1
            continue
        loaded += 1

        for col in ["time_spent", "visitation_count", "time_in_maze_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

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

        # sound vs silent
        roi_names = [f"ROI{i}" for i in range(1, 9)]
        sound_rois = df[df["trial_ID"].isin([2, 4, 6, 8]) & df["ROIs"].isin(roi_names)]
        silent_rois = df[df["trial_ID"].isin([3, 5, 7, 9]) & df["ROIs"].isin(roi_names)]

        def _safe_sum(series):
            v = series.sum()
            return 0 if np.isnan(v) else v

        sound_time = _safe_sum(sound_rois["time_spent"]) if "time_spent" in sound_rois.columns else 0
        sound_visits = _safe_sum(sound_rois["visitation_count"]) if "visitation_count" in sound_rois.columns else 0
        silent_time = _safe_sum(silent_rois["time_spent"]) if "time_spent" in silent_rois.columns else 0
        silent_visits = _safe_sum(silent_rois["visitation_count"]) if "visitation_count" in silent_rois.columns else 0

        avg_sound_dur = sound_time / sound_visits if sound_visits > 0 else 0
        avg_silent_dur = silent_time / silent_visits if silent_visits > 0 else 0
        denom = avg_sound_dur + avg_silent_dur
        pi = (avg_sound_dur - avg_silent_dur) / denom if denom > 0 else np.nan

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
            "hab_time_ms": hab_total,
            "hab_visits": hab_visits,
            "roaming_entropy": roaming_entropy,
        })

        # per-stimulus breakdown
        for _, row in sound_rois.iterrows():
            stim_type = classify_stimulus(row, sess.day)
            if stim_type == "silent":
                continue
            t = row.get("time_spent", 0)
            v = row.get("visitation_count", 0)
            t = 0 if pd.isna(t) else t
            v = 0 if pd.isna(v) else v
            stim_records.append({
                "mouse_id": sess.mouse_id,
                "day": sess.day,
                "trial_ID": row["trial_ID"],
                "ROI": row["ROIs"],
                "stim_type": stim_type,
                "time_spent_ms": t,
                "visitation_count": v,
                "avg_visit_dur_ms": t / v if v > 0 else 0,
            })

    print(f"\nLoaded {loaded} sessions, skipped {skipped}")

    df_pref = pd.DataFrame(records)
    df_stim = pd.DataFrame(stim_records)

    df_pref.to_csv(os.path.join(output_dir, "preference_data.csv"), index=False)
    df_stim.to_csv(os.path.join(output_dir, "stimulus_breakdown.csv"), index=False)
    print(f"Saved preference_data.csv ({len(df_pref)} rows)")
    print(f"Saved stimulus_breakdown.csv ({len(df_stim)} rows)")

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
    print("Plotting Figure 5: Vocalisation contrast...")
    voc_mice = set(df_pi[df_pi["day"] == "w2_vocalisations"]["mouse_id"])
    non_voc_days = [d for d in PI_DAYS if d != "w2_vocalisations"]

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
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
                    "o-", color="#888888", alpha=0.4, ms=5)
        ax.plot(0, df_paired["pi_other"].mean(), "D", color="#457B9D", ms=12, zorder=5)
        ax.plot(1, df_paired["pi_voc"].mean(), "D", color="#E63946", ms=12, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Other days\n(mean)", "Vocalisations"], fontsize=11)
        ax.set_ylabel("Preference Index", fontsize=12)
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(df_paired) >= 6:
            stat, p_wil = wilcoxon(df_paired["pi_voc"], df_paired["pi_other"])
            log_stat(f"\nVocalisation vs Other (Wilcoxon): W={stat:.1f}, p={p_wil:.4f}",
                     stats_lines)
            ax.set_title(f"Paired (n={len(df_paired)}), Wilcoxon p={p_wil:.4f}",
                         fontsize=11)
        else:
            ax.set_title(f"Paired (n={len(df_paired)})", fontsize=11)

    ax = axes5[1]
    df_pi["is_voc"] = df_pi["day"] == "w2_vocalisations"
    df_pi["group"] = df_pi["is_voc"].map({True: "Vocalisations", False: "Other days"})
    sns.violinplot(data=df_pi, x="group", y="preference_index",
                   palette={"Other days": "#457B9D", "Vocalisations": "#E63946"},
                   inner="quart", alpha=0.4, ax=ax, hue="group", legend=False)
    sns.stripplot(data=df_pi, x="group", y="preference_index",
                  color="black", alpha=0.4, size=4, ax=ax)
    ax.axhline(0, color="grey", ls="--", alpha=0.5)
    ax.set_xlabel(""); ax.set_ylabel("Preference Index", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    voc_vals = df_pi[df_pi["is_voc"]]["preference_index"].dropna()
    other_vals = df_pi[~df_pi["is_voc"]]["preference_index"].dropna()
    if len(voc_vals) > 0 and len(other_vals) > 0:
        _, p_mwu = mannwhitneyu(voc_vals, other_vals, alternative="two-sided")
        ax.set_title(f"All sessions (MWU p={p_mwu:.4f})", fontsize=11)
        log_stat(f"Voc vs Other (MWU): p={p_mwu:.4f}", stats_lines)

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

    # ── 9. Complexity stats ──────────────────────────────────────────
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

    # ── 10. One-sample PI tests ──────────────────────────────────────
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

    # ── 11. Save stats ───────────────────────────────────────────────
    stats_path = os.path.join(output_dir, "stats_report.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines))
    print(f"\nStats report saved to {stats_path}")
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
