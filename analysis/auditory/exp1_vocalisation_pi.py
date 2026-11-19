#!/usr/bin/env python3
"""
Experiment 1: Do Mice Prefer Vocalisations?
===========================================

For every mouse and day of Experiment 1, three preference indices from the
sound blocks (trials 2, 4, 6, 8), each the thesis index on mean visit length,
(a - b) / (a + b), with both arms taken from the same blocks:

    vocalisation against the silent arm
    synthetic sounds against the silent arm
    vocalisation against synthetic sounds, directly

Synthetic sounds are every sound arm that is neither the vocalisation nor a
control. The direct index needs a vocalisation arm beside synthetic sounds,
so it exists on days 1 to 3 and the sequence day; day 4 has no silent or
vocalisation arm, and on the vocalisation day every sound is a recording.

Each index is computed twice, because the shared loader caps every visit at
10 s by default. The cap was added against a suspected fault, a visit left
open at the end of a block, which the maze rules out: a new block starts
only once the mouse has left the maze and entered again. The cap therefore
only shortens genuine long stays, and the thesis reports the uncapped
values:

    capped     visits capped at 10 s, as in run_batch_preference.py
    uncapped   no cap; rows longer than a block are still removed

The script also counts, where per-visit logs exist, how many visits over
10 s end within 1 s of their block's end, the visits such a fault would
produce. There are none.

Tests: per day, Wilcoxon signed-rank of each index against 0 across mice;
over days, the same test on each mouse's mean index.

Usage:
    python exp1_vocalisation_pi.py [DATA_DIR]     (the 8_arms_w_voc folder)
    python exp1_vocalisation_pi.py DATA_DIR --thesis-figures <thesis>/figures

Outputs (DATA_DIR/BATCH_ANALYSIS/vocalisation_pi/):
    vocalisation_pi.csv          every mouse, day and cap setting
    vocalisation_pi_stats.txt    the tests
    fig_vocalisation_pi.png/.pdf the three indices per day, every mouse
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SOUND_TRIALS = {2, 4, 6, 8}
ROIS = {f"ROI{i}" for i in range(1, 9)}
NOT_SYNTHETIC = {"silent", "vocalisation", "control", "unknown"}
INDICES = [("voc_vs_silent", "Vocalisation against\nthe silent arm"),
           ("synth_vs_silent", "Synthetic sounds against\nthe silent arm"),
           ("voc_vs_synth", "Vocalisation against\nsynthetic sounds")]
BOUNDARY_S = 1.0


def pi(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else np.nan


def session_indices(df, day, classify, voc_day):
    """Mean visit length per class in the sound blocks, and the three indices."""
    s = df[df["trial_ID"].isin(SOUND_TRIALS) & df["ROIs"].isin(ROIS)].copy()
    s["label"] = [classify(r, day) for _, r in s.iterrows()]
    if voc_day:
        s["cls"] = np.where(s["label"] == "silent", "silent", "vocalisation")
    else:
        s["cls"] = np.where(s["label"] == "silent", "silent",
                   np.where(s["label"] == "vocalisation", "vocalisation",
                   np.where(s["label"].isin(NOT_SYNTHETIC), "other", "synthetic")))
    means = {}
    for cls in ("vocalisation", "synthetic", "silent"):
        g = s[s["cls"] == cls]
        visits = g["visitation_count"].sum()
        means[cls] = g["time_spent"].sum() / visits if visits > 0 else np.nan
    return {"voc_ms": means["vocalisation"], "synth_ms": means["synthetic"],
            "silent_ms": means["silent"],
            "voc_vs_silent": pi(means["vocalisation"], means["silent"]),
            "synth_vs_silent": pi(means["synthetic"], means["silent"]),
            "voc_vs_synth": pi(means["vocalisation"], means["synthetic"])}


def boundary_visits(sess):
    """Visits over 10 s in the sound blocks, and how many end at a block's end."""
    try:
        dv = pd.read_csv(sess.detailed_visits_csv)
        tr = pd.read_csv(sess.trials_csv)
    except Exception:
        return None
    need = {"trial_ID", "sound_on_time", "time_spent_seconds"}
    if not need.issubset(dv.columns) or "end_trial_time" not in tr.columns:
        return None
    ends = tr.groupby("trial_ID")["end_trial_time"].first()
    dv = dv[dv["trial_ID"].isin(SOUND_TRIALS) & dv["ROI_visited"].astype(str).isin(ROIS)]
    long = dv[dv["time_spent_seconds"] > 10]
    if long.empty:
        return 0, 0
    end = long["sound_on_time"] + long["time_spent_seconds"]
    at_boundary = (ends.reindex(long["trial_ID"]).to_numpy() - end.to_numpy()) < BOUNDARY_S
    return len(long), int(at_boundary.sum())


def main():
    import preference_analysis_config as cfg
    from run_batch_preference import classify_stimulus

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR",
                        help="also write the thesis version as fig_exp1_vocalisation_pi")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "vocalisation_pi")
    os.makedirs(out, exist_ok=True)

    sessions = [s for s in cfg.discover_sessions() if s.day != "w1_d4"]
    rows, long_total, long_boundary = [], 0, 0
    for cap_ms, cap_name in ((10000, "capped"), (0, "uncapped")):
        cfg.INDIVIDUAL_VISIT_CLIP_MS = cap_ms      # read by the loader at call time
        for sess in sessions:
            df, source = cfg.load_session_visits(sess)
            if df is None:
                continue
            r = session_indices(df, sess.day, classify_stimulus,
                                voc_day=sess.day == "w2_vocalisations")
            rows.append({"mouse": sess.mouse_id, "day": sess.day, "cap": cap_name,
                         "source": source, **r})
            if cap_name == "uncapped" and source == "detailed_visits":
                b = boundary_visits(sess)
                if b:
                    long_total += b[0]
                    long_boundary += b[1]
    t = pd.DataFrame(rows)
    # a mouse that ran a day twice: keep one value, the mean
    t = t.groupby(["mouse", "day", "cap"], as_index=False).mean(numeric_only=True)
    t.to_csv(os.path.join(out, "vocalisation_pi.csv"), index=False)

    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    say(f"Visits over 10 s in the sound blocks, in sessions with per-visit logs: "
        f"{long_total}; ending within {BOUNDARY_S:g} s of their block's end: {long_boundary}.")
    days = [d for d in cfg.DAY_ORDER if d != "w1_d4"]
    for cap in ("capped", "uncapped"):
        c = t[t["cap"] == cap]
        say(f"\n=== visits {cap} ===")
        for col, name in INDICES:
            say(f"\n{name.replace(chr(10), ' ')}: median [quartiles], mice above 0, "
                "Wilcoxon against 0")
            for day in days:
                x = c.loc[c["day"] == day, col].dropna()
                if len(x) < 5:
                    continue
                p = wilcoxon(x).pvalue
                q1, q3 = np.percentile(x, [25, 75])
                say(f"  {cfg.DAY_SHORT[day]:10s} n={len(x):2d}  {x.median():+.3f} "
                    f"[{q1:+.3f}, {q3:+.3f}]  {int((x > 0).sum()):2d} above 0  p = {p:.4f}")
            per_mouse = c.groupby("mouse")[col].mean().dropna()
            if len(per_mouse) >= 5:
                p = wilcoxon(per_mouse).pvalue
                say(f"  {'all days':10s} n={len(per_mouse):2d}  {per_mouse.median():+.3f} "
                    f"(mean of each mouse)  {int((per_mouse > 0).sum()):2d} above 0  p = {p:.4f}")
    with open(os.path.join(out, "vocalisation_pi_stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    c = t[t["cap"] == "uncapped"]
    draw(c, days, {d: cfg.DAY_SHORT[d] for d in days},
         os.path.join(out, "fig_vocalisation_pi"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6.5,
                             "ytick.labelsize": 6.5, "pdf.fonttype": 42}):
            draw(c, days, THESIS_DAYS,
                 os.path.join(args.thesis_figures, "fig_exp1_vocalisation_pi"), thesis=True)
    print(f"\nAll outputs saved to:\n  {out}")


# the thesis names each day by its stimuli; the two interval days used
# different sets of intervals
THESIS_DAYS = {"w1_d1": "TEM", "w1_d2": "Intervals 1", "w1_d3": "Intervals 2",
               "w2_sequences": "Sequences", "w2_vocalisations": "Vocalisations"}


def draw(c, days, day_names, path, thesis=False):
    """The three indices per day, every mouse, the median and a test against 0."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4) if thesis else (10, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for k, (ax, (col, name)) in enumerate(zip(axes, INDICES)):
        shown = [d for d in days if c.loc[c["day"] == d, col].notna().sum() >= 5]
        for i, day in enumerate(shown):
            x = c.loc[c["day"] == day, col].dropna()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=5 if thesis else 8,
                       color="#9E9E9E", lw=0, zorder=2)
            ax.plot([i - 0.25, i + 0.25], [x.median()] * 2, color="#1F3864", lw=2, zorder=3)
            p = wilcoxon(x).pvalue
            mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if mark:
                ax.text(i, 1.02, mark, ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([day_names[d] for d in shown], rotation=30, ha="right")
        ax.set_title(f"({'abc'[k]}) {name[0].lower() + name[1:]}" if thesis else name,
                     loc="left" if thesis else "center", fontsize=7 if thesis else 8)
        ax.set_ylim(-1.05, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Preference index")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
