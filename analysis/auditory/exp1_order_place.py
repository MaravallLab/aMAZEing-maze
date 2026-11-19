#!/usr/bin/env python3
"""
Experiment 1: Day Order and Place Bias
======================================

Two checks on what the preference indices of Experiment 1 can mean.

Day order. Every mouse was meant to run the six stimulus sets in the same
order; the session dates give the order each actually ran. Because a set's
day and its stimuli coincide, a change from set to set may be habituation to
the maze rather than a response to the stimuli. The indices are therefore
also followed block by block through each session: if avoidance of the
synthetic sounds fades within a day, and comes back the next, habituation
within a session is at work; if it only falls from one day to the next, the
order and the stimuli cannot be told apart.

Place bias. Stimuli moved between arms at every block, so a preference for
an arm, as a place, cannot follow a stimulus. Each arm's share of the time
spent in the arms is measured in the silent habituation block, in the silent
blocks and in the sound blocks, and the consistency of a mouse's arm shares
between the habituation and the sound blocks of the same session
(Spearman correlation over the eight arms) shows how strongly it favours
places regardless of what they play.

The usual silent arm. Only blocks 2 to 4 were reshuffled: the first sound
block of a session always opened on the same layout, with the silent arm in
arm 2 for three days running (arm 7 with the sequences, arm 8 on the
vocalisation day). Had the mice learned that arm 2 was where silence was,
arm 2 would gain on them over the days, already in the silent habituation
block, and in blocks 2 to 4 they would go first to the arm that had been
silent in block 1. Both are tested, the first entries from the per-visit
logs against what each arm's share of the habituation entries predicts; so is
the silent arm of every block against the first entry of the next. And each
arm's share in the shuffled sound blocks is set against its habituation
share: a ratio above 1 for the favoured arms means the preference for places
sharpens once sound begins.

Usage:
    python exp1_order_place.py [DATA_DIR] [--thesis-figures DIR]

Outputs (DATA_DIR/BATCH_ANALYSIS/order_place/):
    order.csv, blocks.csv, arms.csv, arm_trials.csv, first_entries.csv, stats.txt
    fig_order.png/.pdf, fig_place.png/.pdf
    and with --thesis-figures, fig_exp1_order and fig_exp1_place
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp1_stimulus_pi import SETS

ROIS = [f"ROI{i}" for i in range(1, 9)]
CANONICAL = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
SOUND_TRIALS = [2, 4, 6, 8]
SILENT_TRIALS = [3, 5, 7, 9]
NOT_SYNTHETIC = {"silent", "vocalisation", "control", "unknown"}


def session_date(folder):
    m = re.search(r"time_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})", os.path.basename(folder))
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}") if m else pd.NaT


def pi(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else np.nan


def mean_visit(g):
    v = g["visitation_count"].sum()
    return g["time_spent"].sum() / v if v > 0 else np.nan


def block_indices(df, day, classify):
    """Synthetic against silent, and vocalisation against synthetic, in each sound block."""
    voc_day = day == "w2_vocalisations"
    rows = []
    for tr in SOUND_TRIALS:
        b = df[(df["trial_ID"] == tr) & df["ROIs"].isin(ROIS)].copy()
        if b.empty:
            continue
        lab = [classify(r, day) for _, r in b.iterrows()]
        b["cls"] = ["silent" if l == "silent" else "vocalisation" if (l == "vocalisation" or voc_day)
                    else ("synthetic" if l not in NOT_SYNTHETIC else "other") for l in lab]
        m = {c: mean_visit(b[b["cls"] == c]) for c in ("silent", "synthetic", "vocalisation")}
        rows.append({"block": SOUND_TRIALS.index(tr) + 1,
                     "synth_vs_silent": pi(m["synthetic"], m["silent"]),
                     "voc_vs_silent": pi(m["vocalisation"], m["silent"]),
                     "voc_vs_synth": pi(m["vocalisation"], m["synthetic"])})
    return rows


def arm_shares(df):
    """Each arm's share of the time in the arms, in habituation, silent and sound blocks."""
    out = {}
    for phase, trials in (("habituation", [1]), ("silent", SILENT_TRIALS), ("sound", SOUND_TRIALS)):
        g = df[df["trial_ID"].isin(trials) & df["ROIs"].isin(ROIS)]
        t = np.array([g.loc[g["ROIs"] == r, "time_spent"].sum() for r in ROIS], float)
        out[phase] = t / t.sum() if t.sum() > 0 else np.full(8, np.nan)
    return out


def arm_trials(df):
    """Each arm's share of the entries and of the time in the arms, trial by trial."""
    d = df[df["ROIs"].isin(ROIS)].assign(n=lambda x: x["visitation_count"].fillna(0),
                                          t=lambda x: x["time_spent"].fillna(0))
    rows = []
    for trial, g in d.groupby("trial_ID"):
        n = g.groupby("ROIs")["n"].sum().reindex(ROIS, fill_value=0)
        t = g.groupby("ROIs")["t"].sum().reindex(ROIS, fill_value=0)
        for arm in ROIS:
            rows.append({"trial": int(trial), "arm": arm,
                         "n_share": n[arm] / n.sum() if n.sum() else np.nan,
                         "t_share": t[arm] / t.sum() if t.sum() else np.nan})
    return rows


def silent_arms(df, day, classify):
    """The silent arm of each sound block."""
    out = {}
    for tr in SOUND_TRIALS:
        b = df[(df["trial_ID"] == tr) & df["ROIs"].isin(ROIS)]
        arms = [r["ROIs"] for _, r in b.iterrows() if classify(r, day) == "silent"]
        if len(arms) == 1:
            out[SOUND_TRIALS.index(tr) + 1] = arms[0]
    return out


def first_entries(folder):
    """The first arm entered in each trial, from the per-visit log when there is one."""
    logs = glob.glob(os.path.join(folder, "*detailed_visits*.csv"))
    if not logs:
        return {}
    v = pd.read_csv(logs[0])
    v = v[v["ROI_visited"].isin(ROIS)].sort_values("sound_on_time")
    return {int(t): g["ROI_visited"].iloc[0] for t, g in v.groupby("trial_ID")}


# -- figures -----------------------------------------------------------

def fig_order(blocks, path, thesis=False):
    sets = [s for s in SETS.values() if s in set(blocks["set"])]
    panels = [("synth_vs_silent", "Synthetic sounds\nagainst the silent arm"),
              ("voc_vs_synth", "Vocalisation against\nsynthetic sounds"),
              ("voc_vs_silent", "Vocalisation against\nthe silent arm")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5) if thesis else (14, 3.4), sharey=True)
    for k, (ax, (col, name)) in enumerate(zip(axes, panels)):
        x0 = 0
        ticks, labels = [], []
        for s in sets:
            g = blocks[blocks["set"] == s].groupby("block")[col]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            if med.notna().sum() == 0:
                continue                  # a set without the index takes no space

            xs = x0 + med.index.to_numpy()
            ax.fill_between(xs, q1.to_numpy(), q3.to_numpy(), color="#BDBDBD", alpha=0.5, lw=0)
            ax.plot(xs, med.to_numpy(), color="#1F3864", marker="o", ms=3, lw=1.2)
            ticks.append(x0 + 2.5)
            labels.append(s)
            ax.axvline(x0 + 4.5, color="#E0E0E0", lw=0.6)
            x0 += 5
        ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"({'abc'[k]}) {name[0].lower() + name[1:]}" if thesis else name,
                     loc="left" if thesis else "center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Preference index, block by block\n(median and quartiles over mice)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_place(arms, consistency, path, thesis=False):
    """Each arm's share of the time; the batch version adds the histogram of the
    habituation against sound-block correlations, which the thesis gives in the text."""
    if thesis:
        fig, ax = plt.subplots(1, 1, figsize=(4.4, 2.4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    colours = {"habituation": "#9E9E9E", "silent": "#424242", "sound": "#E65100"}
    for j, phase in enumerate(("habituation", "silent", "sound")):
        g = arms[arms["phase"] == phase].groupby("arm")["share"]
        m, se = g.mean() * 100, g.sem() * 100
        x = np.arange(8) + (j - 1) * 0.26
        ax.bar(x, m.reindex(ROIS), yerr=se.reindex(ROIS), width=0.25, color=colours[phase],
               label=phase if phase != "silent" else "silent blocks",
               error_kw={"lw": 0.6, "capsize": 1})
    ax.axhline(12.5, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_ylim(0, 27)                    # room for the legend above the bars
    ax.set_xticks(range(8))
    ax.set_xticklabels([r.replace("ROI", "") for r in ROIS])
    ax.set_xlabel("Arm")
    ax.set_ylabel("Share of the time in the arms (%)")
    ax.legend(fontsize=6 if thesis else 8, frameon=False, ncol=3, loc="upper left")
    if not thesis:
        ax.set_title("Each arm, over all sessions")
        ax = axes[1]
        c = consistency.dropna()
        ax.hist(c, bins=np.linspace(-1, 1, 21), color="#1F3864", alpha=0.8)
        ax.axvline(0, color="black", lw=0.6, ls=(0, (4, 2)))
        ax.axvline(np.median(c), color="#E65100", lw=1.2)
        ax.set_xlabel("Spearman correlation of arm shares,\nhabituation against sound blocks")
        ax.set_ylabel("Sessions")
        ax.set_title("The same arms favoured")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


# -- main --------------------------------------------------------------

def main():
    import preference_analysis_config as cfg
    from run_batch_preference import classify_stimulus
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    cfg.INDIVIDUAL_VISIT_CLIP_MS = 0
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "order_place")
    os.makedirs(out, exist_ok=True)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    sessions = cfg.discover_sessions()
    order_rows, block_rows, arm_rows, cons = [], [], [], []
    trial_rows, first_rows = [], []
    for sess in sessions:
        order_rows.append({"mouse": sess.mouse_id, "day": sess.day,
                           "date": session_date(sess.session_folder)})
        df, _ = cfg.load_session_visits(sess)
        if df is None:
            continue
        name = os.path.basename(sess.session_folder)
        if df.loc[df["ROIs"].isin(ROIS), "visitation_count"].fillna(0).sum() > 0:
            for r in arm_trials(df):          # a session with no visit at all failed
                trial_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day],
                                   "session": name, **r})
        silent = silent_arms(df, sess.day, classify_stimulus)
        for trial, arm in first_entries(sess.session_folder).items():
            first_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day], "session": name,
                               "trial": trial, "first": arm,
                               "silent_block1": silent.get(1),
                               "silent_previous": silent.get((trial - 2) // 2)
                               if trial in (4, 6, 8) else None})
        for r in block_indices(df, sess.day, classify_stimulus):
            block_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day], **r})
        shares = arm_shares(df)
        for phase, sh in shares.items():
            for arm, v in zip(ROIS, sh):
                arm_rows.append({"mouse": sess.mouse_id, "set": SETS[sess.day], "phase": phase,
                                 "arm": arm, "share": v})
        if np.isfinite(shares["habituation"]).all() and np.isfinite(shares["sound"]).all():
            cons.append(spearmanr(shares["habituation"], shares["sound"])[0])

    # -- day order
    order = pd.DataFrame(order_rows).sort_values(["mouse", "date"])
    first = order.groupby(["mouse", "day"], as_index=False)["date"].min()
    seq = first.sort_values("date").groupby("mouse")["day"].apply(list)
    in_order = seq.apply(lambda s: s == [d for d in CANONICAL if d in s])
    say(f"{len(seq)} mice; {int(in_order.sum())} ran their sets in the planned order")
    for mouse, s in seq[~in_order].items():
        say(f"  {mouse}: " + ", ".join(SETS[d] for d in s))
    gaps = first.sort_values(["mouse", "date"]).groupby("mouse")["date"].diff().dt.days.dropna()
    say(f"days between successive sets: median {gaps.median():.0f}, "
        f"range {gaps.min():.0f} to {gaps.max():.0f}")
    order.to_csv(os.path.join(out, "order.csv"), index=False)

    # -- block by block
    blocks = pd.DataFrame(block_rows).groupby(["mouse", "set", "block"], as_index=False).mean(
        numeric_only=True)
    blocks.to_csv(os.path.join(out, "blocks.csv"), index=False)
    say("\nWithin each session: index ~ block (1 to 4) + (1 | mouse); slope per block")
    for col, name in (("synth_vs_silent", "synthetic against silent"),
                      ("voc_vs_synth", "vocalisation against synthetic"),
                      ("voc_vs_silent", "vocalisation against silent")):
        say(f"  {name}")
        for s in SETS.values():
            d = blocks[blocks["set"] == s].dropna(subset=[col])
            if d["mouse"].nunique() < 8:
                continue
            fit = smf.mixedlm(f"{col} ~ block", d, groups=d["mouse"]).fit()
            med = d.groupby("block")[col].median().round(3).tolist()
            say(f"    {s:13s} medians by block {med}  slope {fit.params['block']:+.3f} "
                f"per block, p = {fit.pvalues['block']:.3f}")

    # -- place
    arms = pd.DataFrame(arm_rows)
    arms.to_csv(os.path.join(out, "arms.csv"), index=False)
    say("\nPlace: each arm's share of the time in the arms (%), mean over sessions")
    tab = arms.groupby(["phase", "arm"])["share"].mean().unstack() * 100
    say(tab.round(1).to_string())
    for phase in ("habituation", "silent", "sound"):
        w = arms[arms["phase"] == phase].pivot_table(index=["mouse", "set"], columns="arm",
                                                     values="share").dropna()
        stat, p = friedmanchisquare(*[w[a] for a in ROIS])
        fav = w.max(axis=1) * 100
        say(f"  {phase:12s} arms differ (Friedman chi2(7) = {stat:.1f}, p = {p:.2g}); a session's "
            f"favourite arm took a median {fav.median():.0f}% (quartiles {fav.quantile(.25):.0f} "
            f"to {fav.quantile(.75):.0f}%) against 12.5%")
    c = pd.Series(cons).dropna()
    say(f"  consistency, habituation against sound blocks: median Spearman {c.median():+.2f} over "
        f"{len(c)} sessions, {int((c > 0).sum())} above 0, Wilcoxon p = {wilcoxon(c).pvalue:.2g}")

    # -- the usual silent arm of the first block
    at = pd.DataFrame(trial_rows)
    at.to_csv(os.path.join(out, "arm_trials.csv"), index=False)
    hab = at[at["trial"] == 1].set_index(["session", "arm"])
    say("\nThe usual silent arm (arm 2 in block 1 of TEM and the first two interval sets)")
    say("  habituation share of each arm, before any sound of the day (median over sessions, %):")
    for measure, label in (("t_share", "time"), ("n_share", "entries")):
        tab = at[at["trial"] == 1].groupby(["set", "arm"])[measure].median().unstack() * 100
        say(f"  {label}:")
        say(tab.reindex([s for s in SETS.values() if s in tab.index]).round(1).to_string())
    w = at[(at["trial"] == 1) & (at["arm"] == "ROI2")].pivot_table(index="mouse", columns="set",
                                                                 values="t_share")
    for later in ("Intervals 1", "Intervals 2"):
        pair = w[["TEM", later]].dropna()
        say(f"  arm 2, habituation time share, TEM (the first day) against {later}, same mice: "
            f"{pair['TEM'].median() * 100:.1f}% and {pair[later].median() * 100:.1f}%, "
            f"n = {len(pair)}, Wilcoxon p = {wilcoxon(pair['TEM'], pair[later]).pvalue:.3f}")
    shuffled = at[at["trial"].isin([4, 6, 8])].groupby(["session", "arm"])["t_share"].mean()
    ratio = (shuffled / hab["t_share"]).replace([np.inf, -np.inf], np.nan).dropna()
    say("  each arm's time share in the shuffled sound blocks over its habituation share "
        "(median over sessions):")
    say("    " + "  ".join(f"{a.replace('ROI', 'arm ')} {v:.2f}"
                           for a, v in ratio.groupby(level="arm").median().items()))

    fe = pd.DataFrame(first_rows)
    fe.to_csv(os.path.join(out, "first_entries.csv"), index=False)
    say("  first arm entered in blocks 2 to 4, sessions with per-visit logs, against the arm's "
        "share of the habituation entries:")
    for key, label in (("silent_block1", "the silent arm of block 1"),
                       ("silent_previous", "the silent arm of the block before")):
        f = fe[fe["trial"].isin([4, 6, 8])].dropna(subset=[key]).copy()
        f["hit"] = f["first"] == f[key]
        f["expected"] = [hab["n_share"].get((s_, a), np.nan) for s_, a in zip(f["session"], f[key])]
        pm = f.groupby("mouse")[["hit", "expected"]].mean().dropna()
        diff = pm["hit"] - pm["expected"]
        say(f"    {label}: entered first at {f['hit'].mean() * 100:.0f}% of {len(f)} block starts "
            f"({f['session'].nunique()} sessions), {f['expected'].mean() * 100:.0f}% expected; "
            f"per mouse {diff.median() * 100:+.0f} points, {int((diff > 0).sum())} of {len(diff)} "
            f"above, Wilcoxon p = {wilcoxon(diff).pvalue:.3f}")
        if key == "silent_block1":
            for s_, g in f.groupby("set"):
                say(f"      {s_:13s} {g['hit'].mean() * 100:3.0f}% against "
                    f"{g['expected'].mean() * 100:.0f}% ({len(g)} block starts)")
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    fig_order(blocks, os.path.join(out, "fig_order"))
    fig_place(arms, c, os.path.join(out, "fig_place"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            fig_order(blocks, os.path.join(args.thesis_figures, "fig_exp1_order"), True)
            fig_place(arms, c, os.path.join(args.thesis_figures, "fig_exp1_place"), True)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
