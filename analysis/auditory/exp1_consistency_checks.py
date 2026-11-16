#!/usr/bin/env python3
"""
Experiment 1: Consistency Checks on Consonance and Place
========================================================

Two questions raised on Chapter 4 of the thesis.

1. Consonance in single mice. Over the group, consonant intervals were neither
   preferred nor avoided on any of the three interval sets. Were some mice
   consistent all the same, preferring consonant (or dissonant) intervals on
   every set? Each mouse has one consonant against dissonant index per interval
   set (BATCH_ANALYSIS/stimulus_pi/contrasts.csv, from exp1_stimulus_pi.py).
   If mice differ consistently, a mouse's indices agree from set to set, and the
   spread of the mouse means exceeds what chance gives. Tested by:
     - the Spearman correlation of the indices between each pair of sets;
     - a permutation test of the variance of the mouse means, shuffling the
       indices between mice within each set (the null of no consistent mouse);
     - the number of mice with the same sign on all three sets, against the
       number expected under the same shuffling.

2. Does sound magnify the place preference? The mice favoured arms 1, 2, 7 and
   8 in silence and in sound alike. If sound made favoured arms more favoured
   and avoided arms more avoided, the favoured arms take a larger share of the
   time in the sound blocks than in the silent blocks of the same session (the
   arms 3 to 6 then take a smaller share, as the shares sum to one). From the
   visits of every session (loaded as in exp1_order_place.py), each session
   gives the favoured arms' share of the time in the arms in the sound blocks
   (trials 2, 4, 6, 8) minus their share in the silent blocks (trials 3, 5, 7,
   9), the time pooled over the blocks of each kind, so that a block with
   hardly any time in the arms counts for little; a session with no time in
   the arms in either kind of block is left out. Mice are the unit, by the
   Wilcoxon signed rank test on each mouse's mean difference. Three versions:
     a. the favoured arms are arms 1, 2, 7 and 8;
     b. the favoured arms are each session's four arms with the largest share
        of its silent habituation block (trial 1), which defines them without
        the blocks being compared;
     c. as (a), with only the reshuffled sound blocks (trials 4, 6, 8), as the
        first sound block of a session opened on a fixed layout.

Usage:
    python exp1_consistency_checks.py [DATA_DIR] [--perms 10000] [--thesis-figures DIR]

Output: printed, and in DATA_DIR/BATCH_ANALYSIS/consistency/ stats.txt,
consonance_by_mouse.csv, place_sound_silent.csv and fig_consonance_by_mouse;
with --thesis-figures, fig_exp1_consonance_mice in the thesis style.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SETS = ["Intervals 1", "Intervals 2", "Intervals 3"]
CONTRAST = "consonant vs dissonant"
ROIS = [f"ROI{i}" for i in range(1, 9)]
FAVOURED = ["ROI1", "ROI2", "ROI7", "ROI8"]
SOUND = [2, 4, 6, 8]
SHUFFLED = [4, 6, 8]
SILENT = [3, 5, 7, 9]
SEED = 20261116


def consonance(data_dir, perms, say):
    c = pd.read_csv(os.path.join(data_dir, "BATCH_ANALYSIS", "stimulus_pi", "contrasts.csv"))
    w = c[(c["contrast"] == CONTRAST) & c["set"].isin(SETS)].pivot_table(
        index="mouse", columns="set", values="pi")[SETS]
    say("1. Consonant against dissonant, mouse by mouse")
    for s in SETS:
        say(f"   {s}: {w[s].notna().sum()} mice, median index {w[s].median():+.3f}")
    say("   Spearman correlation between sets, mice with both:")
    for i in range(3):
        for j in range(i + 1, 3):
            pair = w[[SETS[i], SETS[j]]].dropna()
            r, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            say(f"     {SETS[i]} with {SETS[j]}: rho = {r:+.2f}, n = {len(pair)}, p = {p:.2f}")

    full = w.dropna()
    n = len(full)
    x = full.to_numpy()
    observed_var = x.mean(axis=1).var(ddof=1)
    signs = np.sign(x)
    same = int(((signs > 0).all(axis=1) | (signs < 0).all(axis=1)).sum())
    pos = int((signs > 0).all(axis=1).sum())
    neg = int((signs < 0).all(axis=1).sum())
    rng = np.random.default_rng(SEED)
    null_var = np.empty(perms)
    null_same = np.empty(perms)
    for k in range(perms):
        y = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        null_var[k] = y.mean(axis=1).var(ddof=1)
        sy = np.sign(y)
        null_same[k] = ((sy > 0).all(axis=1) | (sy < 0).all(axis=1)).sum()
    p_var = (1 + (null_var >= observed_var).sum()) / (perms + 1)
    p_same = (1 + (null_same >= same).sum()) / (perms + 1)
    # one way intraclass correlation, ICC(1), of the index across the three sets
    k = x.shape[1]
    msb = k * x.mean(axis=1).var(ddof=1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    say(f"   {n} mice ran all three sets")
    say(f"   variance of the mouse means {observed_var:.4f}, against {null_var.mean():.4f} "
        f"when the indices are shuffled between mice ({perms} shuffles), p = {p_var:.2f}")
    say(f"   intraclass correlation ICC(1) = {icc:+.2f}")
    say(f"   same sign on all three sets: {same} of {n} mice ({pos} consonant, {neg} dissonant), "
        f"against {null_same.mean():.1f} expected, p = {p_same:.2f}")
    return w, null_var, observed_var, p_var


def fig_consonance(w, null_var, observed_var, p_var, path):
    """(a) Each mouse's consonant against dissonant index on the three interval
    sets, joined by a line; mice with the same sign on all three are coloured.
    (b) The variance of the mouse means against its distribution when the
    indices are shuffled between mice within each set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp1_stimulus_pi import COLOURS
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(SETS))
    full = w.dropna()
    for _, row in full.iterrows():
        v = row.to_numpy()
        colour = (COLOURS["consonant"] if (v > 0).all() else
                  COLOURS["dissonant"] if (v < 0).all() else "#BDBDBD")
        grey = colour == "#BDBDBD"
        ax.plot(x, v, "-o", color=colour, lw=0.6 if grey else 0.9, ms=2.5,
                alpha=0.8 if grey else 1, zorder=1 if grey else 2)
    partial = w[w.isna().any(axis=1)]
    for _, row in partial.iterrows():
        ax.plot(x, row.to_numpy(), "o", color="#BDBDBD", ms=2.5, mfc="white", zorder=1)
    ax.plot(x, w.median().to_numpy(), "_", color="black", ms=16, mew=1.6, zorder=3)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Interval set {s[-1]}" for s in SETS])
    ax.set_xlim(-0.4, len(SETS) - 0.6)
    ax.set_ylabel("Consonant against dissonant index")
    ax.set_title("(a) each mouse on the three interval sets", loc="left")
    bx.hist(null_var, bins=40, color="#BDBDBD")
    bx.axvline(observed_var, color="black", lw=1)
    bx.text(observed_var, bx.get_ylim()[1] * 0.95, f"  observed\n  p = {p_var:.2f}",
            va="top", ha="left", fontsize=6)
    bx.xaxis.set_major_locator(plt.MaxNLocator(4))
    bx.set_xlabel("Variance of the mouse means")
    bx.set_ylabel("Shuffles")
    bx.set_title("(b) against indices shuffled between mice", loc="left")
    for a in (ax, bx):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def arm_times(cfg):
    """Seconds in each arm, trial by trial, for every session with any visit."""
    from exp1_stimulus_pi import SETS as DAY_TO_SET
    rows = []
    for sess in cfg.discover_sessions():
        df, _ = cfg.load_session_visits(sess)
        if df is None or df.loc[df["ROIs"].isin(ROIS), "visitation_count"].fillna(0).sum() == 0:
            continue
        d = df[df["ROIs"].isin(ROIS)]
        t = d.groupby(["trial_ID", "ROIs"])["time_spent"].sum().unstack().reindex(columns=ROIS)
        rows.append((sess.mouse_id, DAY_TO_SET[sess.day], os.path.basename(sess.session_folder),
                     t.fillna(0)))
    return rows


def place(cfg, say):
    say("\n2. Does sound magnify the place preference?")
    say("   favoured arms' share of the time in the arms, sound blocks against silent blocks,")
    say("   time pooled over the blocks of each kind in a session")
    rows = []
    for mouse, set_, session, t in arm_times(cfg):
        hab_top = (t.loc[1].sort_values(ascending=False).index[:4].tolist()
                   if 1 in t.index and t.loc[1].sum() > 0 else None)

        def share(trials, arms):
            if arms is None:
                return np.nan
            b = t.reindex(trials).fillna(0)
            total = b.to_numpy().sum()
            return b[arms].to_numpy().sum() / total if total > 0 else np.nan

        rows.append({"mouse": mouse, "set": set_, "session": session,
                     "a_sound": share(SOUND, FAVOURED), "a_silent": share(SILENT, FAVOURED),
                     "b_sound": share(SOUND, hab_top), "b_silent": share(SILENT, hab_top),
                     "c_sound": share(SHUFFLED, FAVOURED), "c_silent": share(SILENT, FAVOURED)})
    d = pd.DataFrame(rows)
    out = []
    for v, name in (("a", "arms 1, 2, 7 and 8"),
                    ("b", "each session's four favourite arms in habituation"),
                    ("c", "arms 1, 2, 7 and 8, reshuffled sound blocks only")):
        s = d.dropna(subset=[f"{v}_sound", f"{v}_silent"]).copy()
        s["diff"] = s[f"{v}_sound"] - s[f"{v}_silent"]
        m = s.groupby("mouse")[["diff", f"{v}_sound", f"{v}_silent"]].mean()
        p = wilcoxon(m["diff"]).pvalue
        say(f"   ({v}) {name}:")
        say(f"       {len(s)} sessions, {len(m)} mice; share {m[f'{v}_silent'].median() * 100:.1f}% "
            f"in silent blocks and {m[f'{v}_sound'].median() * 100:.1f}% in sound blocks "
            f"(medians over mice)")
        say(f"       difference per mouse, median {m['diff'].median() * 100:+.1f} points, "
            f"{int((m['diff'] > 0).sum())} of {len(m)} mice above zero, Wilcoxon p = {p:.3f}; "
            f"{int((s['diff'] > 0).sum())} of {len(s)} sessions above zero")
        if v == "a":
            for set_, g in s.groupby("set"):
                say(f"         {set_:13s} median {g['diff'].median() * 100:+.1f} points over "
                    f"{len(g)} sessions, {int((g['diff'] > 0).sum())} above zero")
        out.append(s.assign(version=v))
    return pd.concat(out)


def main():
    import preference_analysis_config as cfg
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    parser.add_argument("--perms", type=int, default=10000)
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    cfg.INDIVIDUAL_VISIT_CLIP_MS = 0
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "consistency")
    os.makedirs(out, exist_ok=True)
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    w, null_var, observed_var, p_var = consonance(args.data_dir, args.perms, say)
    w.to_csv(os.path.join(out, "consonance_by_mouse.csv"))
    fig_consonance(w, null_var, observed_var, p_var, os.path.join(out, "fig_consonance_by_mouse"))
    if args.thesis_figures:
        import matplotlib.pyplot as plt
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            fig_consonance(w, null_var, observed_var, p_var,
                           os.path.join(args.thesis_figures, "fig_exp1_consonance_mice"))
    place(cfg, say).to_csv(os.path.join(out, "place_sound_silent.csv"), index=False)
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
