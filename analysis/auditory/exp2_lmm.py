#!/usr/bin/env python3
"""
Experiment 2: Grammar and Environment, With Place Controlled
============================================================

A reanalysis of the grammar test days in the framework of Experiment 1. The
earlier analysis (model_validation/, and its output folder) is left as it was.

Why a reanalysis. The first sound block of every session was not shuffled
(experiments.py uses the identity permutation for the first active block): in
every session, on both days and in both counterbalancing groups, the EE grammar
at its dominant, secondary and rare levels sat in arms 1, 2 and 3, and the SC
grammar in arms 4, 5 and 6, with the vocalisation in arm 7 and the silent arm in
arm 8. Arms 1, 2, 7 and 8 are the arms the mice favour anyway, from the
habituation day before any sound. In block 1, then, EE against SC is arms 1 to 3
against arms 4 to 6. Only blocks 2 to 4 were reshuffled.

Everything is therefore given for the shuffled blocks 2 to 4 (primary), for all
four blocks, and for block 1 alone, with day 1 primary and day 2 separate:

  block by block: the EE-SC index of time, and the same index for arms 1 to 3
    against arms 4 to 6 whatever they played (place alone), with the habituation
    day as a reference;
  per mouse: EE against SC as an index of total time (the convention of the
    earlier analysis), of mean visit length (staying, the thesis index) and of
    entries per arm (entering), overall and by level of predictability; the
    vocalisation and silent arms against the grammar arms; and EE-SC in each
    counterbalancing group (a preference for one grammar's sound would flip sign
    between groups);
  mixed models (exp2_lmm.R, lme4), with the arm as a random effect.

Exclusions are those of the earlier analysis (model_validation/data_loading.py,
read only): mouse 13533 (light artefact), mice 13603 and 13604 (cage 6339,
identity uncertain and placed with males by mistake), the session with EE and SC swapped,
and duplicate filings.

Usage:
    python exp2_lmm.py [GRAMMAR_DIR] [--thesis-figures DIR]

Outputs (GRAMMAR_DIR/BATCH_ANALYSIS/exp2/):
    arm_blocks.csv, habituation_arms.csv, stats.txt, lmm_results.txt,
    lmm_contrasts.csv, fig_blocks.png/.pdf
    and with --thesis-figures, fig_exp2_blocks
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp1_lmm import rscript
from model_validation.data_loading import load_arm_blocks, EXCLUDED_ANIMALS

DAYS = {"day_1": "day 1", "day_2": "day 2"}
TIERS = ["dominant", "secondary", "rare"]
SUBSETS = {"shuffled": "shuffled blocks 2 to 4", "all": "all four blocks",
           "first": "block 1 alone"}


def subset(d, which):
    return {"all": d, "shuffled": d[d["block"] > 1], "first": d[d["block"] == 1]}[which]


SESSIONS_RUN_IN_ERROR = {
    # day-1 session filed as 13672: stopped after a mix-up of mice (the
    # experimenter's note); no arm was entered and the mouse was not retested
    "13672": "day-1 session run in error, no arm entered",
}


def build_table(root):
    res = load_arm_blocks(root)
    d = res.arm_blocks
    d = d[d["day"].isin(DAYS) & ~d["mouse"].astype(str).isin(SESSIONS_RUN_IN_ERROR)]
    d = d.drop(columns=["melodies"]).copy()
    d["arm"] = d["roi"].astype(str).str.extract(r"(\d)")[0].astype(int)
    d["cls"] = d["arm_type"]
    d["cell"] = np.where(d["cls"] == "grammar", d["environment"] + " " + d["tier"], d["cls"])
    d["mean_s"] = np.where(d["visits"] > 0, d["time_spent_s"] / d["visits"].clip(lower=1), np.nan)
    return d, res.report


def habituation_arms(root):
    """Each arm's share of the time on the habituation day (no sound)."""
    rows = []
    for tp in glob.glob(os.path.join(root, "habituation", "*", "trials_*.csv")):
        m = re.search(r"mouse(\d+)", os.path.basename(os.path.dirname(tp)))
        if m is None or m.group(1) in EXCLUDED_ANIMALS:
            continue
        t = pd.read_csv(tp)
        t = t[t["trial_ID"] == 1]
        t["arm"] = t["ROIs"].astype(str).str.extract(r"(\d)")[0].astype(float)
        t["time"] = pd.to_numeric(t["time_spent"], errors="coerce").fillna(0)
        tot = t["time"].sum()
        if tot > 0:
            for arm, v in t.groupby("arm")["time"].sum().items():
                rows.append({"mouse": m.group(1), "arm": int(arm), "share": v / tot})
    return pd.DataFrame(rows)


def index(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and a + b > 0 else np.nan


def pooled(sub, mask_a, mask_b, measure):
    """Index of a against b for one mouse: total time, mean visit length or entries per arm."""
    a, b = sub[mask_a(sub)], sub[mask_b(sub)]
    if measure == "time":
        return index(a["time_spent_s"].sum(), b["time_spent_s"].sum())
    if measure == "stay":
        va, vb = a["visits"].sum(), b["visits"].sum()
        return index(a["time_spent_s"].sum() / va if va else np.nan,
                     b["time_spent_s"].sum() / vb if vb else np.nan)
    return index(a["visits"].mean(), b["visits"].mean())          # entries per arm-block


def per_mouse(d, mask_a, mask_b, measure):
    return d.groupby("mouse").apply(lambda s: pooled(s, mask_a, mask_b, measure),
                                    include_groups=False).dropna()


def is_env(env, tier=None):
    if tier is None:
        return lambda s: s["environment"] == env
    return lambda s: (s["environment"] == env) & (s["tier"] == tier)


def is_grammar(grammar, tier=None):
    if tier is None:
        return lambda s: s["grammar"] == grammar
    return lambda s: (s["grammar"] == grammar) & (s["tier"] == tier)


def is_cls(cls):
    return lambda s: s["cls"] == cls


def arms(lo, hi):
    return lambda s: s["arm"].between(lo, hi)


def describe(x):
    return (f"median {x.median():+.3f}, mean {x.mean():+.3f}, {int((x > 0).sum())} of {len(x)} "
            f"above 0, Wilcoxon p = {wilcoxon(x).pvalue:.4f}" if len(x) >= 5 else "too few")


def draw_blocks(d, hab, path, thesis=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5) if thesis else (10, 3.6), sharey=True)
    hab_idx = hab.pivot_table(index="mouse", columns="arm", values="share").fillna(0)
    hab_place = ((hab_idx[[1, 2, 3]].sum(axis=1) - hab_idx[[4, 5, 6]].sum(axis=1))
                 / (hab_idx[[1, 2, 3, 4, 5, 6]].sum(axis=1)))
    for ax, (day, label) in zip(axes, DAYS.items()):
        x = d[d["day"] == day]
        ee, pl = [], []
        for b in (1, 2, 3, 4):
            y = x[x["block"] == b]
            e = per_mouse(y, is_env("EE"), is_env("SC"), "time")
            p = per_mouse(y, arms(1, 3), arms(4, 6), "time")
            ee.append((e.mean(), e.sem()))
            pl.append((p.mean(), p.sem()))
        xs = np.arange(1, 5)
        ax.axvspan(0.6, 1.4, color="#FFF3E0", zorder=0, lw=0)
        ax.errorbar(xs - 0.06, [v[0] for v in ee], [v[1] for v in ee], color="#1565C0", marker="o",
                    ms=3.5, capsize=2, lw=1.2, label="EE grammar against SC grammar")
        ax.errorbar(xs + 0.06, [v[0] for v in pl], [v[1] for v in pl], color="#9E9E9E", marker="s",
                    ms=3.5, capsize=2, lw=1.2, ls="--",
                    label="arms 1-3 against arms 4-6, whatever they played")
        ax.axhline(hab_place.mean(), color="#E65100", lw=0.9, ls=":",
                   label="arms 1-3 against arms 4-6, habituation day")
        ax.axhline(0, color="black", lw=0.5)
        ax.text(1, -0.3, "fixed layout:\nEE in arms 1-3,\nSC in arms 4-6", ha="center", va="bottom",
                fontsize=6 if thesis else 7, color="#E65100")
        ax.set_xticks(xs)
        ax.set_xticklabels(["block 1", "block 2", "block 3", "block 4"])
        ax.set_ylim(-0.32, 0.62)
        ax.set_title(f"({'ab'[list(DAYS).index(day)]}) {label}" if thesis else label,
                     loc="left" if thesis else "center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Index of time in the arms\n(mean and SEM over mice)")
    axes[1].legend(fontsize=5.5 if thesis else 7, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    out = os.path.join(args.root, "BATCH_ANALYSIS", "exp2")
    os.makedirs(out, exist_ok=True)

    d, report = build_table(args.root)
    d.to_csv(os.path.join(out, "arm_blocks.csv"), index=False)
    hab = habituation_arms(args.root)
    hab.to_csv(os.path.join(out, "habituation_arms.csv"), index=False)

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say(f"Mice: {d.groupby('day')['mouse'].nunique().to_dict()}; excluded sessions: "
        f"{len(report['excluded_sessions'])} (see the earlier report for reasons)")
    nonexp = d.groupby(["day", "mouse"])["visits"].sum()
    say(f"mice with no visit in a day's sound blocks: {list(nonexp[nonexp == 0].index)}")

    say("\nLayout of block 1 (share of sessions with each cell in its usual arm):")
    b1 = d[d["block"] == 1]
    for cell, g in b1.groupby("cell"):
        c = g["arm"].value_counts()
        say(f"  {cell:16s} arm {c.index[0]} in {c.iloc[0] / c.sum():.0%} of {c.sum()} sessions")
    say("Each arm's share of the time, habituation day (mean over mice, %): "
        + ", ".join(f"arm {a} {v * 100:.1f}" for a, v in
                    hab.groupby("arm")["share"].mean().items()))

    for day, label in DAYS.items():
        x = d[d["day"] == day]
        say(f"\n==================== {label} ====================")
        say("Block by block, index of time (per mouse): EE against SC | arms 1-3 against 4-6")
        for b in (1, 2, 3, 4):
            y = x[x["block"] == b]
            e = per_mouse(y, is_env("EE"), is_env("SC"), "time")
            p = per_mouse(y, arms(1, 3), arms(4, 6), "time")
            say(f"  block {b}: EE-SC {e.mean():+.3f} ({int((e > 0).sum())}/{len(e)}, "
                f"p = {wilcoxon(e).pvalue:.4f})   place {p.mean():+.3f}")
        for which, wlabel in SUBSETS.items():
            y = subset(x, which)
            say(f"\n{wlabel}:")
            say("  time on EE against SC grammar arms per session (s), paired:")
            for tier in [None] + TIERS:
                g = y[y["cls"] == "grammar"] if tier is None else y[(y["cls"] == "grammar") & (y["tier"] == tier)]
                t = g.groupby(["mouse", "environment"])["time_spent_s"].sum().unstack(fill_value=0)
                t = t[(t.sum(axis=1) > 0)]
                diff = t["EE"] - t["SC"]
                p = wilcoxon(t["EE"], t["SC"]).pvalue
                say(f"      {tier or 'all levels':10s} EE median {t['EE'].median():6.1f} s, SC {t['SC'].median():6.1f} s, "
                    f"difference mean {diff.mean():+.1f} s, dz {diff.mean() / diff.std(ddof=1):+.2f}, "
                    f"Wilcoxon p = {p:.2g} (n = {len(t)})")
            for measure, mlabel in (("time", "time"), ("stay", "staying"), ("enter", "entering")):
                say(f"  EE against SC, {mlabel}: " + describe(per_mouse(y, is_env("EE"), is_env("SC"), measure)))
                for tier in TIERS:
                    say(f"      {tier:9s} " + describe(per_mouse(y, is_env("EE", tier), is_env("SC", tier), measure)))
            for a, b in (("vocalisation", "grammar"), ("silent", "grammar"), ("vocalisation", "silent")):
                for measure in ("stay", "enter"):
                    say(f"  {a} against {b}, {'staying' if measure == 'stay' else 'entering'}: "
                        + describe(per_mouse(y, is_cls(a), is_cls(b), measure)))
            for grp in (1, 2):
                z = y[y["group"] == grp]
                say(f"  group {grp} (EE = grammar {'A' if grp == 1 else 'B'}), EE against SC, time: "
                    + describe(per_mouse(z, is_env("EE"), is_env("SC"), "time")))
            for measure, mlabel in (("time", "time"), ("stay", "staying"), ("enter", "entering")):
                say(f"  grammar A (ascending) against B (octave alternation), whatever the cage, "
                    f"{mlabel}: " + describe(per_mouse(y, is_grammar("A"), is_grammar("B"), measure)))
                for tier in TIERS:
                    say(f"      {tier:9s} " + describe(per_mouse(y, is_grammar("A", tier),
                                                             is_grammar("B", tier), measure)))
    # day 1 against day 2: each mouse's index of time over all four blocks
    both = pd.DataFrame({day: per_mouse(d[d["day"] == day], is_env("EE"), is_env("SC"), "time")
                         for day in DAYS}).dropna()
    drop = both["day_2"] - both["day_1"]
    say(f"\nDay 2 against day 1, index of time, all four blocks: lower on day 2 in "
        f"{int((drop < 0).sum())} of {len(drop)} mice, median change {drop.median():+.3f}, "
        f"Wilcoxon p = {wilcoxon(both['day_1'], both['day_2']).pvalue:.4f}")
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    draw_blocks(d, hab, os.path.join(out, "fig_blocks"))
    if args.thesis_figures:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
                             "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                             "pdf.fonttype": 42}):
            draw_blocks(d, hab, os.path.join(args.thesis_figures, "fig_exp2_blocks"), thesis=True)

    print("\nMixed models (lme4):")
    r = subprocess.run([rscript(), os.path.join(HERE, "exp2_lmm.R"),
                        os.path.join(out, "arm_blocks.csv"), out])
    if r.returncode:
        sys.exit(r.returncode)
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
