#!/usr/bin/env python3
"""
Experiment 2: Sample Size and Power
===================================

What the sample of Experiment 2 could detect, and what a replication would need.
Answers the Discussion's limitation on sample and power (thesis Section 7.2.3).

The test is the main one of Experiment 2: on each test day, each mouse's time in
the enriched cage (EE) grammar arms against the standard cage (SC) grammar arms,
paired within the mouse, by the Wilcoxon signed rank test, two sided, at 0.05.
Its effect size is d_z, the mean of the paired differences over their standard
deviation (exp2_lmm.py reports d_z = 0.69 on day 1 and 0.57 on day 2, n = 32).

Power computed from the observed effect only restates the p value, so it is not
reported. Instead:

  1. sensitivity: the smallest d_z that n = 32 detects with 80% and 90% power,
     and the power at n = 16, the size of one counterbalancing group;
  2. sample size: the n a replication needs for 80% and 90% power, for the
     observed d_z, for the lower end of its 95% confidence interval (observed
     effects in small samples tend to overstate the true one), and for day 2;
  3. with the data (GRAMMAR_DIR/BATCH_ANALYSIS/exp2/arm_blocks.csv, written by
     exp2_lmm.py): d_z recomputed from the mice, and the power of the Wilcoxon
     test when the mice are resampled with replacement at each n, which assumes
     nothing about the shape of the differences.

The paired t test gives exact power by the noncentral t distribution
(statsmodels); the Wilcoxon test is simulated, with normal differences in parts
1 and 2. Without the data, parts 1 and 2 run from --dz and --dz-day2.

Usage:
    python exp2_power.py [GRAMMAR_DIR] [--dz 0.69] [--dz-day2 0.57] [--sims 20000]

Output: printed, and GRAMMAR_DIR/BATCH_ANALYSIS/exp2/power.txt when that folder exists.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower

ALPHA = 0.05
N_STUDY = 32
N_GROUP = 16
POWERS = (0.80, 0.90)
SEED = 20261115


def t_power(dz, n):
    return TTestPower().power(effect_size=dz, nobs=n, alpha=ALPHA, alternative="two-sided")


def t_n_needed(dz, power):
    return int(np.ceil(TTestPower().solve_power(effect_size=dz, power=power, alpha=ALPHA,
                                                alternative="two-sided")))


def t_min_dz(n, power):
    return TTestPower().solve_power(nobs=n, power=power, alpha=ALPHA, alternative="two-sided")


def w_power(dz, n, sims, rng, sample=None):
    """Power of the Wilcoxon signed rank test at n mice. Differences are normal with
    mean dz and unit SD, or resampled with replacement from `sample`."""
    if sample is None:
        x = rng.normal(dz, 1.0, size=(sims, n))
    else:
        x = rng.choice(np.asarray(sample, float), size=(sims, n), replace=True)
    p = stats.wilcoxon(x, axis=1, zero_method="wilcox", method="asymptotic").pvalue
    return float(np.mean(p < ALPHA))


def w_n_needed(dz, power, sims, rng, sample=None, n_max=400):
    lo, hi = 4, n_max
    if w_power(dz, hi, sims, rng, sample) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if w_power(dz, mid, sims, rng, sample) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def w_min_dz(n, power, sims, rng):
    lo, hi = 0.0, 2.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if w_power(mid, n, sims, rng) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def dz_interval(dz, n):
    """95% confidence interval of d_z by inverting the noncentral t distribution."""
    t_obs, df = dz * np.sqrt(n), n - 1

    def solve(q):
        a, b = -10.0, 20.0
        for _ in range(80):
            mid = (a + b) / 2
            if stats.nct.cdf(t_obs, df, mid) > q:
                a = mid
            else:
                b = mid
        return (a + b) / 2
    return solve(0.975) / np.sqrt(n), solve(0.025) / np.sqrt(n)


def mouse_differences(root):
    """Per mouse and day, total time in EE minus SC grammar arms over the session,
    from the table of exp2_lmm.py, for all four blocks and for the shuffled blocks."""
    path = os.path.join(root, "BATCH_ANALYSIS", "exp2", "arm_blocks.csv")
    if not os.path.exists(path):
        return None, path
    d = pd.read_csv(path)
    d = d[d["cls"] == "grammar"]
    out = {}
    for label, sub in (("all four blocks", d), ("shuffled blocks 2 to 4", d[d["block"] > 1])):
        for day in ("day_1", "day_2"):
            t = (sub[sub["day"] == day].groupby(["mouse", "environment"])["time_spent_s"]
                 .sum().unstack(fill_value=0))
            t = t[t.sum(axis=1) > 0]
            out[(label, day)] = (t["EE"] - t["SC"]).to_numpy()
    return out, path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--dz", type=float, default=0.69, help="day 1 d_z when the data are absent")
    parser.add_argument("--dz-day2", type=float, default=0.57, help="day 2 d_z when the data are absent")
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    diffs, path = mouse_differences(args.root)
    dz1, dz2 = args.dz, args.dz_day2
    if diffs is None:
        say(f"No table at {path}; d_z taken as given: day 1 {dz1:.2f}, day 2 {dz2:.2f}.")
    else:
        say(f"Table: {path}")
        for (label, day), x in diffs.items():
            say(f"  {label:23s} {day}: n = {len(x)}, d_z = {x.mean() / x.std(ddof=1):+.3f}, "
                f"Wilcoxon p = {stats.wilcoxon(x).pvalue:.2g}")
        x1, x2 = diffs[("all four blocks", "day_1")], diffs[("all four blocks", "day_2")]
        dz1, dz2 = x1.mean() / x1.std(ddof=1), x2.mean() / x2.std(ddof=1)
        say(f"Using all four blocks: day 1 d_z {dz1:.3f}, day 2 d_z {dz2:.3f}.")
    say(f"Two sided, alpha {ALPHA}; Wilcoxon simulated with {args.sims} data sets per point, seed {SEED}.")

    say("\n1. Sensitivity")
    for pw in POWERS:
        say(f"  smallest d_z detected with {pw:.0%} power at n = {N_STUDY}: "
            f"paired t {t_min_dz(N_STUDY, pw):.2f}, Wilcoxon {w_min_dz(N_STUDY, pw, args.sims, rng):.2f}")
    for n in (N_STUDY, N_GROUP):
        say(f"  power for the day 1 d_z ({dz1:.2f}) at n = {n}: "
            f"paired t {t_power(dz1, n):.2f}, Wilcoxon {w_power(dz1, n, args.sims, rng):.2f}")

    lo, hi = dz_interval(dz1, N_STUDY)
    say("\n2. Sample size for a replication")
    say(f"  95% confidence interval of the day 1 d_z at n = {N_STUDY}: {lo:.2f} to {hi:.2f}")
    for name, dz in (("day 1 d_z", dz1), ("lower end of its interval", lo), ("day 2 d_z", dz2)):
        parts = []
        for pw in POWERS:
            wn = w_n_needed(dz, pw, args.sims, rng)
            parts.append(f"{pw:.0%}: t {t_n_needed(dz, pw)}, Wilcoxon {wn if wn else '>400'}")
        say(f"  {name:26s} ({dz:.2f})  n needed for " + "; ".join(parts))

    if diffs is not None:
        say("\n3. Resampling the mice (all four blocks), power of the Wilcoxon test at each n")
        for day in ("day_1", "day_2"):
            x = diffs[("all four blocks", day)]
            row = ", ".join(f"n {n}: {w_power(None, n, args.sims, rng, sample=x):.2f}"
                            for n in (8, 12, 16, 20, 24, 32, 40, 48, 64))
            say(f"  {day}: {row}")
            for pw in POWERS:
                say(f"      n needed for {pw:.0%}: {w_n_needed(None, pw, args.sims, rng, sample=x)}")

    outdir = os.path.dirname(path)
    if os.path.isdir(outdir):
        with open(os.path.join(outdir, "power.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
