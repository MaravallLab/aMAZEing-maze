#!/usr/bin/env python3
"""
Experiment 1: Does Hearing a Recording Every Day Make It Preferred?
===================================================================

The vocalisation arm of days 1 to 5 did not always play the same recording.
The experiment code (archive/auditory_v1/main.py) changed it twice:

    commit 443b4d2, 8 October 2025: the interval days moved from the full
        zenodo5771669 run1_day2 recording to the trimmed run3_day2
    commit d5f5a63, 5 January 2026 09:20: every day moved to the trimmed
        run3_day2-2

so the recording a mouse heard on days 1 to 5 depends on when it ran:

    day                    until 8 Oct 2025   8 Oct 2025 to 5 Jan 2026   from 5 Jan 2026
    1 (TEM)                urine r1           urine r1                   urine r3b
    2 to 4 (intervals)     urine r1           urine r3                   urine r3b
    5 (sequences)          urine r3           urine r3                   urine r3b

All three recordings are among the seven played on the vocalisation day
(day 6), so each can be compared between mice that heard it before and mice
that did not. For each mouse the recording of a day is that of its last
session of the day: an earlier session of the same day is a false start that
was restarted (mouse12246 restarted day 1 on 5 January, after the switch).

Two families of tests, each Holm corrected across its three tests, on the
day 6 index of each recording against the silent arm (stimulus_pi.csv):

    within mice   the familiar recording minus the mean of the other six,
                  Wilcoxon signed-rank against 0, for urine r3b in the mice
                  that heard only it, and for urine r1 and urine r3 in the
                  mice that heard those two
    between mice  each urine recording, mice that heard it on days 1 to 5
                  against mice that did not, Mann-Whitney U

Run exp1_stimulus_pi.py first.

Usage:
    python exp1_familiarity.py [DATA_DIR]     (the 8_arms_w_voc folder)

Outputs (DATA_DIR/BATCH_ANALYSIS/familiarity/):
    familiarity.csv          every mouse: what it heard, its day 6 indices
    familiarity_stats.txt    the tests
"""

import argparse
import collections
import os
import re
import sys

import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INTERVALS_SWITCH = "2025-10-08 12:38"     # commit 443b4d2
ALL_SWITCH = "2026-01-05 09:20"           # commit d5f5a63
URINE = ["urine r1", "urine r3", "urine r3b"]


def control_recording(day, stamp):
    """The recording the vocalisation arm played on a day 1 to 5 session."""
    if stamp >= ALL_SWITCH:
        return "urine r3b"
    if day == "w1_d1":
        return "urine r1"
    if day in ("w1_d2", "w1_d3", "w1_d4"):
        return "urine r1" if stamp < INTERVALS_SWITCH else "urine r3"
    if day == "w2_sequences":
        return "urine r3"
    raise ValueError(day)


def session_stamp(folder):
    m = re.search(r"time_(\d{4}-\d{2}-\d{2})_(\d\d)_(\d\d)", folder)
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}"


def main():
    import preference_analysis_config as cfg
    from exp1_stimulus_pi import label

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", nargs="?", default=cfg.BASE_PATH)
    args = parser.parse_args()
    cfg.BASE_PATH = args.data_dir
    out = os.path.join(args.data_dir, "BATCH_ANALYSIS", "familiarity")
    os.makedirs(out, exist_ok=True)

    # the last session of each mouse and day 1 to 5
    last = {}
    for s in cfg.discover_sessions():
        if s.day == "w2_vocalisations":
            continue
        stamp = session_stamp(s.session_folder)
        key = (s.mouse_id, s.day)
        if key not in last or stamp > last[key]:
            last[key] = stamp
    heard = collections.defaultdict(collections.Counter)
    for (mouse, day), stamp in last.items():
        heard[mouse][control_recording(day, stamp)] += 1

    st = pd.read_csv(os.path.join(args.data_dir, "BATCH_ANALYSIS", "stimulus_pi",
                                  "stimulus_pi.csv"))
    v = st[st["set"] == "Vocalisations"].pivot_table(index="mouse", columns="stimulus",
                                                       values="pi")
    v.columns = [label(c) for c in v.columns]
    recordings = list(v.columns)
    v["heard"] = [" and ".join(sorted(heard[m])) for m in v.index]
    v["days_heard"] = [";".join(f"{k}:{n}" for k, n in sorted(heard[m].items()))
                       for m in v.index]
    v.to_csv(os.path.join(out, "familiarity.csv"))

    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    say("Mice with day 6 data, by the recording(s) heard on days 1 to 5:")
    for grp, n in v["heard"].value_counts().items():
        say(f"  {grp}: {n}")
    say("\nMedian day 6 index of each recording, by group:")
    say(v.groupby("heard")[recordings].median().T.round(3).to_string())

    say("\n1. Within mice: the familiar recording minus the mean of the other six "
        "(Wilcoxon against 0, Holm across the three)")
    rows = []
    for grp, g in v.groupby("heard"):
        for fam in URINE:
            if fam not in grp.split(" and ") or len(g) < 5:
                continue
            others = [c for c in recordings if c != fam]
            d = (g[fam] - g[others].mean(axis=1)).dropna()
            rows.append((grp, fam, len(d), d.median(), int((d > 0).sum()), wilcoxon(d).pvalue))
    holm = multipletests([r[5] for r in rows], method="holm")[1]
    for (grp, fam, n, med, above, p), ph in zip(rows, holm):
        say(f"  heard {grp}: {fam}  n={n}  median {med:+.3f}  {above} above 0  "
            f"p = {p:.4f}  Holm p = {ph:.4f}")

    say("\n2. Between mice: each urine recording, heard on days 1 to 5 or not "
        "(Mann-Whitney U, Holm across the three)")
    rows = []
    for fam in URINE:
        was_heard = v["heard"].apply(lambda h: fam in h.split(" and "))
        yes = v.loc[was_heard, fam].dropna()
        no = v.loc[~was_heard, fam].dropna()
        test = mannwhitneyu(yes, no)
        rows.append((fam, len(yes), yes.median(), len(no), no.median(), test.statistic,
                     test.pvalue))
    holm = multipletests([r[6] for r in rows], method="holm")[1]
    for (fam, ny, my, nn, mn, u, p), ph in zip(rows, holm):
        say(f"  {fam}: heard {my:+.3f} (n={ny}), not heard {mn:+.3f} (n={nn})  "
            f"U = {u:.0f}  p = {p:.4f}  Holm p = {ph:.4f}")

    with open(os.path.join(out, "familiarity_stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
