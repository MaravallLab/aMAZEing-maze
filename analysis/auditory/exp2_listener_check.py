#!/usr/bin/env python3
"""
Experiment 2: which listener reads the secondary melodies as their own grammar?
================================================================================

EXPLORATORY. Not part of the preregistered or confirmatory analyses.

The extended model of Chapter 5 scores each test melody transition by
transition under the full grammars heard in training. Such a listener reads a
secondary melody of one grammar as the other grammar, because grammar A's
secondary step of three tones is grammar B's dominant step, and grammar B's
secondary step of one tone is grammar A's dominant step. The mice preferred the
enriched cage grammar at the secondary level, so they did not read the melodies
this way. This script asks which simple listeners would read them correctly.

Listeners (all start each arm visit at an even belief between the two cages,
keep a context with probability 0.99 from one transition to the next, and value
the enriched cage at 1 and the standard cage at 0, as in the fits of
Section 6.7):

  transition   each transition weighed on its own against the full grammars
               (the listener of the model as fitted)
  pooled       the same, with the belief carried over from one melody to the
               next within the arm (evidence accumulated over the test)
  learning     the same, with each cage's grammar also updated within the test
               from what the arm plays, in proportion to the belief that the
               arm belongs to that cage (counts added to training counts of
               strength N0 per row)
  level        a listener that infers the level of a melody along with its
               grammar: each cage is a mixture of its three level restricted
               chains, the level held fixed within a melody (1 percent of each
               restricted chain is mixed with the full grammar, so that no
               transition is impossible)

For each listener the script computes the mean semantic value S per transition
of a melody of each grammar at each level, averaged over the two
counterbalancing groups (15 and 17 mice), and the difference between the
enriched cage and the standard cage grammar at each level. It compares the sign
and ordering of that difference with the observed day 1 mean shares of time
(Table 6.10, from exp2_model_compare.py), and fits one scale factor by least
squares through the origin to the three observed differences.

Experiment 2 gives only three such differences, so this can reject a listener
whose predicted sign is wrong, not rank listeners whose signs agree.

Usage:
    python exp2_listener_check.py

Output: exp2_listener_check.txt next to this script.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from amazeing.auditory.grammar_stimuli import config as gcfg  # noqa: E402

LEVELS = ["dominant", "secondary", "rare"]
GROUP_N = {1: 15, 2: 17}          # mice per counterbalancing group on day 1
STAY = 0.99                       # persistence of a context between transitions
N_MELODIES = 32                   # melodies per arm visit pooled, about 140 s / 4.4 s
N_SIM = 400                       # simulated arm visits per cell
EPS = 0.01                        # mixture with the full grammar, level listener
N0_VALUES = (20.0, 200.0)         # training strength per row, learning listener

# Observed day 1 mean shares of time (Table 6.10; exp2_model_compare.py output,
# information_criteria.txt): EE dominant, secondary, rare, SC dominant,
# secondary, rare.
OBS_EE = np.array([0.183, 0.166, 0.121])
OBS_SC = np.array([0.116, 0.125, 0.124])


def restricted(M, level):
    """Level restricted, renormalised chain (as sequence_sampler does)."""
    p = gcfg.COMPLEXITY_TIERS[level]
    R = np.where(np.abs(M - p) < gcfg.PROB_TOLERANCE, M, 0.0)
    return R / R.sum(axis=1, keepdims=True)


def sample(M, rng, n=gcfg.MELODY_LENGTH):
    x = [rng.integers(gcfg.N_TONES)]
    for _ in range(n - 1):
        x.append(rng.choice(gcfg.N_TONES, p=M[x[-1]]))
    return x


def s_transition(melodies, Mee, Msc, carry):
    """Mean S per transition for a two context filter on fixed emissions."""
    b, out = 0.5, []
    for mel in melodies:
        if not carry:
            b = 0.5
        for a, c in zip(mel[:-1], mel[1:]):
            bbar = STAY * b + (1 - STAY) * (1 - b)
            le, ls = Mee[a, c], Msc[a, c]
            b = le * bbar / (le * bbar + ls * (1 - bbar))
            out.append(b - bbar)          # S = (V_EE - V_SC)(b - bbar), V_EE - V_SC = 1
    return np.mean(out)


def s_learning(melodies, Mee, Msc, n0):
    """Belief carried over melodies; each cage's chain updated from the arm."""
    Cee, Csc = n0 * Mee.copy(), n0 * Msc.copy()
    b, out = 0.5, []
    for mel in melodies:
        for a, c in zip(mel[:-1], mel[1:]):
            Pe, Ps = Cee / Cee.sum(1, keepdims=True), Csc / Csc.sum(1, keepdims=True)
            bbar = STAY * b + (1 - STAY) * (1 - b)
            le, ls = Pe[a, c], Ps[a, c]
            b = le * bbar / (le * bbar + ls * (1 - bbar))
            out.append(b - bbar)
            Cee[a, c] += b
            Csc[a, c] += 1 - b
    return np.mean(out)


def s_level(melodies, Mee, Msc):
    """Joint belief over cage and level; level fixed within a melody."""
    chains = []
    for M in (Mee, Msc):
        chains.append([(1 - EPS) * restricted(M, lv) + EPS * M for lv in LEVELS])
    out = []
    for mel in melodies:
        w = np.full((2, 3), 1 / 6)        # cage x level, even prior at each melody
        for a, c in zip(mel[:-1], mel[1:]):
            bc = w.sum(1)
            bbar = STAY * bc[0] + (1 - STAY) * bc[1]
            # carry the level split within each cage, mix the cages by persistence
            split = w / w.sum(1, keepdims=True)
            wbar = np.vstack([bbar * split[0], (1 - bbar) * split[1]])
            like = np.array([[chains[k][j][a, c] for j in range(3)] for k in range(2)])
            w = like * wbar
            w = w / w.sum()
            out.append(w[0].sum() - bbar)
    return np.mean(out)


def cell_values(listener, rng, **kw):
    """Mean S for the EE and SC grammar at each level, over the two groups."""
    ee, sc = np.zeros(3), np.zeros(3)
    tot = sum(GROUP_N.values())
    for g, n in GROUP_N.items():
        Mee = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["EE"]]
        Msc = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["SC"]]
        for i, lv in enumerate(LEVELS):
            for arm_M, acc in ((Mee, ee), (Msc, sc)):
                R = restricted(arm_M, lv)
                vals = []
                for _ in range(N_SIM):
                    mels = [sample(R, rng) for _ in range(N_MELODIES)]
                    if listener == "transition":
                        vals.append(s_transition(mels[:1], Mee, Msc, carry=False))
                    elif listener == "pooled":
                        vals.append(s_transition(mels, Mee, Msc, carry=True))
                    elif listener == "learning":
                        vals.append(s_learning(mels, Mee, Msc, kw["n0"]))
                    elif listener == "level":
                        vals.append(s_level(mels[:1], Mee, Msc))
                acc[i] += n / tot * np.mean(vals)
    return ee, sc


WALL = 0.95                       # belief at which the listener counts as decided
MAX_MELODIES = 5                  # stream of the same arm, 11 transitions per melody


def decide(listener, R, Mee, Msc, own_is_ee, rng):
    """Drift diffusion reading of the filter: transitions until the belief in
    one cage reaches WALL. Belief is carried across the melodies of one arm.
    Returns (outcome, transitions): outcome 'right', 'wrong' or 'none'."""
    if listener == "level":
        chains = [[(1 - EPS) * restricted(M, lv) + EPS * M for lv in LEVELS] for M in (Mee, Msc)]
        w = np.full((2, 3), 1 / 6)
    b, n = 0.5, 0
    for _ in range(MAX_MELODIES):
        mel = sample(R, rng)
        for a, c in zip(mel[:-1], mel[1:]):
            n += 1
            if listener == "transition":
                bbar = STAY * b + (1 - STAY) * (1 - b)
                le, ls = Mee[a, c], Msc[a, c]
                b = le * bbar / (le * bbar + ls * (1 - bbar))
            else:
                bc = w.sum(1)
                bbar = STAY * bc[0] + (1 - STAY) * bc[1]
                split = w / w.sum(1, keepdims=True)
                wbar = np.vstack([bbar * split[0], (1 - bbar) * split[1]])
                like = np.array([[chains[k][j][a, c] for j in range(3)] for k in range(2)])
                w = like * wbar
                w = w / w.sum()
                b = w[0].sum()
            right = b if own_is_ee else 1 - b
            if right >= WALL:
                return "right", n
            if right <= 1 - WALL:
                return "wrong", n
    return "none", n


def ddm_report(rng):
    lines = ["",
             "DRIFT DIFFUSION READING (exploratory): how many tone transitions a listener needs",
             f"before it is {WALL:.0%} sure which cage a melody comes from. The belief update of",
             "the model adds a fixed amount of evidence with every transition, so it behaves",
             "like a drift diffusion process with walls at 95% and 5%. Melodies of one arm are",
             f"heard in a row (up to {MAX_MELODIES} melodies, {MAX_MELODIES * 11} transitions); "
             "averaged over both cages and both groups.",
             ""]
    for listener in ("transition", "level"):
        for lv in LEVELS:
            res = {"right": [], "wrong": [], "none": []}
            for g in GROUP_N:
                Mee = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["EE"]]
                Msc = gcfg.GRAMMARS[gcfg.COUNTERBALANCE[g]["SC"]]
                for arm_M, own_is_ee in ((Mee, True), (Msc, False)):
                    R = restricted(arm_M, lv)
                    for _ in range(N_SIM):
                        out, n = decide(listener, R, Mee, Msc, own_is_ee, rng)
                        res[out].append(n)
            tot = sum(len(v) for v in res.values())
            def med(v):
                return f"{np.median(v):.0f}" if v else "-"
            lines.append(f"  {listener:10s} {lv:9s} right wall {len(res['right']) / tot:6.1%} "
                         f"(median {med(res['right'])} transitions), wrong wall "
                         f"{len(res['wrong']) / tot:6.1%} (median {med(res['wrong'])}), "
                         f"undecided {len(res['none']) / tot:6.1%}")
    return lines


def main():
    rng = np.random.default_rng(20261119)
    if len(sys.argv) > 1 and sys.argv[1] == "ddm":
        text = "\n".join(ddm_report(rng))
        print(text)
        with open(os.path.join(HERE, "exp2_listener_ddm.txt"), "w", encoding="utf-8") as f:
            f.write(text.lstrip("\n") + "\n")
        return
    obs_d = OBS_EE - OBS_SC
    lines = ["EXPLORATORY: which listener reads the secondary melodies as their own grammar?",
             "",
             "Observed day 1 mean shares, EE minus SC grammar (dominant, secondary, rare): "
             + ", ".join(f"{v:+.3f}" for v in obs_d),
             "",
             "Predicted mean semantic value S per transition, EE minus SC grammar, and a one",
             "parameter fit (scale through the origin) to the three observed differences:",
             ""]
    runs = [("transition", {}), ("pooled", {})] + \
           [("learning", {"n0": n0}) for n0 in N0_VALUES] + [("level", {})]
    for name, kw in runs:
        ee, sc = cell_values(name, rng, **kw)
        d = ee - sc
        k = float(d @ obs_d / (d @ d)) if d @ d > 0 else 0.0
        pred = k * d
        r2 = 1 - np.sum((obs_d - pred) ** 2) / np.sum((obs_d - obs_d.mean()) ** 2)
        lab = name + (f" (N0 = {kw['n0']:.0f})" if kw else "")
        signs = "".join("+" if v > 0 else "-" for v in d)
        ok = "reads secondary as its own grammar" if d[1] > 0 else "reads secondary as the other grammar"
        lines.append(f"  {lab:22s} S diff {d[0]:+.4f} {d[1]:+.4f} {d[2]:+.4f}  signs {signs}  "
                     f"scale {k:+.3f}  R2 {r2:+.2f}  ({ok})")
    lines += ["",
              "Simulated with seed 20261119, 400 arm visits per cell, 32 melodies pooled per visit",
              "for the pooled and learning listeners. Only three observed differences: a",
              "listener with the wrong sign at the secondary level is rejected; listeners with",
              "the right signs cannot be ranked."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "exp2_listener_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
