#!/usr/bin/env python3
"""
Experiment 2: Thesis Figures
============================

    fig_exp2_session  the session-level preference for the grammar heard in the
                      enriched cage (EE) over the one heard in the standard cage
                      (SC): (a) time on the EE and SC grammar arms per session,
                      day 1; (b) the EE-SC index of time at each level of
                      predictability, day 1, coloured by counterbalancing group;
                      (c) the index on day 1 and day 2
    fig_exp2_model    the value model of Chapter 5 as fitted by the earlier
                      analysis (model_validation/, read only, from its saved
                      results.json and grain_results.json): (a) recovery of a
                      known semantic weight; (b) the semantic weight at three
                      grains of observation; (c) the predicted and observed share
                      of time in each arm type (per-mouse cell means)
    fig_exp2_stimuli  the two grammars and one melody of each at each level
    fig_exp2_moves    the moves each kind of melody makes: (a, b) the secondary
                      melody of each grammar from fig_exp2_stimuli, each step
                      labelled, the moves borrowed from the other grammar in that
                      grammar's colour; (c) the share of each move in all the
                      melodies played at test, for each grammar at each level
                      (also written to BATCH_ANALYSIS/exp2/move_shares.txt)

Usage:
    python exp2_figures.py [GRAMMAR_DIR] --thesis-figures DIR [--only NAME]

    run exp2_lmm.py first (it writes BATCH_ANALYSIS/exp2/arm_blocks.csv).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EE, SC = "#1565C0", "#B71C1C"
GROUP = {1: "#1565C0", 2: "#E65100"}
TIERS = ["dominant", "secondary", "rare"]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def index(a, b):
    return (a - b) / (a + b) if a + b > 0 else np.nan


def per_mouse_index(d, tier=None):
    g = d[d["cls"] == "grammar"]
    if tier is not None:
        g = g[g["tier"] == tier]
    t = g.groupby(["mouse", "environment"])["time_spent_s"].sum().unstack(fill_value=0)
    return t.apply(lambda r: index(r["EE"], r["SC"]), axis=1).dropna()


def fig_session(d, path):
    d1, d2 = d[d["day"] == "day_1"], d[d["day"] == "day_2"]
    groups = d.groupby("mouse")["group"].first()
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [0.8, 1.4, 0.9]})
    rng = np.random.default_rng(0)

    ax = axes[0]
    t = d1[d1["cls"] == "grammar"].groupby(["mouse", "environment"])["time_spent_s"].sum().unstack(fill_value=0)
    t = t[t.sum(axis=1) > 0] / 60
    for _, r in t.iterrows():
        ax.plot([0, 1], [r["EE"], r["SC"]], color="#BDBDBD", lw=0.5, zorder=1)
    ax.scatter(np.zeros(len(t)), t["EE"], s=5, color=EE, zorder=2, lw=0)
    ax.scatter(np.ones(len(t)), t["SC"], s=5, color=SC, zorder=2, lw=0)
    for x, c in ((0, "EE"), (1, "SC")):
        ax.plot([x - 0.2, x + 0.2], [t[c].median()] * 2, color="black", lw=1.6, zorder=3)
    p = wilcoxon(t["EE"], t["SC"]).pvalue
    ax.text(0.5, t.values.max() * 1.02, stars(p), ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["EE\ngrammar", "SC\ngrammar"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Time in the grammar arms\nper session (min)")
    ax.set_title("(a) day 1", loc="left")

    ax = axes[1]
    pis = {tier: per_mouse_index(d1, tier) for tier in TIERS}
    ph = multipletests([wilcoxon(pis[t_]).pvalue for t_ in TIERS], method="holm")[1]
    for i, tier in enumerate(TIERS):
        x = pis[tier]
        cols = [GROUP[int(groups[m])] for m in x.index]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=5, c=cols, lw=0, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [x.median()] * 2, color="black", lw=1.6, zorder=3)
        ax.text(i, 0.93, stars(ph[i]), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels(TIERS)
    ax.set_ylim(-1.0, 1.05)
    ax.set_ylabel("EE-SC index of time")
    ax.set_title("(b) day 1, by level of predictability", loc="left")
    for g_, c in GROUP.items():
        ax.scatter([], [], s=8, color=c, label=f"group {g_} (EE = grammar {'A' if g_ == 1 else 'B'})")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")

    ax = axes[2]
    a, b = per_mouse_index(d1), per_mouse_index(d2)
    both = pd.concat([a.rename("d1"), b.rename("d2")], axis=1)
    for _, r in both.dropna().iterrows():
        ax.plot([0, 1], [r["d1"], r["d2"]], color="#BDBDBD", lw=0.5, zorder=1)
    for x, v in ((0, a), (1, b)):
        ax.scatter(np.full(len(v), x), v, s=5, color="#424242", lw=0, zorder=2)
        ax.plot([x - 0.2, x + 0.2], [v.median()] * 2, color="black", lw=1.6, zorder=3)
        ax.text(x, 0.55, stars(wilcoxon(v).pvalue), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["day 1", "day 2"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.35, 0.65)
    ax.set_ylabel("EE-SC index of time")
    ax.set_title("(c) both test days", loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_model(mv, path):
    res = json.load(open(os.path.join(mv, "results.json")))
    grain = json.load(open(os.path.join(mv, "grain_results.json")))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), gridspec_kw={"width_ratios": [0.9, 1.0, 1.5]})

    ax = axes[0]
    grid = res["recovery"]["parameter_recovery"]["grid"]
    tr = [g["wS_true"] for g in grid]
    mean = [g["wS_recovered_mean"] for g in grid]
    lo = [g["wS_recovered_mean"] - g["ci"][0] for g in grid]
    hi = [g["ci"][1] - g["wS_recovered_mean"] for g in grid]
    ax.plot([-0.2, 2.2], [-0.2, 2.2], color="grey", lw=0.6, ls=(0, (4, 2)))
    ax.errorbar(tr, mean, [lo, hi], fmt="o", ms=3, color="#2E7D32", capsize=2)
    ax.set_xlabel("True semantic weight")
    ax.set_ylabel("Recovered (95% interval)")
    ax.set_title("(a) recovery", loc="left")

    ax = axes[1]
    wt = grain["weight_table"]
    rows = [("per block", wt["A_per_block"]["wS"]), ("per mouse,\ncell means", wt["B_cell_mean"]["wS"]),
            ("per level,\nEE-SC index", wt["C_tier_PI"]["wS"])]
    for i, (lab, w) in enumerate(rows):
        ax.errorbar(w["mean"], i, xerr=[[w["mean"] - w["hdi"][0]], [w["hdi"][1] - w["mean"]]],
                    fmt="o", ms=3.5, color="#2E7D32", capsize=2)
    ax.axvline(0, color="black", lw=0.6, ls=(0, (4, 2)))
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Semantic weight $w_S$ (95% HDI)")
    ax.set_title("(b) by grain", loc="left")

    ax = axes[2]
    cells = ["EE\ndom.", "EE\nsec.", "EE\nrare", "SC\ndom.", "SC\nsec.", "SC\nrare", "silent"]
    obs, pred = grain["grainB_observed_mean"], grain["grainB_ppc_predicted"]
    x = np.arange(7)
    colours = [EE] * 3 + [SC] * 3 + ["#9E9E9E"]
    ax.bar(x - 0.2, obs, 0.38, color=colours, alpha=0.9, label="observed")
    ax.bar(x + 0.2, pred, 0.38, color="white", edgecolor=colours, lw=0.9, label="model")
    ax.axhline(1 / 7, color="grey", lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=5.5)
    ax.set_ylabel("Share of time (mean over mice)")
    ax.set_title("(c) observed and predicted", loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def tint(colour, share):
    """A colour mixed with white, share 1 being the colour itself."""
    from matplotlib.colors import to_rgb
    return tuple(1 - share * (1 - c) for c in to_rgb(colour))


def draw_chain(ax, M, sym, khz, colour):
    """A grammar as a Markov chain: the six tones round a ring, and an arrow for
    every transition, thicker and darker the more likely it is (0.60, 0.12,
    0.07). Staying on a tone (0.02) is a loop outside its node."""
    from matplotlib.patches import Circle, FancyArrowPatch, Arc
    n = len(sym)
    angle = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.c_[np.cos(angle), np.sin(angle)]
    r_node = 0.15
    nodes = [Circle(p, r_node, facecolor=colour, edgecolor="white", lw=0.8, zorder=4) for p in pos]
    style = {0.60: (1.0, 1.7, 7), 0.12: (0.62, 0.9, 5), 0.07: (0.38, 0.6, 4)}
    rad = {1: 0.25, 2: 0.18, 3: 0.22}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = M[i, j]
            share, lw, head = style[min(style, key=lambda s: abs(s - p))]
            step = min((j - i) % n, (i - j) % n)
            r = rad[step]
            ax.add_patch(FancyArrowPatch(pos[i], pos[j], patchA=nodes[i], patchB=nodes[j],
                                         connectionstyle=f"arc3,rad={r}", arrowstyle="-|>",
                                         mutation_scale=head, lw=lw, color=tint(colour, share),
                                         shrinkA=1, shrinkB=1, zorder=2 if p < 0.5 else 3))
            # the label sits on the curve two thirds of the way along, nearer the
            # tone it leads to, so that the labels spread round the nodes
            dx, dy = pos[j] - pos[i]
            ctrl = (pos[i] + pos[j]) / 2 + r * np.array([dy, -dx])
            t = 0.68 if step == 3 else 0.5
            at = (1 - t) ** 2 * pos[i] + 2 * t * (1 - t) * ctrl + t ** 2 * pos[j]
            # every tone has the same pattern turned round the ring, so the
            # probabilities are written on the arrows from the first tone and on
            # the dominant ones; the matrix above gives the rest
            if not (i == 0 or p > 0.5):
                continue
            ax.text(*at, f"{p:.2f}", ha="center", va="center", fontsize=5 if p < 0.5 else 5.4,
                    color=tint(colour, max(share, 0.75)), fontweight="bold" if p > 0.5 else None,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    for k, (p, c) in enumerate(zip(pos, nodes)):
        ax.add_patch(c)
        ax.text(*p, f"{sym[k]}\n{khz[k]:.1f}", ha="center", va="center", fontsize=5,
                color="white", fontweight="bold", zorder=5, linespacing=0.9)
        # staying on the same tone, a loop outside the node
        out = p / np.linalg.norm(p)
        ax.add_patch(Arc(p + out * (r_node + 0.07), 0.15, 0.15, theta1=0, theta2=360,
                         color=tint(colour, 0.38), lw=0.6, zorder=1))
        if k == 0:
            ax.text(*(p + out * (r_node + 0.25)), f"{M[k, k]:.2f}", ha="center", va="center",
                    fontsize=5, color=tint(colour, 0.75))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_stimuli(root, path):
    """The two grammars and one delivered melody of each at each level."""
    import sys
    from matplotlib.gridspec import GridSpec
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from amazeing.auditory.grammar_stimuli import config as gcfg
    from model_validation.data_loading import load_arm_blocks
    arms = load_arm_blocks(root).arm_blocks
    examples = {}
    for (grammar, tier), sub in arms[arms["arm_type"] == "grammar"].groupby(["grammar", "tier"]):
        mels = [m for ms in sub["melodies"] for m in ms]
        if mels:
            examples[(grammar, tier)] = mels[0]
    sym = gcfg.TONE_SYMBOLS
    khz = [gcfg.TONES[s] / 1000 for s in sym]
    colour = {"A": "#00695C", "B": "#6A1B9A"}

    fig = plt.figure(figsize=(6.3, 9.2))
    # the grammars (matrices and chains) and the melodies are two blocks, each
    # with its own spacing between rows
    outer = GridSpec(2, 1, figure=fig, height_ratios=[5.4, 2.4], hspace=0.08)
    top = outer[0].subgridspec(2, 6, height_ratios=[2.3, 2.9], hspace=0.38, wspace=0.9)
    bottom = outer[1].subgridspec(2, 6, hspace=0.6, wspace=0.9)

    class _Cells:
        def __getitem__(self, key):
            r, c = key
            return top[r, c] if r < 2 else bottom[r - 2, c]
    gs = _Cells()
    titles = []          # (axes, column group, title), placed once everything is drawn
    # one distinct shade for each of the four probabilities, the same shades as
    # the arrows of the chains below, so that the matrix and the chain read alike
    shade = {0.60: 1.0, 0.12: 0.62, 0.07: 0.38, 0.02: 0.12}
    for k, g in enumerate(("A", "B")):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        M = gcfg.GRAMMARS[g]
        img = np.array([[tint(colour[g], shade[min(shade, key=lambda s: abs(s - p))])
                         for p in row] for row in M])
        ax.imshow(img)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.4 else "black",
                        fontweight="bold" if M[i, j] > 0.4 else None)
        ax.set_xticks(range(6)); ax.set_xticklabels(sym)
        ax.set_yticks(range(6)); ax.set_yticklabels(sym)
        ax.set_xlabel("next tone"); ax.set_ylabel("current tone")
        name = "ascending cycle" if g == "A" else "alternation three steps apart"
        titles.append((ax, k, f"({'ab'[k]}) grammar {g}, {name}"))
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        draw_chain(ax, M, sym, khz, colour[g])
        titles.append((ax, k, f"({'cd'[k]}) grammar {g} as a Markov chain"))
    letters = iter("efghij")
    for row, g in enumerate(("A", "B"), start=2):
        for col, tier in enumerate(("dominant", "secondary", "rare")):
            ax = fig.add_subplot(gs[row, 2 * col:2 * col + 2])
            mel = examples.get((g, tier), "")
            for i, s in enumerate(mel):
                y = sym.index(s)
                ax.barh(y, 0.15, left=i * 0.2, height=0.6, color=colour[g])
            ax.set_xlim(0, 2.4); ax.set_ylim(-0.6, 5.6)
            ax.set_yticks(range(6))
            ax.set_yticklabels([f"{s} {f:.1f}" for s, f in zip(sym, khz)] if col == 0 else [])
            if row == 3:
                ax.set_xlabel("Time (s)")
            # the first melody of a row lines up with the left column above
            titles.append((ax, 0 if col == 0 else f"m{row}{col}", f"({next(letters)}) {g}, {tier}"))
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.02, 0.24, "Tone (kHz)", rotation=90, va="center", fontsize=7)
    # Panel letters in a column share one left edge, the leftmost drawn edge
    # (tick labels and axis labels included) of the panels in that column, so
    # that (a), (c) and (e) line up, and (b) and (d).
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    left = {}
    for ax, group, _ in titles:
        x0 = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).x0
        left[group] = min(left.get(group, x0), x0)
    for ax, group, text in titles:
        top = ax.get_position().y1
        fig.text(left[group], top + 0.008, text, ha="left", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"])
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


MOVES = ["+1", "+2", "+3", "-1", "-2", "repeat"]


def move_of(a, b, n=6):
    """The step between two tones, counted round the ring of six tones."""
    d = (b - a) % n
    return {0: "repeat", 1: "+1", 2: "+2", 3: "+3", 4: "-2", 5: "-1"}[d]


def fig_moves(root, path):
    """The moves of each kind of melody: the secondary melodies borrow the other
    grammar's most common move, yet each melody kind has its own set of moves."""
    import sys
    from matplotlib.gridspec import GridSpec
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from amazeing.auditory.grammar_stimuli import config as gcfg
    from model_validation.data_loading import load_arm_blocks
    arms = load_arm_blocks(root).arm_blocks
    sym = gcfg.TONE_SYMBOLS
    khz = [gcfg.TONES[s] / 1000 for s in sym]
    colour = {"A": "#00695C", "B": "#6A1B9A"}
    examples, counts = {}, {}
    for (grammar, tier), sub in arms[arms["arm_type"] == "grammar"].groupby(["grammar", "tier"]):
        mels = [m for ms in sub["melodies"] for m in ms]
        if not mels:
            continue
        examples[(grammar, tier)] = mels[0]          # the melody shown in fig_exp2_stimuli
        c = dict.fromkeys(MOVES, 0)
        for m in mels:
            idx = [sym.index(s) for s in m]
            for a, b in zip(idx[:-1], idx[1:]):
                c[move_of(a, b)] += 1
        counts[(grammar, tier)] = c
    # each grammar's most common move, drawn in that grammar's colour wherever it appears
    move_colour = {"+1": colour["A"], "+3": colour["B"], "+2": "#9E9E9E",
                   "-1": "#CFCFCF", "-2": "#E3E3E3", "repeat": "#F2F2F2"}

    fig = plt.figure(figsize=(6.3, 5.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.18)
    for k, g in enumerate(("A", "B")):
        ax = fig.add_subplot(gs[0, k])
        mel = examples[(g, "secondary")]
        idx = [sym.index(s) for s in mel]
        for i, y in enumerate(idx):
            ax.barh(y, 0.15, left=i * 0.2, height=0.6, color=colour[g], zorder=3)
        for i in range(len(idx) - 1):
            mv = move_of(idx[i], idx[i + 1])
            borrowed = (g == "A" and mv == "+3") or (g == "B" and mv == "+1")
            c = colour["B" if g == "A" else "A"] if borrowed else "#9E9E9E"
            x0, x1 = i * 0.2 + 0.15, (i + 1) * 0.2
            ax.annotate("", xy=(x1, idx[i + 1]), xytext=(x0, idx[i]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4 if borrowed else 0.8,
                                        mutation_scale=7, shrinkA=0, shrinkB=0), zorder=2)
            ax.text((x0 + x1) / 2, max(idx[i], idx[i + 1]) + 0.55, mv, ha="center", va="bottom",
                    fontsize=5.5, color=c, fontweight="bold" if borrowed else None)
        ax.set_xlim(-0.05, 2.45); ax.set_ylim(-0.6, 6.4)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"{s} {f:.1f}" for s, f in zip(sym, khz)] if k == 0 else [])
        ax.set_xlabel("Time (s)")
        if k == 0:
            ax.set_ylabel("Tone (kHz)")
        other = "B" if g == "A" else "A"
        ax.set_title(f"({'ab'[k]}) grammar {g}, secondary\nmoves borrowed from grammar {other} in colour",
                     loc="left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    kinds = [(g, t) for g in ("A", "B") for t in TIERS]
    y = np.arange(len(kinds))[::-1]
    for yi, kind in zip(y, kinds):
        c = counts[kind]
        tot = sum(c.values())
        left = 0.0
        for mv in MOVES:
            share = c[mv] / tot if tot else 0
            if share > 0:
                ax.barh(yi, share, left=left, height=0.62, color=move_colour[mv],
                        edgecolor="white", lw=0.6)
                if share > 0.06:
                    ax.text(left + share / 2, yi, mv, ha="center", va="center", fontsize=6,
                            color="white" if mv in ("+1", "+3", "+2") else "black")
            left += share
    ax.set_yticks(y)
    ax.set_yticklabels([f"grammar {g}, {t}" for g, t in kinds])
    for lab, (g, _) in zip(ax.get_yticklabels(), kinds):
        lab.set_color(colour[g])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the moves in the melodies played at test")
    ax.set_title("(c) the moves each kind of melody makes", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = ["Share of the moves in the melodies played at test, by kind of melody"]
    for kind in kinds:
        c = counts[kind]; tot = sum(c.values())
        lines.append(f"  grammar {kind[0]}, {kind[1]:9s} n = {tot:6d}  " +
                     "  ".join(f"{mv} {c[mv] / tot:.3f}" for mv in MOVES))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--thesis-figures", metavar="DIR", required=True)
    parser.add_argument("--only", choices=["stimuli", "moves", "session", "model"],
                        help="draw only this figure")
    args = parser.parse_args()
    rc = {"font.family": "serif", "font.size": 7, "axes.titlesize": 7,
          "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42}
    want = (lambda name: args.only in (None, name))
    with plt.rc_context(rc):
        if want("stimuli"):
            fig_stimuli(args.root, os.path.join(args.thesis_figures, "fig_exp2_stimuli"))
        if want("moves"):
            txt = fig_moves(args.root, os.path.join(args.thesis_figures, "fig_exp2_moves"))
            print(txt)
            with open(os.path.join(args.root, "BATCH_ANALYSIS", "exp2", "move_shares.txt"),
                      "w", encoding="utf-8") as f:
                f.write(txt + "\n")
        if want("session") or want("model"):
            d = pd.read_csv(os.path.join(args.root, "BATCH_ANALYSIS", "exp2", "arm_blocks.csv"))
            d["mouse"] = d["mouse"].astype(str)
            if want("session"):
                fig_session(d, os.path.join(args.thesis_figures, "fig_exp2_session"))
            if want("model"):
                fig_model(os.path.join(args.root, "model_validation_output"),
                          os.path.join(args.thesis_figures, "fig_exp2_model"))
    print("written to", args.thesis_figures)


if __name__ == "__main__":
    main()
