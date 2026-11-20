#!/usr/bin/env python3
"""
Experiment 2: How the Value Model Is Fitted, Step by Step
=========================================================

An explanatory figure for the fitting of the model of Chapter 5 to the first
test day of Experiment 2 (Section 2.7.1 of the thesis), for readers who do not
fit models themselves. Every number in it comes from the analysis.

    a  the three scores of each kind of arm, on the scale the fit uses (each
       term divided by its standard deviation over the grammar arms), the mean
       over mice
    b  a worked example, the two dominant arms: each weight times its score,
       their sum (the arm's value), and the share of time the model predicts
       against the share the mice gave
    c  the shares of time in the seven kinds of arm, observed (mean and
       standard error over mice) and predicted without and with semantic value
    d  recovery of a known semantic weight from simulated mice
    e  how often the comparison chose the model with semantic value, for
       simulated mice made three ways
    f  AIC of the four nested models, against the model of Brielmann and Dayan

Panels a to c use exp2_model_compare.py (its features, its shares of whole
sessions and its Dirichlet fit); d and e read the output of model_validation
(results.json); f reads information_criteria.csv, written by
exp2_model_compare.py.

Usage:
    python exp2_model_explainer.py [GRAMMAR_DIR] [--thesis-figures DIR]

Output: GRAMMAR_DIR/BATCH_ANALYSIS/exp2_model/fig_model_explainer.png and .pdf,
and stats_explainer.txt
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp2_model_compare as mc
from model_validation.config import validated_config
from model_validation.data_loading import load_arm_blocks, load_day
from model_validation import latent_regressors as lr

EE, SC = "#1565C0", "#B71C1C"   # the arm colours of Chapter 6
SILENT = "#757575"              # darker than Chapter 6's grey, for contrast on white
INK, MUTED, LIGHT = "#222222", "#8C8C8C", "#D5D8DC"
STYLE = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
    "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
}
WIDTH = 426.79 / 72.27  # the thesis text width in inches, so that 8 pt prints as 8 pt

# the seven kinds of arm in the order of exp2_model_compare.CELLS (EE dominant,
# secondary, rare, SC dominant, secondary, rare, silent), drawn in pairs by level
CENTRES = np.array([0.0, 1.25, 2.5])
XPOS = np.r_[CENTRES - 0.19, CENTRES + 0.19, 3.55]
COLOURS = [EE] * 3 + [SC] * 3 + [SILENT]
BAR = 0.34


def letter(ax, text, x=-0.02, y=1.02):
    # cmr10 has no bold weight, so the letter is set in bold maths
    ax.text(x, y, rf"$\mathbf{{{text}}}$", transform=ax.transAxes, fontsize=11,
            ha="right", va="bottom")


def num(x, digits=2):
    """A signed number as mathtext, so that its minus is a true minus sign."""
    return f"${x:+.{digits}f}$"


def arm_axis(ax, labels=("Dominant", "Secondary", "Rare", "Silent")):
    ax.set_xticks(np.r_[CENTRES, XPOS[6]])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 3.95)


def panel_scores(axes, scores):
    for k, (ax, title) in enumerate(zip(axes, ("Fluency", "Learning", "Semantic value"))):
        ax.bar(XPOS, scores[:, k], BAR, color=COLOURS, zorder=2)
        ax.axhline(0, color=INK, lw=0.6, zorder=3)
        # the silent arm scores zero on every term; a short mark shows it is not missing
        ax.plot([XPOS[6] - BAR / 2, XPOS[6] + BAR / 2], [0, 0], color=SILENT, lw=2.4,
                solid_capstyle="butt", zorder=4)
        ax.set_title(title, pad=4)
        arm_axis(ax, ("Dom.", "Sec.", "Rare", "Silent"))
        if k:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Score (standard deviations)")


def panel_example(ax, scores, w, pred, obs):
    names = ("Fluency", "Learning", "Semantic", "Value")
    ticks, labels = [], []
    for g, (cell, colour, head) in enumerate(((0, EE, "Enriched cage grammar, dominant level"),
                                              (3, SC, "Standard cage grammar, dominant level"))):
        parts = w * scores[cell]
        vals = np.r_[parts, parts.sum()]
        top = 4.0 - g * 5.4
        ys = top - np.arange(4)
        for i, (y, v) in enumerate(zip(ys, vals)):
            ax.barh(y, v, 0.62, color=colour, alpha=1.0 if i == 3 else 0.42, zorder=2)
        ax.text(-0.33, top + 0.72, head, ha="left", va="center", fontsize=8.5, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.text(0.78, ys[2], f"Predicted {pred[cell] * 100:.1f}%", ha="right", va="center", fontsize=8.5)
        ax.text(0.78, ys[3], f"Mice {obs[cell] * 100:.1f}%", ha="right", va="center", fontsize=8.5)
        ticks += list(ys)
        labels += list(names)
    ax.axvline(0, color=INK, lw=0.6, zorder=3)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-0.2, 0, 0.2])
    ax.set_xlim(-0.33, 0.78)
    ax.set_ylim(-5.0, 5.2)
    ax.set_xlabel("Weight $\\times$ score")
    ax.spines["left"].set_visible(False)


def panel_shares(ax, obs, sem, base, full):
    ax.bar(XPOS, obs, BAR, color=COLOURS, zorder=2)
    ax.errorbar(XPOS, obs, sem, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=3)
    ax.plot(XPOS, base, "o", ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=4)
    ax.plot(XPOS, full, "D", ms=3.8, color=INK, zorder=5)
    arm_axis(ax)
    ax.set_ylim(0, 0.25)
    ax.set_ylabel("Share of the session's time")
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=INK, mew=0.8,
               label="Predicted without semantic value"),
        Line2D([], [], ls="none", marker="D", ms=3.8, color=INK, label="Predicted with semantic value")],
        loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.1)


def panel_recovery(ax, grid, fpr):
    true = np.array([g["wS_true"] for g in grid])
    rec = np.array([g["wS_recovered_mean"] for g in grid])
    lo = rec - np.array([g["ci"][0] for g in grid])
    hi = np.array([g["ci"][1] for g in grid]) - rec
    ax.plot([-0.2, 2.3], [-0.2, 2.3], color=LIGHT, lw=1.2, zorder=1)
    ax.errorbar(true, rec, [lo, hi], fmt="o", ms=3.5, color=INK, elinewidth=0.8, capsize=2, zorder=2)
    ax.set_xticks(true)
    ax.set_xticklabels(["0", "0.5", "1", "2"])
    ax.set_yticks(true)
    ax.set_yticklabels(["0", "0.5", "1", "2"])
    ax.set_xlim(-0.2, 2.3)
    ax.set_ylim(-0.2, 2.3)
    ax.set_xlabel("Semantic weight put in")
    ax.set_ylabel("Weight recovered")
    ax.text(2.25, -0.1, f"False positives\n{fpr * 100:.1f}%", ha="right", va="bottom", fontsize=8.5)


def panel_bars(ax, labels, values, xlabel, fmt, xlim, highlight):
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, 0.6, color=[INK if i == highlight else MUTED for i in range(len(values))],
            zorder=2)
    ax.axvline(0, color=INK, lw=0.6, zorder=3)
    span = xlim[1] - xlim[0]
    for yy, v in zip(y, values):
        # every value is written to the right of zero, clear of the row labels
        ax.text(max(v, 0) + 0.03 * span, yy, fmt(v), ha="left", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("grammar_dir", nargs="?", default="D:/AuditoryExperiments/grammar")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    out = os.path.join(args.grammar_dir, "BATCH_ANALYSIS", "exp2_model")
    os.makedirs(out, exist_ok=True)

    # the analysis of exp2_model_compare.py, at the grain of whole sessions
    cfg = validated_config()
    day1 = load_day(load_arm_blocks(args.grammar_dir).arm_blocks, "primary")
    day1 = day1[day1.groupby("mouse")["time_spent_s"].transform("sum") > 0]
    feat = lr.build_features(day1, cfg, emission="tier_restricted")
    X, Y = mc.session_data(feat)
    fits = {m: mc.fit_dirichlet(X, Y, mc.MODELS[m])[1] for m in ("baseline", "full")}
    w = fits["full"][:3]
    base = mc.predict(X, fits["baseline"][:2], mc.MODELS["baseline"]).mean(axis=0)
    full = mc.predict(X, w, mc.MODELS["full"]).mean(axis=0)
    obs, sem = Y.mean(axis=0), Y.std(axis=0, ddof=1) / np.sqrt(len(Y))
    scores = X.mean(axis=0)

    val = json.load(open(os.path.join(args.grammar_dir, "model_validation_output", "results.json")))
    rec = val["recovery"]["parameter_recovery"]
    conf = val["recovery"]["model_confusion"]
    ic = pd.read_csv(os.path.join(out, "information_criteria.csv"))
    ic = ic[(ic["grain"] == "session") & (ic["likelihood"] == "Dirichlet")].set_index("model")
    d_aic = (ic["AIC"] - ic.loc["baseline", "AIC"]).loc[["intercept", "fluency", "baseline", "full"]]
    choice = [conf[k]["prefer_full_rate"] * 100 for k in ("bd_baseline", "full", "intrinsic_grammar")]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 6.6))
        outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 1.0], hspace=0.5,
                                 top=0.93)
        top = outer[0].subgridspec(1, 3, wspace=0.12)
        ax_a = [fig.add_subplot(top[0, 0])]
        ax_a += [fig.add_subplot(top[0, k], sharey=ax_a[0]) for k in (1, 2)]
        panel_scores(ax_a, scores)
        letter(ax_a[0], "a", x=-0.36)
        mid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.42)
        ax_b = fig.add_subplot(mid[0, 0])
        panel_example(ax_b, scores, w, full, obs)
        letter(ax_b, "b", x=-0.24)
        ax_c = fig.add_subplot(mid[0, 1])
        panel_shares(ax_c, obs, sem, base, full)
        letter(ax_c, "c", x=-0.2)
        bot = outer[2].subgridspec(1, 3, width_ratios=[0.72, 0.85, 0.85], wspace=1.05)
        ax_d = fig.add_subplot(bot[0, 0])
        panel_recovery(ax_d, rec["grid"], rec["false_positive_rate_at_null"])
        letter(ax_d, "d", x=-0.3)
        ax_e = fig.add_subplot(bot[0, 1])
        panel_bars(ax_e, ["Without semantic\nvalue", "With semantic\nvalue", "One grammar\nliked"],
                   choice, "Simulations choosing\nsemantic value", lambda v: f"{v:.0f}%", (0, 140), 1)
        letter(ax_e, "e", x=-0.72)
        ax_f = fig.add_subplot(bot[0, 2])
        panel_bars(ax_f, ["Baseline alone", "Fluency", "Brielmann\nand Dayan", "With semantic\nvalue"],
                   d_aic.to_numpy(), "Change in AIC\nfrom Brielmann\nand Dayan",
                   lambda v: "$0$" if abs(v) < 0.05 else num(v, 1), (-40, 26), 3)
        letter(ax_f, "f", x=-0.72)
        fig.legend(handles=[Patch(color=EE, label="Enriched cage grammar"),
                            Patch(color=SC, label="Standard cage grammar"),
                            Patch(color=SILENT, label="Silent arm")],
                   loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.2, columnspacing=1.6)
        targets = [out] + ([args.thesis_figures] if args.thesis_figures else [])
        for t in targets:
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(t, f"fig_exp2_model_explainer.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Panel c on its own, for Section 6.7 (it replaces the table of observed and
        # predicted shares): the bars are the mice, the markers the two models, and
        # the index of EE over SC of each is written beside the legend.
        def ee_sc(v):
            return (v[:3].sum() - v[3:6].sum()) / (v[:3].sum() + v[3:6].sum())
        # full text width, legend above the plot, so the figure is not shrunk on
        # the page and its text prints at its size
        fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
        panel_shares(ax, obs, sem, base, full)
        ax.legend(handles=[
            Patch(color=EE, label="EE grammar, mice"), Patch(color=SC, label="SC grammar, mice"),
            Patch(color=SILENT, label="Silent arm, mice"),
            Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK, mew=0.8,
                   label="Predicted without semantic value"),
            Line2D([], [], ls="none", marker="D", ms=4.2, color=INK, label="Predicted with semantic value")],
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
            handletextpad=0.4, columnspacing=1.2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.text(0.99, 0.98, "Index, EE over SC\n"
                f"mice {num(ee_sc(obs), 3)}\nwithout {num(ee_sc(base), 3)}\nwith {num(ee_sc(full), 3)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        for t in targets:
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(t, f"fig_exp2_shares.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    names = ["EE dominant", "EE secondary", "EE rare", "SC dominant", "SC secondary", "SC rare", "silent"]
    lines = [f"Mice: {len(Y)}; weights of the full model (Dirichlet, whole sessions): "
             f"wr {w[0]:+.3f}, wV {w[1]:+.3f}, wS {w[2]:+.3f}",
             "Scores (fluency, learning, semantic; standard deviations), shares observed (SE), "
             "predicted without and with semantic value:"]
    for i, n in enumerate(names):
        lines.append(f"  {n:13s} {scores[i, 0]:+.3f} {scores[i, 1]:+.3f} {scores[i, 2]:+.3f} | "
                     f"{obs[i]:.3f} ({sem[i]:.3f}) | {base[i]:.3f} | {full[i]:.3f}")
    for cell in (0, 3):
        parts = w * scores[cell]
        lines.append(f"  {names[cell]}: weight x score {parts[0]:+.3f}, {parts[1]:+.3f}, "
                     f"{parts[2]:+.3f}; value {parts.sum():+.3f}")
    lines.append("Recovery: " + "; ".join(f"put in {g['wS_true']:g}, recovered {g['wS_recovered_mean']:+.3f} "
                                          f"({g['ci'][0]:+.3f} to {g['ci'][1]:+.3f})" for g in rec["grid"])
                 + f"; found when absent in {rec['false_positive_rate_at_null'] * 100:.1f}%")
    lines.append("Simulations choosing the model with semantic value: " + ", ".join(
        f"{k} {v:.0f}%" for k, v in zip(("no semantic value", "semantic value", "one grammar liked"), choice)))
    lines.append("AIC against Brielmann and Dayan: " + ", ".join(f"{k} {v:+.1f}" for k, v in d_aic.items()))
    with open(os.path.join(out, "stats_explainer.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nAll outputs saved to:\n  {out}")


if __name__ == "__main__":
    main()
