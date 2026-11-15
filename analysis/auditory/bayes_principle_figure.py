#!/usr/bin/env python3
"""
The Principle of Bayesian Model Fitting, for General Methods
============================================================

A schematic for readers with no background in modelling (Section 2.7.1 of the
thesis). It uses no data; the curves and bars are illustrative, and the axes
carry no numbers on purpose. The results themselves are in Chapter 6.

    a  the procedure: take one possible value of a weight, let the model
       predict each arm's share of the time, ask how likely what the mice did
       is under that prediction (the likelihood), weigh that by how plausible
       the value seemed beforehand (the prior), and repeat for every value
    b  what the likelihood means for one weight: three values of a weight
       and the time each predicts for two arms, against the mice
    c  Bayes' rule as three curves: the prior (normal, mean 0, standard
       deviation 1, as in the fit) times the likelihood gives the posterior,
       with its 95% highest density interval

Usage:
    python bayes_principle_figure.py [OUT_DIR] [--thesis-figures DIR]

Output: OUT_DIR/fig_bayes_principle.png and .pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.stats import norm

# two arms in greys, so that the blue and red of the grammars keep their
# meaning for Chapter 6, where the figure's terms are introduced
ARM_A, ARM_B = "#4D4D4D", "#B3B3B3"
INK, MUTED = "#222222", "#8C8C8C"
# the fitting itself in purple (checked for colour blind separation from the
# blue and red of the grammars in the later chapters)
PURPLE, PURPLE_LIGHT, PURPLE_FILL, PURPLE_SHADE = "#7A1E9C", "#B48CC8", "#F1E7F6", "#DFC8EA"
STYLE = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
    "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
}
WIDTH = 426.79 / 72.27  # the thesis text width in inches, so that 8 pt prints as 8 pt

# illustrative values, chosen only to make the principle visible. The
# likelihood peaks at LIKE_MEAN; with a normal prior of mean 0 and standard
# deviation 1, the posterior is normal, pulled towards 0 and narrower than both.
# The likelihood is kept broad enough for the pull to be visible; with a much
# sharper one the posterior would look the same as the likelihood
LIKE_MEAN, LIKE_SD = 2.0, 0.7
POST_SD = (1 + LIKE_SD ** -2) ** -0.5
POST_MEAN = LIKE_MEAN * LIKE_SD ** -2 * POST_SD ** 2
CANDIDATES = (("None", 0.0), ("Moderate", LIKE_MEAN), ("Large", 4.5))
DELTA_S = 0.2027  # difference in score between the two arms, so that the
MICE = 0.60       # moderate weight predicts the mice's 60% on arm A


def letter(ax, text, x=-0.02, y=1.02):
    # cmr10 has no bold weight, so the letter is set in bold maths
    ax.text(x, y, rf"$\mathbf{{{text}}}$", transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom")


def box(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=PURPLE_FILL, ec=PURPLE, lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontsize=8, linespacing=1.25)


def arrow(ax, start, end, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def panel_loop(ax):
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.25, 1)
    ax.axis("off")
    xs, y, w, h = (0.115, 0.375, 0.635, 0.885), 0.6, 0.215, 0.72
    texts = ("Take one possible\nvalue of the weight",
             "The model predicts\neach arm's share\nof the time",
             "How likely is what\nthe mice did, if\nthis prediction\nwere true?",
             "Weigh that by how\nplausible the value\nseemed beforehand")
    for x, t in zip(xs, texts):
        box(ax, x, y, w, h, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w / 2 + 0.004, y), (b - w / 2 - 0.004, y))
    # the return path, under the boxes from the last back to the first
    low = y - h / 2 - 0.14
    ax.plot([xs[3], xs[3], xs[0]], [y - h / 2, low, low], color=INK, lw=0.8, solid_capstyle="butt")
    arrow(ax, (xs[0], low), (xs[0], y - h / 2))
    ax.text(0.5, low - 0.05, "Repeat for every possible value; the weighed values, drawn as a curve, "
            "are the posterior in (c)", ha="center", va="top", fontsize=7.5)


def panel_candidates(ax):
    groups = [(name, 1 / (1 + np.exp(-w * DELTA_S))) for name, w in CANDIDATES] + [("Mice", MICE)]
    for i, (name, ee) in enumerate(groups):
        x = i * 1.15
        ax.bar([x - 0.2, x + 0.2], [ee, 1 - ee], 0.36, color=[ARM_A, ARM_B], zorder=2)
        ax.text(x, -0.07, name, ha="center", va="top", fontsize=7.5)
        if name != "Mice":
            fit = "Good fit" if name == "Moderate" else "Poor fit"
            ax.text(x, 0.86, fit, ha="center", va="bottom", fontsize=7)
    ax.axvline(3 * 1.15 - 0.575, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(1.15, -0.2, "Value of the weight tried", ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.5, 3 * 1.15 + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Share of the time on\nthe two arms")


def curve_axis(ax, title):
    ax.set_xlim(-3, 4.2)
    ax.set_xticks([0])
    ax.set_xticklabels(["0"])
    ax.set_yticks([])
    ax.set_xlabel("Value of the weight")
    ax.set_title(title, linespacing=1.2)
    ax.axvline(0, color=MUTED, lw=0.5, zorder=0)


def panel_rule(axes, signs):
    x = np.linspace(-3, 4.2, 1200)
    prior, post = norm.pdf(x, 0, 1), norm.pdf(x, POST_MEAN, POST_SD)
    like = norm.pdf(x, LIKE_MEAN, LIKE_SD)
    ax_p, ax_l, ax_q = axes
    ax_p.plot(x, prior, color=PURPLE_LIGHT, lw=1.2, ls=(0, (4, 2)))
    curve_axis(ax_p, "Before the data\n(prior)")
    ax_p.set_ylabel("Probability density")
    ax_l.plot(x, like, color=INK, lw=1.2)
    curve_axis(ax_l, "How well each value\npredicts the mice (likelihood)")
    ax_l.set_ylim(0, like.max() * 1.15)
    lo, hi = norm.ppf([0.025, 0.975], POST_MEAN, POST_SD)
    inside = (x >= lo) & (x <= hi)
    # the prior and the likelihood again, faintly, so that the posterior is
    # seen to lie between them (the likelihood scaled to the posterior's height)
    ax_q.plot(x, prior, color=PURPLE_LIGHT, lw=0.8, ls=(0, (4, 2)), alpha=0.8, zorder=0)
    ax_q.plot(x, like / like.max() * post.max(), color=MUTED, lw=0.8, zorder=0)
    ax_q.text(LIKE_MEAN + LIKE_SD * 1.3, post.max() * 0.8, "likelihood", fontsize=6.5, color=MUTED,
              ha="left", va="center")
    ax_q.fill_between(x[inside], post[inside], color=PURPLE_SHADE, zorder=1)
    ax_q.plot(x, post, color=PURPLE, lw=1.4, zorder=2)
    curve_axis(ax_q, "After the data\n(posterior)")
    ax_q.annotate("95% interval", xy=(POST_MEAN - 0.35, 0.18), xytext=(-2.9, 0.62),
                  fontsize=7, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1, shrinkB=0))
    # the prior and posterior share one density scale, so the narrower curve is the taller
    top = post.max() * 1.15
    ax_p.set_ylim(0, top)
    ax_q.set_ylim(0, top)
    for ax, sym in zip(signs, (r"$\times$", r"$=$")):
        ax.axis("off")
        ax.text(0.5, 0.5, sym, ha="center", va="center", fontsize=14, transform=ax.transAxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", nargs="?", default="D:/AuditoryExperiments/grammar/BATCH_ANALYSIS/bayes_principle")
    parser.add_argument("--thesis-figures", metavar="DIR")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 6.3))
        gs = fig.add_gridspec(5, 1, height_ratios=[0.95, 0.15, 1.2, 0.85, 1.15], hspace=0.0)
        ax_a = fig.add_subplot(gs[0])
        panel_loop(ax_a)
        letter(ax_a, "a", x=0.0, y=0.98)

        mid = gs[2].subgridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.05)
        ax_b = fig.add_subplot(mid[0, 0])
        panel_candidates(ax_b)
        letter(ax_b, "b", x=-0.14, y=1.04)
        ax_key = fig.add_subplot(mid[0, 1])
        ax_key.axis("off")
        ax_key.legend(handles=[Patch(color=ARM_A, label="Arm A"),
                               Patch(color=ARM_B, label="Arm B")],
                      loc="center left", frameon=False, handlelength=1.2)

        bot = gs[4].subgridspec(1, 5, width_ratios=[1, 0.16, 1, 0.16, 1], wspace=0.05)
        curves = [fig.add_subplot(bot[0, k]) for k in (0, 2, 4)]
        signs = [fig.add_subplot(bot[0, k]) for k in (1, 3)]
        panel_rule(curves, signs)
        letter(curves[0], "c", x=-0.14, y=1.18)

        targets = [args.out_dir] + ([args.thesis_figures] if args.thesis_figures else [])
        for t in targets:
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(t, f"fig_bayes_principle.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
