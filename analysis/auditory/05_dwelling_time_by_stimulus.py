"""
05 — Dwelling time by stimulus type
====================================

Single-figure summary of how long mice dwell (per visit) in each stimulus
arm, with every stimulus type on the x-axis and dwelling time on the y-axis.

Data source
-----------
BATCH_ANALYSIS/stimulus_breakdown.csv  (produced by 01_preference_analysis.py)
Each row is one  mouse x day x trial x ROI  with:
  - avg_visit_dur_ms   : mean time per visit to that arm  ("dwelling time")
  - time_spent_ms      : total time in that arm across the trial
  - visitation_count   : number of visits
  - stim_type          : the stimulus presented in that arm

Unit of observation
--------------------
We reduce to ONE value per (mouse, stimulus): each mouse's mean dwelling
time for a given stimulus, pooled over its trials/days.  This avoids
pseudoreplication — without it, mice that contributed more trials would
dominate the test, and the per-(trial,ROI) rows are not independent.
Rows where the mouse never entered the arm (visitation_count == 0) carry no
dwell to measure and are dropped before averaging (configurable below).

Why SEM (not SD) on the plot
----------------------------
The question is whether the *mean* dwelling time differs between stimuli, so
the relevant error bar is the precision of each group mean → SEM
(= SD / sqrt(n_mice)).  SD (sample spread) is still written to the
descriptives CSV for reference.  The box itself shows the full distribution
(median, IQR, whiskers); the black marker overlays mean ± SEM.

Statistics
----------
  - Omnibus: Kruskal-Wallis across all stimuli (non-parametric; dwelling
    times are right-skewed).  Caveat: the same mice appear under several
    stimuli, so groups are not fully independent — a repeated-measures
    omnibus (Friedman) is not usable here because mice did not all receive
    every stimulus (different experiment days).  K-W on per-mouse means is
    the pragmatic choice and matches the rest of this pipeline.
  - Post-hoc: pairwise Mann-Whitney U, Holm-Bonferroni corrected across all
    pairs.  Full table saved to CSV; significant pairs printed.

Outputs (saved to BATCH_ANALYSIS/)
----------------------------------
  - fig_dwelling_time_by_stimulus.png / .pdf
  - dwelling_time_by_stimulus_descriptives.csv
  - dwelling_time_pairwise_tests.csv
  - dwelling_time_stats_report.txt

Run:
    python 05_dwelling_time_by_stimulus.py
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Make the sibling config importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preference_analysis_config import OUTPUT_DIR

# ── config ────────────────────────────────────────────────────────────

# Which column is "dwelling time".  avg_visit_dur_ms = mean time per visit.
METRIC = "avg_visit_dur_ms"
METRIC_LABEL = "Mean dwelling time per visit (ms)"

# Require at least one real visit for a row to count toward a mouse's mean.
# Set to 0 to include never-visited arms (avg_visit_dur == 0) as zeros.
MIN_VISITS = 1

# Drop a stimulus from the plot/tests if fewer than this many mice have it.
MIN_MICE = 3

# Stimuli we never want on the axis (controls / unlabelled).
EXCLUDE_STIM = {"silent", "control", "unknown", "nan", ""}

INPUT_CSV = os.path.join(OUTPUT_DIR, "stimulus_breakdown.csv")

# ── stimulus grouping (for ordering + colour) ─────────────────────────

STIM_GROUPS = {
    "smooth": "Texture",
    "rough": "Texture",
    "rough_complex": "Texture",
    "consonant": "Interval",
    "dissonant": "Interval",
    "vocalisation": "Vocalisation (synthetic)",
    "AAAAA": "Sequence",
    "AoAo": "Sequence",
    "ABAB": "Sequence",
    "ABCABC": "Sequence",
    "BABA": "Sequence",
    "ABBA": "Sequence",
}
# Anything not listed above (the named social-vocalisation files) -> this group.
DEFAULT_GROUP = "Social vocalisation"

# Order acoustic categories logically; social-voc files appended (sorted).
PREFERRED_ORDER = [
    "smooth", "rough", "rough_complex",
    "consonant", "dissonant",
    "vocalisation",
    "AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA",
]

GROUP_COLORS = {
    "Texture": "#457B9D",
    "Interval": "#2A9D8F",
    "Vocalisation (synthetic)": "#E9C46A",
    "Sequence": "#F4A261",
    "Social vocalisation": "#E63946",
}

GROUP_ORDER = [
    "Texture", "Interval", "Vocalisation (synthetic)",
    "Sequence", "Social vocalisation",
]


def stim_group(stim):
    return STIM_GROUPS.get(stim, DEFAULT_GROUP)


# ── load ──────────────────────────────────────────────────────────────

if not os.path.isfile(INPUT_CSV):
    sys.exit(
        f"ERROR: {INPUT_CSV} not found.\n"
        f"Run 01_preference_analysis.py first to generate stimulus_breakdown.csv."
    )

df = pd.read_csv(INPUT_CSV)
df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
df["visitation_count"] = pd.to_numeric(df["visitation_count"], errors="coerce")
df["stim_type"] = df["stim_type"].astype(str).str.strip()

df = df[~df["stim_type"].str.lower().isin(EXCLUDE_STIM)]
df = df.dropna(subset=[METRIC])
if MIN_VISITS > 0:
    df = df[df["visitation_count"] >= MIN_VISITS]

print(f"Loaded {len(df)} rows across {df['stim_type'].nunique()} stimulus types "
      f"and {df['mouse_id'].nunique()} mice.")

# ── reduce to one value per (mouse, stimulus) ─────────────────────────

per_mouse = (
    df.groupby(["mouse_id", "stim_type"])[METRIC]
    .mean()
    .reset_index()
)
per_mouse["group"] = per_mouse["stim_type"].map(stim_group)

# Drop stimuli seen by too few mice.
counts = per_mouse.groupby("stim_type")["mouse_id"].nunique()
keep = counts[counts >= MIN_MICE].index
dropped = sorted(set(counts.index) - set(keep))
if dropped:
    print(f"Dropping {len(dropped)} stimuli with < {MIN_MICE} mice: {dropped}")
per_mouse = per_mouse[per_mouse["stim_type"].isin(keep)]

# Final plotting order: preferred categories first, then social-voc files.
present = list(per_mouse["stim_type"].unique())
order = [s for s in PREFERRED_ORDER if s in present]
order += sorted(s for s in present if s not in PREFERRED_ORDER)

# ── descriptive statistics ────────────────────────────────────────────

desc_rows = []
for stim in order:
    vals = per_mouse.loc[per_mouse["stim_type"] == stim, METRIC].values
    n = len(vals)
    sd = np.std(vals, ddof=1) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    desc_rows.append({
        "stim_type": stim,
        "group": stim_group(stim),
        "n_mice": n,
        "mean_ms": np.mean(vals),
        "sem_ms": sem,
        "sd_ms": sd,
        "median_ms": np.median(vals),
        "q1_ms": np.percentile(vals, 25),
        "q3_ms": np.percentile(vals, 75),
    })
desc = pd.DataFrame(desc_rows)
desc.to_csv(
    os.path.join(OUTPUT_DIR, "dwelling_time_by_stimulus_descriptives.csv"),
    index=False,
)
print("\nPer-stimulus dwelling time (mean +/- SEM, ms):")
for _, r in desc.iterrows():
    print(f"  {r['stim_type']:>34s}: {r['mean_ms']:8.1f} +/- {r['sem_ms']:6.1f}  "
          f"(median {r['median_ms']:7.1f}, n={int(r['n_mice'])})")

# ── statistics ────────────────────────────────────────────────────────

report = []


def log(msg):
    print(msg)
    report.append(msg)


log("=" * 64)
log("DWELLING TIME BY STIMULUS TYPE — STATISTICS")
log("=" * 64)
log(f"Metric: {METRIC} ({METRIC_LABEL})")
log(f"Unit:   per-mouse mean dwelling time per stimulus "
    f"(rows with >= {MIN_VISITS} visit(s))")
log(f"Stimuli tested ({len(order)}): {', '.join(order)}")

# Omnibus Kruskal-Wallis
groups = [per_mouse.loc[per_mouse["stim_type"] == s, METRIC].values for s in order]
groups = [g for g in groups if len(g) > 0]
if len(groups) >= 2:
    H, p_kw = kruskal(*groups)
    log("\nOmnibus Kruskal-Wallis (across all stimuli):")
    log(f"  H = {H:.3f}, df = {len(groups) - 1}, p = {p_kw:.4g}")
    if p_kw < 0.05:
        log("  -> Dwelling time differs significantly among stimuli.")
    else:
        log("  -> No significant overall difference among stimuli.")
    log("  Caveat: same mice contribute to several stimuli, so groups are not")
    log("  fully independent; treat the omnibus p as approximate.")

# Pairwise Mann-Whitney U with Holm correction
pairs = list(combinations(order, 2))
raw_p, stat_u, pair_rows = [], [], []
for a, b in pairs:
    ga = per_mouse.loc[per_mouse["stim_type"] == a, METRIC].values
    gb = per_mouse.loc[per_mouse["stim_type"] == b, METRIC].values
    u, p = mannwhitneyu(ga, gb, alternative="two-sided")
    raw_p.append(p)
    stat_u.append(u)
    pair_rows.append((a, b, np.median(ga), np.median(gb), len(ga), len(gb)))

if raw_p:
    reject, p_holm, _, _ = multipletests(raw_p, method="holm")
    pw = pd.DataFrame({
        "stim_a": [r[0] for r in pair_rows],
        "stim_b": [r[1] for r in pair_rows],
        "median_a_ms": [r[2] for r in pair_rows],
        "median_b_ms": [r[3] for r in pair_rows],
        "n_a": [r[4] for r in pair_rows],
        "n_b": [r[5] for r in pair_rows],
        "U": stat_u,
        "p_raw": raw_p,
        "p_holm": p_holm,
        "significant_holm": reject,
    }).sort_values("p_holm").reset_index(drop=True)
    pw.to_csv(
        os.path.join(OUTPUT_DIR, "dwelling_time_pairwise_tests.csv"), index=False
    )

    n_sig = int(pw["significant_holm"].sum())
    log(f"\nPairwise Mann-Whitney U (Holm-corrected, {len(pw)} pairs): "
        f"{n_sig} significant at alpha=0.05.")
    if n_sig:
        log("  Significant pairs (median_a vs median_b, ms):")
        for _, r in pw[pw["significant_holm"]].iterrows():
            log(f"    {r['stim_a']:>20s} vs {r['stim_b']:<20s}  "
                f"{r['median_a_ms']:7.0f} vs {r['median_b_ms']:7.0f}  "
                f"p_holm={r['p_holm']:.4g}")

# ── figure ────────────────────────────────────────────────────────────

palette = {s: GROUP_COLORS[stim_group(s)] for s in order}

fig_w = max(12.0, 0.78 * len(order) + 3.0)
fig, ax = plt.subplots(figsize=(fig_w, 7))

sns.boxplot(
    data=per_mouse, x="stim_type", y=METRIC, order=order,
    palette=palette, hue="stim_type", legend=False,
    showfliers=False, width=0.65, ax=ax,
    boxprops=dict(alpha=0.55), linewidth=1.0,
)
sns.stripplot(
    data=per_mouse, x="stim_type", y=METRIC, order=order,
    color="black", alpha=0.35, size=3.5, jitter=0.18, ax=ax,
)

# Overlay mean +/- SEM (black diamonds with error bars).
xpos = np.arange(len(order))
means = desc.set_index("stim_type").loc[order, "mean_ms"].values
sems = desc.set_index("stim_type").loc[order, "sem_ms"].values
ax.errorbar(
    xpos, means, yerr=sems, fmt="D", color="black", markersize=6,
    capsize=4, elinewidth=1.4, zorder=10, label="Mean ± SEM",
)

ax.set_xlabel("Stimulus type", fontsize=12)
ax.set_ylabel(METRIC_LABEL, fontsize=12)
ax.set_title(
    "Dwelling time by stimulus type\n"
    "(box = distribution across mice; black = mean ± SEM; "
    "points = individual mice)",
    fontsize=13,
)
ax.tick_params(axis="x", rotation=55)
for lbl in ax.get_xticklabels():
    lbl.set_horizontalalignment("right")
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.15)

# Legend: stimulus-group colours + the mean±SEM marker.
group_handles = [
    mpatches.Patch(color=GROUP_COLORS[g], label=g, alpha=0.6)
    for g in GROUP_ORDER if any(stim_group(s) == g for s in order)
]
mean_handle = ax.get_legend_handles_labels()[0][-1]
ax.legend(
    handles=group_handles + [mean_handle],
    loc="upper right", fontsize=9, frameon=False, title="Stimulus group",
)

plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, "fig_dwelling_time_by_stimulus.png")
out_pdf = os.path.join(OUTPUT_DIR, "fig_dwelling_time_by_stimulus.pdf")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved {out_png}")
print(f"Saved {out_pdf}")

# ── save stats report ─────────────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, "dwelling_time_stats_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"Saved {report_path}")
print(f"\nAll outputs in: {OUTPUT_DIR}")
