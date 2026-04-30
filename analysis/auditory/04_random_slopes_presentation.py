#!/usr/bin/env python3
"""
04 - Random-Slope Mixed Model + Presentation Trajectories
==========================================================

Self-contained follow-up to ``03_completers_lmm.py``. For each of three
preference outcomes, fits a single linear mixed model with random
intercept AND random slope on session number per mouse:

    outcome ~ session_num + (1 + session_num | mouse_id)

This lets each mouse have its own baseline AND its own learning
trajectory. Designed for slide-deck style presentation figures.

Outcomes
--------
    preference_index    Overall PI
    voc_pi              Vocalisation PI
    other_sounds_pi     Other-sounds PI

Per-outcome reports
-------------------
    * Fixed-effect estimate, SE, z, p, and 95 % CI (intercept + session_num)
    * Random-effect variance components:
        Var(intercept|mouse), Var(slope|mouse),
        Cov(intercept,slope|mouse), Var(resid)
    * Variance decomposition (% from mouse intercept / mouse slope / residual)
    * Convergence + boundary / singular-fit flag
    * Per-mouse BLUPs for both intercept and slope

Outputs
-------
    completers_random_slopes_report.txt
    fig_completers_random_slopes.png/pdf      caterpillar of slopes per outcome
    fig_presentation_trajectories.png/pdf     3 outcomes x 3 panels (slide style)
    completers_random_slopes_blups.csv        per-mouse intercept + slope BLUPs

Quick start
-----------
    python 04_random_slopes_presentation.py "C:\\path\\to\\8_arms_w_voc"
    python 04_random_slopes_presentation.py --csv preference_data.csv

Dependencies: numpy, pandas, matplotlib, scipy, statsmodels.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("ERROR: statsmodels is required. Install with:  pip install statsmodels")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── experiment structure (kept in sync with 03_completers_lmm.py) ──────────
PI_DAYS = ["w1_d1", "w1_d2", "w1_d3", "w2_sequences", "w2_vocalisations"]
DAY_SHORT = {
    "w1_d1": "D1 (TEM)",
    "w1_d2": "D2 (Int)",
    "w1_d3": "D3 (Int)",
    "w2_sequences": "W2 (Seq)",
    "w2_vocalisations": "W2 (Voc)",
}

# (column_name, pretty_label)
OUTCOMES = [
    ("preference_index", "Overall PI"),
    ("voc_pi",           "Vocalisation PI"),
    ("other_sounds_pi",  "Other Sounds PI"),
]

# Wong (2011) colour-blind-safe palette
COLOUR_POSITIVE = "#D55E00"   # vermillion -- "increasing"
COLOUR_NEGATIVE = "#0072B2"   # blue       -- "decreasing"
COLOUR_NEUTRAL  = "#999999"   # grey       -- ~zero / fallback
COLOUR_GROUP    = "#000000"   # black      -- group mean / fixed-effect line
COLOUR_SCATTER_HIGH = "#D55E00"
COLOUR_SCATTER_LOW  = "#0072B2"

SINGULAR_TOL = 1e-6   # variance/eigenvalue threshold for "boundary" flag


# ─── helpers ────────────────────────────────────────────────────────────────


def log(msg: str, lines: list[str]) -> None:
    print(msg)
    lines.append(msg)


def safe_savefig(fig, path: str, **kwargs) -> None:
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: could not write {os.path.basename(path)} (file locked)")


def safe_random_effects(result):
    """``result.random_effects`` but returns {} if the RE covariance is
    singular (statsmodels raises LinAlgError -> ValueError in that case)."""
    try:
        return dict(result.random_effects)
    except (np.linalg.LinAlgError, ValueError, Exception):
        return {}


def safe_random_effects_cov(result):
    try:
        return dict(result.random_effects_cov)
    except (np.linalg.LinAlgError, ValueError, Exception):
        return {}


def find_completers(df: pd.DataFrame, required_days: list[str]) -> list[str]:
    """Mice present on every day in ``required_days``."""
    counts = (df.groupby("mouse_id")["day"]
                .apply(lambda s: set(s.unique()))
                .to_dict())
    needed = set(required_days)
    return sorted([m for m, d in counts.items() if needed.issubset(d)])


def fit_random_slope(df: pd.DataFrame, outcome: str):
    """Fit ``outcome ~ session_num + (1 + session_num | mouse_id)`` on the
    completer subset. Returns (result, info) where ``info`` flags
    convergence and boundary / singular fits."""
    d = df.dropna(subset=[outcome, "session_num", "mouse_id"]).copy()

    info = {
        "n_obs":      int(len(d)),
        "n_groups":   int(d["mouse_id"].nunique()),
        "converged":  None,
        "singular":   None,
        "fit_method": None,
        "error":      None,
    }
    if info["n_groups"] < 3 or info["n_obs"] < 6:
        info["error"] = (f"insufficient data "
                         f"(n_obs={info['n_obs']}, n_groups={info['n_groups']})")
        return None, info

    md = smf.mixedlm(f"{outcome} ~ session_num", d,
                     groups=d["mouse_id"],
                     re_formula="~session_num")
    try:
        res = md.fit(reml=True, method="lbfgs")
        info["fit_method"] = "lbfgs"
    except Exception:
        try:
            res = md.fit(reml=True)
            info["fit_method"] = "default"
        except Exception as e:
            info["error"] = f"fit failed: {e}"
            return None, info

    info["converged"] = bool(res.converged) if hasattr(res, "converged") else None

    # Boundary / singular check: any near-zero variance or near-zero eigenvalue
    cov_re = np.asarray(res.cov_re)
    diag = np.diag(cov_re).astype(float)
    try:
        eigvals = np.linalg.eigvalsh(cov_re)
    except np.linalg.LinAlgError:
        eigvals = np.array([np.nan])
    info["singular"] = bool(
        np.any(diag < SINGULAR_TOL) or np.any(eigvals < SINGULAR_TOL)
    )

    return res, info


def fixed_effect_table(res) -> pd.DataFrame:
    """Estimate, SE, z, p, 95 % CI for each fixed effect."""
    ci = res.conf_int()
    ci.columns = ["ci_lo", "ci_hi"]
    out = pd.DataFrame({
        "estimate": res.fe_params,
        "std_err":  res.bse_fe,
        "z":        res.tvalues[res.fe_params.index],
        "p":        res.pvalues[res.fe_params.index],
    }).join(ci.loc[res.fe_params.index])
    out.insert(0, "term", out.index)
    return out.reset_index(drop=True)


def variance_components(res) -> dict:
    """Return Var(int), Var(slope), Cov(int,slope), Var(resid)."""
    cov_re = np.asarray(res.cov_re)
    var_int   = float(cov_re[0, 0]) if cov_re.shape[0] >= 1 else np.nan
    var_slope = float(cov_re[1, 1]) if cov_re.shape[0] >= 2 else np.nan
    cov_is    = float(cov_re[0, 1]) if cov_re.shape[0] >= 2 else np.nan
    var_resid = float(res.scale)
    return {
        "var_intercept": var_int,
        "var_slope":     var_slope,
        "cov_int_slope": cov_is,
        "var_resid":     var_resid,
    }


def variance_decomposition(vc: dict) -> dict:
    """Simple % decomposition: each component divided by the sum of the
    diagonal (intercept + slope + residual). Note this is a quick
    interpretive summary, not a formal ICC -- the contribution of the
    random slope to outcome variance also depends on the scale of
    ``session_num`` (so units matter)."""
    a = vc["var_intercept"]
    b = vc["var_slope"]
    c = vc["var_resid"]
    total = a + b + c
    if total <= 0 or not np.isfinite(total):
        return {"pct_intercept": np.nan, "pct_slope": np.nan,
                "pct_resid": np.nan, "pct_between": np.nan}
    return {
        "pct_intercept": 100.0 * a / total,
        "pct_slope":     100.0 * b / total,
        "pct_resid":     100.0 * c / total,
        "pct_between":   100.0 * (a + b) / total,
    }


def per_mouse_blups(res) -> pd.DataFrame:
    """Per-mouse intercept + slope BLUPs with posterior SDs (95 % CI)."""
    re_dict = safe_random_effects(res)
    cov_dict = safe_random_effects_cov(res)

    rows = []
    for g, vec in re_dict.items():
        # Index by re_formula term name when possible; statsmodels stores them
        # under design names like "Group" (intercept) and "session_num".
        try:
            int_blup = float(vec.iloc[0])
            slp_blup = float(vec.iloc[1]) if len(vec) > 1 else np.nan
        except Exception:
            arr = np.asarray(vec)
            int_blup = float(arr[0]) if arr.size > 0 else np.nan
            slp_blup = float(arr[1]) if arr.size > 1 else np.nan

        int_se = slp_se = np.nan
        if g in cov_dict:
            try:
                pcov = np.asarray(cov_dict[g])
                int_se = float(np.sqrt(pcov[0, 0])) if pcov.shape[0] >= 1 else np.nan
                slp_se = float(np.sqrt(pcov[1, 1])) if pcov.shape[0] >= 2 else np.nan
            except Exception:
                pass
        rows.append({
            "mouse_id": g,
            "intercept_blup": int_blup,
            "intercept_se":   int_se,
            "slope_blup":     slp_blup,
            "slope_se":       slp_se,
        })
    return pd.DataFrame(rows)


# ─── plotting: caterpillar of slopes ────────────────────────────────────────


def plot_slope_caterpillar(blups_by_outcome: dict, output_dir: str) -> None:
    """One panel per outcome, slopes ranked with 95 % CI."""
    n = len(OUTCOMES)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), squeeze=False)
    z95 = norm.ppf(0.975)

    for ax, (col, label) in zip(axes[0], OUTCOMES):
        bdf = blups_by_outcome.get(col)
        if bdf is None or bdf.empty or bdf["slope_blup"].isna().all():
            ax.set_title(f"{label}\n(slopes not identifiable)")
            ax.text(0.5, 0.5,
                    "Random-slope variance ~ 0:\nBLUPs not identifiable",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            continue

        d = bdf.sort_values("slope_blup").reset_index(drop=True)
        y = np.arange(len(d))
        slopes = d["slope_blup"].to_numpy(dtype=float)
        ses    = d["slope_se"].to_numpy(dtype=float)
        # colour by sign of point estimate
        colours = np.where(slopes >= 0, COLOUR_POSITIVE, COLOUR_NEGATIVE)

        ax.errorbar(slopes, y, xerr=z95 * ses,
                    fmt="none", ecolor="#aaaaaa", elinewidth=1, capsize=2)
        ax.scatter(slopes, y, c=colours, s=42, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(d["mouse_id"].tolist(), fontsize=8)
        ax.set_xlabel("Slope BLUP\n(change in PI per session)")
        ax.set_title(f"{label}\nper-mouse trajectory")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-mouse random slopes (random-slope LMM)", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "fig_completers_random_slopes.png"),
                 dpi=200, bbox_inches="tight")
    safe_savefig(fig, os.path.join(output_dir, "fig_completers_random_slopes.pdf"),
                 bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_completers_random_slopes")


# ─── plotting: 3 x 3 presentation panel ─────────────────────────────────────


def _shared_ylim(df_completers: pd.DataFrame, outcome: str,
                 blups: pd.DataFrame, sess_min: int, sess_max: int) -> tuple:
    """Compute a single y-axis range usable across all 3 panels of one row."""
    ys = df_completers[outcome].dropna().to_numpy(dtype=float)
    if blups is not None and not blups.empty:
        # fitted endpoints from BLUPs span a similar range to the raw data;
        # include them to be safe
        a = blups["intercept_blup"].to_numpy(dtype=float)
        b = blups["slope_blup"].to_numpy(dtype=float)
        # intercept_blup in statsmodels is the deviation from the grand mean,
        # so for endpoint range we just use the raw data extremes
    if len(ys) == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(ys))
    hi = float(np.nanmax(ys))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return (lo - 0.1, hi + 0.1) if np.isfinite(lo) else (-1.0, 1.0)
    pad = 0.08 * (hi - lo)
    return (lo - pad, hi + pad)


def _strip_ax(ax) -> None:
    """Minimal slide-deck aesthetic: no top/right spines, no grid."""
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_presentation_trajectories(df_c: pd.DataFrame,
                                   blups_by_outcome: dict,
                                   fe_by_outcome: dict,
                                   output_dir: str) -> None:
    """3 outcomes (rows) x 3 panel types (columns) for slide presentation.

    Cols:  spaghetti  |  fan (BLUP-fitted)  |  first-vs-last scatter
    """
    plt.rcParams.update({
        "font.size":        14,
        "axes.titlesize":   15,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    n_rows = len(OUTCOMES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5.6 * n_rows), squeeze=False)

    # X axis: 1-based session numbers for display
    sessions_disp = np.arange(1, len(PI_DAYS) + 1)
    sess_ticks = sessions_disp
    sess_labels = [DAY_SHORT[d] for d in PI_DAYS]

    for r, (col, label) in enumerate(OUTCOMES):
        ax_sp, ax_fan, ax_sc = axes[r]
        d = df_c.dropna(subset=[col]).copy()
        # session_num in df_c is 0-based; for display we show 1-based
        d["session_disp"] = d["session_num"].astype(int) + 1
        ymin, ymax = _shared_ylim(d, col, blups_by_outcome.get(col),
                                  sess_min=1, sess_max=len(PI_DAYS))

        # ── Panel 1: spaghetti (raw) ───────────────────────────────────────
        for mouse, g in d.groupby("mouse_id"):
            g = g.sort_values("session_disp")
            ax_sp.plot(g["session_disp"], g[col],
                       color=COLOUR_NEUTRAL, alpha=0.45, lw=1.2,
                       marker="o", ms=3.5, mfc=COLOUR_NEUTRAL, mec="white",
                       mew=0.4)
        # group mean line
        grp = (d.groupby("session_disp")[col]
                 .mean().reindex(sessions_disp).reset_index())
        ax_sp.plot(grp["session_disp"], grp[col],
                   color=COLOUR_GROUP, lw=3.2, marker="o", ms=7,
                   mfc=COLOUR_GROUP, mec="white", mew=1.0,
                   label="Group mean", zorder=5)
        ax_sp.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax_sp.text(sessions_disp[-1], 0,
                   "  No preference",
                   va="center", ha="left", fontsize=11, color="#555555")
        ax_sp.set_xticks(sess_ticks)
        ax_sp.set_xticklabels(sess_labels, rotation=20, ha="right")
        ax_sp.set_ylim(ymin, ymax)
        ax_sp.set_ylabel(f"{label}")
        ax_sp.set_title("Individual trajectories across sessions")
        ax_sp.legend(loc="best", frameon=False)
        _strip_ax(ax_sp)

        # ── Panel 2: fan plot (BLUP-fitted) ───────────────────────────────
        bdf = blups_by_outcome.get(col)
        fe_df = fe_by_outcome.get(col)
        # Population fixed effects
        if fe_df is not None:
            try:
                fe_int = float(fe_df.loc[fe_df["term"] == "Intercept",
                                         "estimate"].iloc[0])
                fe_slp = float(fe_df.loc[fe_df["term"] == "session_num",
                                         "estimate"].iloc[0])
            except Exception:
                fe_int = fe_slp = np.nan
        else:
            fe_int = fe_slp = np.nan

        if bdf is not None and not bdf.empty and not bdf["slope_blup"].isna().all():
            # statsmodels BLUPs are deviations from the fixed effects, so
            # per-mouse line = (fe_int + int_blup) + (fe_slp + slope_blup) * x
            x_model = np.array([0, len(PI_DAYS) - 1])  # 0-based model x
            x_disp  = x_model + 1                       # 1-based display x
            n_pos = n_neg = 0
            for _, row in bdf.iterrows():
                a = fe_int + float(row["intercept_blup"])
                b = fe_slp + float(row["slope_blup"])
                col_line = (COLOUR_POSITIVE if row["slope_blup"] >= 0
                            else COLOUR_NEGATIVE)
                if row["slope_blup"] >= 0:
                    n_pos += 1
                else:
                    n_neg += 1
                ax_fan.plot(x_disp, a + b * x_model,
                            color=col_line, alpha=0.55, lw=1.6)
            # population line
            if np.isfinite(fe_int) and np.isfinite(fe_slp):
                ax_fan.plot(x_disp, fe_int + fe_slp * x_model,
                            color=COLOUR_GROUP, lw=3.4,
                            label="Population fixed effect")
            # legend swatches
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=COLOUR_POSITIVE, lw=2.4,
                       label=f"Increasing (n={n_pos})"),
                Line2D([0], [0], color=COLOUR_NEGATIVE, lw=2.4,
                       label=f"Decreasing (n={n_neg})"),
                Line2D([0], [0], color=COLOUR_GROUP, lw=3.0,
                       label="Population mean"),
            ]
            ax_fan.legend(handles=handles, loc="best", frameon=False)
        else:
            ax_fan.text(0.5, 0.5,
                        "Random-slope variance ~ 0:\nBLUPs not identifiable",
                        ha="center", va="center", transform=ax_fan.transAxes,
                        fontsize=12, color="#888888")
        ax_fan.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax_fan.set_xticks(sess_ticks)
        ax_fan.set_xticklabels(sess_labels, rotation=20, ha="right")
        ax_fan.set_ylim(ymin, ymax)
        ax_fan.set_title("Model-estimated individual trajectories")
        _strip_ax(ax_fan)

        # ── Panel 3: first vs last scatter ────────────────────────────────
        first_day = PI_DAYS[0]
        last_day  = PI_DAYS[-1]
        # mean per mouse on each (in case there are multiple sessions on the same day)
        first = (d[d["day"] == first_day]
                 .groupby("mouse_id")[col].mean())
        last  = (d[d["day"] == last_day]
                 .groupby("mouse_id")[col].mean())
        merged = (pd.concat([first.rename("first"), last.rename("last")],
                            axis=1)
                    .dropna())
        if not merged.empty:
            xs = merged["first"].to_numpy(dtype=float)
            ys = merged["last"].to_numpy(dtype=float)
            lo = float(np.nanmin([xs.min(), ys.min()]))
            hi = float(np.nanmax([xs.max(), ys.max()]))
            pad = 0.08 * (hi - lo) if hi > lo else 0.1
            lo -= pad; hi += pad
            # diagonal
            ax_sc.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.4,
                       label="No change (y = x)")
            colours = np.where(ys >= xs, COLOUR_SCATTER_HIGH, COLOUR_SCATTER_LOW)
            ax_sc.scatter(xs, ys, c=colours, s=110, edgecolor="white",
                          linewidth=1.2, zorder=4)
            for mouse, x_, y_ in zip(merged.index, xs, ys):
                ax_sc.annotate(str(mouse), (x_, y_),
                               xytext=(5, 5), textcoords="offset points",
                               fontsize=9, color="#444444")
            ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
            ax_sc.legend(loc="best", frameon=False)
        else:
            ax_sc.text(0.5, 0.5, "No first/last paired data",
                       ha="center", va="center", transform=ax_sc.transAxes,
                       fontsize=12, color="#888888")

        ax_sc.set_xlabel("PI at start")
        ax_sc.set_ylabel("PI at end")
        ax_sc.set_title("Did individual mice change?")
        _strip_ax(ax_sc)

        # row label on the leftmost panel via y-label already set above

    fig.suptitle("Random-slope mixed model -- presentation panel",
                 y=1.005, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=2.0, w_pad=2.5)
    safe_savefig(fig, os.path.join(output_dir, "fig_presentation_trajectories.png"),
                 dpi=220, bbox_inches="tight")
    safe_savefig(fig, os.path.join(output_dir, "fig_presentation_trajectories.pdf"),
                 bbox_inches="tight")
    plt.close(fig)
    # restore default rc
    plt.rcdefaults()
    print("  Saved fig_presentation_trajectories")


# ─── reporting ──────────────────────────────────────────────────────────────


def report_outcome(label: str, outcome: str, res, info: dict,
                   stats_lines: list[str]) -> tuple:
    """Run reporting for a single outcome. Returns (fe_df, blups_df) or
    (None, None) when the model failed."""
    log("=" * 70, stats_lines)
    log(f"Outcome: {label}  ({outcome})", stats_lines)
    log(f"Formula: {outcome} ~ session_num + (1 + session_num | mouse_id)",
        stats_lines)
    log("-" * 70, stats_lines)
    log(f"  n_obs = {info['n_obs']}, n_groups = {info['n_groups']}",
        stats_lines)

    if res is None:
        log(f"  MODEL DID NOT FIT  ({info.get('error', 'unknown error')})",
            stats_lines)
        log("", stats_lines)
        return None, None

    # Convergence + singular flag
    log(f"  Converged           : {info['converged']}", stats_lines)
    log(f"  Optimiser           : {info['fit_method']}", stats_lines)
    if info.get("singular"):
        log("  *** SINGULAR FIT *** : random-effects covariance is on the boundary "
            "(near-zero variance / eigenvalue). Estimates are still reported, "
            "but per-mouse BLUPs may be unidentifiable.", stats_lines)
    log("", stats_lines)

    # Fixed effects
    fe_df = fixed_effect_table(res)
    log("  Fixed effects:", stats_lines)
    log("    {:<14s} {:>10s} {:>10s} {:>9s} {:>9s} {:>10s} {:>10s}".format(
        "term", "estimate", "std_err", "z", "p", "ci_lo", "ci_hi"),
        stats_lines)
    for _, row in fe_df.iterrows():
        log("    {:<14s} {:>10.4f} {:>10.4f} {:>9.3f} {:>9.4f} {:>10.4f} {:>10.4f}"
            .format(str(row["term"]), row["estimate"], row["std_err"],
                    row["z"], row["p"], row["ci_lo"], row["ci_hi"]),
            stats_lines)
    log("", stats_lines)

    # Variance components
    vc = variance_components(res)
    log("  Random-effect variance components:", stats_lines)
    log(f"    Var(intercept | mouse) = {vc['var_intercept']:.5f}", stats_lines)
    log(f"    Var(slope     | mouse) = {vc['var_slope']:.5f}", stats_lines)
    log(f"    Cov(int, slope| mouse) = {vc['cov_int_slope']:.5f}", stats_lines)
    log(f"    Var(residual)          = {vc['var_resid']:.5f}", stats_lines)
    if vc["var_intercept"] > 0 and vc["var_slope"] > 0:
        corr_is = vc["cov_int_slope"] / np.sqrt(vc["var_intercept"]
                                                * vc["var_slope"])
        log(f"    Corr(int, slope)       = {corr_is:.3f}", stats_lines)
    log("", stats_lines)

    # Variance decomposition
    vd = variance_decomposition(vc)
    log("  Variance decomposition (% of intercept+slope+residual):",
        stats_lines)
    log(f"    From mouse intercept : {vd['pct_intercept']:.1f} %", stats_lines)
    log(f"    From mouse slope     : {vd['pct_slope']:.1f} %", stats_lines)
    log(f"    Residual             : {vd['pct_resid']:.1f} %", stats_lines)
    log(f"    -> total between-mouse: {vd['pct_between']:.1f} %", stats_lines)
    log("    (Note: this is a quick decomposition; the random-slope "
        "contribution\n     scales with session_num units. ICC at session=0 "
        "is", stats_lines)
    if vc["var_intercept"] + vc["var_resid"] > 0:
        icc0 = vc["var_intercept"] / (vc["var_intercept"] + vc["var_resid"])
        log(f"     Var(int) / (Var(int)+Var(resid)) = {icc0:.3f}.)",
            stats_lines)
    else:
        log("     Var(int) / (Var(int)+Var(resid)) = NA.)", stats_lines)
    log("", stats_lines)

    # Per-mouse BLUPs
    blups_df = per_mouse_blups(res)
    if blups_df.empty:
        log("  Per-mouse BLUPs       : NOT IDENTIFIABLE (singular RE covariance)",
            stats_lines)
    else:
        blups_sorted = blups_df.sort_values("slope_blup")
        log("  Per-mouse BLUPs (sorted by slope, ascending):", stats_lines)
        log("    {:<12s} {:>12s} {:>10s} {:>12s} {:>10s}".format(
            "mouse_id", "intercept", "int_se", "slope", "slope_se"),
            stats_lines)
        for _, row in blups_sorted.iterrows():
            log("    {:<12s} {:>12.4f} {:>10.4f} {:>12.4f} {:>10.4f}".format(
                str(row["mouse_id"]),
                row["intercept_blup"],
                row["intercept_se"] if np.isfinite(row["intercept_se"])
                                    else np.nan,
                row["slope_blup"],
                row["slope_se"] if np.isfinite(row["slope_se"])
                                else np.nan),
                stats_lines)
    log("", stats_lines)
    return fe_df, blups_df


# ─── main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Random-slope LMM + presentation trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to 8_arms_w_voc root (BATCH_ANALYSIS/ inside).")
    parser.add_argument("--csv", default=None,
                        help="Direct path to preference_data.csv.")
    parser.add_argument("--required-days", default=",".join(PI_DAYS),
                        help=f"Comma-separated completer days. "
                             f"Default: {','.join(PI_DAYS)}")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: alongside input CSV).")
    args = parser.parse_args()

    # Locate input CSV (mirrors 03_completers_lmm.py)
    if args.csv:
        csv_path = args.csv
        default_outdir = os.path.dirname(os.path.abspath(csv_path))
    else:
        base_path = args.data_path or os.environ.get("MAZE_DATA_DIR")
        if base_path is None:
            print("Enter the path to the 8_arms_w_voc data folder")
            print("  (it must contain BATCH_ANALYSIS/preference_data.csv):")
            base_path = input("> ").strip().strip('"').strip("'")
        if not os.path.isdir(base_path):
            print(f"ERROR: directory not found: {base_path}")
            sys.exit(1)
        csv_path = os.path.join(base_path, "BATCH_ANALYSIS",
                                "preference_data.csv")
        default_outdir = os.path.join(base_path, "BATCH_ANALYSIS")

    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run `run_batch_preference.py` first.")
        sys.exit(1)

    output_dir = args.output_dir or default_outdir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    # Load + restrict to completers
    df = pd.read_csv(csv_path)
    required_days = [d.strip() for d in args.required_days.split(",")
                     if d.strip()]
    print(f"Required days for 'completer' status: {required_days}")
    completers = find_completers(df, required_days)
    if not completers:
        print("\nERROR: no mice completed all required days.")
        sys.exit(1)

    df_c = df[df["mouse_id"].isin(completers)
              & df["day"].isin(required_days)].copy()
    day_to_num = {d: i for i, d in enumerate(PI_DAYS)}
    df_c["session_num"] = df_c["day"].map(day_to_num).astype(float)
    print(f"Completer mice (n={len(completers)}): {', '.join(completers)}")
    print(f"Total sessions for completers: {len(df_c)}\n")

    stats_lines: list[str] = []
    log("=" * 70, stats_lines)
    log("RANDOM-SLOPE LMM REPORT", stats_lines)
    log("=" * 70, stats_lines)
    log(f"Required days       : {required_days}", stats_lines)
    log(f"Completer mice (n={len(completers)}): {', '.join(completers)}",
        stats_lines)
    log(f"Total sessions      : {len(df_c)}", stats_lines)
    log("Model               : outcome ~ session_num + "
        "(1 + session_num | mouse_id)", stats_lines)
    log("Estimation          : REML, statsmodels MixedLM (lbfgs)", stats_lines)
    log("", stats_lines)

    fe_by_outcome: dict = {}
    blups_by_outcome: dict = {}
    all_fe_rows: list = []
    all_blup_rows: list = []
    all_var_rows: list = []

    for outcome, label in OUTCOMES:
        print(f"=== Fitting {label} ({outcome}) ===")
        res, info = fit_random_slope(df_c, outcome)
        fe_df, blups_df = report_outcome(label, outcome, res, info,
                                         stats_lines)
        fe_by_outcome[outcome] = fe_df
        blups_by_outcome[outcome] = blups_df

        if fe_df is not None:
            tmp = fe_df.copy()
            tmp.insert(0, "outcome", outcome)
            all_fe_rows.append(tmp)
        if blups_df is not None and not blups_df.empty:
            tmp = blups_df.copy()
            tmp.insert(0, "outcome", outcome)
            all_blup_rows.append(tmp)
        if res is not None:
            vc = variance_components(res)
            vd = variance_decomposition(vc)
            all_var_rows.append({
                "outcome":         outcome,
                "n_obs":           info["n_obs"],
                "n_groups":        info["n_groups"],
                "converged":       info["converged"],
                "singular":        info["singular"],
                **vc,
                **vd,
            })

    # Save tables
    if all_fe_rows:
        pd.concat(all_fe_rows, ignore_index=True).to_csv(
            os.path.join(output_dir,
                         "completers_random_slopes_fixed_effects.csv"),
            index=False)
        print("Saved completers_random_slopes_fixed_effects.csv")
    if all_blup_rows:
        pd.concat(all_blup_rows, ignore_index=True).to_csv(
            os.path.join(output_dir, "completers_random_slopes_blups.csv"),
            index=False)
        print("Saved completers_random_slopes_blups.csv")
    if all_var_rows:
        pd.DataFrame(all_var_rows).to_csv(
            os.path.join(output_dir,
                         "completers_random_slopes_variance.csv"),
            index=False)
        print("Saved completers_random_slopes_variance.csv")

    # Save report
    report_path = os.path.join(output_dir,
                               "completers_random_slopes_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print(f"Saved {os.path.basename(report_path)}")

    # Plots
    print("\nPlotting...")
    plot_slope_caterpillar(blups_by_outcome, output_dir)
    plot_presentation_trajectories(df_c, blups_by_outcome, fe_by_outcome,
                                   output_dir)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
