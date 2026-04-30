#!/usr/bin/env python3
"""
03 - Completers-Only Linear Mixed Model Analysis
=================================================

Restricts the dataset to mice that completed ALL experimental days, then
fits linear mixed-effects models to assess the joint effect of session
(day) and mouse identity on every preference / exploration outcome.

Why "completers only"?
----------------------
With unbalanced data, mouse-level random intercepts can absorb day effects
(or vice versa). Restricting to mice with a complete set of sessions makes
each fixed-effect contrast a true within-mouse comparison and the random
intercepts directly comparable across mice.

Models fitted (per outcome y)
-----------------------------
    M0:  y ~ 1            + (1 | mouse_id)         null  (random intercepts only)
    M1:  y ~ C(day)       + (1 | mouse_id)         day as fixed effect (the user's request)
    M2:  y ~ session_num  + (1 | mouse_id)         day as a linear trend
    M3:  y ~ session_num  + (1 + session_num | mouse_id)
                                                   random slopes (mice differ in trend)

For each outcome we report:
    - fixed-effect coefficients (estimate, SE, z, p, 95 % CI)
    - random-intercept variance, residual variance, ICC(mouse)
    - marginal R^2 (fixed only) and conditional R^2 (fixed + random)
    - LRT comparing M0 vs M1 (does day matter?)
                   M1 vs M3 (do mice differ in their day-trend?)
    - per-mouse BLUPs (Best Linear Unbiased Predictors)

Outcomes modelled
-----------------
    preference_index    - overall sound vs silent
    voc_pi              - vocalisation vs silent
    other_sounds_pi     - non-vocalisation sounds vs silent
    roaming_entropy     - full-habituation RE
    re_first_min        - first-minute habituation RE

Inputs
------
    BATCH_ANALYSIS/preference_data.csv
        Produced by `run_batch_preference.py`. Run that first.

Outputs (all to BATCH_ANALYSIS/, prefixed `completers_`)
--------------------------------------------------------
    completers_summary.csv              List of completer mice
    completers_lmm_fixed_effects.csv    Fixed-effect table (all outcomes/models)
    completers_lmm_variance.csv         ICC + R^2 + LRT per outcome/model
    completers_lmm_random_intercepts.csv Per-mouse BLUPs per outcome
    completers_stats_report.txt         Full text report
    fig_completers_caterpillar.png/pdf  Per-mouse intercept ranking, by outcome
    fig_completers_day_estimates.png/pdf Day estimated marginal means w/ 95 % CI
    fig_completers_spaghetti.png/pdf    Individual trajectories + model fit

Quick start
-----------
    # First run the batch analysis to produce preference_data.csv
    python run_batch_preference.py "C:\\path\\to\\8_arms_w_voc"

    # Then run this script (same data path or BATCH_ANALYSIS path)
    python 03_completers_lmm.py "C:\\path\\to\\8_arms_w_voc"
    # or point directly at the CSV:
    python 03_completers_lmm.py --csv "C:\\path\\to\\8_arms_w_voc\\BATCH_ANALYSIS\\preference_data.csv"

Dependencies
------------
    numpy  pandas  matplotlib  seaborn  scipy  statsmodels
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("ERROR: statsmodels is required. Install with:  pip install statsmodels")
    sys.exit(1)

# Suppress harmless statsmodels convergence warnings (we report convergence
# explicitly per model)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── experiment structure (kept in-sync with preference_analysis_config.py) ──
DAY_ORDER = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
PI_DAYS   = ["w1_d1", "w1_d2", "w1_d3", "w2_sequences", "w2_vocalisations"]
DAY_SHORT = {
    "w1_d1": "D1 (TEM)", "w1_d2": "D2 (Int)", "w1_d3": "D3 (Int)",
    "w1_d4": "D4 (Int)", "w2_sequences": "W2 (Seq)", "w2_vocalisations": "W2 (Voc)",
}

# Outcomes to model: (column_name, pretty_label, scale_note)
OUTCOMES = [
    ("preference_index", "Overall PI",          "[-1, 1]"),
    ("voc_pi",           "Vocalisation PI",     "[-1, 1]"),
    ("other_sounds_pi",  "Other Sounds PI",     "[-1, 1]"),
    ("roaming_entropy",  "Roaming Entropy",     "[0, 1]"),
    ("re_first_min",     "RE (first minute)",   "[0, 1]"),
]


# ─── helpers ────────────────────────────────────────────────────────────────


def log(msg, lines):
    print(msg)
    lines.append(msg)


def safe_savefig(fig, path, **kwargs):
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"  WARNING: could not write {os.path.basename(path)} (file locked)")


def find_completers(df: pd.DataFrame, required_days: list[str]) -> list[str]:
    """Return mouse_ids present on EVERY day in required_days."""
    counts = (df.groupby("mouse_id")["day"]
                .apply(lambda s: set(s.unique()))
                .to_dict())
    needed = set(required_days)
    return sorted([m for m, days in counts.items() if needed.issubset(days)])


def variance_components(model) -> dict:
    """Extract var(random intercept), var(residual), and ICC."""
    try:
        var_re = float(model.cov_re.iloc[0, 0])
    except Exception:
        var_re = float(model.cov_re) if not hasattr(model.cov_re, "iloc") else np.nan
    var_resid = float(model.scale)
    icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else np.nan
    return {"var_re": var_re, "var_resid": var_resid, "icc": icc}


def safe_random_effects(result):
    """Return result.random_effects as a dict, or {} if the random-effects
    covariance is singular / cannot be inverted.

    statsmodels.MixedLMResults.random_effects calls np.linalg.inv(self.cov_re),
    which raises LinAlgError -> ValueError when the between-group variance
    estimate collapses to (near-)zero. That is common with small completer
    samples on outcomes where almost all variation is within-mouse.
    """
    try:
        return dict(result.random_effects)
    except (np.linalg.LinAlgError, ValueError, Exception):
        return {}


def safe_random_effects_cov(result):
    """Same as above for result.random_effects_cov."""
    try:
        return dict(result.random_effects_cov)
    except (np.linalg.LinAlgError, ValueError, Exception):
        return {}


def nakagawa_r2(model, df, outcome) -> dict:
    """Approximate marginal & conditional R^2 (Nakagawa & Schielzeth, 2013).
    var_f = variance of fitted fixed-effect predictions
    var_u = random-intercept variance
    var_e = residual variance
    R2_marginal    = var_f / (var_f + var_u + var_e)
    R2_conditional = (var_f + var_u) / (var_f + var_u + var_e)
    """
    try:
        # Fixed-effect predictions only (population level): use design matrix * fixed params
        try:
            fitted_full = np.asarray(model.fittedvalues)
        except Exception:
            fitted_full = np.asarray(model.predict(df))

        # Subtract per-row random intercept to get pure fixed-effect prediction.
        # If the RE covariance is singular, treat all BLUPs as 0 (collapses to
        # the fixed-effect-only prediction, which is correct for marginal R^2).
        re_dict = safe_random_effects(model)
        per_row_re = np.array([
            float(re_dict[g].iloc[0]) if g in re_dict and len(re_dict[g]) > 0 else 0.0
            for g in df["mouse_id"].values
        ])
        fitted_fe = fitted_full - per_row_re

        var_f = float(np.var(fitted_fe, ddof=0))
        vc = variance_components(model)
        var_u = vc["var_re"]
        var_e = vc["var_resid"]
        denom = var_f + var_u + var_e
        if denom <= 0:
            return {"r2_marginal": np.nan, "r2_conditional": np.nan}
        return {
            "r2_marginal":    var_f / denom,
            "r2_conditional": (var_f + var_u) / denom,
        }
    except Exception:
        return {"r2_marginal": np.nan, "r2_conditional": np.nan}


def lrt(small, large) -> dict:
    """Likelihood-ratio test of nested mixed models (must be fit with REML=False)."""
    try:
        d_ll = 2.0 * (large.llf - small.llf)
        d_df = large.df_modelwc - small.df_modelwc
        p = chi2.sf(d_ll, d_df) if d_df > 0 else np.nan
        return {"lrt_chi2": d_ll, "lrt_df": d_df, "lrt_p": p}
    except Exception:
        return {"lrt_chi2": np.nan, "lrt_df": np.nan, "lrt_p": np.nan}


def fit_models(df_outcome: pd.DataFrame, outcome: str) -> dict:
    """Fit M0/M1/M2/M3 for one outcome on the completers subset.

    Returns a dict keyed by model label, each containing:
        result, fixed_effects (DataFrame), variance, r2, lrt_vs_m0
    """
    out = {}

    df_o = df_outcome.dropna(subset=[outcome]).copy()
    if df_o["mouse_id"].nunique() < 3 or len(df_o) < 6:
        return out

    # session_num = ordinal position in PI_DAYS (0..k-1) for completers
    day_to_num = {d: i for i, d in enumerate(PI_DAYS)}
    df_o["session_num"] = df_o["day"].map(day_to_num).astype(float)
    df_o["day_cat"] = pd.Categorical(df_o["day"], categories=PI_DAYS)

    def _fit(formula, re_formula=None, reml=True):
        try:
            m = smf.mixedlm(formula, df_o, groups=df_o["mouse_id"],
                            re_formula=re_formula).fit(reml=reml,
                                                       method="lbfgs")
            return m
        except Exception:
            try:
                m = smf.mixedlm(formula, df_o, groups=df_o["mouse_id"],
                                re_formula=re_formula).fit(reml=reml)
                return m
            except Exception as e:
                return None

    # REML fits for inference (CI + variance components)
    m0 = _fit(f"{outcome} ~ 1",                      reml=True)
    m1 = _fit(f"{outcome} ~ C(day_cat)",             reml=True)
    m2 = _fit(f"{outcome} ~ session_num",            reml=True)
    m3 = _fit(f"{outcome} ~ session_num",
              re_formula="~session_num",             reml=True)

    # ML fits for LRT
    m0_ml = _fit(f"{outcome} ~ 1",                   reml=False)
    m1_ml = _fit(f"{outcome} ~ C(day_cat)",          reml=False)
    m2_ml = _fit(f"{outcome} ~ session_num",         reml=False)
    m3_ml = _fit(f"{outcome} ~ session_num",
                 re_formula="~session_num",          reml=False)

    def _pack(label, m, ml_for_lrt=None, vs_label=None):
        if m is None:
            return None
        try:
            ci = m.conf_int()
            ci.columns = ["ci_lo", "ci_hi"]
            fe_df = pd.DataFrame({
                "estimate": m.fe_params,
                "std_err":  m.bse_fe,
                "z":        m.tvalues[m.fe_params.index],
                "p":        m.pvalues[m.fe_params.index],
            }).join(ci.loc[m.fe_params.index])
            fe_df.insert(0, "term", fe_df.index)
            fe_df.insert(0, "model", label)
            fe_df.insert(0, "outcome", outcome)
            vc = variance_components(m)
            r2 = nakagawa_r2(m, df_o, outcome)
            entry = {
                "result":        m,
                "fixed_effects": fe_df.reset_index(drop=True),
                "variance":      vc,
                "r2":            r2,
                "n_obs":         int(m.nobs),
                "n_groups":      df_o["mouse_id"].nunique(),
                "converged":     bool(m.converged) if hasattr(m, "converged") else None,
            }
            return entry
        except Exception:
            return None

    out["M0"] = _pack("M0", m0)
    out["M1"] = _pack("M1", m1)
    out["M2"] = _pack("M2", m2)
    out["M3"] = _pack("M3", m3)

    # LRTs
    if out.get("M0") and m0_ml is not None and m1_ml is not None:
        out["M1"]["lrt_vs_M0"] = lrt(m0_ml, m1_ml) if out.get("M1") else None
    if out.get("M2") and m0_ml is not None and m2_ml is not None:
        out["M2"]["lrt_vs_M0"] = lrt(m0_ml, m2_ml)
    if out.get("M3") and m2_ml is not None and m3_ml is not None:
        out["M3"]["lrt_vs_M2"] = lrt(m2_ml, m3_ml)

    return out


# ─── plotting ───────────────────────────────────────────────────────────────


def plot_caterpillar(model_results: dict, df_completers: pd.DataFrame,
                     output_dir: str):
    """One panel per outcome: per-mouse random intercepts, ranked, with 95% CI.

    The 95% CI here is approximated as BLUP +/- 1.96 * sqrt(Var_BLUP);
    statsmodels exposes only the conditional posterior variance per group.
    """
    n_panels = sum(1 for o, _, _ in OUTCOMES if o in model_results)
    if n_panels == 0:
        return
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    panel = 0
    for out_col, label, _ in OUTCOMES:
        if out_col not in model_results:
            continue
        r, c = divmod(panel, ncols)
        ax = axes[r][c]
        panel += 1

        models = model_results[out_col]
        m1 = models.get("M1")
        if m1 is None:
            ax.set_title(f"{label}\n(model failed)")
            ax.set_visible(False)
            continue
        result = m1["result"]
        re_dict = safe_random_effects(result)
        if not re_dict:
            ax.set_title(f"{label}\n(singular RE covariance)")
            ax.text(0.5, 0.5,
                    "Random-intercept variance ~ 0:\nBLUPs not identifiable",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        items = [(g, float(v.iloc[0])) for g, v in re_dict.items()]
        items.sort(key=lambda kv: kv[1])
        mice = [k for k, _ in items]
        vals = np.array([v for _, v in items])

        # SE per group: posterior SD of random intercept
        pcov = safe_random_effects_cov(result)
        if pcov:
            try:
                ses = np.array([np.sqrt(float(pcov[g].iloc[0, 0]))
                                for g in mice])
            except Exception:
                ses = np.full_like(vals, np.nan)
        else:
            ses = np.full_like(vals, np.nan)

        y = np.arange(len(mice))
        ax.errorbar(vals, y, xerr=1.96 * ses, fmt="o", color="#457B9D",
                    ecolor="#999999", elinewidth=1, capsize=2, ms=5)
        ax.axvline(0, color="black", ls="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(mice, fontsize=8)
        ax.set_xlabel(f"BLUP (deviation from grand mean)")
        ax.set_title(f"{label}\nper-mouse random intercept (M1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # hide leftover axes
    for idx in range(panel, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Completers: per-mouse random intercepts (caterpillar plot)\n"
                 "M1: outcome ~ day + (1 | mouse)",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_completers_caterpillar.png"),
                dpi=200, bbox_inches="tight")
    safe_savefig(fig, os.path.join(output_dir, "fig_completers_caterpillar.pdf"),
                 bbox_inches="tight")
    print("  Saved fig_completers_caterpillar")


def plot_day_estimates(model_results: dict, df_completers: pd.DataFrame,
                       output_dir: str):
    """One panel per outcome: estimated marginal mean per day with 95% CI from M1.
    Reference category is the first level (w1_d1); we reconstruct each day's
    estimate by adding the appropriate dummy contrast to the intercept.
    """
    n_panels = sum(1 for o, _, _ in OUTCOMES if o in model_results)
    if n_panels == 0:
        return
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    panel = 0
    for out_col, label, _ in OUTCOMES:
        if out_col not in model_results:
            continue
        r, c = divmod(panel, ncols)
        ax = axes[r][c]
        panel += 1

        models = model_results[out_col]
        m1 = models.get("M1")
        if m1 is None:
            ax.set_visible(False)
            continue
        result = m1["result"]
        params = result.fe_params
        cov = result.cov_params()
        # find Intercept and C(day_cat)[T.<day>] terms
        intercept = float(params.get("Intercept", np.nan))
        days_present = [d for d in PI_DAYS if d in df_completers["day"].values]

        means = []
        ses = []
        for d in days_present:
            term = f"C(day_cat)[T.{d}]"
            if d == days_present[0] or term not in params.index:
                # reference level
                est = intercept
                # SE of the intercept
                idx = list(params.index).index("Intercept")
                se = float(np.sqrt(cov.iloc[idx, idx]))
            else:
                est = intercept + float(params[term])
                # SE of (intercept + dummy) = sqrt(var(int) + var(dummy) + 2 cov)
                i_int = list(params.index).index("Intercept")
                i_d = list(params.index).index(term)
                v_int = cov.iloc[i_int, i_int]
                v_d   = cov.iloc[i_d, i_d]
                cov_id = cov.iloc[i_int, i_d]
                se = float(np.sqrt(v_int + v_d + 2.0 * cov_id))
            means.append(est)
            ses.append(se)

        means = np.asarray(means)
        ses = np.asarray(ses)
        x = np.arange(len(days_present))

        # raw per-mouse points (jittered)
        for d_idx, d in enumerate(days_present):
            sub = df_completers[df_completers["day"] == d][out_col].dropna()
            if len(sub) == 0:
                continue
            jitter = (np.random.rand(len(sub)) - 0.5) * 0.25
            ax.scatter(np.full(len(sub), d_idx) + jitter, sub.values,
                       s=15, alpha=0.4, color="#888")

        ax.errorbar(x, means, yerr=1.96 * ses, fmt="D-", color="#E63946",
                    ecolor="#E63946", elinewidth=1.5, capsize=4, ms=8,
                    zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([DAY_SHORT[d] for d in days_present],
                           rotation=20, fontsize=9)
        ax.set_ylabel(label)
        ax.axhline(0, color="grey", ls="--", alpha=0.4)
        ax.set_title(f"{label}: estimated mean per day (95% CI)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(panel, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Completers: M1 estimated marginal means by day",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_completers_day_estimates.png"),
                dpi=200, bbox_inches="tight")
    safe_savefig(fig, os.path.join(output_dir, "fig_completers_day_estimates.pdf"),
                 bbox_inches="tight")
    print("  Saved fig_completers_day_estimates")


def plot_spaghetti(df_completers: pd.DataFrame, output_dir: str):
    """One panel per outcome: individual mouse trajectories across days."""
    panels = [(o, l) for o, l, _ in OUTCOMES if o in df_completers.columns]
    if not panels:
        return
    ncols = min(len(panels), 3)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    day_x = {d: i for i, d in enumerate(PI_DAYS)}
    days_present = [d for d in PI_DAYS if d in df_completers["day"].values]
    mice = sorted(df_completers["mouse_id"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(mice), 1)))

    for idx, (out_col, label) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for i, m in enumerate(mice):
            sub = df_completers[df_completers["mouse_id"] == m].copy()
            sub["x"] = sub["day"].map(day_x)
            sub = sub.dropna(subset=[out_col]).sort_values("x")
            if len(sub) == 0:
                continue
            ax.plot(sub["x"], sub[out_col], "o-",
                    color=cmap[i % len(cmap)], alpha=0.6, lw=1.2, ms=4)
        # group mean
        means = [df_completers[df_completers["day"] == d][out_col].mean()
                 for d in days_present]
        xs = [day_x[d] for d in days_present]
        ax.plot(xs, means, "D-", color="black", lw=2, ms=8, zorder=5,
                label="Group mean")
        ax.axhline(0, color="grey", ls="--", alpha=0.4)
        ax.set_xticks(list(day_x.values()))
        ax.set_xticklabels([DAY_SHORT[d] for d in PI_DAYS], rotation=20, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=9, frameon=False)

    for idx in range(len(panels), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Completers: individual trajectories across days",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_completers_spaghetti.png"),
                dpi=200, bbox_inches="tight")
    safe_savefig(fig, os.path.join(output_dir, "fig_completers_spaghetti.pdf"),
                 bbox_inches="tight")
    print("  Saved fig_completers_spaghetti")


# ─── main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Completers-only LMM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_path", nargs="?", default=None,
        help="Path to 8_arms_w_voc root (BATCH_ANALYSIS/ must exist inside).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Direct path to preference_data.csv (overrides data_path).",
    )
    parser.add_argument(
        "--required-days", default=",".join(PI_DAYS),
        help=("Comma-separated list of day keys that define a 'completer'. "
              f"Default: {','.join(PI_DAYS)}"),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write outputs. Default: alongside the input CSV.",
    )
    args = parser.parse_args()

    # --- locate input CSV --------------------------------------------------
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
        csv_path = os.path.join(base_path, "BATCH_ANALYSIS", "preference_data.csv")
        default_outdir = os.path.join(base_path, "BATCH_ANALYSIS")

    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run `run_batch_preference.py` first to generate preference_data.csv.")
        sys.exit(1)

    output_dir = args.output_dir or default_outdir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input  : {csv_path}")
    print(f"Output : {output_dir}\n")

    # --- load + filter to completers ---------------------------------------
    df = pd.read_csv(csv_path)
    required_days = [d.strip() for d in args.required_days.split(",") if d.strip()]
    print(f"Required days for 'completer' status: {required_days}")

    completers = find_completers(df, required_days)
    if not completers:
        print("\nERROR: no mice completed all required days.")
        # show how close each mouse got
        print("\nDays per mouse:")
        for m, days in (df.groupby("mouse_id")["day"]
                          .apply(lambda s: sorted(set(s)))
                          .items()):
            print(f"  {m}: {days}")
        sys.exit(1)

    df_c = df[df["mouse_id"].isin(completers) &
              df["day"].isin(required_days)].copy()
    print(f"Completer mice (n={len(completers)}): {', '.join(completers)}")
    print(f"Total sessions for completers: {len(df_c)}")

    stats_lines: list[str] = []
    log("=" * 70, stats_lines)
    log("COMPLETERS-ONLY LINEAR MIXED MODEL ANALYSIS", stats_lines)
    log("=" * 70, stats_lines)
    log(f"Required days: {required_days}", stats_lines)
    log(f"Completer mice (n={len(completers)}): {', '.join(completers)}", stats_lines)
    log(f"Sessions analysed: {len(df_c)}", stats_lines)
    log("", stats_lines)

    # write completers summary
    summary = (df_c.groupby("mouse_id")
                  .agg(n_sessions=("day", "count"),
                       days_present=("day", lambda s: ",".join(sorted(set(s)))))
                  .reset_index())
    summary.to_csv(os.path.join(output_dir, "completers_summary.csv"), index=False)
    print(f"Saved completers_summary.csv ({len(summary)} mice)\n")

    # --- fit models per outcome -------------------------------------------
    model_results: dict = {}
    fe_rows = []
    var_rows = []
    blup_rows = []

    for out_col, label, scale in OUTCOMES:
        if out_col not in df_c.columns:
            log(f"\n[skip] outcome '{out_col}' not in CSV — skipping.", stats_lines)
            continue
        n_obs = df_c[out_col].notna().sum()
        if n_obs < 6:
            log(f"\n[skip] outcome '{out_col}': only {n_obs} non-NaN rows.",
                stats_lines)
            continue

        log("\n" + "=" * 70, stats_lines)
        log(f"OUTCOME: {label}  ({out_col}, scale {scale})", stats_lines)
        log("=" * 70, stats_lines)

        results = fit_models(df_c, out_col)
        if not results:
            log(f"  All models failed to fit for {out_col}.", stats_lines)
            continue
        model_results[out_col] = results

        # report per model
        for model_label in ("M0", "M1", "M2", "M3"):
            entry = results.get(model_label)
            if entry is None:
                log(f"\n  [{model_label}] failed to fit.", stats_lines)
                continue
            log(f"\n  --- {model_label} ---", stats_lines)
            log(f"  n_obs={entry['n_obs']}, n_mice={entry['n_groups']}, "
                f"converged={entry['converged']}", stats_lines)

            # fixed effects
            for _, row in entry["fixed_effects"].iterrows():
                log(f"    {row['term']:<28s} est={row['estimate']:+.4f}  "
                    f"SE={row['std_err']:.4f}  z={row['z']:+.2f}  "
                    f"p={row['p']:.4f}  CI=[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]",
                    stats_lines)
                fe_rows.append(row.to_dict())

            # variance + R^2
            v = entry["variance"]
            r2 = entry["r2"]
            log(f"    Var(mouse)={v['var_re']:.4f}  Var(resid)={v['var_resid']:.4f}  "
                f"ICC={v['icc']:.3f}", stats_lines)
            log(f"    Marginal R^2={r2['r2_marginal']:.3f}  "
                f"Conditional R^2={r2['r2_conditional']:.3f}", stats_lines)
            var_rows.append({
                "outcome": out_col, "model": model_label,
                "n_obs": entry["n_obs"], "n_groups": entry["n_groups"],
                "var_re": v["var_re"], "var_resid": v["var_resid"], "icc": v["icc"],
                "r2_marginal": r2["r2_marginal"],
                "r2_conditional": r2["r2_conditional"],
                "lrt_chi2": entry.get("lrt_vs_M0", entry.get("lrt_vs_M2", {})).get("lrt_chi2", np.nan),
                "lrt_df":   entry.get("lrt_vs_M0", entry.get("lrt_vs_M2", {})).get("lrt_df",   np.nan),
                "lrt_p":    entry.get("lrt_vs_M0", entry.get("lrt_vs_M2", {})).get("lrt_p",    np.nan),
            })

            # LRT messages
            if "lrt_vs_M0" in entry and entry["lrt_vs_M0"]:
                lr = entry["lrt_vs_M0"]
                log(f"    LRT vs M0: chi2={lr['lrt_chi2']:.3f}, df={lr['lrt_df']}, "
                    f"p={lr['lrt_p']:.4f}", stats_lines)
            if "lrt_vs_M2" in entry and entry["lrt_vs_M2"]:
                lr = entry["lrt_vs_M2"]
                log(f"    LRT vs M2: chi2={lr['lrt_chi2']:.3f}, df={lr['lrt_df']}, "
                    f"p={lr['lrt_p']:.4f}", stats_lines)

        # BLUPs from M1 (the user's primary model). When the random-intercept
        # variance is degenerate, statsmodels cannot invert cov_re and
        # random_effects raises -- in that case we skip BLUP rows and log it.
        m1 = results.get("M1")
        if m1 is not None:
            re_dict = safe_random_effects(m1["result"])
            if not re_dict:
                log(f"  [{out_col}] M1: random-intercept covariance singular; "
                    f"BLUPs not identifiable -- skipping per-mouse intercepts.",
                    stats_lines)
            for g, v in re_dict.items():
                blup_rows.append({
                    "outcome": out_col, "model": "M1",
                    "mouse_id": g,
                    "intercept_blup": float(v.iloc[0]),
                })

    # --- save tables -------------------------------------------------------
    if fe_rows:
        pd.DataFrame(fe_rows).to_csv(
            os.path.join(output_dir, "completers_lmm_fixed_effects.csv"),
            index=False,
        )
        print("Saved completers_lmm_fixed_effects.csv")
    if var_rows:
        pd.DataFrame(var_rows).to_csv(
            os.path.join(output_dir, "completers_lmm_variance.csv"),
            index=False,
        )
        print("Saved completers_lmm_variance.csv")
    if blup_rows:
        pd.DataFrame(blup_rows).to_csv(
            os.path.join(output_dir, "completers_lmm_random_intercepts.csv"),
            index=False,
        )
        print("Saved completers_lmm_random_intercepts.csv")

    # --- figures -----------------------------------------------------------
    print("\nPlotting...")
    plot_caterpillar(model_results, df_c, output_dir)
    plot_day_estimates(model_results, df_c, output_dir)
    plot_spaghetti(df_c, output_dir)

    # --- save stats report -------------------------------------------------
    report_path = os.path.join(output_dir, "completers_stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    print(f"\nSaved {os.path.basename(report_path)}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
