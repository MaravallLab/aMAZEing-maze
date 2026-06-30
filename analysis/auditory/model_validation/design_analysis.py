"""Pre-registered model-free test — independent of the HMM being correct.

Primary model (per-arm dwell on grammar arms, day 1)::

    log1p(dwell) ~ C(environment) * C(tier) * C(group) + (1 | mouse)

Reports:

  * environment (EE vs SC) main effect      -> S(t) exists
  * tier (complexity) main effect           -> r(t)
  * the FORM of the environment x tier interaction, and specifically the
    SIMPLE effect of environment within each tier. The IC-tier design predicts
    the EE-SC effect is attenuated-but-non-zero at high complexity (rare); a
    per-tier simple effect distinguishes "attenuated but present" from "absent
    at high complexity". This is the empirical fingerprint of the tier design.
  * the environment x GROUP interaction (point 8): a semantic effect is
    consistent across counterbalancing groups; an intrinsic-grammar confound
    (e.g. preferring the octave-trill dominant Grammar B over the ascending
    Grammar A regardless of association) flips or concentrates in one group.
    The dominant-tier EE-SC result is PROVISIONAL until this is clean.

Secondary (day 2 + day-1 robustness): within-session block time-course::

    PI ~ block + (1 | mouse)

Monotonic decay across blocks (continuing into day 2) is the extinction
signature; flat is a stable association.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False

TIERS = ["dominant", "secondary", "rare"]


def _fit_mixedlm(formula: str, data: pd.DataFrame, groups: str):
    res = smf.mixedlm(formula, data, groups=data[groups]).fit(reml=False)
    rows = []
    ci = res.conf_int()
    for name in res.fe_params.index:
        rows.append({
            "term": name,
            "estimate": float(res.fe_params[name]),
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
            "p": float(res.pvalues.get(name, np.nan)),
        })
    return pd.DataFrame(rows), res


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["arm_type"] == "grammar"].dropna(subset=["environment", "tier", "group"]).copy()
    g["log_dwell"] = np.log1p(g["time_spent_s"].astype(float))
    g["environment"] = pd.Categorical(g["environment"], categories=["SC", "EE"])  # EE is the +level
    g["tier"] = pd.Categorical(g["tier"], categories=TIERS)
    g["group"] = g["group"].astype(int)
    return g


def design_analysis(df: pd.DataFrame) -> Dict[str, object]:
    """Run the full model-free design analysis on one day's arm-block dataframe."""
    if not _HAVE_SM:
        return {"error": "statsmodels not available"}

    g = _prep(df)
    out: Dict[str, object] = {}
    out["n_obs"] = int(len(g))
    out["n_mice"] = int(g["mouse"].nunique())
    out["group_balance"] = g.drop_duplicates("mouse").groupby("group").size().to_dict()

    # ---- full 3-way model ----
    try:
        tbl, _ = _fit_mixedlm(
            "log_dwell ~ C(environment) * C(tier) * C(group)", g, "mouse")
        out["full_model"] = tbl.to_dict(orient="records")
    except Exception as e:
        out["full_model_error"] = str(e)

    # ---- simple effect of environment within each tier (+ env x group) ----
    simple: List[Dict[str, object]] = []
    for tier in TIERS:
        sub = g[g["tier"] == tier]
        if sub["environment"].nunique() < 2 or len(sub) < 8:
            continue
        try:
            tbl, _ = _fit_mixedlm(
                "log_dwell ~ C(environment) * C(group)", sub, "mouse")
            env_row = tbl[tbl["term"].str.contains("environment") &
                          ~tbl["term"].str.contains("group")]
            inter = tbl[tbl["term"].str.contains("environment") &
                        tbl["term"].str.contains("group")]
            rec = {"tier": tier}
            if len(env_row):
                r = env_row.iloc[0]
                rec.update(env_effect=float(r["estimate"]),
                           env_ci=[float(r["ci_low"]), float(r["ci_high"])],
                           env_p=float(r["p"]))
            if len(inter):
                r = inter.iloc[0]
                rec.update(env_x_group=float(r["estimate"]),
                           env_x_group_p=float(r["p"]))
            simple.append(rec)
        except Exception as e:
            simple.append({"tier": tier, "error": str(e)})
    out["simple_effects_by_tier"] = simple

    # ---- environment effect within each group (consistency check) ----
    per_group: List[Dict[str, object]] = []
    for grp in sorted(g["group"].unique()):
        sub = g[g["group"] == grp]
        try:
            tbl, _ = _fit_mixedlm("log_dwell ~ C(environment) * C(tier)", sub, "mouse")
            env_row = tbl[(tbl["term"] == "C(environment)[T.EE]")]
            if len(env_row):
                r = env_row.iloc[0]
                per_group.append({
                    "group": int(grp),
                    "env_effect_EE_minus_SC": float(r["estimate"]),
                    "ci": [float(r["ci_low"]), float(r["ci_high"])],
                    "p": float(r["p"]),
                })
        except Exception as e:
            per_group.append({"group": int(grp), "error": str(e)})
    out["environment_effect_by_group"] = per_group
    # consistent sign across groups => semantic; opposite => intrinsic-grammar confound
    signs = [np.sign(r.get("env_effect_EE_minus_SC", np.nan))
             for r in per_group if "env_effect_EE_minus_SC" in r]
    out["group_consistent"] = bool(len(signs) == 2 and signs[0] == signs[1] and signs[0] != 0)

    return out


def block_timecourse(block_pi_df: pd.DataFrame) -> Dict[str, object]:
    """PI ~ block + (1|mouse): monotonic decay = extinction; flat = stable."""
    if not _HAVE_SM:
        return {"error": "statsmodels not available"}
    d = block_pi_df.dropna(subset=["PI"]).copy()
    if d["block"].nunique() < 2 or d["mouse"].nunique() < 2:
        return {"error": "insufficient blocks/mice for time-course"}
    d["block_c"] = d["block"].astype(float)
    try:
        tbl, _ = _fit_mixedlm("PI ~ block_c", d, "mouse")
        slope = tbl[tbl["term"] == "block_c"].iloc[0]
        return {
            "slope_per_block": float(slope["estimate"]),
            "ci": [float(slope["ci_low"]), float(slope["ci_high"])],
            "p": float(slope["p"]),
            "interpretation": ("decay (possible extinction)"
                               if slope["estimate"] < 0 and slope["p"] < 0.05
                               else "flat / stable"),
            "mean_pi_by_block": d.groupby("block")["PI"].mean().to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}
