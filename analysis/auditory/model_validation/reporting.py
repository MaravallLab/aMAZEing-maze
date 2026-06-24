"""Markdown + JSON reporting for the validation pipeline.

Emits a single human-readable report (cohort/exclusions stated at the top, every
gate carrying a go/no-go) plus a machine-readable results.json holding every raw
number, so nothing computed is lost.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)


def _go(flag: bool) -> str:
    return "✅ GO" if flag else "🛑 NO-GO"


def render_markdown(results: Dict[str, object]) -> str:
    L: List[str] = []
    L.append("# Model-validation report\n")
    L.append(f"Results root: `{results.get('results_root', '?')}`  ")
    L.append(f"Generated: {results.get('generated', '?')}\n")

    # ---- cohort ----
    cohort = results.get("cohort", {})
    L.append("## Cohort & exclusions\n")
    L.append(f"- Sessions found: **{cohort.get('n_sessions_found', '?')}**")
    L.append(f"- Day-1 distinct mice (after exclusions): "
             f"**{cohort.get('day1_distinct_mice', '?')}**")
    L.append(f"- Day-1 group balance: `{cohort.get('day1_group_balance', '?')}`")
    excl = cohort.get("excluded_sessions", [])
    L.append(f"- Excluded sessions: **{len(excl)}**")
    for e in excl:
        L.append(f"    - {e.get('mouse')}/{e.get('day')}/{e.get('ts')} — {e.get('reason')}")
    if cohort.get("duplicates_dropped"):
        L.append(f"- Duplicate filings dropped: `{cohort['duplicates_dropped']}`")
    L.append(f"- Grammar arms missing melodies: "
             f"{cohort.get('missing_melody_grammar_arms', '?')}")
    L.append(f"- trials↔grammar_samples label mismatches: "
             f"{cohort.get('trials_vs_samples_label_mismatches', '?')}")
    integ = cohort.get("integrity", {})
    inc = integ.get("cross_session_group_inconsistent", [])
    L.append(f"- Cross-session group inconsistencies: "
             f"{'none' if not inc else inc}")
    L.append("")

    # ---- sanity ----
    san = results.get("sanity")
    if san:
        L.append(f"## Sanity gate — {_go(san.get('passed', False))}\n")
        for n in san.get("notes", []):
            L.append(f"- {n}")
        for k, v in san.get("checks", {}).items():
            L.append(f"- **{k}**: `{v}`")
        if san.get("non_explorers"):
            L.append(f"- Non-explorers: `{san['non_explorers']}`")
        L.append("")

    # ---- design analysis ----
    des = results.get("design")
    if des:
        L.append("## Model-free design analysis (day 1)\n")
        L.append(f"- n_obs={des.get('n_obs')}, n_mice={des.get('n_mice')}, "
                 f"group_balance=`{des.get('group_balance')}`")
        L.append(f"- Group-consistent EE−SC sign across groups: "
                 f"**{des.get('group_consistent')}** "
                 f"(consistent ⇒ semantic; flips ⇒ intrinsic-grammar confound)")
        L.append("- Environment effect by group:")
        for r in des.get("environment_effect_by_group", []):
            L.append(f"    - group {r.get('group')}: "
                     f"EE−SC={r.get('env_effect_EE_minus_SC'):+.3f} "
                     f"CI={r.get('ci')} p={r.get('p')}")
        L.append("- Simple effect of environment within each tier "
                 "(attenuated-but-present vs absent at high complexity):")
        for r in des.get("simple_effects_by_tier", []):
            if "env_effect" in r:
                L.append(f"    - {r['tier']}: EE−SC={r['env_effect']:+.3f} "
                         f"CI={r.get('env_ci')} p={r.get('env_p')} "
                         f"(env×group p={r.get('env_x_group_p')})")
        L.append("")

    # ---- collinearity ----
    col = results.get("collinearity")
    if col:
        L.append("## Feature collinearity / variance\n")
        L.append(f"- between-arm variance: `{col.get('variance')}`")
        L.append(f"- corr(r, S) = {col.get('corr_r_S')}")
        L.append(f"- VIF: `{col.get('vif')}`")
        if col.get("flag_high_rS_collinearity"):
            L.append("- ⚠️ **HIGH r–S collinearity**: the design did not separate "
                     "fluency from semantics in practice; recovery decides if wS "
                     "is estimable at all.")
        L.append("")

    # ---- recovery ----
    rec = results.get("recovery")
    if rec and "error" not in rec:
        L.append(f"## Recovery gate — {_go(rec.get('gate_passed', False))}\n")
        pr = rec.get("parameter_recovery", {})
        L.append(f"- Parameter recovery passed: **{pr.get('passed')}** "
                 f"(n_sim={pr.get('n_sim')})")
        for row in pr.get("grid", []):
            L.append(f"    - wS_true={row['wS_true']:.2f} → "
                     f"recovered={row['wS_recovered_mean']:+.3f} "
                     f"CI={row['ci']} covers_truth={row['covers_truth']}")
        cf = rec.get("model_confusion", {})
        L.append(f"- Model confusion passed: **{cf.get('passed')}**")
        for gname in ("bd_baseline", "full", "intrinsic_grammar"):
            if gname in cf:
                g = cf[gname]
                L.append(f"    - gen={gname}: prefer_full={g['prefer_full_rate']:.2f}, "
                         f"wS(full)={g['wS_full_mean']:+.3f}, "
                         f"wS(full+grammar)={g['wS_full_grammar_mean']:+.3f}")
        L.append(f"- _{cf.get('intrinsic_grammar_note', '')}_")
        L.append("")
    elif rec:
        L.append(f"## Recovery gate — skipped\n- {rec.get('error')}\n")

    # ---- day-2 secondary ----
    sec = results.get("secondary")
    if sec:
        L.append("## Day-2 secondary (model-free + block time-course)\n")
        tc = sec.get("timecourse", {})
        L.append(f"- block time-course PI~block slope: "
                 f"{tc.get('slope_per_block')} CI={tc.get('ci')} "
                 f"→ {tc.get('interpretation')}")
        L.append(f"- mean PI by block: `{tc.get('mean_pi_by_block')}`")
        L.append("")

    # ---- Phase 2 (Bayesian) ----
    p2 = results.get("phase2")
    if p2:
        L.append("## Phase 2 — Bayesian nested-model comparison (day 1)\n")
        if "error" in p2:
            L.append(f"- ERROR: {p2['error']}")
        else:
            L.append(f"- MCMC: `{p2.get('mcmc')}`")
            L.append("- LOO comparison (rank 0 = best out-of-sample):")
            for row in p2.get("loo_compare", []):
                L.append(f"    - {row.get('model', '?')}: rank={row.get('rank')}, "
                         f"elpd_loo={row.get('elpd_loo')}, weight={row.get('weight')}")
            ws = p2.get("wS_posterior", {})
            if ws and ws.get("wS_mean") is not None:
                L.append(f"- wS posterior: mean={ws.get('wS_mean'):+.3f}, "
                         f"HDI95={ws.get('wS_hdi95')}, "
                         f"P(wS>0)={ws.get('wS_p_positive')}")
            ind = p2.get("individual_diffs", {})
            if ind:
                L.append(f"- wS leave-one-mouse-out: all-mice={ind.get('wS_full_all_mice')}, "
                         f"LOO mean={ind.get('wS_leave_one_out_mean')}, "
                         f"range={ind.get('wS_leave_one_out_range')}, "
                         f"sign_stable={ind.get('sign_stable')}")
            ppc = p2.get("posterior_predictive", {})
            if ppc:
                L.append(f"- posterior-predictive arm pattern: "
                         f"`{ppc.get('predicted_pattern')}`")
        L.append("")

    L.append("## Verdict\n")
    L.append(results.get("verdict", "_pending — see gates above_"))
    L.append("")
    return "\n".join(L)


def save_report(out_dir: str, results: Dict[str, object]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")
    json_path = os.path.join(out_dir, "results.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(results))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    return {"report": md_path, "results": json_path}
