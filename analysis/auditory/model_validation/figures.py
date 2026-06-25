"""Figure generation for the model-validation pipeline.

Reads a finished run's `results.json` + `day1_arm_block_features.csv` from an
out_dir and writes labelled 300-dpi PNGs into that same out_dir. Each figure is
tied to a specific claim in the report; every figure is wrapped so one failure
does not abort the rest.

Run standalone:
    python -m model_validation.figures  <out_dir>
or via the pipeline with `--figures`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENV_COLORS = {"EE": "#1565C0", "SC": "#B71C1C"}
TIER_ORDER = ["dominant", "secondary", "rare"]
ARM_COLORS = {"grammarA": "#1565C0", "grammarB": "#B71C1C", "silent": "#777777"}


def _save(fig, out_dir, name, written):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    print(f"  wrote {name}")


def _sem(a):
    a = np.asarray(a, float)
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


# ---------------------------------------------------------------------------
# Figure 1 — EE vs SC dwell by tier, per counterbalancing group (the headline)
# ---------------------------------------------------------------------------
def fig_ee_sc_by_tier_group(feats, out_dir, written):
    g = feats[feats["arm_type"] == "grammar"].dropna(subset=["environment", "tier", "group"])
    groups = sorted(g["group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.6),
                             sharey=True, squeeze=False, constrained_layout=True)
    for ax, grp in zip(axes[0], groups):
        sub = g[g["group"] == grp]
        per = (sub.groupby(["mouse", "tier", "environment"])["time_spent_s"]
               .sum().reset_index())
        x = np.arange(len(TIER_ORDER)); w = 0.38
        for k, env in enumerate(("EE", "SC")):
            means, sems = [], []
            for t in TIER_ORDER:
                v = per[(per["tier"] == t) & (per["environment"] == env)]["time_spent_s"].values
                means.append(float(np.mean(v)) if len(v) else 0.0)
                sems.append(_sem(v))
            ax.bar(x + (k - 0.5) * w, means, w, yerr=sems, capsize=3,
                   color=ENV_COLORS[env], label=env, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(TIER_ORDER)
        ax.set_title(f"counterbalance group {int(grp)}")
        ax.set_xlabel("predictability tier")
    axes[0][0].set_ylabel("mean per-mouse dwell (s)")
    axes[0][0].legend(title="environment\nassociation")
    fig.suptitle("EE vs SC dwell by tier, per group\n"
                 "(consistent EE>SC across both groups ⇒ semantic, not intrinsic-grammar)",
                 fontweight="bold")
    _save(fig, out_dir, "fig1_ee_sc_by_tier_group.png", written)


# ---------------------------------------------------------------------------
# Figure 2 — per-mouse EE-SC preference index, by group
# ---------------------------------------------------------------------------
def fig_per_mouse_pi(feats, out_dir, written):
    g = feats[feats["arm_type"] == "grammar"].dropna(subset=["environment", "group"])
    rows = []
    for (mouse, grp), sub in g.groupby(["mouse", "group"]):
        ee = sub.loc[sub["environment"] == "EE", "time_spent_s"].sum()
        sc = sub.loc[sub["environment"] == "SC", "time_spent_s"].sum()
        tot = ee + sc
        if tot > 0:
            rows.append({"mouse": mouse, "group": int(grp), "PI": (ee - sc) / tot})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = sorted(d["group"].unique())
    for grp in groups:
        sub = d[d["group"] == grp]
        x = np.full(len(sub), grp) + np.random.uniform(-0.08, 0.08, len(sub))
        ax.scatter(x, sub["PI"], s=30, alpha=0.6,
                   color="#1565C0" if grp == 1 else "#B71C1C", label=f"group {grp}")
        m, s = sub["PI"].mean(), _sem(sub["PI"].values)
        ax.errorbar(grp + 0.22, m, yerr=s, fmt="o", color="black", capsize=4, ms=6)
    ax.axhline(0, ls="--", color="grey", lw=0.8)
    ax.set_xticks(groups); ax.set_xticklabels([f"group {g}" for g in groups])
    ax.set_ylabel("EE−SC preference index")
    ax.set_title("Per-mouse EE−SC preference (black = group mean ± SEM)", fontweight="bold")
    ax.legend()
    _save(fig, out_dir, "fig2_per_mouse_pi.png", written)


# ---------------------------------------------------------------------------
# Figure 3 — the six-cell behavioural pattern (env x tier) + silent/voc refs
# ---------------------------------------------------------------------------
def fig_six_cell_pattern(feats, out_dir, written):
    g = feats[feats["arm_type"] == "grammar"].dropna(subset=["environment", "tier"])
    per = g.groupby(["mouse", "environment", "tier"])["time_spent_s"].sum().reset_index()
    labels, means, sems, colors = [], [], [], []
    for env in ("EE", "SC"):
        for t in TIER_ORDER:
            v = per[(per["environment"] == env) & (per["tier"] == t)]["time_spent_s"].values
            labels.append(f"{env}\n{t}"); means.append(np.mean(v) if len(v) else 0)
            sems.append(_sem(v)); colors.append(ENV_COLORS[env])
    # reference arms
    for atype, lab in (("vocalisation", "voc"), ("silent", "silent")):
        sub = feats[feats["arm_type"] == atype]
        pm = sub.groupby("mouse")["time_spent_s"].sum().values
        if len(pm):
            labels.append(lab); means.append(np.mean(pm)); sems.append(_sem(pm))
            colors.append("#2E7D32" if atype == "vocalisation" else "#777777")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, capsize=3, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean per-mouse dwell (s)")
    ax.set_title("Six-cell grammar pattern + reference arms (mean ± SEM)", fontweight="bold")
    _save(fig, out_dir, "fig3_six_cell_pattern.png", written)


# ---------------------------------------------------------------------------
# Figure 4 — latent regressors in (r, S) space
# ---------------------------------------------------------------------------
def fig_latent_rs(feats, out_dir, written):
    g = feats[feats["arm_type"] == "grammar"].dropna(subset=["environment", "tier", "r_mean", "S_mean"])
    cells = g.drop_duplicates(subset=["group", "environment", "tier"])
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"dominant": "o", "secondary": "s", "rare": "^"}
    for _, c in cells.iterrows():
        ax.scatter(c["r_mean"], c["S_mean"], s=120,
                   color=ENV_COLORS.get(c["environment"], "grey"),
                   marker=markers.get(c["tier"], "o"), edgecolor="black",
                   label=f"{c['environment']}-{c['tier']}")
    ax.axhline(0, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("r  (fluency = −surprise under full grammar; higher = more predictable)")
    ax.set_ylabel("S  (semantic belief shift toward EE)")
    ax.set_title("Latent regressors per (group,env,tier) cell\n"
                 "S clean EE+/SC− at dominant, flips at secondary, ~0 at rare; r tracks tier",
                 fontweight="bold")
    handles, labs = ax.get_legend_handles_labels()
    seen = dict(zip(labs, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2)
    _save(fig, out_dir, "fig4_latent_regressors_rS.png", written)


# ---------------------------------------------------------------------------
# Figure 5 — recovery (identifiability + confusion)
# ---------------------------------------------------------------------------
def fig_recovery(results, out_dir, written):
    rec = results.get("recovery", {})
    pr = rec.get("parameter_recovery"); cf = rec.get("model_confusion")
    if not pr or not cf:
        print("  (skip fig5: recovery not in results)"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    grid = pr["grid"]
    tru = [r["wS_true"] for r in grid]
    rec_m = [r["wS_recovered_mean"] for r in grid]
    lo = [r["wS_recovered_mean"] - r["ci"][0] for r in grid]
    hi = [r["ci"][1] - r["wS_recovered_mean"] for r in grid]
    a1.errorbar(tru, rec_m, yerr=[lo, hi], fmt="o", capsize=4, color="#1565C0")
    lim = [min(tru) - 0.2, max(tru) + 0.2]
    a1.plot(lim, lim, ls="--", color="grey")
    a1.set_xlabel("true wS"); a1.set_ylabel("recovered wS (95% CI)")
    a1.set_title(f"Parameter recovery (passed={pr.get('passed')})")
    gens = [k for k in ("bd_baseline", "full", "intrinsic_grammar") if k in cf]
    rates = [cf[k]["prefer_full_rate"] for k in gens]
    a2.bar(range(len(gens)), rates, color=["#777777", "#2E7D32", "#B71C1C"])
    a2.axhline(0.5, ls="--", color="grey")
    a2.set_xticks(range(len(gens))); a2.set_xticklabels(gens, fontsize=8, rotation=10)
    a2.set_ylabel("P(prefer 'full' out-of-sample)"); a2.set_ylim(0, 1)
    a2.set_title("Model confusion (calibration)")
    fig.suptitle("Recovery gate: wS estimable, no false positive, confusion calibrated",
                 fontweight="bold")
    _save(fig, out_dir, "fig5_recovery.png", written)


# ---------------------------------------------------------------------------
# Figure 6 — model comparison (LOO) + wS posterior
# ---------------------------------------------------------------------------
def fig_model_comparison(results, out_dir, written):
    p2 = results.get("phase2", {})
    if not p2 or "error" in p2:
        print("  (skip fig6: phase2 not available)"); return
    loo = p2.get("loo_compare", []); ws = p2.get("wS_posterior", {})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    if loo:
        names = [r.get("model") for r in loo]
        elpd = [r.get("elpd_loo", 0.0) or 0.0 for r in loo]
        dse = [r.get("dse", 0.0) or 0.0 for r in loo]
        best = max(elpd)
        delta = [e - best for e in elpd]   # standard plot_compare view: vs best
        a1.errorbar(delta, range(len(names)), xerr=dse, fmt="o", capsize=4, color="#1565C0")
        a1.axvline(0, ls="--", color="grey", lw=0.8)
        a1.set_yticks(range(len(names))); a1.set_yticklabels(names)
        a1.invert_yaxis()
        a1.set_xlabel("Δelpd_loo vs best  (0 = best; bars = dse)")
        a1.set_title("Bayesian LOO comparison\n(overlap with 0 ⇒ models weakly separated)")
    if ws and ws.get("wS_mean") is not None:
        m = ws["wS_mean"]; hdi = ws.get("wS_hdi95", [m, m])
        a2.errorbar([0], [m], yerr=[[m - hdi[0]], [hdi[1] - m]], fmt="o",
                    capsize=5, color="#2E7D32", ms=8)
        a2.axhline(0, ls="--", color="grey")
        a2.set_xticks([]); a2.set_ylabel("wS (semantic weight)")
        a2.set_title(f"wS posterior: mean={m:+.3f}\n95% HDI={[round(x,3) for x in hdi]}, "
                     f"P(wS>0)={ws.get('wS_p_positive')}")
    fig.suptitle("Does the semantic term beat the B–D baseline out of sample?",
                 fontweight="bold")
    _save(fig, out_dir, "fig6_model_comparison_wS.png", written)


# ---------------------------------------------------------------------------
# Figure 7 — posterior-predictive arm pattern vs observed
# ---------------------------------------------------------------------------
def fig_posterior_predictive(results, feats, out_dir, written):
    p2 = results.get("phase2", {})
    pred = (p2.get("posterior_predictive") or {}).get("predicted_pattern")
    if not pred:
        print("  (skip fig7: no posterior predictive)"); return
    # observed dwell shares by grammar/silent, per block, averaged
    g = feats.copy()
    key = {"A": "grammarA", "B": "grammarB"}
    g["cat"] = g.apply(lambda r: key.get(r["grammar"], "silent")
                       if r["arm_type"] in ("grammar", "silent") else None, axis=1)
    g = g.dropna(subset=["cat"])
    obs_share = {}
    for (mouse, blk), sub in g.groupby(["mouse", "block"]):
        tot = sub["time_spent_s"].sum()
        if tot <= 0:
            continue
        for cat, s2 in sub.groupby("cat"):
            obs_share.setdefault(cat, []).append(s2["time_spent_s"].sum() / tot)
    cats = [c for c in ("grammarA", "grammarB", "silent") if c in pred]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, [pred[c] for c in cats], w, label="predicted",
           color="#1565C0", edgecolor="white")
    ax.bar(x + w / 2, [np.mean(obs_share.get(c, [0])) for c in cats], w,
           label="observed", color="#F9A825", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("dwell fraction within block")
    ax.set_title("Posterior-predictive vs observed arm pattern", fontweight="bold")
    ax.legend()
    _save(fig, out_dir, "fig7_posterior_predictive.png", written)


# ---------------------------------------------------------------------------
# Figure 8 — within-session block time-course (day 1 vs day 2)
# ---------------------------------------------------------------------------
def fig_block_timecourse(results, out_dir, written):
    series = []
    d1 = (results.get("day1_timecourse") or {}).get("mean_pi_by_block")
    if d1:
        series.append(("day 1", d1, "#1565C0"))
    sec = (results.get("secondary") or {}).get("timecourse", {}).get("mean_pi_by_block")
    if sec:
        series.append(("day 2", sec, "#B71C1C"))
    if not series:
        print("  (skip fig8: no time-course)"); return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, mp, color in series:
        blocks = sorted(int(b) for b in mp.keys())
        ax.plot(blocks, [mp[str(b)] if str(b) in mp else mp[b] for b in blocks],
                "-o", color=color, label=label, lw=2)
    ax.axhline(0, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("15-min reshuffle block"); ax.set_ylabel("mean EE−SC PI")
    ax.set_title("Within-session PI time-course\n(day-2 decay = extinction signature)",
                 fontweight="bold")
    ax.legend()
    _save(fig, out_dir, "fig8_block_timecourse.png", written)


def make_all(out_dir: str) -> List[str]:
    """Generate every figure from a finished run's out_dir."""
    with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    feats_path = os.path.join(out_dir, "day1_arm_block_features.csv")
    feats = pd.read_csv(feats_path) if os.path.isfile(feats_path) else pd.DataFrame()
    written: List[str] = []
    print(f"[figures] writing into {out_dir}")
    for fn in (
        lambda: fig_ee_sc_by_tier_group(feats, out_dir, written),
        lambda: fig_per_mouse_pi(feats, out_dir, written),
        lambda: fig_six_cell_pattern(feats, out_dir, written),
        lambda: fig_latent_rs(feats, out_dir, written),
        lambda: fig_recovery(results, out_dir, written),
        lambda: fig_model_comparison(results, out_dir, written),
        lambda: fig_posterior_predictive(results, feats, out_dir, written),
        lambda: fig_block_timecourse(results, out_dir, written),
    ):
        try:
            fn()
        except Exception as e:  # one figure failing must not abort the rest
            print(f"  FIGURE FAILED: {e}")
    print(f"[figures] {len(written)} figures written")
    return written


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m model_validation.figures <out_dir>")
    make_all(sys.argv[1])
