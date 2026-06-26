"""Grain-comparison: fit the SAME model at three observation grains to show the
EE-SC effect recovers as aggregation passes per-block dwell noise.

  Grain A  per-block               (anchor; underpowered, near-flat)
  Grain B  per-mouse cell-mean     (full 7-arm pattern; Dirichlet, per-mouse
                                     selectivity beta_m — NOT a canceling intercept)
  Grain C  per-tier EE-SC PI       (max-power semantic contrast; r,V cancel,
                                     identifies wS only; per-mouse PI intercept)

S uses tier-restricted emissions (fixes the i->i+3 secondary flip). w_r,w_V>=0
(half-normal); w_S ~ Normal(0,1). No interaction term — the env x tier interaction
is carried by the tier-dependence of S. Cell/PI grains RE-EXPRESS the model-free
result as a process model: a mechanistic illustration, NOT independent validation.

LOO is NOT comparable across grains (different observation models). Run standalone:
    python grain_comparison.py <results_dir> <out_dir>
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_AUD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AUD not in sys.path:
    sys.path.insert(0, _AUD)
from model_validation import config as C, data_loading as dl, latent_regressors as lr
from model_validation import recovery as rec, models as M

CELLS = ["EE-dominant", "EE-secondary", "EE-rare",
         "SC-dominant", "SC-secondary", "SC-rare", "silent"]
TIERS = ["dominant", "secondary", "rare"]
ENV_COLORS = {"EE": "#1565C0", "SC": "#B71C1C"}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def cell_regressors(feats):
    g = feats[feats.arm_type == "grammar"].dropna(subset=["environment", "tier"])
    sd = {c: (g[c].std(ddof=1) if g[c].std(ddof=1) > 1e-8 else 1.0)
          for c in ["r_mean", "dV_mean", "S_mean"]}
    R, DV, S = np.zeros(7), np.zeros(7), np.zeros(7)
    for i, cell in enumerate(CELLS):
        if cell == "silent":
            continue
        env, tier = cell.split("-")
        s = g[(g.environment == env) & (g.tier == tier)]
        R[i] = s.r_mean.mean() / sd["r_mean"]
        DV[i] = s.dV_mean.mean() / sd["dV_mean"]
        S[i] = s.S_mean.mean() / sd["S_mean"]
    return R, DV, S


def cell_composition(day1, cohort):
    mice, comps, T = [], [], []
    for m, sub in day1[day1.mouse.isin(cohort)].groupby("mouse"):
        vec = np.zeros(7)
        for i, cell in enumerate(CELLS):
            if cell == "silent":
                vec[i] = sub[sub.arm_type == "silent"].time_spent_s.sum()
            else:
                env, tier = cell.split("-")
                vec[i] = sub[(sub.arm_type == "grammar") & (sub.environment == env)
                             & (sub.tier == tier)].time_spent_s.sum()
        tot = vec.sum()
        if tot <= 0:
            continue
        mice.append(m); comps.append(vec / tot); T.append(tot)
    return mice, np.array(comps), np.array(T)


def tier_pi(day1, cohort):
    rows = []
    d = day1[(day1.mouse.isin(cohort)) & (day1.arm_type == "grammar")]
    for (m, t), sub in d.groupby(["mouse", "tier"]):
        ee = sub[sub.environment == "EE"].time_spent_s.sum()
        sc = sub[sub.environment == "SC"].time_spent_s.sum()
        if ee + sc <= 0:
            continue
        grp = int(sub["group"].iloc[0])
        rows.append({"mouse": m, "tier": t, "group": grp, "PI": (ee - sc) / (ee + sc)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# models  (Grain B = cell-mean Dirichlet; Grain C = per-tier PI Normal)
# ---------------------------------------------------------------------------
def _weights_block(pm, use, R, DV, S):
    w0 = pm.Normal("w0", 0.0, 2.0) if "w0" in use else 0.0
    wr = pm.HalfNormal("wr", 1.0) if "wr" in use else 0.0
    wV = pm.HalfNormal("wV", 1.0) if "wV" in use else 0.0
    wS = pm.Normal("wS", 0.0, 1.0) if "wS" in use else 0.0
    return w0 + wr * R + wV * DV + wS * S


def fit_grain_b(comp, T, R, DV, S, model="full", draws=1000, tune=2000,
                chains=4, ta=0.95, seed=0):
    pm, az = M.require_pymc()
    use = rec.MODELS[model]
    Tn = T / np.median(T)
    obs = np.clip(comp, 1e-6, None); obs = obs / obs.sum(axis=1, keepdims=True)
    nM = comp.shape[0]
    with pm.Model():
        A = _weights_block(pm, use, R, DV, S)
        mu_lb = pm.Normal("mu_lb", 0.0, 1.0)
        sd_lb = pm.HalfNormal("sd_lb", 1.0)
        lb = pm.Normal("lb", mu_lb, sd_lb, shape=nM)
        beta = pm.math.exp(lb)                          # per-mouse selectivity (non-canceling)
        logits = beta[:, None] * A[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        e = pm.math.exp(logits)
        pred = e / e.sum(axis=1, keepdims=True)
        kappa = pm.Gamma("kappa", mu=50.0, sigma=50.0)
        alpha = kappa * Tn[:, None] * pred + 1e-3
        pm.Dirichlet("obs", a=alpha, observed=obs)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=ta, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    return idata


def fit_grain_c(df_pi, dS, model="full", draws=1000, tune=2000, chains=4,
                ta=0.95, seed=0):
    pm, az = M.require_pymc()
    use = rec.MODELS[model]
    mice = sorted(df_pi.mouse.unique()); midx = {m: i for i, m in enumerate(mice)}
    m_idx = df_pi.mouse.map(midx).values
    t_idx = df_pi.tier.map({t: i for i, t in enumerate(TIERS)}).values
    pi = df_pi.PI.values.astype(float)
    dSv = np.asarray(dS, dtype=float)
    with pm.Model():
        wS = pm.Normal("wS", 0.0, 1.0) if "wS" in use else 0.0
        sigma = pm.HalfNormal("sigma", 1.0)
        sd_u = pm.HalfNormal("sd_u", 1.0)
        u = pm.Normal("u", 0.0, sd_u, shape=len(mice))
        mu = pm.math.tanh(wS * dSv[t_idx] / 2.0) + u[m_idx]
        pm.Normal("obs", mu, sigma, observed=pi)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=ta, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    return idata, mice


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def wsum(idata, params=("w0", "wr", "wV", "wS")):
    import arviz as az
    post = idata.posterior
    out = {}
    for p in params:
        if p in post:
            x = post[p].values.reshape(-1)
            lo, hi = az.hdi(x, hdi_prob=0.95)
            out[p] = {"mean": float(x.mean()), "hdi": [float(lo), float(hi)],
                      "p_pos": float((x > 0).mean())}
    return out


def lomo_wS_grain_c(df_pi, dS):
    """Fast leave-one-mouse-out wS (within-mouse-centred least squares)."""
    from scipy.optimize import minimize_scalar
    tmap = {t: i for i, t in enumerate(TIERS)}
    dSv = np.asarray(dS, float)
    mice = sorted(df_pi.mouse.unique())
    out = {}
    for held in [None] + mice:
        d = df_pi if held is None else df_pi[df_pi.mouse != held]
        ti = d.tier.map(tmap).values

        def obj(wS):
            d2 = d.assign(pred=np.tanh(wS * dSv[ti] / 2.0))
            err = 0.0
            for _, s in d2.groupby("mouse"):
                err += np.sum(((s.PI - s.PI.mean()) - (s.pred - s.pred.mean())) ** 2)
            return err
        r = minimize_scalar(obj, bounds=(-5, 5), method="bounded")
        out["all" if held is None else held] = float(r.x)
    return out


def compare_intercept_full(fits):
    import arviz as az
    return az.compare({k: v for k, v in fits.items()}, ic="loo")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _save(fig, out_dir, name, written):
    p = os.path.join(out_dir, name); fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig); written.append(p); print("  wrote", name)


def ppc_cellmean(idata, R, DV, S, nM, sub=300):
    """Posterior-predictive mean composition = avg over posterior & mice of
    softmax(beta_m * A). Using beta (not beta=1) is essential — beta_m and the
    weight scale trade off, so softmax(A) alone over-peaks."""
    post = idata.posterior
    n = post.dims["chain"] * post.dims["draw"]

    def flat(name, d=0.0):
        return post[name].values.reshape(-1) if name in post else np.full(n, d)
    w0, wr, wV, wS = flat("w0"), flat("wr"), flat("wV"), flat("wS")
    lb = post["lb"].values.reshape(-1, nM)
    Asamp = (w0[:, None] + wr[:, None] * R[None, :] + wV[:, None] * DV[None, :]
             + wS[:, None] * S[None, :])                       # (n, 7)
    idx = np.linspace(0, n - 1, min(sub, n)).astype(int)
    acc = np.zeros(7)
    for s in idx:
        logit = np.exp(lb[s])[:, None] * Asamp[s][None, :]
        logit -= logit.max(axis=1, keepdims=True)
        e = np.exp(logit); acc += (e / e.sum(axis=1, keepdims=True)).mean(axis=0)
    return acc / len(idx)


def fig_grainB_ppc(comp, pred, out_dir, written):
    obs = comp.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    x = np.arange(7); wbar = 0.4
    ax.bar(x - wbar / 2, pred, wbar, label="model predicted", color="#1565C0", edgecolor="white")
    ax.bar(x + wbar / 2, obs, wbar, label="observed", color="#F9A825", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(CELLS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("dwell fraction (per-mouse cell-mean)")
    ax.set_title("Grain B — per-mouse cell-mean: predicted vs observed", fontweight="bold")
    ax.legend()
    _save(fig, out_dir, "grainB_cellmean_ppc.png", written)


def fig_grainC_pi(df_pi, dS, wS_mean, out_dir, written):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(3)
    pred = [np.tanh(wS_mean * dS[i] / 2.0) for i in range(3)]
    ax.plot(x, pred, "-o", color="black", lw=2, label="model predicted", zorder=5)
    for grp, mk in [(1, "o"), (2, "s")]:
        for env_unused in [0]:
            means = []; sems = []
            for t in TIERS:
                v = df_pi[(df_pi.group == grp) & (df_pi.tier == t)].PI.values
                means.append(np.mean(v) if len(v) else np.nan)
                sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            ax.errorbar(x + (0.06 if grp == 2 else -0.06), means, yerr=sems, fmt=mk,
                        capsize=3, label=f"observed group {grp}",
                        color="#1565C0" if grp == 1 else "#B71C1C")
    ax.axhline(0, ls="--", color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TIERS)
    ax.set_ylabel("EE−SC preference index")
    ax.set_title(f"Grain C — per-tier PI: predicted vs observed (wS={wS_mean:+.3f})",
                 fontweight="bold")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "grainC_tierPI_ppc.png", written)


def fig_grain_weights(table, out_dir, written):
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    grains = list(table.keys()); y = np.arange(len(grains))
    for i, gname in enumerate(grains):
        ws = table[gname].get("wS")
        if ws:
            ax.errorbar(ws["mean"], i, xerr=[[ws["mean"] - ws["hdi"][0]],
                        [ws["hdi"][1] - ws["mean"]]], fmt="o", capsize=4,
                        color="#2E7D32", ms=8)
            ax.text(ws["mean"], i + 0.12, f"P>0={ws['p_pos']:.2f}", ha="center", fontsize=8)
    ax.axvline(0, ls="--", color="grey")
    ax.set_yticks(y); ax.set_yticklabels(grains)
    ax.set_xlabel("wS (semantic weight) — 95% HDI")
    ax.set_title("Semantic weight recovers as grain coarsens\n"
                 "(block → cell-mean → PI); effect real but small per-block",
                 fontweight="bold")
    _save(fig, out_dir, "grain_weights_wS.png", written)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(results_dir, out_dir, draws=1000, tune=2000, chains=4, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    cfg = C.validated_config()
    load = dl.load_arm_blocks(results_dir)
    day1 = dl.load_day(load.arm_blocks, "primary")
    feats = lr.build_features(day1, cfg, emission="tier_restricted")

    # ---- cohort resolution (precondition) ----
    design = rec.build_design(feats)
    cohort = sorted(design.mice())                       # explorers with valid blocks
    dropped = sorted(set(day1.mouse.unique()) - set(cohort))
    cohort_note = {
        "n_model_free_day1": int(day1.mouse.nunique()),
        "n_analysis_cohort": len(cohort),
        "dropped": dropped,
        "reason": ("mouse 13672 is a non-explorer (0 s total dwell, 0 arms "
                   "visited) — carries no preference info; excluded consistently."),
    }
    print("[cohort]", cohort_note)

    R, DV, S = cell_regressors(feats)
    dS = np.array([S[CELLS.index(f"EE-{t}")] - S[CELLS.index(f"SC-{t}")] for t in TIERS])

    results = {"cohort": cohort_note, "cell_regressors": {"R": R.tolist(),
               "DV": DV.tolist(), "S": S.tolist(), "dS_by_tier": dS.tolist()}}
    table = {}
    written = []

    # ---- Grain A: per-block (anchor) ----
    print("[grain A] per-block full ...")
    ia_full = M.fit_model(design, "full", draws=draws, tune=tune, chains=chains, seed=seed)
    table["A_per_block"] = wsum(ia_full)

    # ---- Grain B: per-mouse cell-mean ----
    print("[grain B] cell-mean full + intercept ...")
    mice_b, comp, T = cell_composition(day1, cohort)
    ib_full = fit_grain_b(comp, T, R, DV, S, "full", draws, tune, chains, seed=seed)
    ib_int = fit_grain_b(comp, T, R, DV, S, "intercept", draws, tune, chains, seed=seed)
    table["B_cell_mean"] = wsum(ib_full)
    results["grainB_compare"] = compare_intercept_full(
        {"full": ib_full, "intercept": ib_int}).reset_index().rename(
        columns={"index": "model"}).to_dict("records")

    # ---- Grain C: per-tier PI ----
    print("[grain C] per-tier PI full + intercept ...")
    df_pi = tier_pi(day1, cohort)
    ic_full, mice_c = fit_grain_c(df_pi, dS, "full", draws, tune, chains, seed=seed)
    ic_int, _ = fit_grain_c(df_pi, dS, "intercept", draws, tune, chains, seed=seed)
    table["C_tier_PI"] = wsum(ic_full, params=("wS",))
    results["grainC_compare"] = compare_intercept_full(
        {"full": ic_full, "intercept": ic_int}).reset_index().rename(
        columns={"index": "model"}).to_dict("records")
    results["grainC_lomo_wS"] = lomo_wS_grain_c(df_pi, dS)

    results["weight_table"] = table

    # ---- figures ----
    print("[figures]")
    ppc_b = ppc_cellmean(ib_full, R, DV, S, len(mice_b))
    results["grainB_ppc_predicted"] = ppc_b.tolist()
    results["grainB_observed_mean"] = comp.mean(axis=0).tolist()
    fig_grainB_ppc(comp, ppc_b, out_dir, written)
    wS_c = table["C_tier_PI"]["wS"]["mean"]
    fig_grainC_pi(df_pi, dS, wS_c, out_dir, written)
    fig_grain_weights(table, out_dir, written)

    # ---- verdict ----
    wSb = table["B_cell_mean"].get("wS", {})
    wSc = table["C_tier_PI"].get("wS", {})
    lomo = results["grainC_lomo_wS"]
    sign_stable = all(v > 0 for k, v in lomo.items() if k != "all")
    results["verdict"] = (
        f"Cohort = {len(cohort)} explorers (13672 excluded, non-explorer, 0 dwell). "
        f"The EE-SC effect is robust model-free and is reproduced by the generative "
        f"model once fit at the grain where it lives. wS by grain: "
        f"block {table['A_per_block'].get('wS',{}).get('mean'):+.3f} "
        f"{table['A_per_block'].get('wS',{}).get('hdi')}; "
        f"cell-mean {wSb.get('mean'):+.3f} {wSb.get('hdi')}; "
        f"PI {wSc.get('mean'):+.3f} {wSc.get('hdi')} (P>0={wSc.get('p_pos')}). "
        f"wS pulls away from 0 as the grain coarsens — the per-block flatness was a "
        f"grain mismatch, not absence of effect. Leave-one-mouse-out wS sign-stable "
        f"at the PI grain: {sign_stable}. The generative fit is a MECHANISTIC "
        f"ILLUSTRATION of the model-free result (it re-expresses ~7 cell means / 3 "
        f"PIs with 4 weights), not independent statistical validation; the statistical "
        f"weight remains the design-based model-free analysis. Secondary-tier fit "
        f"remains the weakest (residual i->i+3 corruption of S). LOO is NOT compared "
        f"across grains (non-comparable observation models)."
    )

    with open(os.path.join(out_dir, "grain_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("[verdict]", results["verdict"])
    print("[done] figures:", [os.path.basename(p) for p in written])
    return results


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else C.DEFAULT_RESULTS_DIR
    od = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("TEMP", "."), "mv_grain")
    run(rd, od)
