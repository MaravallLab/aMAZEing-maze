#!/usr/bin/env python3
"""
Experiment 2: Information Criteria for the Nested Value Models
===============================================================

The earlier analysis (model_validation/, read only) compared its nested models
by Bayesian leave-one-out only, and compared the model with a semantic term
against the model of Brielmann and Dayan (2022) only block by block. This script
adds AIC and BIC for all four nested models, with the earlier analysis's own
features (tier-restricted emissions, model_validation.latent_regressors), at the
grain of the session-level preference on day 1: each mouse's share of its time
in each of the seven kinds of arm (EE and SC grammar at three levels, and the
silent arm) over the whole session.

Each model predicts the shares as the softmax of the arms' values,
A = w0 + wr r + wV dV + wS S (w0 cancels in the softmax; the silent arm has all
three terms zero; the vocalisation is held out). Two likelihoods: a Dirichlet
with a fitted concentration (shares clipped at 1e-6 and renormalised, as in the
earlier models), and a Gaussian on the shares (least squares), so that the
comparison does not rest on one choice. The full model is also refitted once
without each mouse, to show that its semantic weight does not rest on a few
animals. Parameters counted: the free weights
and the concentration or the residual variance. Fitted by maximum likelihood
(L-BFGS-B; wr and wV constrained non-negative, as in the earlier priors).

    intercept     no weight: an even spread over the arms
    fluency       wr
    baseline      wr, wV     (Brielmann and Dayan 2022)
    full          wr, wV, wS

Usage:
    python exp2_model_compare.py [GRAMMAR_DIR]

Output: GRAMMAR_DIR/BATCH_ANALYSIS/exp2_model/information_criteria.csv and
information_criteria.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_validation.config import validated_config
from model_validation.data_loading import load_arm_blocks, load_day
from model_validation import latent_regressors as lr

CELLS = [("EE", "dominant"), ("EE", "secondary"), ("EE", "rare"),
         ("SC", "dominant"), ("SC", "secondary"), ("SC", "rare"), ("silent", None)]
MODELS = {"intercept": [], "fluency": ["wr"], "baseline": ["wr", "wV"],
          "full": ["wr", "wV", "wS"]}
COL = {"wr": 0, "wV": 1, "wS": 2}


def softmax(a):
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def session_data(feat):
    """Per mouse: features (7 x 3) and shares (7) over the whole session."""
    g = feat[feat["arm_type"] == "grammar"]
    sd = {c: g[c].std(ddof=1) for c in ("r_mean", "dV_mean", "S_mean")}
    X, Y = [], []
    for mouse, sub in feat.groupby("mouse"):
        x, y = np.zeros((7, 3)), np.zeros(7)
        for i, (env, tier) in enumerate(CELLS):
            s = sub[sub["arm_type"] == "silent"] if env == "silent" else \
                sub[(sub["environment"] == env) & (sub["tier"] == tier)]
            y[i] = s["time_spent_s"].sum()
            if env != "silent":
                x[i] = [s["r_mean"].mean() / sd["r_mean"], s["dV_mean"].mean() / sd["dV_mean"],
                        s["S_mean"].mean() / sd["S_mean"]]
        if y.sum() > 0:
            X.append(x); Y.append(y / y.sum())
    return np.array(X), np.array(Y)


def block_data(feat):
    """Per block: features (7 x 3) and shares over its grammar and silent arms."""
    use = feat[feat["arm_type"].isin(["grammar", "silent"])]
    g = use[use["arm_type"] == "grammar"]
    sd = {c: g[c].std(ddof=1) for c in ("r_mean", "dV_mean", "S_mean")}
    X, Y = [], []
    for _, sub in use.groupby(["mouse", "block"]):
        if len(sub) != 7 or sub["time_spent_s"].sum() <= 0:
            continue
        x = np.column_stack([np.nan_to_num(sub[c].to_numpy()) / sd[c]
                             for c in ("r_mean", "dV_mean", "S_mean")])
        X.append(x); Y.append((sub["time_spent_s"] / sub["time_spent_s"].sum()).to_numpy())
    return np.array(X), np.array(Y)


def predict(X, w, names):
    a = np.zeros(X.shape[:2])
    for n, v in zip(names, w):
        a = a + v * X[:, :, COL[n]]
    return softmax(a)


def fit_dirichlet(X, Y, names):
    Yc = np.clip(Y, 1e-6, None); Yc = Yc / Yc.sum(axis=1, keepdims=True)
    logY = np.log(Yc)

    def nll(theta):
        w, kappa = theta[:-1], np.exp(theta[-1])
        alpha = kappa * predict(X, w, names) + 1e-9
        ll = gammaln(alpha.sum(axis=1)) - gammaln(alpha).sum(axis=1) + ((alpha - 1) * logY).sum(axis=1)
        return -ll.sum()

    bounds = [(0, None) if n in ("wr", "wV") else (None, None) for n in names] + [(-5, 10)]
    best = None
    for start in (0.0, 0.5):
        r = minimize(nll, np.r_[np.full(len(names), start), np.log(20.0)], method="L-BFGS-B",
                     bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def fit_gauss(X, Y, names):
    n = Y.size

    def sse(w):
        return np.sum((Y - predict(X, w, names)) ** 2)

    bounds = [(0, None) if nm in ("wr", "wV") else (None, None) for nm in names]
    if names:
        r = minimize(sse, np.zeros(len(names)), method="L-BFGS-B", bounds=bounds)
        s, w = r.fun, r.x
    else:
        s, w = sse(np.zeros(0)), np.zeros(0)
    ll = -0.5 * n * (np.log(2 * np.pi * s / n) + 1)
    return ll, w


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "D:/AuditoryExperiments/grammar"
    out = os.path.join(root, "BATCH_ANALYSIS", "exp2_model")
    os.makedirs(out, exist_ok=True)
    cfg = validated_config()
    day1 = load_day(load_arm_blocks(root).arm_blocks, "primary")
    day1 = day1[day1.groupby("mouse")["time_spent_s"].transform("sum") > 0]
    feat_all = lr.build_features(day1, cfg, emission="tier_restricted")

    rows, lines = [], []
    for blocks in ("all",):
        feat = feat_all
        for grain, (X, Y) in (("session", session_data(feat)),):
            for like, fitter in (("Dirichlet", fit_dirichlet), ("Gaussian", fit_gauss)):
                for model, names in MODELS.items():
                    ll, theta = fitter(X, Y, names)
                    k = len(names) + 1
                    n = len(Y) if like == "Dirichlet" else Y.size
                    rows.append({"blocks": blocks, "grain": grain, "likelihood": like, "model": model,
                                 "k": k, "n": n, "loglik": ll, "AIC": 2 * k - 2 * ll,
                                 "BIC": k * np.log(n) - 2 * ll,
                                 **{nm: v for nm, v in zip(names, theta[:len(names)])}})
    t = pd.DataFrame(rows)

    # head to head at the session grain: the full model against the baseline
    from scipy.stats import chi2
    X, Y = session_data(feat_all)
    lines.append("Session grain, the model with semantic value against Brielmann and Dayan's "
                 "model (fluency and learning):")
    for like in ("Dirichlet", "Gaussian"):
        g = t[(t["grain"] == "session") & (t["likelihood"] == like)].set_index("model")
        lr_stat = 2 * (g.loc["full", "loglik"] - g.loc["baseline", "loglik"])
        d_aic = g.loc["full", "AIC"] - g.loc["baseline", "AIC"]
        weight = 1 / (1 + np.exp(d_aic / 2))
        lines.append(f"  {like}: likelihood ratio chi2(1) = {lr_stat:.1f}, p = {chi2.sf(lr_stat, 1):.1e}; "
                     f"Akaike weight of the full model against the baseline {weight:.6f}")
    obs = Y.mean(axis=0)
    names_cells = ["EE dominant", "EE secondary", "EE rare", "SC dominant", "SC secondary", "SC rare", "silent"]
    preds = {}
    for model in ("baseline", "full"):
        _, theta = fit_dirichlet(X, Y, MODELS[model])
        P = predict(X, theta[:len(MODELS[model])], MODELS[model])
        preds[model] = P.mean(axis=0)
        r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)
        r2_cells = 1 - np.sum((obs - preds[model]) ** 2) / np.sum((obs - obs.mean()) ** 2)
        lines.append(f"  {model}: variance explained, each mouse's shares {r2:.3f}; the seven mean "
                     f"shares {r2_cells:.3f}")
    lines.append("  mean share of time (observed | baseline | full):")
    for i, c in enumerate(names_cells):
        lines.append(f"    {c:13s} {obs[i]:.3f} | {preds['baseline'][i]:.3f} | {preds['full'][i]:.3f}")
    # leave one mouse out: the full model refitted without each mouse in turn
    mice = sorted(feat_all["mouse"].astype(str).unique())
    ws = []
    for m in mice:
        Xm, Ym = session_data(feat_all[feat_all["mouse"].astype(str) != m])
        ws.append(fit_dirichlet(Xm, Ym, MODELS["full"])[1][COL["wS"]])
    ws = np.array(ws)
    lines.append(f"  leave one mouse out, Dirichlet ({len(ws)} refits of the full model): wS from "
                 f"{ws.min():+.3f} to {ws.max():+.3f}, {int((ws > 0).sum())} of {len(ws)} positive")
    ee = [0, 1, 2]; sc = [3, 4, 5]
    for lab, v in (("observed", obs), ("baseline", preds["baseline"]), ("full", preds["full"])):
        idx = (v[ee].sum() - v[sc].sum()) / (v[ee].sum() + v[sc].sum())
        lines.append(f"    EE-SC index of the mean shares, {lab}: {idx:+.3f}")
    lines.append("")

    for (blocks, grain, like), g in t.groupby(["blocks", "grain", "likelihood"], sort=False):
        base = g.set_index("model").loc["baseline"]
        lines.append(f"\n{'all four blocks' if blocks == 'all' else 'blocks 2 to 4'}, {grain} grain, "
                     f"{like} likelihood (n = {int(g['n'].iloc[0])}); differences from the baseline:")
        for _, r in g.iterrows():
            w = ", ".join(f"{nm} {r[nm]:+.3f}" for nm in ("wr", "wV", "wS") if nm in r and pd.notna(r[nm]))
            lines.append(f"  {r['model']:10s} k={r['k']}  logL {r['loglik']:10.2f}  AIC {r['AIC']:10.2f} "
                         f"(dAIC {r['AIC'] - base['AIC']:+7.2f})  BIC {r['BIC']:10.2f} "
                         f"(dBIC {r['BIC'] - base['BIC']:+7.2f})  {w}")
    print("\n".join(lines))
    t.to_csv(os.path.join(out, "information_criteria.csv"), index=False)
    with open(os.path.join(out, "information_criteria.txt"), "w", encoding="utf-8") as f:
        f.write("AIC and BIC of the nested value models, day 1 (lower is better; "
                "a difference of more than 10 is strong support)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
