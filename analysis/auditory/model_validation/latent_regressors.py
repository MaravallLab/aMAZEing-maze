"""Bigram HMM context filter + Brielmann-Dayan system-state updates.

Pure NumPy so it is fast enough to run thousands of times inside the recovery
loop.

DESIGN CHOICE (documented per spec commitment #3): the latent regressors are
properties of the *stimulus*, not of the mouse's trajectory. For each arm-block
we run the filters over the logged melodies of that arm from the fixed prior,
summarise to per-arm-block scalars, and only then predict preference. This
avoids the chicken-and-egg of belief depending on dwell depending on belief.

We compute features PER MELODY (each 12-tone melody filtered independently from
b0) and average across melodies. Because melodies are i.i.d. draws from a
(grammar, tier) cell, the MEAN features converge to a per-cell constant and are
NOT biased by the known over-logging of melodies (render() logs 20 melodies per
arm entry regardless of how long the mouse stayed; see experiments.py). The
accumulated *sum* variants are reported but are sensitive to that over-logging,
so the report flags them.

EMISSIONS ARE BIGRAM: the hidden context z in {EE, SC} selects which grammar
governs the tone transition. A single-tone emission would carry no EE/SC
information (both grammars share a uniform tone marginal) and force S == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import ValidatedConfig, symbol_to_index


# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------

def parse_symbols(symbols: str, sym2idx: Dict[str, int]) -> np.ndarray:
    """Convert a melody letter string (e.g. 'CDEFABCDEFAB') to a tone-index array."""
    return np.array([sym2idx[s] for s in symbols], dtype=np.int64)


# ---------------------------------------------------------------------------
# HMM forward filter (bigram emissions over the EE/SC context)
# ---------------------------------------------------------------------------

@dataclass
class FilterTrace:
    b_prior: np.ndarray   # (T-1, 2) predicted belief b̄ before each transition
    b_post: np.ndarray    # (T-1, 2) posterior belief b after each transition
    S: np.ndarray         # (T-1,)   semantic signal  V·(b_post - b_prior)
    r_hmm: np.ndarray     # (T-1,)   log marginal likelihood of the transition (fluency)


def hmm_filter(seq: np.ndarray, M_EE: np.ndarray, M_SC: np.ndarray,
               cfg: ValidatedConfig) -> FilterTrace:
    """Forward filter over context z given a tone-index sequence.

    For each transition o_{t-1} -> o_t:
        predict : b̄ = A_ctxᵀ b_prev
        likelihood per context k:  L_k = M^{(k)}[o_{t-1}, o_t]
        update  : b ∝ b̄ ⊙ L
        S       : V · (b - b̄)           (= b_EE - b̄_EE since V = [1, 0])
        r_hmm   : log Σ_k L_k b̄_k        (marginal log-lik = negative surprise)
    """
    A_ctx = cfg.A_ctx
    V = cfg.V
    b_prev = cfg.b0.copy()

    n = len(seq)
    if n < 2:
        z = np.zeros((0, 2))
        return FilterTrace(z, z.copy(), np.zeros(0), np.zeros(0))

    b_prior = np.empty((n - 1, 2))
    b_post = np.empty((n - 1, 2))
    S = np.empty(n - 1)
    r_hmm = np.empty(n - 1)

    for i in range(n - 1):
        a, b = int(seq[i]), int(seq[i + 1])
        b_bar = A_ctx.T @ b_prev
        L = np.array([M_EE[a, b], M_SC[a, b]], dtype=np.float64)
        unnorm = b_bar * L
        Z = unnorm.sum()
        if Z <= 0.0:
            post = b_bar.copy()           # transition impossible under both: stay put
            Z = max(Z, 1e-300)
        else:
            post = unnorm / Z
        b_prior[i] = b_bar
        b_post[i] = post
        S[i] = float(V @ (post - b_bar))
        r_hmm[i] = float(np.log(Z))
        b_prev = post

    return FilterTrace(b_prior, b_post, S, r_hmm)


# ---------------------------------------------------------------------------
# Brielmann-Dayan system-state updates (belief-weighted, per context)
# ---------------------------------------------------------------------------

def bd_updates(seq: np.ndarray, b_post: np.ndarray, cfg: ValidatedConfig,
               n_contexts: int = 2) -> np.ndarray:
    """Per-step ΔV(t) from the B-D system-state value, weighted by context belief.

    State per context k: X_k = N(mu_k, sigma2). Observation o at transition i is
    the landing tone's log2-frequency. Update (after the filter posterior b)::

        mu_k <- mu_k + alpha * b_post_k * (o - mu_k)

    Value::

        V(t) = Σ_k b_post_k * ( -KL( N(mu_k, sigma2) ‖ p_T ) )
        ΔV(t) = V(t) - V(t-1)

    At n_contexts == 1 this reduces to the plain B-D single-state update
    (asserted in tests). ΔV averaged over a steady-state block is expected to be
    small — within-context learning plateaus — which is fine; the cross-arm
    signal then lives mainly in r (complexity) and S (grammar).
    """
    n = len(seq)
    if n < 2:
        return np.zeros(0)

    logf = cfg.tone_logfreq
    mu = np.full(n_contexts, cfg.p_T_mu, dtype=np.float64)   # start at target mean
    sig2, mu_T, sig2_T = cfg.sigma2, cfg.p_T_mu, cfg.p_T_sigma2

    def _value(mu_vec: np.ndarray, w: np.ndarray) -> float:
        # -Σ_k w_k KL( N(mu_k, sig2) ‖ N(mu_T, sig2_T) )   (1-D Gaussians)
        kl = (np.log(np.sqrt(sig2_T / sig2))
              + (sig2 + (mu_vec - mu_T) ** 2) / (2.0 * sig2_T) - 0.5)
        return float(-np.sum(w * kl))

    dV = np.empty(n - 1)
    if n_contexts == 1:
        w_seq = np.ones((n - 1, 1))
    else:
        w_seq = b_post

    V_prev = _value(mu, w_seq[0])
    for i in range(n - 1):
        o = logf[int(seq[i + 1])]
        w = w_seq[i]
        mu = mu + cfg.alpha * w * (o - mu)
        V_now = _value(mu, w)
        dV[i] = V_now - V_prev
        V_prev = V_now
    return dV


# ---------------------------------------------------------------------------
# Per-arm-block feature summaries
# ---------------------------------------------------------------------------

@dataclass
class ArmFeatures:
    r_mean: float
    S_mean: float
    dV_mean: float
    r_sum: float       # over-logging-sensitive; report-only
    S_sum: float
    dV_sum: float
    n_melodies: int
    n_transitions: int


def tier_restricted_emissions(M: np.ndarray, tier: str, cfg: ValidatedConfig,
                              eps: float = 1e-3) -> np.ndarray:
    """Restrict a grammar matrix to its tier columns and renormalise (with eps
    smoothing on out-of-tier transitions).

    These emissions MATCH the generative process — the stimuli were drawn from
    exactly these tier-restricted distributions. Scoring transitions this way
    fixes the secondary-tier S sign-flip that the full matrix produces (a
    secondary melody's i->i+3 / i->i+1 steps overlap the OTHER grammar's dominant
    step, so under the full matrix a secondary melody looks like the wrong
    grammar). Under tier-restricted emissions a melody is correctly most likely
    under its own grammar's tier distribution.
    """
    prob = cfg.complexity_tiers[tier]
    R = np.full_like(M, eps)
    mask = np.abs(M - prob) < 1e-3
    R[mask] = M[mask]
    return R / R.sum(axis=1, keepdims=True)


def arm_block_features(melodies: Sequence[str], group: int,
                       cfg: ValidatedConfig, tier: Optional[str] = None,
                       emission: str = "full") -> ArmFeatures:
    """Summarise a grammar arm-block's delivered melodies into scalar features.

    `melodies` is the list of logged melody strings for this (mouse, block, ROI).
    Each melody is filtered independently from b0; per-melody means are averaged
    across melodies (the per-cell constant), and per-melody sums are accumulated
    (over-logging-sensitive).

    `emission`: "full" scores transitions under the full learned grammar matrices;
    "tier_restricted" scores them under the tier-restricted generative
    distributions (requires `tier`) — this fixes the secondary-tier S sign-flip.
    """
    M_EE, M_SC = cfg.emission_matrices(group)
    if emission == "tier_restricted" and tier in cfg.complexity_tiers:
        M_EE = tier_restricted_emissions(M_EE, tier, cfg)
        M_SC = tier_restricted_emissions(M_SC, tier, cfg)
    sym2idx = symbol_to_index(cfg)

    r_means: List[float] = []
    S_means: List[float] = []
    dV_means: List[float] = []
    r_total = S_total = dV_total = 0.0
    n_trans = 0

    for mel in melodies:
        seq = parse_symbols(mel, sym2idx)
        tr = hmm_filter(seq, M_EE, M_SC, cfg)
        dV = bd_updates(seq, tr.b_post, cfg, n_contexts=2)
        if tr.S.size == 0:
            continue
        r_means.append(float(np.mean(tr.r_hmm)))
        S_means.append(float(np.mean(tr.S)))
        dV_means.append(float(np.mean(dV)) if dV.size else 0.0)
        r_total += float(np.sum(tr.r_hmm))
        S_total += float(np.sum(tr.S))
        dV_total += float(np.sum(dV))
        n_trans += tr.S.size

    if not r_means:
        return ArmFeatures(np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0, 0)

    return ArmFeatures(
        r_mean=float(np.mean(r_means)),
        S_mean=float(np.mean(S_means)),
        dV_mean=float(np.mean(dV_means)),
        r_sum=r_total, S_sum=S_total, dV_sum=dV_total,
        n_melodies=len(r_means), n_transitions=n_trans,
    )


_CELL_POOL_CAP = 3000   # max melodies pooled per (group, environment, tier) cell


def build_features(df, cfg: ValidatedConfig, pool_cap: int = _CELL_POOL_CAP,
                   emission: str = "full"):
    """Attach CANONICAL (group, environment, tier) cell features to every arm.

    The latent regressors are properties of the *stimulus*: within a
    (group, environment, tier) cell every melody is an i.i.d. draw from the same
    restricted grammar. So we pool the logged melodies across all arm-blocks of a
    cell, compute the cell's mean r/S/dV ONCE, and assign them to every arm-block
    in that cell — including arms the mouse never visited that block (which log no
    melodies). This (a) removes the dependence on the known melody over-logging
    and on whether an arm happened to be visited, and (b) uses only delivered
    sequences, never a regenerated one. `n_melodies_logged` keeps per-arm
    provenance; `n_cell_melodies` records the pool size behind each cell estimate.

    Non-grammar arms get NaN features (silent -> A = w0 in the link; vocalisation
    is held out of the fit). Returns a new dataframe; does not mutate the input.
    """
    import pandas as pd  # local import keeps the numpy core import-light

    out = df.copy()
    gram = out[out["arm_type"] == "grammar"]

    # 1. pool melodies per (group, environment, tier) cell; compute features once
    cell_feat: Dict[tuple, ArmFeatures] = {}
    for key, sub in gram.groupby(["group", "environment", "tier"]):
        grp = key[0]
        pooled: List[str] = [m for mels in sub["melodies"] for m in mels]
        if grp not in (1, 2) or not pooled:
            cell_feat[tuple(key)] = None
            continue
        if len(pooled) > pool_cap:                # unbiased deterministic subsample
            idx = np.linspace(0, len(pooled) - 1, pool_cap).astype(int)
            pooled = [pooled[i] for i in idx]
        tier = key[2]
        cell_feat[tuple(key)] = arm_block_features(pooled, int(grp), cfg,
                                                   tier=tier, emission=emission)

    # 2. assign cell features to each row
    cols = ["r_mean", "S_mean", "dV_mean", "n_melodies_logged", "n_cell_melodies"]
    vals = {c: [] for c in cols}
    for _, row in out.iterrows():
        f = cell_feat.get((row["group"], row["environment"], row["tier"])) \
            if row.get("arm_type") == "grammar" else None
        logged = len(row.get("melodies") or []) if row.get("arm_type") == "grammar" else 0
        if f is not None:
            vals["r_mean"].append(f.r_mean)
            vals["S_mean"].append(f.S_mean)
            vals["dV_mean"].append(f.dV_mean)
            vals["n_cell_melodies"].append(f.n_melodies)
        else:
            vals["r_mean"].append(np.nan)
            vals["S_mean"].append(np.nan)
            vals["dV_mean"].append(np.nan)
            vals["n_cell_melodies"].append(0)
        vals["n_melodies_logged"].append(logged)
    for c in cols:
        out[c] = vals[c]
    return out


# ---------------------------------------------------------------------------
# Collinearity / variance diagnostics
# ---------------------------------------------------------------------------

def feature_diagnostics(df) -> Dict[str, object]:
    """Between-arm variance, correlation matrix, and VIF for r_mean/dV_mean/S_mean.

    A feature with near-zero between-arm variance cannot explain preference and
    that is stated, not hidden. High ``|corr(r, S)|`` is flagged loudly: it means
    the design did not separate fluency from semantics in practice, and recovery
    will confirm whether wS is estimable at all.
    """
    import pandas as pd

    feats = ["r_mean", "dV_mean", "S_mean"]
    sub = df[df["arm_type"] == "grammar"][feats].dropna()
    out: Dict[str, object] = {}
    out["n"] = int(len(sub))
    out["variance"] = sub.var(ddof=1).to_dict()
    out["corr"] = sub.corr().to_dict()

    # Variance Inflation Factor for each feature (regress on the others).
    vif: Dict[str, float] = {}
    X = sub.values
    for j, name in enumerate(feats):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif[name] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else np.inf
        except np.linalg.LinAlgError:
            vif[name] = np.nan
    out["vif"] = vif
    out["corr_r_S"] = float(sub["r_mean"].corr(sub["S_mean"])) if len(sub) > 1 else np.nan
    out["flag_high_rS_collinearity"] = bool(
        np.isfinite(out["corr_r_S"]) and abs(out["corr_r_S"]) > 0.9
    )
    return out
