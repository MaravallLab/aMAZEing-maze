"""Parameter recovery + model-confusion harness. RECOVERY GATES INTERPRETATION.

No fitted wS is reported or interpreted until this passes: wS must be separable
from wr and w_V given the ACTUAL delivered sequences, and model comparison must
not false-positive on baseline-generated data.

Everything here uses the real per-arm features (from latent_regressors) and a
fast NumPy/scipy estimator so the loop runs thousands of times. This is the
frequentist sibling of the Phase 2 PyMC/LOO comparison; if it fails, the correct
output of the whole pipeline is "the design cannot separate semantic value from
fluency" — a real, reportable result, not a bug.

The link is SOFTMAX here (weights are identified; the matching law is invariant
to a positive rescaling of all weights, so it identifies only sign and ratios).
Sign is what the EE>SC claim needs; softmax additionally pins magnitude, which
makes recovery interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

# free-parameter sets per nested model
MODELS: Dict[str, Tuple[str, ...]] = {
    "intercept":    ("w0",),
    "fluency":      ("w0", "wr"),
    "bd_baseline":  ("w0", "wr", "wV"),
    "full":         ("w0", "wr", "wV", "wS"),
    "full_grammar": ("w0", "wr", "wV", "wS", "wg"),   # + intrinsic-grammar nuisance
}
_ALL_PARAMS = ("w0", "wr", "wV", "wS", "wg")


@dataclass
class Block:
    r: np.ndarray
    dV: np.ndarray
    S: np.ndarray
    g: np.ndarray          # grammar sign: +1 (A), -1 (B), 0 (silent)
    obs_frac: np.ndarray   # observed dwell fractions over the arms in this block
    mouse: str


@dataclass
class Design:
    blocks: List[Block]
    scales: Dict[str, float] = field(default_factory=dict)

    def mice(self) -> List[str]:
        return sorted({b.mouse for b in self.blocks})


# ---------------------------------------------------------------------------
# Build the design from the features dataframe (day 1, grammar + silent arms)
# ---------------------------------------------------------------------------

def build_design(feat_df) -> Design:
    """Construct the per-(mouse, block) design. Vocalisation is held out of the fit."""
    import pandas as pd  # noqa

    use = feat_df[feat_df["arm_type"].isin(["grammar", "silent"])].copy()
    gram = use[use["arm_type"] == "grammar"]
    # scale-only normalisation: divide by sd of grammar-arm features so the silent
    # arm's structural 0 stays 0 (silent -> A = w0).
    scales = {}
    for c in ("r_mean", "dV_mean", "S_mean"):
        sd = float(gram[c].std(ddof=1))
        # guard near-zero variance (e.g. a single-tier subset where r is
        # constant): a tiny SD would blow up feature/scale. Leave such a
        # feature unscaled rather than dividing by ~0.
        scales[c] = sd if sd > 1e-8 else 1.0

    blocks: List[Block] = []
    for (mouse, blk), sub in use.groupby(["mouse", "block"]):
        tot = sub["time_spent_s"].sum()
        if tot <= 0 or len(sub) < 2:
            continue
        r = np.nan_to_num(sub["r_mean"].values) / scales["r_mean"]
        dV = np.nan_to_num(sub["dV_mean"].values) / scales["dV_mean"]
        S = np.nan_to_num(sub["S_mean"].values) / scales["S_mean"]
        g = sub["grammar"].map({"A": 1.0, "B": -1.0}).fillna(0.0).values
        obs = (sub["time_spent_s"] / tot).values
        blocks.append(Block(r, dV, S, g, obs, str(mouse)))
    return Design(blocks=blocks, scales=scales)


# ---------------------------------------------------------------------------
# Model value + prediction
# ---------------------------------------------------------------------------

def _A(block: Block, w: Dict[str, float]) -> np.ndarray:
    return (w.get("w0", 0.0)
            + w.get("wr", 0.0) * block.r
            + w.get("wV", 0.0) * block.dV
            + w.get("wS", 0.0) * block.S
            + w.get("wg", 0.0) * block.g)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _pred_frac(block: Block, w: Dict[str, float]) -> np.ndarray:
    return _softmax(_A(block, w))


def _sse(blocks: Sequence[Block], w: Dict[str, float]) -> float:
    return float(sum(np.sum((b.obs_frac - _pred_frac(b, w)) ** 2) for b in blocks))


# ---------------------------------------------------------------------------
# Fast fit (SSE on dwell fractions)
# ---------------------------------------------------------------------------

def fit(blocks: Sequence[Block], model: str) -> Dict[str, float]:
    params = MODELS[model]
    x0 = np.zeros(len(params))

    def obj(x):
        w = {p: v for p, v in zip(params, x)}
        return _sse(blocks, w)

    res = minimize(obj, x0, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    return {p: float(v) for p, v in zip(params, res.x)}


def lomo_cv_sse(design: Design, model: str) -> float:
    """Leave-one-mouse-out cross-validated SSE (lower is better)."""
    mice = design.mice()
    total = 0.0
    for held in mice:
        train = [b for b in design.blocks if b.mouse != held]
        test = [b for b in design.blocks if b.mouse == held]
        if not train or not test:
            continue
        w = fit(train, model)
        total += _sse(test, w)
    return total


# ---------------------------------------------------------------------------
# Synthetic data generation (uses the real design's features)
# ---------------------------------------------------------------------------

def simulate(design: Design, w_true: Dict[str, float],
             kappa: float, rng: np.random.Generator) -> Design:
    """Generate synthetic dwell fractions from w_true with Dirichlet count noise."""
    new_blocks = []
    for b in design.blocks:
        p = _pred_frac(b, w_true)
        noisy = rng.dirichlet(np.clip(kappa * p, 1e-3, None))
        new_blocks.append(Block(b.r, b.dV, b.S, b.g, noisy, b.mouse))
    return Design(blocks=new_blocks, scales=design.scales)


# ---------------------------------------------------------------------------
# Harness 1: parameter recovery
# ---------------------------------------------------------------------------

def parameter_recovery(design: Design,
                       wS_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
                       w_base: Optional[Dict[str, float]] = None,
                       n_sim: int = 200, kappa: float = 30.0,
                       seed: int = 0) -> Dict[str, object]:
    """Recover wS over a grid (incl. wS=0) from full-model-generated data."""
    rng = np.random.default_rng(seed)
    w_base = w_base or {"w0": 0.0, "wr": 1.0, "wV": 0.5}
    rows = []
    for wS_true in wS_grid:
        w_true = {**w_base, "wS": float(wS_true)}
        recov = []
        for _ in range(n_sim):
            sim = simulate(design, w_true, kappa, rng)
            w_hat = fit(sim.blocks, "full")
            recov.append(w_hat["wS"])
        recov = np.array(recov)
        lo, hi = np.percentile(recov, [2.5, 97.5])
        rows.append({
            "wS_true": float(wS_true),
            "wS_recovered_mean": float(recov.mean()),
            "ci": [float(lo), float(hi)],
            "covers_truth": bool(lo <= wS_true <= hi),
        })
    # pass: truth covered everywhere AND wS=0 CI includes 0 (no false positive)
    null_row = next(r for r in rows if r["wS_true"] == 0.0)
    passed = (all(r["covers_truth"] for r in rows)
              and null_row["ci"][0] <= 0.0 <= null_row["ci"][1])
    return {"grid": rows, "passed": bool(passed),
            "n_sim": n_sim, "kappa": kappa,
            "false_positive_rate_at_null": _fp_rate(design, w_base, n_sim, kappa, seed)}


def _fp_rate(design, w_base, n_sim, kappa, seed) -> float:
    """Fraction of wS=0 simulations whose recovered wS 95% interval excludes 0."""
    rng = np.random.default_rng(seed + 1)
    w_true = {**w_base, "wS": 0.0}
    # per-sim we cannot get a CI cheaply; approximate FP as |recovered| beyond the
    # null spread. Use the empirical null SD and count |wS_hat| > 1.96 SD.
    recov = np.array([fit(simulate(design, w_true, kappa, rng).blocks, "full")["wS"]
                      for _ in range(n_sim)])
    sd = recov.std(ddof=1) or 1e-9
    return float(np.mean(np.abs(recov - recov.mean()) > 1.96 * sd))


# ---------------------------------------------------------------------------
# Harness 2: model confusion (incl. intrinsic-grammar generator, point 8)
# ---------------------------------------------------------------------------

def model_confusion(design: Design, n_sim: int = 50, kappa: float = 30.0,
                    seed: int = 0,
                    wS_effect: float = 1.5, wg_effect: float = 1.5) -> Dict[str, object]:
    """Does the comparison prefer the right model on each generator?

    Generators:
      * bd_baseline (wS=0): comparison must NOT prefer `full`.
      * full (wS>0):        comparison MUST prefer `full`.
      * intrinsic-grammar (wg>0, wS=0): a preference aligned with the physical
        grammar (octave-trill vs sweep), which flips EE/SC across groups. `full`
        (S only) may spuriously fit it in an unbalanced sample; `full_grammar`
        (S + grammar nuisance) should attribute it to wg with wS ~ 0.
    """
    rng = np.random.default_rng(seed)
    base = {"w0": 0.0, "wr": 1.0, "wV": 0.5}
    gens = {
        "bd_baseline": {**base, "wS": 0.0},
        "full":        {**base, "wS": wS_effect},
        "intrinsic_grammar": {**base, "wS": 0.0, "wg": wg_effect},
    }
    out: Dict[str, object] = {"n_sim": n_sim}
    for gname, w_true in gens.items():
        prefer_full = 0
        wS_full = []
        wS_fullg = []
        for _ in range(n_sim):
            sim = simulate(design, w_true, kappa, rng)
            cv_bd = lomo_cv_sse(sim, "bd_baseline")
            cv_full = lomo_cv_sse(sim, "full")
            prefer_full += int(cv_full < cv_bd)
            wS_full.append(fit(sim.blocks, "full")["wS"])
            wS_fullg.append(fit(sim.blocks, "full_grammar")["wS"])
        out[gname] = {
            "prefer_full_rate": prefer_full / n_sim,
            "wS_full_mean": float(np.mean(wS_full)),
            "wS_full_grammar_mean": float(np.mean(wS_fullg)),
        }
    # calibration summary
    out["passed"] = bool(
        out["bd_baseline"]["prefer_full_rate"] < 0.5
        and out["full"]["prefer_full_rate"] > 0.5
    )
    out["intrinsic_grammar_note"] = (
        "wS_full vs wS_full_grammar on the intrinsic-grammar generator: if "
        "wS_full is positive but wS_full_grammar ~ 0, the grammar nuisance term "
        "is REQUIRED to avoid a spurious semantic conclusion."
    )
    return out


def run_recovery(feat_df, n_sim_recovery: int = 200, n_sim_confusion: int = 50,
                 seed: int = 0) -> Dict[str, object]:
    """Build the design from features and run both harnesses."""
    design = build_design(feat_df)
    if len(design.blocks) < 4 or len(design.mice()) < 3:
        return {"error": "insufficient blocks/mice to run recovery",
                "n_blocks": len(design.blocks), "n_mice": len(design.mice())}
    recov = parameter_recovery(design, n_sim=n_sim_recovery, seed=seed)
    confusion = model_confusion(design, n_sim=n_sim_confusion, seed=seed)
    return {
        "n_blocks": len(design.blocks),
        "n_mice": len(design.mice()),
        "feature_scales": design.scales,
        "parameter_recovery": recov,
        "model_confusion": confusion,
        "gate_passed": bool(recov["passed"] and confusion["passed"]),
    }
