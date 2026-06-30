"""Phase 2 — nested Bayesian models (PyMC). Lazy-imported.

Four nested models over arm-block behaviour, all sharing the softmax link and a
Dirichlet observation on per-block dwell fractions, with a per-mouse random
intercept, so LOO is comparable across them::

    1. intercept     — w0 only
    2. fluency       — w0, wr
    3. bd_baseline   — w0, wr, wV            (this is K=1, wS=0: Brielmann & Dayan)
    4. full          — w0, wr, wV, wS
    (+ full_grammar  — adds the intrinsic-grammar nuisance wg; see recovery.py)

Priors are weakly-informative; wr and wV are half-normal (the Brielmann-Dayan
non-negativity constraint on fluency and learning), and the wS prior is centred at
0 (the null) so a positive posterior is evidence, not assumption.

Requires `pymc` and `arviz` (not installed in the Phase-1 environment). Import is
deferred so the rest of the package runs without them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .recovery import Design, MODELS


def require_pymc():
    try:
        import pymc as pm
        import arviz as az
        return pm, az
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Phase 2 needs pymc + arviz. Install with `pip install pymc arviz`. "
            f"(import error: {e})"
        )


def build_model(design: Design, model_name: str, *,
                kappa_prior: float = 50.0):
    """Construct a PyMC model for one nested specification.

    Vectorised: blocks sharing the modal arm count (usually 7 = 6 grammar + the
    silent arm; vocalisation is held out) are stacked into one batched Dirichlet
    over their dwell fractions, so the model compiles and samples quickly and
    each block is one pointwise LOO unit. A per-mouse random intercept is added.
    """
    from collections import Counter
    pm, _ = require_pymc()
    params = MODELS[model_name]

    lengths = [len(b.r) for b in design.blocks]
    K = Counter(lengths).most_common(1)[0][0]
    blocks = [b for b in design.blocks if len(b.r) == K]
    mice = sorted({b.mouse for b in blocks})
    midx = {m: i for i, m in enumerate(mice)}

    R = np.array([b.r for b in blocks])
    DV = np.array([b.dV for b in blocks])
    S = np.array([b.S for b in blocks])
    G = np.array([b.g for b in blocks])
    OBS = np.clip(np.array([b.obs_frac for b in blocks]), 1e-6, None)
    OBS = OBS / OBS.sum(axis=1, keepdims=True)
    m_idx = np.array([midx[b.mouse] for b in blocks])

    with pm.Model() as model:
        w0 = pm.Normal("w0", 0.0, 2.0) if "w0" in params else 0.0
        # Brielmann-Dayan sign constraints: fluency r and learning ΔV contribute
        # NON-NEGATIVELY to value (half-normal priors). wS stays null-centred, so
        # a positive wS posterior is evidence, not an assumption.
        wr = pm.HalfNormal("wr", 1.0) if "wr" in params else 0.0
        wV = pm.HalfNormal("wV", 1.0) if "wV" in params else 0.0
        wS = pm.Normal("wS", 0.0, 1.0) if "wS" in params else 0.0   # null-centred
        wg = pm.Normal("wg", 0.0, 1.0) if "wg" in params else 0.0
        sd_mouse = pm.HalfNormal("sd_mouse", 1.0)
        # non-centered random intercept (avoids the hierarchical funnel that
        # otherwise produces divergences + low ESS when sd_mouse is small)
        u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=len(mice))
        u = pm.Deterministic("u_mouse", u_raw * sd_mouse)
        kappa = pm.Gamma("kappa", mu=kappa_prior, sigma=kappa_prior)

        A = (w0 + wr * R + wV * DV + wS * S + wg * G) + u[m_idx][:, None]
        A = A - A.max(axis=1, keepdims=True)
        e = pm.math.exp(A)
        frac = e / e.sum(axis=1, keepdims=True)
        pm.Dirichlet("obs", a=kappa * frac + 1e-3, observed=OBS)
    return model


def fit_model(design: Design, model_name: str, *, draws: int = 1000,
              tune: int = 1000, chains: int = 4, cores: int = 1,
              target_accept: float = 0.9, seed: int = 0):
    # cores=1 keeps sampling in-process (robust on Windows / no C compiler).
    pm, az = require_pymc()
    with build_model(design, model_name):
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept,
                          random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": True})
    return idata
