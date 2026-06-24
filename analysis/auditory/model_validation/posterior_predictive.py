"""Phase 2 — posterior predictive over the six-arm pattern.

From the fitted `full` model, generate predicted dwell fractions for all six
grammar arms (+ silent/vocalisation as reference) and overlay on the observed
six-arm pattern. This is where behavioural validity actually lives: can the
model reproduce the preference pattern it is meant to explain. Requires pymc +
arviz (lazy).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .recovery import Design, _A, _softmax


def predicted_pattern_from_weights(design: Design, w: Dict[str, float]) -> Dict[str, float]:
    """Mean predicted dwell fraction per arm, given a point estimate of weights.

    Usable with a posterior-mean weight dict (Phase 2) or a fast-fit dict
    (Phase 1) — keeps posterior-predictive logic independent of the sampler.
    """
    acc: Dict[str, list] = {}
    for b in design.blocks:
        frac = _softmax(_A(b, w))
        # label each arm by its grammar sign as a coarse key (EE/SC resolved upstream)
        for gi, f in zip(b.g, frac):
            key = {1.0: "grammarA", -1.0: "grammarB", 0.0: "silent"}.get(float(gi), "other")
            acc.setdefault(key, []).append(float(f))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def posterior_predictive(idata, design: Design) -> Dict[str, object]:
    """Posterior-predictive six-arm pattern (requires a fitted PyMC idata)."""
    post = idata.posterior
    w = {p: float(post[p].values.mean()) for p in ("w0", "wr", "wV", "wS")
         if p in post}
    return {"predicted_pattern": predicted_pattern_from_weights(design, w),
            "weights_posterior_mean": w}
