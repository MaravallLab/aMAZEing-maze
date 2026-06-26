"""Phase 2 — out-of-sample model comparison (PyMC LOO + leave-one-mouse-out).

The claim is supported iff `full` beats `bd_baseline` out of sample AND the wS
posterior is reliably positive with the EE>SC sign. In-sample likelihood does
not test the claim. Requires pymc + arviz (lazy). The frequentist
leave-one-mouse-out CV (recovery.lomo_cv_sse) is available without pymc and is
reported by the Phase-1 recovery harness already.
"""

from __future__ import annotations

from typing import Dict, Sequence

from .recovery import Design, lomo_cv_sse
from . import models as _models

_DEFAULT = ("intercept", "fluency", "bd_baseline", "full")


def leave_one_mouse_out(design: Design,
                        model_names: Sequence[str] = _DEFAULT) -> Dict[str, float]:
    """Frequentist out-of-sample CV (no pymc): held-out-mouse SSE per model."""
    return {m: lomo_cv_sse(design, m) for m in model_names}


def compare_loo(design: Design, model_names: Sequence[str] = _DEFAULT, **fit_kw):
    """Bayesian LOO comparison via ArviZ (requires pymc + arviz)."""
    pm, az = _models.require_pymc()
    idatas = {m: _models.fit_model(design, m, **fit_kw) for m in model_names}
    table = az.compare(idatas, ic="loo")
    summary = {}
    if "full" in idatas:
        full_id = idatas["full"]
        post = full_id.posterior
        if "wS" in post:
            wS = post["wS"].values.reshape(-1)
            summary["wS_mean"] = float(wS.mean())
            summary["wS_hdi95"] = [float(x) for x in az.hdi(wS, hdi_prob=0.95)]
            summary["wS_p_positive"] = float((wS > 0).mean())
        # convergence diagnostics over the weights (the inference targets)
        try:
            ss = az.summary(full_id, var_names=["w0", "wr", "wV", "wS", "sd_mouse", "kappa"])
            if "wS" in ss.index:
                summary["wS_rhat"] = float(ss.loc["wS", "r_hat"])
                summary["wS_ess_bulk"] = float(ss.loc["wS", "ess_bulk"])
            summary["max_rhat_weights"] = float(ss["r_hat"].max())
            summary["min_ess_bulk_weights"] = float(ss["ess_bulk"].min())
        except Exception as e:  # pragma: no cover
            summary["diag_error"] = str(e)
        try:
            summary["n_divergences_full"] = int(full_id.sample_stats["diverging"].values.sum())
        except Exception:
            pass
    return {"compare_table": table, "wS_summary": summary, "idata": idatas}
