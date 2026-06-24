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
        post = idatas["full"].posterior
        if "wS" in post:
            wS = post["wS"].values.reshape(-1)
            summary["wS_mean"] = float(wS.mean())
            summary["wS_hdi95"] = [float(x) for x in az.hdi(wS, hdi_prob=0.95)]
            summary["wS_p_positive"] = float((wS > 0).mean())
    return {"compare_table": table, "wS_summary": summary, "idata": idatas}
