"""Per-mouse semantic weight + leave-one-mouse-out stability of the group wS.

Answers whether the effect is consistent or driven by a few animals. The
leave-one-mouse-out stability of the group wS uses the fast NumPy fitter and so
runs WITHOUT pymc; the per-mouse hierarchical posterior (random slope on S) is
Phase 2 and requires pymc.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .recovery import Design, fit


def loo_mouse_wS_stability(design: Design) -> Dict[str, object]:
    """Refit the `full` model dropping each mouse; track the group wS estimate."""
    mice = design.mice()
    full_w = fit(design.blocks, "full")
    drops: List[Dict[str, object]] = []
    for held in mice:
        train = [b for b in design.blocks if b.mouse != held]
        if not train:
            continue
        w = fit(train, "full")
        drops.append({"held_out": held, "wS": float(w["wS"])})
    wS_vals = np.array([d["wS"] for d in drops]) if drops else np.array([np.nan])
    return {
        "wS_full_all_mice": float(full_w["wS"]),
        "wS_leave_one_out_mean": float(np.nanmean(wS_vals)),
        "wS_leave_one_out_range": [float(np.nanmin(wS_vals)), float(np.nanmax(wS_vals))],
        "sign_stable": bool(np.all(np.sign(wS_vals) == np.sign(full_w["wS"]))
                            if np.isfinite(full_w["wS"]) else False),
        "per_drop": drops,
    }
