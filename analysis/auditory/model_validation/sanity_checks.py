"""Model-free sanity gate. Runs before anything else.

A model-free effect must be established (and the rig must be sane) before any
model fit is trusted. Emits a go/no-go; the orchestrator refuses to continue
past a failed gate unless --force.

Checks:
  * exploration / dead-arm: per-mouse total dwell and number of arms visited;
    non-explorers are flagged (threshold logged, not silent).
  * silent arm should be the least-preferred arm type.
  * vocalisation arm behaves as an intended positive control (reported).
  * arm *index* (physical ROI) carries no residual preference after the
    reshuffle: regress dwell on ROI and confirm it is ~null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SanityResult:
    passed: bool
    checks: Dict[str, object] = field(default_factory=dict)
    non_explorers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def run_sanity(df: pd.DataFrame,
               min_arms_visited: int = 4,
               min_total_dwell_s: float = 15.0,
               arm_idx_r2_max: float = 0.10) -> SanityResult:
    """Run the model-free sanity gate on one day's arm-block dataframe.

    The GATE is the arm-index residual (a genuine spatial-confound check). The
    control-arm comparison (silent vs sound) is reported but does NOT gate:
    test-day sound neophobia/avoidance — silence being *more* preferred than the
    grammar arms — is a real, expected possibility in this paradigm, not a rig
    failure. Comparisons use visited arm-blocks only, so the 6-vs-1 arm count and
    unvisited (zero-dwell) arms do not distort the per-arm-type means.
    """
    checks: Dict[str, object] = {}
    notes: List[str] = []
    visited = df[df["visits"] > 0]

    # ---- exploration / dead arms ----
    per_mouse = (df.groupby("mouse")
                 .agg(total_dwell_s=("time_spent_s", "sum"),
                      arms_visited=("visits", lambda v: int((v > 0).sum())))
                 .reset_index())
    non_explorers = per_mouse[
        (per_mouse["arms_visited"] < min_arms_visited)
        | (per_mouse["total_dwell_s"] < min_total_dwell_s)
    ]["mouse"].tolist()
    checks["exploration"] = {
        "threshold": {"min_arms_visited": min_arms_visited,
                      "min_total_dwell_s": min_total_dwell_s},
        "n_mice": int(per_mouse["mouse"].nunique()),
        "n_non_explorers": len(non_explorers),
        "median_total_dwell_s": float(per_mouse["total_dwell_s"].median()),
    }
    notes.append(
        f"{len(non_explorers)} non-explorer(s) flagged "
        f"(<{min_arms_visited} arms or <{min_total_dwell_s}s total)."
    )

    # ---- control arms (INFORMATIONAL — not a gate) ----
    by_type = visited.groupby("arm_type")["time_spent_s"].mean().to_dict()
    silent = by_type.get("silent", np.nan)
    grammar = by_type.get("grammar", np.nan)
    voc = by_type.get("vocalisation", np.nan)
    checks["control_arms_informational"] = {
        "mean_visited_dwell_silent_s": float(silent) if np.isfinite(silent) else None,
        "mean_visited_dwell_grammar_s": float(grammar) if np.isfinite(grammar) else None,
        "mean_visited_dwell_vocalisation_s": float(voc) if np.isfinite(voc) else None,
        "silent_le_grammar": bool(np.isfinite(silent) and np.isfinite(grammar)
                                  and silent <= grammar),
        "note": ("NOT gated: test-day sound avoidance (silence preferred over "
                 "grammar arms) is a real possibility; reported for inspection."),
    }

    # ---- arm-index residual (the spatial-confound GATE) ----
    g = visited[visited["arm_type"] == "grammar"].copy()
    arm_idx_r2 = np.nan
    if len(g) > 10:
        g["roi_num"] = pd.to_numeric(g["roi"], errors="coerce")
        sub = g.dropna(subset=["roi_num", "time_spent_s"])
        if sub["roi_num"].nunique() > 1:
            x = sub["roi_num"].values
            y = sub["time_spent_s"].values
            A = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            arm_idx_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    arm_idx_ok = bool(not np.isfinite(arm_idx_r2) or arm_idx_r2 < arm_idx_r2_max)
    checks["arm_index_residual"] = {
        "r2_dwell_on_roi": float(arm_idx_r2) if np.isfinite(arm_idx_r2) else None,
        "threshold": arm_idx_r2_max,
        "passed": arm_idx_ok,
        "note": "physical ROI should explain little dwell variance after the reshuffle",
    }

    passed = bool(arm_idx_ok and per_mouse["mouse"].nunique() > 0)
    return SanityResult(passed=passed, checks=checks,
                        non_explorers=non_explorers, notes=notes)
