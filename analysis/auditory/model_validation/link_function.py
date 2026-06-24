"""Link from per-arm aesthetic value A(t) to predicted behaviour.

A(t) is instantaneous; the data are aggregate dwell, so the link is a real
modelling commitment. Two are exposed; default is the matching law.

    A_i = w0 + wr * r_i + wV * dV_i + wS * S_i           (per arm)

  matching law : predicted dwell fraction of arm i within a block
                 ∝ A_i / Σ_j A_j   over simultaneously available arms
  softmax      : P(arm i) ∝ exp(beta * A_i)

The silent arm gets A = w0 (no tones, so r = dV = S = 0). The vocalisation arm
is held out of fitting (positive control only). Predicted dwell fractions are
converted to a predicted PI with the SAME formula as the observed EE-SC PI, so
model and data live on one scale:  PI = (EE - SC) / (EE + SC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

_EPS = 1e-9


@dataclass
class Weights:
    w0: float = 0.0
    wr: float = 0.0
    wV: float = 0.0
    wS: float = 0.0
    beta: float = 1.0      # softmax inverse-temperature (unused by matching)

    def vec(self) -> np.ndarray:
        return np.array([self.w0, self.wr, self.wV, self.wS], dtype=np.float64)


def aesthetic_value(r: np.ndarray, dV: np.ndarray, S: np.ndarray,
                    w: Weights) -> np.ndarray:
    """A_i for an array of arms. r/dV/S are 0 for the silent arm by construction."""
    r = np.nan_to_num(np.asarray(r, dtype=np.float64))
    dV = np.nan_to_num(np.asarray(dV, dtype=np.float64))
    S = np.nan_to_num(np.asarray(S, dtype=np.float64))
    return w.w0 + w.wr * r + w.wV * dV + w.wS * S


def dwell_fractions(A: np.ndarray, link: str = "matching",
                    beta: float = 1.0) -> np.ndarray:
    """Predicted dwell fractions over the arms available in one block."""
    A = np.asarray(A, dtype=np.float64)
    if link == "matching":
        pos = np.clip(A, _EPS, None)        # matching law needs non-negative value
        return pos / pos.sum()
    if link == "softmax":
        z = beta * (A - A.max())
        e = np.exp(z)
        return e / e.sum()
    raise ValueError(f"unknown link {link!r}; use 'matching' or 'softmax'")


def predicted_pi_from_fractions(frac: np.ndarray, is_ee: Sequence[bool],
                                is_sc: Sequence[bool]) -> float:
    """EE-SC preference index from predicted dwell fractions (same formula as data)."""
    frac = np.asarray(frac, dtype=np.float64)
    ee = float(frac[np.asarray(is_ee, dtype=bool)].sum())
    sc = float(frac[np.asarray(is_sc, dtype=bool)].sum())
    tot = ee + sc
    return (ee - sc) / tot if tot > 0 else np.nan


def predict_block_pi(r: np.ndarray, dV: np.ndarray, S: np.ndarray,
                     is_ee: Sequence[bool], is_sc: Sequence[bool],
                     w: Weights, link: str = "matching") -> float:
    """Convenience: A -> dwell fractions -> predicted EE-SC PI for one block."""
    A = aesthetic_value(r, dV, S, w)
    frac = dwell_fractions(A, link=link, beta=w.beta)
    return predicted_pi_from_fractions(frac, is_ee, is_sc)
