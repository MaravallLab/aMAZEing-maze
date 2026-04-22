"""Sample symbol sequences from a grammar at a given complexity tier.

Design rule (see README.md): the 6x6 transition matrix is never modified.
Complexity is controlled by restricting sampling on each transition to the
columns of the current row whose probability matches a given tier value,
then renormalising within that subset.

- ``dominant``:  1 column / row -> always the same next tone -> deterministic
- ``secondary``: 2 columns / row -> 1 bit of within-row entropy
- ``rare``:      2 columns / row -> 1 bit of within-row entropy

Because only one tier is active per melody (on training days: dominant only;
on test day: the tier assigned to the arm), the *effective* matrix used for
a given melody is a restricted, renormalised version of the stored row -
but the stored row itself is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as cfg


# ---------------------------------------------------------------------------
# Tier -> per-row target columns
# ---------------------------------------------------------------------------

def get_tier_targets(matrix: np.ndarray, tier: str) -> Dict[int, np.ndarray]:
    """Return, for each row, the column indices that match ``tier``.

    Matching is by probability value against ``config.COMPLEXITY_TIERS[tier]``
    within ``config.PROB_TOLERANCE``. The returned mapping has ``N_TONES``
    keys, each mapping to a 1-D array of column indices.
    """
    if tier not in cfg.COMPLEXITY_TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}. Valid: {list(cfg.COMPLEXITY_TIERS)}"
        )
    target = cfg.COMPLEXITY_TIERS[tier]

    out: Dict[int, np.ndarray] = {}
    for i in range(matrix.shape[0]):
        cols = np.where(np.abs(matrix[i] - target) < cfg.PROB_TOLERANCE)[0]
        if cols.size == 0:
            raise RuntimeError(
                f"Row {i} has no entries matching tier {tier!r} "
                f"(value {target}). Matrix may be malformed."
            )
        out[i] = cols
    return out


# ---------------------------------------------------------------------------
# Information content
# ---------------------------------------------------------------------------

def _row_entropy(p: np.ndarray) -> float:
    """Shannon entropy (bits) of a probability vector."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Left-eigenvector of ``matrix`` for eigenvalue 1, normalised."""
    # Power iteration; our matrices are doubly stochastic so the uniform
    # distribution is stationary, but we compute it generically for safety.
    n = matrix.shape[0]
    pi = np.ones(n) / n
    for _ in range(500):
        pi_next = pi @ matrix
        if np.allclose(pi_next, pi, atol=1e-12):
            pi = pi_next
            break
        pi = pi_next
    return pi / pi.sum()


def compute_entropy_rate(matrix: np.ndarray) -> float:
    """Entropy rate of a stationary Markov chain (bits per transition)."""
    pi = _stationary_distribution(matrix)
    return float(sum(pi[i] * _row_entropy(matrix[i]) for i in range(matrix.shape[0])))


def information_content_per_tier(tier: str) -> float:
    """Surprise (-log2 p) of the *unrestricted* tier entry.

    This is the information content an ideal observer would assign to a
    single tone drawn at this tier under the full matrix, ignoring the
    restriction trick. Useful as a per-tone difficulty score.
    """
    p = cfg.COMPLEXITY_TIERS[tier]
    return float(-np.log2(p))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class SampleMeta:
    """Metadata about a single sampled melody.

    Used for logging; consumed by :mod:`session_runner`.
    """

    grammar_name: str
    tier: str
    start_symbol: str
    symbols: List[str]
    # Per-tone surprise under the *restricted* distribution actually used.
    per_step_bits: List[float]

    @property
    def mean_bits(self) -> float:
        return float(np.mean(self.per_step_bits)) if self.per_step_bits else 0.0

    @property
    def total_bits(self) -> float:
        return float(np.sum(self.per_step_bits))


class MarkovSampler:
    """Sample tone sequences from a frozen grammar under a chosen tier.

    Parameters
    ----------
    grammar_name:
        ``'A'`` or ``'B'``; selects the matrix from :data:`config.GRAMMARS`.
    tier:
        One of ``'dominant'``, ``'secondary'``, ``'rare'``.
    rng:
        Optional ``numpy.random.Generator``. Created from ``seed`` if None.
    seed:
        Optional integer seed (only used if ``rng`` is None).
    """

    def __init__(
        self,
        grammar_name: str,
        tier: str,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> None:
        if grammar_name not in cfg.GRAMMARS:
            raise ValueError(
                f"Unknown grammar {grammar_name!r}. Valid: {list(cfg.GRAMMARS)}"
            )
        if tier not in cfg.COMPLEXITY_TIERS:
            raise ValueError(
                f"Unknown tier {tier!r}. Valid: {list(cfg.COMPLEXITY_TIERS)}"
            )

        self.grammar_name = grammar_name
        self.tier = tier
        self.matrix = cfg.GRAMMARS[grammar_name]
        self._targets = get_tier_targets(self.matrix, tier)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    # -- low-level transitions --------------------------------------------
    def next_tone(self, current_idx: int) -> Tuple[int, float]:
        """Return (next_index, surprise_bits) given current tone index."""
        cols = self._targets[current_idx]
        if cols.size == 1:
            nxt = int(cols[0])
            p_eff = 1.0
        else:
            # Renormalise within the tier subset.
            probs = self.matrix[current_idx, cols]
            probs = probs / probs.sum()
            nxt = int(self.rng.choice(cols, p=probs))
            # In practice all tier values within a row are equal (q=q, r=r),
            # so p_eff = 1/len(cols), but we compute from probs for safety.
            p_eff = float(probs[np.where(cols == nxt)[0][0]])
        bits = float(-np.log2(p_eff)) if p_eff > 0 else 0.0
        bits = 0.0 if bits == 0.0 else bits  # collapse -0.0 to 0.0
        return nxt, bits

    # -- convenience ------------------------------------------------------
    @staticmethod
    def get_state_label(index: int) -> str:
        return cfg.TONE_SYMBOLS[index]

    def sample_melody(
        self,
        length: int = cfg.MELODY_LENGTH,
        start_symbol: Optional[str] = None,
    ) -> SampleMeta:
        """Sample a melody of ``length`` tones.

        If ``start_symbol`` is None, the first tone is uniform across the
        6 tones.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")

        if start_symbol is None:
            cur = int(self.rng.integers(cfg.N_TONES))
        else:
            if start_symbol not in cfg.TONES:
                raise KeyError(f"Unknown start symbol {start_symbol!r}")
            cur = cfg.TONE_SYMBOLS.index(start_symbol)

        symbols: List[str] = [self.get_state_label(cur)]
        per_step: List[float] = [0.0]  # first tone has no transition

        for _ in range(length - 1):
            cur, bits = self.next_tone(cur)
            symbols.append(self.get_state_label(cur))
            per_step.append(bits)

        return SampleMeta(
            grammar_name=self.grammar_name,
            tier=self.tier,
            start_symbol=symbols[0],
            symbols=symbols,
            per_step_bits=per_step,
        )

    def get_information_content(self) -> float:
        """Mean surprise per transition under the restricted distribution."""
        # Each tier-restricted row has equal probabilities across its columns,
        # so surprise per step is log2(n_cols). Average across rows.
        bits = [float(np.log2(cols.size)) if cols.size > 1 else 0.0
                for cols in self._targets.values()]
        return float(np.mean(bits))
