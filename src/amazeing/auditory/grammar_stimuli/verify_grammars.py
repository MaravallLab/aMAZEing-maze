"""Standalone verification of grammar integrity and tier behaviour.

Run as::

    python -m grammar_stimuli.verify_grammars

Eleven checks (each prints ``[ok]`` on success or raises ``AssertionError``):

1.  Both matrices are 6x6.
2.  Every row of each matrix sums to 1 (+/- tolerance).
3.  Every column of each matrix sums to 1 (doubly stochastic).
4.  Each row contains exactly the tier-count pattern {1xp, 2xq, 2xr, 1xe}.
5.  The two grammars are not equal.
6.  ``dominant`` tier yields exactly one eligible column per row.
7.  ``secondary`` tier yields exactly two eligible columns per row.
8.  ``rare``      tier yields exactly two eligible columns per row.
9.  Sampling 10 000 tones from each (grammar, tier) combination reproduces
    the expected restricted distribution within 3 standard errors.
10. Entropy rate of each grammar matches the closed-form value
    (sum_i pi_i * H(row_i)) to within 1e-6 bits.
11. Training sessions (dominant tier only) produce deterministic sequences
    given a fixed start symbol.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from . import config as cfg
from .sequence_sampler import (
    MarkovSampler,
    compute_entropy_rate,
    get_tier_targets,
)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _check_1_shape() -> None:
    for name, M in cfg.GRAMMARS.items():
        assert M.shape == (cfg.N_TONES, cfg.N_TONES), f"Grammar {name} wrong shape"
    _ok("Both grammars are 6x6.")


def _check_2_rows_sum_to_one() -> None:
    for name, M in cfg.GRAMMARS.items():
        sums = M.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=cfg.PROB_TOLERANCE), (
            f"Grammar {name} rows: {sums}"
        )
    _ok("Row sums = 1 for both grammars.")


def _check_3_cols_sum_to_one() -> None:
    for name, M in cfg.GRAMMARS.items():
        sums = M.sum(axis=0)
        assert np.allclose(sums, 1.0, atol=cfg.PROB_TOLERANCE), (
            f"Grammar {name} cols: {sums}"
        )
    _ok("Column sums = 1 for both grammars (doubly stochastic).")


def _check_4_tier_counts() -> None:
    expected = Counter([cfg._p, cfg._q, cfg._q, cfg._r, cfg._r, cfg._e])
    for name, M in cfg.GRAMMARS.items():
        for i, row in enumerate(M):
            # Round to avoid float hash issues.
            vals = Counter(round(x, 6) for x in row)
            expected_r = Counter(round(x, 6) for x in expected.elements())
            assert vals == expected_r, (
                f"Grammar {name} row {i} tier counts wrong: {vals}"
            )
    _ok("Every row has {1xp, 2xq, 2xr, 1xe}.")


def _check_5_distinct() -> None:
    assert not np.allclose(cfg.GRAMMAR_A, cfg.GRAMMAR_B), "Grammars identical"
    _ok("Grammar A != Grammar B.")


def _check_6_7_8_tier_sizes() -> None:
    for tier, expected_size in [("dominant", 1), ("secondary", 2), ("rare", 2)]:
        for name, M in cfg.GRAMMARS.items():
            targets = get_tier_targets(M, tier)
            for i, cols in targets.items():
                assert cols.size == expected_size, (
                    f"Grammar {name}, tier {tier}, row {i}: got "
                    f"{cols.size} cols, expected {expected_size}"
                )
        _ok(f"Tier {tier!r}: every row has exactly {expected_size} eligible column(s).")


def _check_9_sampled_distribution() -> None:
    # For each (grammar, tier), sample many transitions starting from uniform
    # states and verify that the empirical conditional distribution matches
    # the tier-restricted renormalised row.
    n_draws = 10_000
    rng = np.random.default_rng(0)

    for name, M in cfg.GRAMMARS.items():
        for tier in cfg.COMPLEXITY_TIERS:
            sampler = MarkovSampler(name, tier, rng=rng)
            targets = get_tier_targets(M, tier)

            counts = np.zeros_like(M)
            for _ in range(n_draws):
                i = int(rng.integers(cfg.N_TONES))
                j, _bits = sampler.next_tone(i)
                counts[i, j] += 1

            for i in range(cfg.N_TONES):
                cols = targets[i]
                row_total = counts[i].sum()
                if row_total == 0:
                    continue
                emp = counts[i, cols] / row_total
                exp = np.full(cols.size, 1.0 / cols.size)
                # Binomial SE for each bin.
                se = np.sqrt(exp * (1 - exp) / row_total)
                assert np.all(np.abs(emp - exp) < 3 * se + 1e-3), (
                    f"Grammar {name} tier {tier} row {i}: emp={emp} exp={exp}"
                )
    _ok("Empirical transition frequencies match tier-restricted distribution "
        "(10 000 draws, within 3 SE).")


def _check_10_entropy_rate() -> None:
    # For a doubly stochastic matrix whose rows all have the same entropy,
    # the entropy rate equals that per-row entropy.
    row_H = -(cfg._p * math.log2(cfg._p) +
              2 * cfg._q * math.log2(cfg._q) +
              2 * cfg._r * math.log2(cfg._r) +
              cfg._e * math.log2(cfg._e))
    for name, M in cfg.GRAMMARS.items():
        H = compute_entropy_rate(M)
        assert abs(H - row_H) < 1e-6, f"Grammar {name}: H={H} vs expected {row_H}"
    _ok(f"Entropy rate = {row_H:.4f} bits/transition for both grammars.")


def _check_11_training_determinism() -> None:
    # Dominant tier has 1 eligible column per row, so given a start symbol
    # the entire melody is deterministic.
    for name in cfg.GRAMMARS:
        melodies = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            s = MarkovSampler(name, "dominant", rng=rng)
            meta = s.sample_melody(length=cfg.MELODY_LENGTH, start_symbol="A")
            melodies.append(meta.symbols)
        assert all(m == melodies[0] for m in melodies), (
            f"Grammar {name} training not deterministic: {melodies}"
        )
    _ok("Dominant-tier melodies are deterministic given a fixed start.")


def main() -> int:
    print("Verifying grammar stimuli module...\n")
    _check_1_shape()
    _check_2_rows_sum_to_one()
    _check_3_cols_sum_to_one()
    _check_4_tier_counts()
    _check_5_distinct()
    _check_6_7_8_tier_sizes()
    _check_9_sampled_distribution()
    _check_10_entropy_rate()
    _check_11_training_determinism()
    print("\nAll 11 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
