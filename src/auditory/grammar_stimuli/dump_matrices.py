"""Pretty-print both grammar transition matrices for human inspection.

Writes a readable text file showing every transition with its probability
and tier label. Confirms that each row contains the expected mixture:

    1 x 0.60 (dominant)
    2 x 0.12 (secondary)
    2 x 0.07 (rare)
    1 x 0.02 (self)

Usage from ``src/auditory/``::

    python -m grammar_stimuli.dump_matrices                    # to stdout
    python -m grammar_stimuli.dump_matrices grammars.txt       # to file
    python -m grammar_stimuli.dump_matrices grammars.txt --samples 20

The optional ``--samples N`` flag also appends N sampled melodies per
grammar/tier so you can inspect actual sequences (useful to see what
'secondary' or 'rare' tiers look like, since training only uses 'dominant').
"""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from . import config as cfg
from .sequence_sampler import MarkovSampler


def _tier_label(p: float) -> str:
    if abs(p - 0.60) < cfg.PROB_TOLERANCE:
        return "DOM "
    if abs(p - 0.12) < cfg.PROB_TOLERANCE:
        return "sec "
    if abs(p - 0.07) < cfg.PROB_TOLERANCE:
        return "RARE"
    if abs(p - 0.02) < cfg.PROB_TOLERANCE:
        return "self"
    return " ?  "


def _dump_matrix(name: str, out: TextIO) -> None:
    M = cfg.GRAMMARS[name]
    out.write(f"\n{'=' * 78}\n")
    out.write(f"GRAMMAR {name}\n")
    out.write(f"{'=' * 78}\n\n")

    # Column header: tone symbols + frequencies
    # NB: keep the backslash out of the f-string expression itself -- a
    # backslash inside ``{...}`` is a SyntaxError before Python 3.12.
    header_label = "from\\to"
    out.write(f"{header_label:>10}")
    for col_symbol in cfg.TONE_SYMBOLS:
        out.write(f"{col_symbol:>11}")
    out.write("\n")
    out.write(" " * 10)
    for col_symbol in cfg.TONE_SYMBOLS:
        out.write(f"{int(cfg.TONES[col_symbol]):>9}Hz")
    out.write("\n\n")

    # Numeric rows
    for i, row_symbol in enumerate(cfg.TONE_SYMBOLS):
        out.write(f"{row_symbol:>10}")
        for j in range(cfg.N_TONES):
            out.write(f"{M[i, j]:>11.3f}")
        out.write(f"   sum={M[i].sum():.3f}\n")
    out.write("\n")

    # Tier-label rows (which entry is dom/sec/rare/self)
    out.write(f"{'tiers:':>10}\n")
    for i, row_symbol in enumerate(cfg.TONE_SYMBOLS):
        out.write(f"{row_symbol:>10}")
        for j in range(cfg.N_TONES):
            out.write(f"{_tier_label(M[i, j]):>11}")
        out.write("\n")
    out.write("\n")

    # Per-row breakdown sorted by probability
    out.write("Per-row breakdown (sorted by probability):\n")
    for i, row_symbol in enumerate(cfg.TONE_SYMBOLS):
        ranked = sorted(
            ((float(M[i, j]), cfg.TONE_SYMBOLS[j]) for j in range(cfg.N_TONES)),
            reverse=True,
        )
        out.write(f"  from {row_symbol}: ")
        out.write("  ".join(
            f"->{col} p={p:.2f} [{_tier_label(p).strip()}]" for p, col in ranked
        ))
        out.write("\n")
    out.write("\n")

    # Column sums (should also be 1.0 because doubly stochastic)
    out.write("Column sums (should all be 1.000 — doubly stochastic):  ")
    out.write("  ".join(f"{cfg.TONE_SYMBOLS[j]}={M[:, j].sum():.3f}"
                       for j in range(cfg.N_TONES)))
    out.write("\n")


def _dump_samples(name: str, n_samples: int, out: TextIO) -> None:
    import numpy as np
    out.write(f"\n{'-' * 78}\n")
    out.write(f"Sample melodies from GRAMMAR {name} (length={cfg.MELODY_LENGTH} tones)\n")
    out.write(f"{'-' * 78}\n")
    # 'all' first since it's what training uses; then the test tiers.
    tiers = [cfg.FULL_TIER] + list(cfg.COMPLEXITY_TIERS)
    for tier in tiers:
        out.write(f"\n  tier = {tier!r}"
                  f"{'  ← training uses this' if tier == cfg.TRAINING_COMPLEXITY else ''}\n")
        sampler = MarkovSampler(grammar_name=name, tier=tier, seed=0)
        for k in range(n_samples):
            meta = sampler.sample_melody()
            out.write(f"    {k:>3}: {' '.join(meta.symbols)}"
                      f"   mean_bits={meta.mean_bits:.3f}\n")

    # Empirical transition counts under 'all': confirms all four
    # probabilities (0.60 / 0.12 / 0.07 / 0.02) actually fire in training.
    out.write(f"\n  Empirical transition counts under tier='all' "
              f"(100k transitions, seed=0):\n")
    sampler = MarkovSampler(grammar_name=name, tier=cfg.FULL_TIER, seed=0)
    M = cfg.GRAMMARS[name]
    counts = np.zeros_like(M)
    cur = 0
    for _ in range(100_000):
        nxt, _ = sampler.next_tone(cur)
        counts[cur, nxt] += 1
        cur = nxt
    totals = counts.sum(axis=1, keepdims=True)
    empirical = counts / np.where(totals > 0, totals, 1)
    # Aggregate by tier label
    from collections import defaultdict
    tier_freq: Dict[str, list] = defaultdict(list)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            tier_freq[_tier_label(M[i, j]).strip()].append(empirical[i, j])
    expected = {"DOM": 0.60, "sec": 0.12, "RARE": 0.07, "self": 0.02}
    for label in ("DOM", "sec", "RARE", "self"):
        vals = tier_freq.get(label, [])
        if vals:
            out.write(f"    {label:>5}: mean empirical p = {sum(vals)/len(vals):.4f}"
                      f"   (expected {expected[label]:.2f})\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="grammar_stimuli.dump_matrices")
    p.add_argument("path", nargs="?", default=None,
                   help="Output file (omit for stdout)")
    p.add_argument("--samples", type=int, default=0,
                   help="Append N sample melodies per grammar per tier (default 0)")
    args = p.parse_args(argv)

    out: TextIO = open(args.path, "w", encoding="utf-8") if args.path else sys.stdout
    try:
        out.write("Grammar transition matrices\n")
        out.write("Tier values: DOM=0.60, sec=0.12, RARE=0.07, self=0.02\n")
        out.write(f"Tone inventory: {dict((s, int(f)) for s, f in cfg.TONES.items())}\n")
        for name in cfg.GRAMMARS:
            _dump_matrix(name, out)
        if args.samples > 0:
            for name in cfg.GRAMMARS:
                _dump_samples(name, args.samples, out)
    finally:
        if args.path:
            out.close()
            print(f"Wrote {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
