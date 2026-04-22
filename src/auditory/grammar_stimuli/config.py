"""Grammar stimulus configuration.

Defines the acoustic primitives (pure tones), the two first-order Markov
grammars (A and B) over those tones, the probability tiers used to produce
complexity variation, and the test-day arm assignment.

The grammar matrices themselves are never modified across sessions or
conditions: complexity is varied by sampling from restricted tiers of each
row, not by reweighting the matrix. See ``sequence_sampler.MarkovSampler``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tone inventory
# ---------------------------------------------------------------------------

# Six pure tones, spaced on a logarithmic grid spanning ~1.66 octaves
# (8 kHz -> 25.4 kHz) so that adjacent states are a perceptually comparable
# frequency ratio (2^(1/3) ~= 1.26).
TONES: Dict[str, float] = {
    "A":  8000.0,
    "B": 10079.0,
    "C": 12699.0,
    "D": 16000.0,
    "E": 20159.0,
    "F": 25398.0,
}
TONE_SYMBOLS: List[str] = list(TONES.keys())
N_TONES: int = len(TONE_SYMBOLS)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

TONE_DURATION_MS: int = 150
INTER_TONE_GAP_MS: int = 50           # silence inside a melody between tones
MELODY_LENGTH: int = 12               # tones per melody
MELODY_DURATION_MS: int = MELODY_LENGTH * (TONE_DURATION_MS + INTER_TONE_GAP_MS)  # 2400
INTER_MELODY_GAP_MS: int = 2000       # silence between successive melodies
MELODY_CYCLE_MS: int = MELODY_DURATION_MS + INTER_MELODY_GAP_MS                   # 4400
RAMP_MS: float = 5.0                  # cosine-squared on/off ramp per tone


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 44100
AMPLITUDE: float = 0.5                # peak amplitude of each pure tone in [-1, 1]
DTYPE: str = "float32"


# ---------------------------------------------------------------------------
# Session structure
# ---------------------------------------------------------------------------

SESSION_DURATION_HOURS: float = 4.0
SESSION_DURATION_S: float = SESSION_DURATION_HOURS * 3600.0
N_SOUND_ARMS: int = 7                 # test day: 6 grammar arms + vocalisation arm
# (+ 1 silent arm = 8 arms total; the silent arm plays nothing)

# Training sessions use only the dominant tier.
TRAINING_COMPLEXITY: str = "dominant"


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------
# Each row of a 6x6 transition matrix contains, in some permutation of
# columns:
#   - one "dominant" entry with probability p = 0.60
#   - two "secondary" entries with probability q = 0.12 each
#   - two "rare" entries with probability r = 0.07 each
#   - one "self" entry (diagonal) with probability e = 0.02
# The six entries sum to exactly 1.0. Rows are a permutation so every column
# also sums to 1.0, making each grammar doubly stochastic.

_p: float = 0.60
_q: float = 0.12
_r: float = 0.07
_e: float = 0.02

PROB_TOLERANCE: float = 1e-3


def _row(dom: int, sec: Tuple[int, int], rar: Tuple[int, int], self_idx: int) -> np.ndarray:
    """Build a single row of a transition matrix given tier column indices."""
    row = np.zeros(N_TONES, dtype=np.float64)
    row[dom] = _p
    row[sec[0]] = _q
    row[sec[1]] = _q
    row[rar[0]] = _r
    row[rar[1]] = _r
    row[self_idx] = _e
    return row


# -- Grammar A: ascending-cycle dominant structure --------------------------
# Dominant transition is i -> (i+1) mod 6 (cycle A->B->C->D->E->F->A).
# Secondaries are +2, +3; rares are -1, -2; self stays on i.
GRAMMAR_A: np.ndarray = np.stack([
    _row(dom=(i + 1) % N_TONES,
         sec=((i + 2) % N_TONES, (i + 3) % N_TONES),
         rar=((i - 1) % N_TONES, (i - 2) % N_TONES),
         self_idx=i)
    for i in range(N_TONES)
])

# -- Grammar B: skip-3 dominant structure -----------------------------------
# Dominant transition is i -> (i+3) mod 6 (A<->D, B<->E, C<->F).
# Secondaries +1, +2; rares -1, -2; self stays on i.
GRAMMAR_B: np.ndarray = np.stack([
    _row(dom=(i + 3) % N_TONES,
         sec=((i + 1) % N_TONES, (i + 2) % N_TONES),
         rar=((i - 1) % N_TONES, (i - 2) % N_TONES),
         self_idx=i)
    for i in range(N_TONES)
])

GRAMMARS: Dict[str, np.ndarray] = {"A": GRAMMAR_A, "B": GRAMMAR_B}


# ---------------------------------------------------------------------------
# Complexity tiers
# ---------------------------------------------------------------------------
# For a given tier, the sampler considers only the columns of a row whose
# probability matches the tier value, then renormalises within that subset.
# This preserves the grammar's structure exactly: we never touch the matrix,
# we only restrict which columns of a row are eligible on each draw.

COMPLEXITY_TIERS: Dict[str, float] = {
    "dominant":  _p,   # 0.60: one column per row -> fully predictable
    "secondary": _q,   # 0.12: two columns per row -> moderate uncertainty
    "rare":      _r,   # 0.07: two columns per row -> high uncertainty
}


# ---------------------------------------------------------------------------
# Test-day arm plan
# ---------------------------------------------------------------------------
# On test day the animal encounters 8 arms. Six grammar arms cross
# {trained grammar, novel grammar} x {dominant, secondary, rare}. One arm
# plays a conspecific vocalisation; one arm is silent.

TEST_ARM_PLAN: List[Dict[str, str]] = [
    {"arm_id": "arm1", "kind": "grammar", "grammar_role": "trained", "tier": "dominant"},
    {"arm_id": "arm2", "kind": "grammar", "grammar_role": "trained", "tier": "secondary"},
    {"arm_id": "arm3", "kind": "grammar", "grammar_role": "trained", "tier": "rare"},
    {"arm_id": "arm4", "kind": "grammar", "grammar_role": "novel",   "tier": "dominant"},
    {"arm_id": "arm5", "kind": "grammar", "grammar_role": "novel",   "tier": "secondary"},
    {"arm_id": "arm6", "kind": "grammar", "grammar_role": "novel",   "tier": "rare"},
    {"arm_id": "arm7", "kind": "vocalisation", "grammar_role": None, "tier": None},
    {"arm_id": "arm8", "kind": "silent",       "grammar_role": None, "tier": None},
]


# ---------------------------------------------------------------------------
# Counterbalancing
# ---------------------------------------------------------------------------
# Group 1: enriched environment trained on Grammar A, standard-conditions
#          controls trained on Grammar B.
# Group 2: the reverse. Grammars themselves are identical in structure; only
# the assignment to housing condition flips.

COUNTERBALANCE: Dict[int, Dict[str, str]] = {
    1: {"EE": "A", "SC": "B"},
    2: {"EE": "B", "SC": "A"},
}


def trained_grammar_for(group: int, condition: str) -> str:
    """Return 'A' or 'B' given counterbalance group and housing condition."""
    if group not in COUNTERBALANCE:
        raise ValueError(f"group must be 1 or 2, got {group!r}")
    if condition not in ("EE", "SC"):
        raise ValueError(f"condition must be 'EE' or 'SC', got {condition!r}")
    return COUNTERBALANCE[group][condition]


def novel_grammar_for(group: int, condition: str) -> str:
    trained = trained_grammar_for(group, condition)
    return "B" if trained == "A" else "A"


# ---------------------------------------------------------------------------
# Integrity assertions (evaluated at import time)
# ---------------------------------------------------------------------------

def _check_grammar(name: str, M: np.ndarray) -> None:
    assert M.shape == (N_TONES, N_TONES), f"{name} has wrong shape {M.shape}"
    # Each row sums to 1.
    row_sums = M.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=PROB_TOLERANCE), (
        f"{name} rows do not sum to 1: {row_sums}"
    )
    # Doubly stochastic: each column also sums to 1.
    col_sums = M.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=PROB_TOLERANCE), (
        f"{name} columns do not sum to 1: {col_sums}"
    )
    # Each row has exactly one p, two q, two r, one e.
    for i, row in enumerate(M):
        vals = sorted(row.tolist())
        expected = sorted([_p, _q, _q, _r, _r, _e])
        assert np.allclose(vals, expected, atol=PROB_TOLERANCE), (
            f"{name} row {i} tier counts are wrong: sorted={vals}"
        )


_check_grammar("GRAMMAR_A", GRAMMAR_A)
_check_grammar("GRAMMAR_B", GRAMMAR_B)

# Grammars must be distinct.
assert not np.allclose(GRAMMAR_A, GRAMMAR_B), "GRAMMAR_A and GRAMMAR_B are identical"

# Timing sanity.
assert MELODY_DURATION_MS == 2400
assert MELODY_CYCLE_MS == 4400


@dataclass
class GrammarSessionConfig:
    """Runtime knobs for a grammar session.

    The dataclass mirrors the style of ``code/auditory/updated_version/config.py``
    so that downstream code can consume either.
    """

    sample_rate: int = SAMPLE_RATE
    amplitude: float = AMPLITUDE
    tone_duration_ms: int = TONE_DURATION_MS
    inter_tone_gap_ms: int = INTER_TONE_GAP_MS
    melody_length: int = MELODY_LENGTH
    inter_melody_gap_ms: int = INTER_MELODY_GAP_MS
    ramp_ms: float = RAMP_MS
    session_duration_s: float = SESSION_DURATION_S

    device_id: int = 3                 # sounddevice output device
    output_dir: str = "./sessions"     # where per-session logs are written

    # Behaviour knobs
    dry_run: bool = False              # if True, do not actually play audio
    seed: int | None = None            # RNG seed for reproducibility
    group: int = 1                     # counterbalancing group (1 or 2)
    condition: str = "EE"              # 'EE' or 'SC'

    # Mode: 'training' or 'test'.
    mode: str = "training"
    # Only used when mode == 'training'. Defaults to the trained grammar for
    # (group, condition). Override to run a specific grammar.
    training_grammar: str | None = None

    def resolve_training_grammar(self) -> str:
        return self.training_grammar or trained_grammar_for(self.group, self.condition)
