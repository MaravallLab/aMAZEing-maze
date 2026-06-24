"""Frozen structural configuration for the model-validation pipeline.

HARD COMMITMENTS (baked in — not options):
  * Structural parameters are FIXED from the experimental design, never fitted:
    the two grammar bigram matrices, the context-level transition matrix A_ctx,
    the filter prior b0, the B-D learning rate alpha, and the long-run target
    p_T. Only the linear weights (w0, wr, wV, wS) and the link parameter(s) are
    free, and they live in the model/recovery modules, not here.
  * V_EE = 1.0, V_SC = 0.0. Only the product wS * V_EE is identifiable, so the
    value scale is absorbed into wS. wS is therefore the combined semantic
    weight; we do not estimate context values.

EMISSIONS ARE BIGRAM (the central correction). Both grammars are doubly
stochastic and share a uniform single-tone marginal, so a single-tone emission
carries zero information about EE vs SC and would force S(t) == 0. The hidden
context instead selects which grammar governs the tone *transition*:

    P(o_t | o_{t-1}, z = k) = M^{(k)}[o_{t-1}, o_t]

where M^{(EE)} / M^{(SC)} are the two grammar matrices the mouse learned. Which
physical grammar (A or B) is EE vs SC is per-mouse (counterbalancing group), so
the emission pair is resolved per mouse via `emission_matrices(group)`.

Structural constants are imported from the experiment's own grammar module so
there is a single source of truth; they are never re-declared here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Import the experiment's structural constants (single source of truth)
# ---------------------------------------------------------------------------
# This package lives at <repo>/analysis/auditory/model_validation/. The grammar
# module lives at <repo>/src/auditory/grammar_stimuli/. Put src/auditory on the
# path so `grammar_stimuli` is importable, then import the frozen constants.
def _add_src_auditory_to_path() -> Path:
    here = Path(__file__).resolve()
    # parents[3] == <repo> (model_validation -> auditory -> analysis -> repo)
    for cand in (here.parents[3] / "src" / "auditory",):
        if (cand / "grammar_stimuli" / "config.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise ImportError(
        "Could not locate src/auditory/grammar_stimuli relative to "
        f"{here}. Expected <repo>/src/auditory/grammar_stimuli/config.py."
    )


_SRC_AUDITORY = _add_src_auditory_to_path()

import grammar_stimuli.config as gcfg  # noqa: E402  (after sys.path setup)


# ---------------------------------------------------------------------------
# Default results location (read-only source)
# ---------------------------------------------------------------------------
# The recordings live on the Desktop by default; callers normally override with
# --results_dir. Kept here only as a convenience default.
DEFAULT_RESULTS_DIR = str(
    Path.home() / "Desktop" / "auditory_maze_experiments" / "maze_recordings" / "grammar"
)

CONTEXTS: Tuple[str, str] = ("EE", "SC")   # canonical ordering for belief / value vectors


@dataclass(frozen=True)
class ValidatedConfig:
    """Frozen structural configuration shared by every downstream module."""

    # --- tone inventory (from the experiment) ---
    tone_symbols: Tuple[str, ...]
    n_tones: int
    tone_logfreq: np.ndarray            # log2 Hz per tone index (B-D observation feature)
    melody_length: int

    # --- grammar bigram matrices (emission / transition structure) ---
    grammar_A: np.ndarray               # 6x6 row-stochastic
    grammar_B: np.ndarray
    complexity_tiers: Dict[str, float]  # dominant/secondary/rare -> matrix prob

    # --- HMM context layer ---
    contexts: Tuple[str, str]
    A_ctx: np.ndarray                   # 2x2 context transition (stable within test)
    b0: np.ndarray                      # prior over contexts, default uniform
    V: np.ndarray                       # value per context, [V_EE, V_SC] = [1, 0]

    # --- Brielmann-Dayan system-state layer (fixed) ---
    alpha: float                        # learning rate
    p_T_mu: float                       # long-run target mean (log2 Hz)
    p_T_sigma2: float                   # long-run target variance
    sigma2: float                       # system-state variance

    # ----- per-mouse emission resolution -----
    def emission_matrices(self, group: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (M_EE, M_SC) bigram matrices for a counterbalancing group.

        group 1: EE<-A, SC<-B ; group 2: EE<-B, SC<-A. Mirrors
        grammar_stimuli.config.COUNTERBALANCE so EE/SC are consistent with the
        experiment, and S(t) is therefore group-invariant by construction.
        """
        if group == 1:
            return self.grammar_A, self.grammar_B
        if group == 2:
            return self.grammar_B, self.grammar_A
        raise ValueError(f"group must be 1 or 2, got {group!r}")

    def full_grammar(self, grammar_name: str) -> np.ndarray:
        if grammar_name == "A":
            return self.grammar_A
        if grammar_name == "B":
            return self.grammar_B
        raise ValueError(f"grammar must be 'A' or 'B', got {grammar_name!r}")


def validated_config(
    *,
    ctx_self_transition: float = 0.99,
    prior: Tuple[float, float] = (0.5, 0.5),
    alpha: float = 0.05,
    sigma2: float = 1.0,
) -> ValidatedConfig:
    """Build the frozen structural config from the experiment's constants.

    Defaults encode the design commitments; everything here is FIXED, not
    fitted. `ctx_self_transition ~ 0.99` makes the maze context stable within a
    test session (the mouse does not believe the EE/SC association flips
    moment-to-moment).
    """
    tone_symbols = tuple(gcfg.TONE_SYMBOLS)
    n_tones = int(gcfg.N_TONES)
    tone_logfreq = np.array(
        [np.log2(gcfg.TONES[s]) for s in tone_symbols], dtype=np.float64
    )

    s = float(ctx_self_transition)
    A_ctx = np.array([[s, 1.0 - s], [1.0 - s, s]], dtype=np.float64)

    b0 = np.array(prior, dtype=np.float64)
    b0 = b0 / b0.sum()

    V = np.array([1.0, 0.0], dtype=np.float64)   # [V_EE, V_SC]

    # p_T target: centred on the mean tone (log2 Hz), broad. Fixed by design.
    mu = float(np.mean(tone_logfreq))
    sig2_T = float(np.var(tone_logfreq) * 4.0 + 1e-6)

    return ValidatedConfig(
        tone_symbols=tone_symbols,
        n_tones=n_tones,
        tone_logfreq=tone_logfreq,
        melody_length=int(gcfg.MELODY_LENGTH),
        grammar_A=np.array(gcfg.GRAMMAR_A, dtype=np.float64),
        grammar_B=np.array(gcfg.GRAMMAR_B, dtype=np.float64),
        complexity_tiers=dict(gcfg.COMPLEXITY_TIERS),
        contexts=CONTEXTS,
        A_ctx=A_ctx,
        b0=b0,
        V=V,
        alpha=float(alpha),
        p_T_mu=mu,
        p_T_sigma2=sig2_T,
        sigma2=float(sigma2),
    )


def symbol_to_index(cfg: ValidatedConfig) -> Dict[str, int]:
    return {s: i for i, s in enumerate(cfg.tone_symbols)}
