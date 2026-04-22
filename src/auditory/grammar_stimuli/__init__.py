"""Acoustic grammar stimulus generation.

Public API::

    from grammar_stimuli import (
        GrammarSessionConfig,
        MarkovSampler,
        SessionRunner,
        generate_melody,
        generate_tone,
    )
"""

from .config import (
    COMPLEXITY_TIERS,
    COUNTERBALANCE,
    GRAMMARS,
    GRAMMAR_A,
    GRAMMAR_B,
    GrammarSessionConfig,
    TEST_ARM_PLAN,
    TONES,
    TONE_SYMBOLS,
    novel_grammar_for,
    trained_grammar_for,
)
from .sequence_sampler import (
    MarkovSampler,
    SampleMeta,
    compute_entropy_rate,
    get_tier_targets,
    information_content_per_tier,
)
from .session_runner import SessionRunner, SessionSummary
from .tone_generator import (
    generate_melody,
    generate_silence,
    generate_silence_gap,
    generate_tone,
    generate_tone_unit,
)

__all__ = [
    # config
    "COMPLEXITY_TIERS",
    "COUNTERBALANCE",
    "GRAMMARS",
    "GRAMMAR_A",
    "GRAMMAR_B",
    "GrammarSessionConfig",
    "TEST_ARM_PLAN",
    "TONES",
    "TONE_SYMBOLS",
    "novel_grammar_for",
    "trained_grammar_for",
    # sampling
    "MarkovSampler",
    "SampleMeta",
    "compute_entropy_rate",
    "get_tier_targets",
    "information_content_per_tier",
    # synthesis
    "generate_melody",
    "generate_silence",
    "generate_silence_gap",
    "generate_tone",
    "generate_tone_unit",
    # runner
    "SessionRunner",
    "SessionSummary",
]
