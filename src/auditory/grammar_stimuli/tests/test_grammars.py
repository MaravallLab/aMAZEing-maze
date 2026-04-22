"""pytest suite for grammar_stimuli."""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from grammar_stimuli import (
    COMPLEXITY_TIERS,
    GRAMMARS,
    GRAMMAR_A,
    GRAMMAR_B,
    GrammarSessionConfig,
    MarkovSampler,
    SessionRunner,
    TEST_ARM_PLAN,
    TONES,
    TONE_SYMBOLS,
    compute_entropy_rate,
    generate_melody,
    generate_silence_gap,
    generate_tone,
    generate_tone_unit,
    get_tier_targets,
    information_content_per_tier,
    novel_grammar_for,
    trained_grammar_for,
)
from grammar_stimuli import config as cfg
from grammar_stimuli.run import main as run_main


# ---------------------------------------------------------------------------
# Config / grammar matrices
# ---------------------------------------------------------------------------

class TestConfig:
    def test_tone_inventory_has_six_entries(self):
        assert len(TONES) == 6
        assert TONE_SYMBOLS == list(TONES.keys())

    def test_tone_frequencies_are_ascending(self):
        vals = list(TONES.values())
        assert vals == sorted(vals)

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_grammar_shape(self, name):
        assert GRAMMARS[name].shape == (6, 6)

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_rows_sum_to_one(self, name):
        assert np.allclose(GRAMMARS[name].sum(axis=1), 1.0, atol=cfg.PROB_TOLERANCE)

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_cols_sum_to_one(self, name):
        assert np.allclose(GRAMMARS[name].sum(axis=0), 1.0, atol=cfg.PROB_TOLERANCE)

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_tier_counts_per_row(self, name):
        M = GRAMMARS[name]
        expected = sorted([cfg._p, cfg._q, cfg._q, cfg._r, cfg._r, cfg._e])
        for row in M:
            assert np.allclose(sorted(row.tolist()), expected, atol=cfg.PROB_TOLERANCE)

    def test_grammars_distinct(self):
        assert not np.allclose(GRAMMAR_A, GRAMMAR_B)

    def test_probabilities_sum_exactly(self):
        assert abs(cfg._p + 2 * cfg._q + 2 * cfg._r + cfg._e - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Counterbalancing
# ---------------------------------------------------------------------------

class TestCounterbalance:
    @pytest.mark.parametrize("group,cond,expected", [
        (1, "EE", "A"), (1, "SC", "B"),
        (2, "EE", "B"), (2, "SC", "A"),
    ])
    def test_trained_mapping(self, group, cond, expected):
        assert trained_grammar_for(group, cond) == expected

    def test_novel_is_complement(self):
        for group in (1, 2):
            for cond in ("EE", "SC"):
                t = trained_grammar_for(group, cond)
                n = novel_grammar_for(group, cond)
                assert {t, n} == {"A", "B"}

    def test_invalid_group(self):
        with pytest.raises(ValueError):
            trained_grammar_for(3, "EE")

    def test_invalid_condition(self):
        with pytest.raises(ValueError):
            trained_grammar_for(1, "BAD")


# ---------------------------------------------------------------------------
# Tier targets & information content
# ---------------------------------------------------------------------------

class TestTiers:
    @pytest.mark.parametrize("name", ["A", "B"])
    def test_dominant_has_one_col_per_row(self, name):
        targets = get_tier_targets(GRAMMARS[name], "dominant")
        assert all(cols.size == 1 for cols in targets.values())

    @pytest.mark.parametrize("name,tier", [
        ("A", "secondary"), ("A", "rare"),
        ("B", "secondary"), ("B", "rare"),
    ])
    def test_two_col_tiers(self, name, tier):
        targets = get_tier_targets(GRAMMARS[name], tier)
        assert all(cols.size == 2 for cols in targets.values())

    def test_info_content_values(self):
        assert information_content_per_tier("dominant") == pytest.approx(-math.log2(0.60), abs=1e-6)
        assert information_content_per_tier("secondary") == pytest.approx(-math.log2(0.12), abs=1e-6)
        assert information_content_per_tier("rare") == pytest.approx(-math.log2(0.07), abs=1e-6)

    def test_bad_tier_raises(self):
        with pytest.raises(ValueError):
            get_tier_targets(GRAMMAR_A, "nonsense")


# ---------------------------------------------------------------------------
# Entropy rate
# ---------------------------------------------------------------------------

class TestEntropyRate:
    def test_entropy_rate_matches_closed_form(self):
        closed = -(cfg._p * math.log2(cfg._p)
                   + 2 * cfg._q * math.log2(cfg._q)
                   + 2 * cfg._r * math.log2(cfg._r)
                   + cfg._e * math.log2(cfg._e))
        for M in (GRAMMAR_A, GRAMMAR_B):
            assert compute_entropy_rate(M) == pytest.approx(closed, abs=1e-6)

    def test_grammars_have_same_entropy_rate(self):
        assert compute_entropy_rate(GRAMMAR_A) == pytest.approx(
            compute_entropy_rate(GRAMMAR_B), abs=1e-9
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class TestSampler:
    def test_dominant_deterministic_from_start(self):
        for name in ("A", "B"):
            runs = []
            for seed in range(4):
                s = MarkovSampler(name, "dominant", seed=seed)
                runs.append(s.sample_melody(length=12, start_symbol="A").symbols)
            assert all(r == runs[0] for r in runs)

    def test_melody_length(self):
        s = MarkovSampler("A", "secondary", seed=0)
        meta = s.sample_melody(length=12)
        assert len(meta.symbols) == 12
        assert len(meta.per_step_bits) == 12

    def test_empirical_distribution_secondary(self):
        # 20 000 transitions should recover the uniform-within-tier pattern.
        sampler = MarkovSampler("A", "secondary", seed=1)
        M = GRAMMAR_A
        targets = get_tier_targets(M, "secondary")
        counts = np.zeros_like(M)
        rng = np.random.default_rng(2)
        for _ in range(20_000):
            i = int(rng.integers(6))
            j, _ = sampler.next_tone(i)
            counts[i, j] += 1
        for i in range(6):
            cols = targets[i]
            row = counts[i, cols]
            if row.sum() == 0:
                continue
            emp = row / row.sum()
            # Uniform across 2 columns.
            assert np.all(np.abs(emp - 0.5) < 0.05)

    def test_rare_never_dominant(self):
        # Under rare tier, the sampler must never produce the dominant column.
        sampler = MarkovSampler("A", "rare", seed=0)
        dom_targets = get_tier_targets(GRAMMAR_A, "dominant")
        for i in range(6):
            dom_col = int(dom_targets[i][0])
            seen = set()
            for _ in range(1000):
                j, _ = sampler.next_tone(i)
                seen.add(j)
            assert dom_col not in seen

    def test_start_symbol_validated(self):
        s = MarkovSampler("A", "dominant", seed=0)
        with pytest.raises(KeyError):
            s.sample_melody(length=4, start_symbol="Z")


# ---------------------------------------------------------------------------
# Tone generator
# ---------------------------------------------------------------------------

class TestToneGenerator:
    def test_tone_length(self):
        n = int(round(cfg.TONE_DURATION_MS * 1e-3 * cfg.SAMPLE_RATE))
        assert generate_tone(8000.0).shape == (n,)

    def test_tone_dtype(self):
        assert generate_tone(8000.0).dtype == np.float32

    def test_tone_amplitude_bounded(self):
        t = generate_tone(8000.0, amplitude=0.5)
        # With ramps, peak can only be <= amplitude.
        assert np.max(np.abs(t)) <= 0.5 + 1e-6

    def test_melody_length(self):
        syms = ["A", "B", "C", "D", "E", "F"] * 2
        wave = generate_melody(syms)
        unit_n = int(round((cfg.TONE_DURATION_MS + cfg.INTER_TONE_GAP_MS) * 1e-3 * cfg.SAMPLE_RATE))
        assert wave.shape == (unit_n * 12,)

    def test_silence_gap_length(self):
        n = int(round(cfg.INTER_MELODY_GAP_MS * 1e-3 * cfg.SAMPLE_RATE))
        gap = generate_silence_gap()
        assert gap.shape == (n,)
        assert np.all(gap == 0)

    def test_tone_unit_is_tone_plus_gap(self):
        unit = generate_tone_unit("A")
        n_tone = int(round(cfg.TONE_DURATION_MS * 1e-3 * cfg.SAMPLE_RATE))
        n_gap = int(round(cfg.INTER_TONE_GAP_MS * 1e-3 * cfg.SAMPLE_RATE))
        assert unit.shape == (n_tone + n_gap,)
        assert np.all(unit[-n_gap:] == 0)

    def test_bad_symbol(self):
        with pytest.raises(KeyError):
            generate_tone_unit("Z")


# ---------------------------------------------------------------------------
# Session runner (dry run)
# ---------------------------------------------------------------------------

class TestSessionRunner:
    def test_training_dry_run_produces_log(self, tmp_path):
        c = GrammarSessionConfig(
            output_dir=str(tmp_path),
            dry_run=True, seed=0, group=1, condition="EE",
            mode="training",
            session_duration_s=44.0,  # 10 melody cycles
        )
        summary = SessionRunner(c).run_training_session()
        assert summary.mode == "training"
        assert summary.trained_grammar == "A"
        assert summary.n_melodies["training"] == 10
        path = summary.arm_logs["training"]
        assert os.path.exists(path)
        with open(path) as fh:
            lines = fh.readlines()
        # header + 10 rows
        assert len(lines) == 11

    def test_test_dry_run_all_arms(self, tmp_path):
        c = GrammarSessionConfig(
            output_dir=str(tmp_path),
            dry_run=True, seed=0, group=2, condition="SC",
            mode="test",
            session_duration_s=44.0 * len(TEST_ARM_PLAN),
        )
        summary = SessionRunner(c).run_test_session()
        assert summary.trained_grammar == "A"
        assert summary.novel_grammar == "B"
        assert set(summary.arm_logs) == {a["arm_id"] for a in TEST_ARM_PLAN}
        # Grammar arms should have >0 melodies; voc and silent 0.
        for arm in TEST_ARM_PLAN:
            n = summary.n_melodies[arm["arm_id"]]
            if arm["kind"] == "grammar":
                assert n > 0
            else:
                assert n == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_dry_run_training(self, tmp_path):
        rc = run_main([
            "--mode", "training",
            "--group", "1", "--condition", "EE",
            "--duration-seconds", "44",
            "--dry-run",
            "--output-dir", str(tmp_path),
            "--seed", "0",
        ])
        assert rc == 0
        csvs = [p for p in os.listdir(tmp_path) if p.endswith(".csv")]
        assert len(csvs) == 1
