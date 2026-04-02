"""Tests for ExperimentFactory — trial generation for all experiment modes."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ── Expected DataFrame columns per experiment type ─────────────────
COMMON_TRACKING_COLS = {"time_spent", "visitation_count", "time_in_maze_ms",
                        "trial_start_time", "end_trial_time"}
COMMON_TRIAL_COLS = {"trial_ID", "ROIs", "wave_arrays"}


class TestIntervalLookup:
    """Test the _get_interval helper."""

    def test_known_intervals(self):
        from experiments import ExperimentFactory

        val, s = ExperimentFactory._get_interval("unison")
        assert val == pytest.approx(1.0)
        assert s == "1/1"

        val, s = ExperimentFactory._get_interval("octave")
        assert val == pytest.approx(2.0)
        assert s == "2/1"

        val, s = ExperimentFactory._get_interval("perf_5")
        assert val == pytest.approx(3/2)
        assert s == "3/2"

    def test_unknown_interval_raises(self):
        from experiments import ExperimentFactory
        with pytest.raises(KeyError):
            ExperimentFactory._get_interval("nonexistent")


class TestSimpleSmoothTrials:
    """Test _create_simple_trials_logic."""

    def test_output_shape(self, mock_audio):
        from experiments import ExperimentFactory

        rois = [f"ROI{i+1}" for i in range(4)]
        frequencies = [10000, 12000, 14000, 16000]

        df, wave_arrays = ExperimentFactory._create_simple_trials_logic(
            rois, frequencies, mock_audio, total_repetitions=3
        )

        # 3 repetitions × 4 ROIs = 12 rows
        assert len(df) == 12
        assert len(wave_arrays) == 12

    def test_silent_trials_are_even(self, mock_audio):
        """Even-indexed trials (0, 2, 4...) should be silent (freq=0)."""
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        frequencies = [10000, 12000]

        df, _ = ExperimentFactory._create_simple_trials_logic(
            rois, frequencies, mock_audio, total_repetitions=5
        )

        # trial_ID 1, 3, 5 are silent (i=0, 2, 4)
        for trial_id in [1, 3, 5]:
            trial_rows = df[df["trial_ID"] == trial_id]
            assert all(trial_rows["frequency"] == 0), f"Trial {trial_id} should be silent"

    def test_active_trial_has_correct_frequencies(self, mock_audio):
        """First active trial (trial_ID=2) preserves original frequency order."""
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2", "ROI3"]
        frequencies = [10000, 12000, 14000]

        df, _ = ExperimentFactory._create_simple_trials_logic(
            rois, frequencies, mock_audio, total_repetitions=3
        )

        trial_2 = df[df["trial_ID"] == 2]
        assert list(trial_2["frequency"]) == frequencies

    def test_has_tracking_columns(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        df, _ = ExperimentFactory._create_simple_trials_logic(
            rois, [10000, 12000], mock_audio, total_repetitions=3
        )

        for col in COMMON_TRACKING_COLS:
            assert col in df.columns, f"Missing tracking column: {col}"

    def test_wave_arrays_are_numpy(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        df, wave_arrays = ExperimentFactory._create_simple_trials_logic(
            rois, [10000, 12000], mock_audio, total_repetitions=3
        )

        for arr in wave_arrays:
            assert isinstance(arr, np.ndarray)


class TestTEMTrials:
    """Test _create_tem_trials_logic."""

    def test_output_columns(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        frequencies = [10000, 20000]
        temporal_modulation = ["none", 50]
        sound_type = ["smooth", "rough"]
        sounds_arrays = [
            mock_audio.generate_sound_data(10000),
            mock_audio.generate_simple_tem_sound_data(20000, modulated_frequency=50),
        ]

        df, _ = ExperimentFactory._create_tem_trials_logic(
            rois, frequencies, temporal_modulation, sound_type, sounds_arrays,
            mock_audio, total_repetitions=3
        )

        expected_cols = {"trial_ID", "ROIs", "frequency", "sound_type",
                         "temporal_modulation", "wave_arrays"} | COMMON_TRACKING_COLS
        assert set(df.columns) == expected_cols

    def test_silent_trials_marked(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1"]
        df, _ = ExperimentFactory._create_tem_trials_logic(
            rois, [10000], ["none"], ["smooth"],
            [mock_audio.generate_sound_data(10000)],
            mock_audio, total_repetitions=3
        )

        silent_rows = df[df["trial_ID"] == 1]
        assert all(silent_rows["sound_type"] == "silent_trial")


class TestComplexIntervalsTrials:
    """Test _create_complex_intervals_trials_logic."""

    def test_output_columns(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        s1 = mock_audio.generate_sound_data(15000)
        s2 = mock_audio.generate_sound_data(22500)

        df, _ = ExperimentFactory._create_complex_intervals_trials_logic(
            rois,
            frequency=[[15000, 22500], [15000, 20000]],
            interval_numerical_list=["3/2", "4/3"],
            interval_string_names=["perf_5", "perf_4"],
            sound_type=["consonant", "consonant"],
            sounds_arrays=[[s1, s2], [s1, s1]],
            audio=mock_audio,
            total_repetitions=3,
        )

        expected_cols = {"trial_ID", "ROIs", "frequency", "interval_type",
                         "interval_ratio", "interval_name", "wave_arrays"} | COMMON_TRACKING_COLS
        assert set(df.columns) == expected_cols

    def test_silent_trials_use_tuple(self, mock_audio):
        """Silent trials store (0, 0) as wave_arrays."""
        from experiments import ExperimentFactory

        rois = ["ROI1"]
        s = mock_audio.generate_sound_data(15000)

        df, _ = ExperimentFactory._create_complex_intervals_trials_logic(
            rois, [[15000, 22500]], ["3/2"], ["perf_5"],
            ["consonant"], [[s, s]], mock_audio, total_repetitions=3,
        )

        silent_rows = df[df["trial_ID"] == 1]
        for wa in silent_rows["wave_arrays"]:
            assert wa == (0, 0)


class TestSequenceTrials:
    """Test _create_sequence_trials_logic."""

    def test_sequence_output_shape(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2"]
        # Two patterns: one normal, one silence
        sequence_of_frequencies = [
            [10000, 12000, 10000, 12000],  # ABAB-like
            [0] * 4,                        # silence
        ]
        patterns = ["ABAB", "silence"]

        df, wave_arrays = ExperimentFactory._create_sequence_trials_logic(
            rois, sequence_of_frequencies, patterns, mock_audio,
            total_repetitions=3,
        )

        assert len(df) == 6  # 3 reps × 2 ROIs
        assert "pattern" in df.columns

    def test_vocalisation_sentinel(self, mock_audio):
        from experiments import ExperimentFactory

        rois = ["ROI1"]
        df, _ = ExperimentFactory._create_sequence_trials_logic(
            rois, ["vocalisation"], ["vocalisation"], mock_audio,
            path_to_voc=None, total_repetitions=3,
        )

        # Active trial should have "vocalisation" as pattern
        active = df[df["trial_ID"] == 2]
        assert active["pattern"].values[0] == "vocalisation"


class TestGenerateTrialsRouter:
    """Test that generate_trials dispatches correctly."""

    def test_unknown_mode_raises(self, mock_audio, experiment_config):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "nonexistent_mode"
        with pytest.raises(ValueError, match="Unknown experiment mode"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    def test_simple_smooth_mode(self, mock_audio, experiment_config):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 4

        df, wave_arrays = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        assert isinstance(df, pd.DataFrame)
        assert "trial_ID" in df.columns
        assert "frequency" in df.columns
        assert len(wave_arrays) > 0


class TestInfoHelpers:
    """Test the _get_info_*_hard_coded helpers."""

    def test_get_info_intervals_hard_coded(self):
        from experiments import ExperimentFactory

        rois = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6", "ROI7", "ROI8"]
        tonal_centre = 10000
        intervals_list = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]

        freqs, nums, names = ExperimentFactory._get_info_intervals_hard_coded(
            rois, tonal_centre, intervals_list
        )

        # Should have: 1 unison + 6 intervals + 1 silent = 8
        assert len(freqs) == 8
        assert names[0] == "unison"
        assert names[-1] == "no_interval"
        assert freqs[-1] == [0, 0]

    def test_get_info_tem_hard_coded(self, mock_audio, tmp_path):
        from experiments import ExperimentFactory

        # Create a dummy WAV path (load_wav will return silence for missing files)
        dummy_voc = str(tmp_path / "dummy.wav")

        freqs, mods, types, arrays = ExperimentFactory._get_info_tem_hard_coded(
            rois_number=8,
            controls=["vocalisation", "silent"],
            smooth_freqs=[10000, 20000],
            constant_rough_freqs=[10000, 20000],
            complex_rough_freqs=[10000, 20000],
            audio=mock_audio,
            path_to_voc=dummy_voc,
        )

        assert len(freqs) == 8
        assert types[0] == "control"
        assert types[1] == "control"
        assert "smooth" in types
        assert "rough" in types
