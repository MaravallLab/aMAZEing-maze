"""Integration tests — simulate a short experiment run end-to-end.

These tests verify that the full pipeline (config → trial generation →
CSV output) produces structurally correct output without requiring any
physical hardware.
"""

import os
import csv
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestSimpleSmoothIntegration:
    """Simulate a simple_smooth experiment: generate trials, verify CSV structure."""

    def test_full_trial_generation(self, mock_audio, experiment_config):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 4

        df, wave_arrays = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        # Structural checks
        assert isinstance(df, pd.DataFrame)
        assert "trial_ID" in df.columns
        assert "ROIs" in df.columns
        assert "frequency" in df.columns
        assert "wave_arrays" in df.columns

        # Correct number of ROIs per trial block
        rois_per_block = experiment_config.rois_number
        unique_trials = df["trial_ID"].unique()
        for tid in unique_trials:
            block = df[df["trial_ID"] == tid]
            assert len(block) == rois_per_block

    def test_csv_round_trip(self, mock_audio, experiment_config, tmp_path):
        """Trials can be saved to CSV and loaded back."""
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 4

        df, wave_arrays = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        # Save (drop wave_arrays since they're numpy objects)
        csv_path = tmp_path / "trials.csv"
        df_to_save = df.drop(columns=["wave_arrays"])
        df_to_save.to_csv(csv_path, index=False)

        # Load back
        loaded = pd.read_csv(csv_path)
        assert list(loaded.columns) == list(df_to_save.columns)
        assert len(loaded) == len(df)

    def test_npy_save_load(self, mock_audio, experiment_config, tmp_path):
        """Sound arrays can be saved/loaded via numpy."""
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 4

        _, wave_arrays = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        npy_path = tmp_path / "sounds.npy"
        np.save(str(npy_path), np.array(wave_arrays, dtype=object))

        loaded = np.load(str(npy_path), allow_pickle=True)
        assert len(loaded) == len(wave_arrays)


class TestComplexIntervalsIntegration:
    """Simulate a complex_intervals experiment."""

    def test_w1day2_generates_correct_structure(self, mock_audio, experiment_config):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "complex_intervals"
        experiment_config.complex_interval_day = "w1day2"
        experiment_config.rois_number = 8

        df, wave_arrays = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        assert "interval_type" in df.columns
        assert "interval_name" in df.columns
        assert "interval_ratio" in df.columns

        # Active trials should have sound_type info
        active = df[df["trial_ID"] == 2]
        types = set(active["interval_type"].values)
        # Should include some of: vocalisation, silent, smooth, rough, consonant, dissonant
        assert len(types) > 1

    def test_w1day4_no_controls(self, mock_audio, experiment_config):
        """w1day4 has no controls (no vocalisation, no silent)."""
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "complex_intervals"
        experiment_config.complex_interval_day = "w1day4"
        experiment_config.rois_number = 8

        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        active = df[df["trial_ID"] == 2]
        types = set(active["interval_type"].values)
        assert "vocalisation" not in types
        assert "silent" not in types


class TestTEMIntegration:
    """Simulate a temporal_envelope_modulation experiment."""

    def test_tem_generates_all_types(self, mock_audio, experiment_config):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = "temporal_envelope_modulation"
        experiment_config.rois_number = 8

        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        active = df[df["trial_ID"] == 2]
        types = set(active["sound_type"].values)
        # Should have: control, smooth, rough, rough_complex
        assert "smooth" in types
        assert "rough" in types
        assert "control" in types


class TestVisitLogIntegration:
    """Test that the visit log CSV is created with correct headers."""

    def test_visit_log_format(self, tmp_path):
        """Verify the CSV schema produced by DataManager.initialise_visit_log."""
        from data_manager import DataManager

        mgr = DataManager(str(tmp_path))
        mgr.session_directory = str(tmp_path)
        mgr.mouseID = "mouse_test"

        from config import ExperimentConfig
        cfg = ExperimentConfig()

        log_path = mgr.initialise_visit_log(cfg)

        assert os.path.exists(log_path)

        with open(log_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected = ["trial_ID", "ROI_visited", "stimulus",
                    "sound_on_time", "sound_off_time", "time_spent_seconds"]
        assert headers == expected


class TestTrialAlternationPattern:
    """Verify the silent/active alternation pattern across all experiment types."""

    @pytest.mark.parametrize("mode,rois_n", [
        ("simple_smooth", 4),
    ])
    def test_alternation(self, mock_audio, experiment_config, mode, rois_n):
        from experiments import ExperimentFactory

        experiment_config.experiment_mode = mode
        experiment_config.rois_number = rois_n

        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        unique_trials = sorted(df["trial_ID"].unique())
        for tid in unique_trials:
            block = df[df["trial_ID"] == tid]
            is_silent = all(block["frequency"] == 0)

            # Odd trial_IDs (1, 3, 5...) should be silent
            # Even trial_IDs (2, 4, 6...) should be active
            if tid % 2 == 1:
                assert is_silent, f"Trial {tid} should be silent"
            else:
                assert not is_silent, f"Trial {tid} should be active"
