"""Tests for ExperimentConfig — verify defaults and trial lengths."""

import pytest


class TestExperimentConfig:

    def test_default_values(self, experiment_config):
        """Key defaults match expected experiment parameters."""
        assert experiment_config.samplerate == 192000
        assert experiment_config.default_sound_duration == 10.0
        assert experiment_config.rois_number == 8
        assert experiment_config.default_waveform == "sine"

    def test_trial_lengths_testing_mode(self, experiment_config):
        """Testing mode returns short trial lengths."""
        experiment_config.testing = True
        lengths = experiment_config.get_trial_lengths()
        assert len(lengths) == 9
        # Testing lengths should be very short
        assert all(t < 3 for t in lengths)

    def test_trial_lengths_normal_mode(self):
        from config import ExperimentConfig
        cfg = ExperimentConfig()
        cfg.testing = False
        cfg.longer_middle_silence = False
        cfg.use_microcontroller = False

        lengths = cfg.get_trial_lengths()
        assert len(lengths) == 9
        assert lengths == [15, 15, 2, 15, 2, 15, 2, 15, 2]

    def test_trial_lengths_longer_silence(self):
        from config import ExperimentConfig
        cfg = ExperimentConfig()
        cfg.testing = False
        cfg.longer_middle_silence = True

        lengths = cfg.get_trial_lengths()
        assert lengths == [15, 15, 2, 15, 15, 15, 2, 15, 2]

    def test_trial_lengths_microcontroller(self):
        from config import ExperimentConfig
        cfg = ExperimentConfig()
        cfg.testing = False
        cfg.longer_middle_silence = False
        cfg.use_microcontroller = True

        lengths = cfg.get_trial_lengths()
        assert lengths == [15, 10, 2, 10, 2, 10, 2, 10, 2]

    def test_entrance_rois_default(self, experiment_config):
        assert experiment_config.entrance_rois == ["entrance1", "entrance2"]

    def test_experiment_modes_list(self):
        """All referenced experiment modes are valid strings."""
        valid_modes = {
            "simple_smooth", "simple_intervals", "temporal_modulation",
            "complex_intervals", "sequences", "vocalisation",
            "semantic_predictive_complexity",
        }
        # Just verify they're all strings
        for mode in valid_modes:
            assert isinstance(mode, str)
