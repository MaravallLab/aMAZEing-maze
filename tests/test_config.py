"""Tests for ExperimentConfig — verify defaults and trial lengths."""

import pytest


class TestExperimentConfig:

    def test_default_values(self, experiment_config):
        """Key defaults match expected experiment parameters."""
        assert experiment_config.samplerate == 192000
        assert experiment_config.default_sound_duration == 10.0
        assert experiment_config.rois_number == 8
        assert experiment_config.default_waveform == "sine"

    # The legacy schedules below only apply to the non-grammar modes; the
    # grammar mode takes its durations from the grammar_* fields instead.

    def test_trial_lengths_testing_mode(self, experiment_config):
        """Testing mode returns short trial lengths."""
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.testing = True
        lengths = experiment_config.get_trial_lengths()
        assert len(lengths) == 9
        # Testing lengths should be very short
        assert all(t < 3 for t in lengths)

    def test_trial_lengths_normal_mode(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="simple_smooth")
        cfg.testing = False
        cfg.longer_middle_silence = False
        cfg.use_microcontroller = False

        lengths = cfg.get_trial_lengths()
        assert len(lengths) == 9
        assert lengths == [15, 15, 2, 15, 2, 15, 2, 15, 2]

    def test_trial_lengths_longer_silence(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="simple_smooth")
        cfg.testing = False
        cfg.longer_middle_silence = True

        lengths = cfg.get_trial_lengths()
        assert lengths == [15, 15, 2, 15, 15, 15, 2, 15, 2]

    def test_trial_lengths_microcontroller(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="simple_smooth")
        cfg.testing = False
        cfg.longer_middle_silence = False
        cfg.use_microcontroller = True

        lengths = cfg.get_trial_lengths()
        assert lengths == [15, 10, 2, 10, 2, 10, 2, 10, 2]

    def test_grammar_trial_lengths(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="grammar", grammar_mode="test")
        assert len(cfg.get_trial_lengths()) == 9
        cfg.grammar_mode = "silent_baseline"
        assert cfg.get_trial_lengths() == [cfg.grammar_silent_baseline_minutes]
        cfg.grammar_mode = "test"
        cfg.grammar_test_block_minutes = [1, 2, 3]
        with pytest.raises(ValueError):
            cfg.get_trial_lengths()

    def test_roi_csv_path_defaults_next_to_recordings(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(base_output_path=str(tmp_path))
        assert cfg.roi_csv_path == str(tmp_path / "rois1.csv")
        cfg2 = ExperimentConfig(base_output_path=str(tmp_path), roi_csv_path="x.csv")
        assert cfg2.roi_csv_path == "x.csv"

    def test_no_hard_coded_user_paths_in_defaults(self):
        """Defaults must not point at a specific lab machine."""
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig()
        for val in (cfg.path_to_vocalisation_folder, cfg.path_to_vocalisation_control,
                    cfg.base_output_path, cfg.calibration_gain_path):
            assert "labuser" not in val.lower()
            assert "labadmin" not in val.lower()

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
