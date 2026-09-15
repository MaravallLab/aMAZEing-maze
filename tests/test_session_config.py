"""Tests for the YAML session config, the custom stimulus mode and the manifest."""

import json
import os

import numpy as np
import pandas as pd
import pytest


class TestSessionConfigFile:

    def test_round_trip(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        from amazeing.auditory.session_config import load_config, save_config, SCHEMA_VERSION

        cfg = ExperimentConfig(experiment_mode="custom", rois_number=3,
                               base_output_path=str(tmp_path / "rec"),
                               custom_stimuli=[{"roi": "1", "kind": "tone", "frequency": 12000}],
                               custom_block_minutes=[1, 2, 1, 2, 1, 2, 1, 2, 1])
        path = save_config(cfg, str(tmp_path / "s.yaml"))
        assert os.path.exists(path)
        text = open(path, encoding="utf-8").read()
        assert text.startswith(f"schema_version: {SCHEMA_VERSION}")

        back = load_config(path)
        assert back == cfg
        assert back.roi_csv_path == cfg.roi_csv_path

    def test_partial_file_keeps_defaults(self, tmp_path):
        from amazeing.auditory.session_config import load_config
        p = tmp_path / "s.yaml"
        p.write_text("experiment_mode: simple_smooth\nrois_number: 4\n", encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg.experiment_mode == "simple_smooth"
        assert cfg.rois_number == 4
        assert cfg.samplerate == 192000

    def test_unknown_key_is_an_error(self, tmp_path):
        from amazeing.auditory.session_config import load_config
        p = tmp_path / "s.yaml"
        p.write_text("rois_numbre: 4\n", encoding="utf-8")
        with pytest.raises(ValueError, match="rois_numbre"):
            load_config(str(p))

    def test_wrong_schema_version_is_an_error(self, tmp_path):
        from amazeing.auditory.session_config import load_config
        p = tmp_path / "s.yaml"
        p.write_text("schema_version: 99\n", encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_config(str(p))

    def test_custom_block_minutes_validation(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="custom", custom_block_minutes=[1, 2, 3])
        with pytest.raises(ValueError):
            cfg.get_trial_lengths()
        cfg.custom_block_minutes = [0, 5, 0, 5, 0, 5, 0, 5, 0]
        assert cfg.get_trial_lengths() == [0, 5, 0, 5, 0, 5, 0, 5, 0]
        cfg.custom_block_minutes = None
        assert len(cfg.get_trial_lengths()) == 9


class TestCustomMode:

    def test_stimuli_mapped_to_rois(self, mock_audio, experiment_config, tmp_path):
        from amazeing.auditory.experiments import ExperimentFactory

        experiment_config.experiment_mode = "custom"
        experiment_config.rois_number = 4
        experiment_config.custom_stimuli = [
            {"roi": "1", "kind": "tone", "frequency": 10000, "label": "low"},
            {"roi": "2", "kind": "am_tone", "frequency": 20000, "mod_freq": 40},
            {"roi": "4", "kind": "silent"},
            # ROI 3 deliberately omitted -> silent
        ]
        df, waves = ExperimentFactory.generate_trials(experiment_config, mock_audio)

        assert set(df.columns) >= {"trial_ID", "ROIs", "frequency", "sound_type",
                                   "stimulus_label", "wave_arrays", "time_spent"}
        assert len(df) == 9 * 4
        first_active = df[df["trial_ID"] == 2].set_index("ROIs")
        assert first_active.loc["1", "frequency"] == 10000
        assert first_active.loc["1", "stimulus_label"] == "low"
        assert first_active.loc["2", "sound_type"] == "am_tone"
        assert first_active.loc["3", "sound_type"] == "silent"
        assert first_active.loc["4", "sound_type"] == "silent"
        # silent blocks are all zero
        assert (df[df["trial_ID"] == 1]["frequency"] == 0).all()
        # a tone wave actually has energy; a silent one does not
        assert np.max(np.abs(first_active.loc["1", "wave_arrays"])) > 0
        assert np.max(np.abs(first_active.loc["3", "wave_arrays"])) == 0

    def test_shuffle_keeps_stimulus_set(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "custom"
        experiment_config.rois_number = 3
        experiment_config.custom_stimuli = [
            {"roi": "1", "kind": "tone", "frequency": 8000},
            {"roi": "2", "kind": "tone", "frequency": 9000},
            {"roi": "3", "kind": "tone", "frequency": 11000},
        ]
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        for tid in (2, 4, 6, 8):
            assert sorted(df[df["trial_ID"] == tid]["frequency"]) == [8000, 9000, 11000]

    def test_bad_entries_raise(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "custom"
        experiment_config.rois_number = 2

        experiment_config.custom_stimuli = [{"roi": "9", "kind": "tone", "frequency": 1}]
        with pytest.raises(ValueError, match="ROI '9'"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

        experiment_config.custom_stimuli = [{"roi": "1", "kind": "laser"}]
        with pytest.raises(ValueError, match="Unknown custom stimulus kind"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

        experiment_config.custom_stimuli = [{"roi": "1", "kind": "wav", "path": "nope.wav"}]
        with pytest.raises(FileNotFoundError):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

        experiment_config.custom_stimuli = [{"roi": "1", "kind": "tone"}]
        with pytest.raises(ValueError, match="frequency"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    def test_stimulus_string_uses_custom_columns(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        from amazeing.auditory.data_manager import DataManager
        experiment_config.experiment_mode = "custom"
        experiment_config.rois_number = 1
        experiment_config.custom_stimuli = [{"roi": "1", "kind": "tone", "frequency": 5000}]
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        s = DataManager.get_stimulus_string(df, 2, "1")
        assert "frequency:5000" in s and "sound_type:tone" in s


class TestManifest:

    def test_manifest_written_with_units(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        from amazeing.auditory.data_manager import DataManager

        cfg = ExperimentConfig(experiment_mode="simple_smooth", base_output_path=str(tmp_path))
        files = {"trials_csv": str(tmp_path / "trials_x.csv"), "video": None}
        path = DataManager.write_manifest(str(tmp_path), cfg, files, status="running",
                                          extra={"mouse_id": "mouse1"})
        m = json.load(open(path, encoding="utf-8"))
        assert m["status"] == "running"
        assert m["mouse_id"] == "mouse1"
        assert m["config"]["experiment_mode"] == "simple_smooth"
        assert m["files"]["trials_csv"] == "trials_x.csv"
        assert m["files"]["video"] is None
        assert m["columns"]["trials_csv"]["time_spent"].startswith("seconds")
        assert m["columns"]["detailed_visits_csv"]["time_spent_seconds"] == "seconds"

        DataManager.write_manifest(str(tmp_path), cfg, files, status="completed")
        assert json.load(open(path, encoding="utf-8"))["status"] == "completed"
