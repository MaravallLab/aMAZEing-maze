"""Tests for DataManager — session setup, metadata, visit logging.

Note: DataManager currently has bugs (os.join.path, os.path.exist, etc.).
These tests document the EXPECTED behaviour after bugfixes. Tests that
exercise the buggy code paths are marked with xfail until Step 4 fixes them.
"""

import os
import csv
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestGetStimulusString:
    """Test the static get_stimulus_string method (this one works as-is)."""

    def _make_df(self):
        return pd.DataFrame({
            "trial_ID": [1, 1, 1],
            "ROIs": ["ROI1", "ROI2", "ROI3"],
            "frequency": [10000, 0, "vocalisation"],
            "sound_type": ["smooth", "silent_trial", "control"],
        })

    def test_known_stimulus(self):
        from data_manager import DataManager
        df = self._make_df()
        result = DataManager.get_stimulus_string(df, 1, "ROI1")
        assert "frequency:10000" in result
        assert "sound_type:smooth" in result

    def test_silent_stimulus(self):
        from data_manager import DataManager
        df = self._make_df()
        result = DataManager.get_stimulus_string(df, 1, "ROI2")
        assert "frequency:0" in result

    def test_unknown_roi_returns_unknown(self):
        from data_manager import DataManager
        df = self._make_df()
        result = DataManager.get_stimulus_string(df, 1, "ROI_NONEXISTENT")
        assert result == "Unknown_Stimulus"

    def test_empty_df_returns_unknown(self):
        from data_manager import DataManager
        df = pd.DataFrame(columns=["trial_ID", "ROIs", "frequency"])
        result = DataManager.get_stimulus_string(df, 1, "ROI1")
        assert result == "Unknown_Stimulus"


class TestLogIndividualVisit:
    """Test CSV visit logging."""

    def test_log_appends_row(self, tmp_path):
        from data_manager import DataManager

        csv_path = tmp_path / "visits.csv"
        # Create the file with headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_ID", "ROI_visited", "stimulus",
                             "sound_on_time", "sound_off_time", "time_spent_seconds"])

        DataManager.log_individual_visit(
            str(csv_path), 1, "ROI1", "frequency:10000", 100.0, 110.0, 10.0
        )

        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["trial_ID"] == 1
        assert df.iloc[0]["ROI_visited"] == "ROI1"
        assert df.iloc[0]["time_spent_seconds"] == 10.0

    def test_log_multiple_visits(self, tmp_path):
        from data_manager import DataManager

        csv_path = tmp_path / "visits.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_ID", "ROI_visited", "stimulus",
                             "sound_on_time", "sound_off_time", "time_spent_seconds"])

        for i in range(5):
            DataManager.log_individual_visit(
                str(csv_path), i, f"ROI{i}", "test", 0, 1, 1.0
            )

        df = pd.read_csv(csv_path)
        assert len(df) == 5


class TestSetupSession:
    """Test session directory creation.

    These tests verify the INTENDED behaviour. The actual code has typos
    (os.join.path, os.path.exist) that will cause crashes. Tests are marked
    xfail until Step 4 fixes them.
    """

    def test_setup_creates_directory(self, tmp_path, experiment_config):
        from data_manager import DataManager

        experiment_config.base_output_path = str(tmp_path)
        experiment_config.experiment_mode = "simple_smooth"

        mgr = DataManager(str(tmp_path))

        with patch("builtins.input", return_value="1234"):
            session_dir, mouse_id = mgr.setup_session(experiment_config)

        assert os.path.isdir(session_dir)
        assert "mouse1234" in mouse_id
