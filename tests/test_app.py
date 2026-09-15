"""Tests for the graphical interface's non-visual logic (run headless)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class TestLauncher:

    def test_command_from_source(self, monkeypatch):
        import sys
        from amazeing.app import launcher
        monkeypatch.delattr(sys, "frozen", raising=False)
        cmd = launcher.command_for("auditory", ["--config", "x.yaml"])
        assert cmd[0] == sys.executable
        assert cmd[-4:] == ["-m", "amazeing.auditory.main", "--config", "x.yaml"]

    def test_command_when_frozen(self, monkeypatch):
        import sys
        from amazeing.app import launcher
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        cmd = launcher.command_for("summary", ["--all", "d"])
        assert cmd == [sys.executable, "--entry", "summary", "--all", "d"]

    def test_unknown_entry(self):
        from amazeing.app import launcher
        with pytest.raises(KeyError):
            launcher.command_for("nope")

    def test_every_entry_module_imports(self):
        import importlib
        from amazeing.app.launcher import ENTRY_MODULES
        for name, mod in ENTRY_MODULES.items():
            if name == "tactile":      # runs a session at import time
                continue
            importlib.import_module(mod)


class TestConfigForm:

    def test_round_trip_default(self, qapp):
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        form = ConfigForm()
        form.from_config(ExperimentConfig())
        assert form.to_config() == ExperimentConfig()

    def test_round_trip_custom(self, qapp, tmp_path):
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(
            experiment_mode="custom", rois_number=3, base_output_path=str(tmp_path),
            channel_id=5, samplerate=96000, grammar_seed=7, experiment_day="day_2",
            custom_stimuli=[
                {"roi": "1", "kind": "tone", "frequency": 12000.0, "label": "low"},
                {"roi": "2", "kind": "wav", "path": "c:/x.wav"},
                {"roi": "3", "kind": "silent"},
            ],
            custom_block_minutes=[1, 2, 1, 2, 1, 2, 1, 2, 1],
        )
        form = ConfigForm()
        form.from_config(cfg)
        back = form.to_config()
        assert back == cfg

    def test_bad_schedule_reported(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.w["experiment_mode"].setCurrentText("custom")
        form.w["custom_block_minutes"].setText("1,2,3")
        with pytest.raises(ValueError, match="9 entries"):
            form.to_config()
        form.w["custom_block_minutes"].setText("1,x")
        with pytest.raises(ValueError, match="not a number"):
            form.to_config()

    def test_fill_rows_matches_roi_count(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.w["rois_number"].setValue(4)
        form._fill_rows()
        assert [s["roi"] for s in form._stim_rows()] == ["1", "2", "3", "4"]


class TestCalibrationTab:

    def test_load_edit_save(self, qapp, tmp_path):
        import pandas as pd
        from amazeing.app.calibration_tab import CalibrationTab
        tab = CalibrationTab()
        assert tab.table.rowCount() > 2          # shipped curve loaded
        tab._add_row(70.0, -4.0)
        out = tmp_path / "speaker.csv"
        got = []
        tab.saved.connect(got.append)
        tab._save_to(str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["Frequency_kHz", "Attenuation_dB"]
        assert df["Frequency_kHz"].is_monotonic_increasing
        assert 70.0 in df["Frequency_kHz"].values
        assert got == [str(out)]


class TestMainWindow:

    def test_window_builds(self, qapp):
        from amazeing.app.main_window import MainWindow
        win = MainWindow()
        assert win.centralWidget().count() == 5
