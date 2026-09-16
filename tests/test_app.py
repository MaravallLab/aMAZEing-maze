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
            block_minutes=[1, 2, 1, 2, 1, 2, 1, 2, 1],
        )
        form = ConfigForm()
        form.from_config(cfg)
        back = form.to_config()
        assert back == cfg

    def test_bad_schedule_reported(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.set_mode("custom")
        form.w["block_minutes"].setText("1,2,3")
        with pytest.raises(ValueError, match="9 entries"):
            form.to_config()
        form.w["block_minutes"].setText("1,x")
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


class TestModePanels:
    """Only the selected mode's parameters are shown, and they are editable."""

    def test_only_the_selected_mode_panel_is_visible(self, qapp):
        from amazeing.app.config_form import MODE_BOX, ConfigForm
        form = ConfigForm()
        form.show()
        for mode, box_key in MODE_BOX.items():
            form.set_mode(mode)
            assert form.boxes[box_key].isVisible(), f"{mode} panel hidden"
            for other, other_key in MODE_BOX.items():
                if other_key != box_key:
                    assert not form.boxes[other_key].isVisible(),                         f"{other} panel visible while in {mode}"

    def test_arduino_rows_follow_the_checkbox(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.show()
        form.w["use_microcontroller"].setChecked(False)
        assert not form.w["arduino_port"].isVisible()
        assert not form.w["arduino_baud"].isVisible()
        form.w["use_microcontroller"].setChecked(True)
        assert form.w["arduino_port"].isVisible()
        assert form.w["arduino_baud"].isVisible()

    def test_every_mode_parameter_round_trips(self, qapp, tmp_path):
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(
            experiment_mode="temporal_envelope_modulation",
            base_output_path=str(tmp_path),
            smooth_frequencies=[1111, 2222],
            simple_interval_tonal_centre=9000.0,
            simple_intervals_list=["octave", "min_3"],
            tem_controls=["silent"],
            tem_smooth_freqs=[3000], tem_constant_rough_freqs=[4000],
            tem_complex_rough_freqs=[5000], tem_constant_mod_freq=77.0,
            tem_complex_mod_freqs=[11, 22], tem_mod_depth=0.25,
            sequence_patterns=["ABAB", "silence"],
            sequence_tone_map={"A": 8000.0, "B": 9000.0},
            sequence_repetitions=17,
            vocalisation_include_silent_arm=False,
        )
        form = ConfigForm()
        form.from_config(cfg)
        assert form.to_config() == cfg

    def test_interval_order_is_preserved(self, qapp):
        """Arm assignment follows list order, so a round trip must not re-sort."""
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        order = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]
        form = ConfigForm()
        form.from_config(ExperimentConfig(simple_intervals_list=order))
        assert form.to_config().simple_intervals_list == order

    def test_complex_preset_versus_override(self, qapp):
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        form = ConfigForm()
        form.from_config(ExperimentConfig(complex_interval_day="w1day3"))
        assert form.w["complex_use_preset"].isChecked()
        cfg = form.to_config()
        assert cfg.complex_consonant_intervals == []       # still preset-driven
        assert cfg.resolve_complex_interval_day()["consonant"] == ["maj_6", "min_3"]

        form.w["complex_use_preset"].setChecked(False)
        form.w["complex_consonant_intervals"].set_checked(["octave"])
        cfg2 = form.to_config()
        assert cfg2.complex_consonant_intervals == ["octave"]
        assert cfg2.resolve_complex_interval_day()["consonant"] == ["octave"]

    def test_reset_to_defaults(self, qapp, tmp_path):
        from amazeing.app.config_form import ConfigForm
        from amazeing.auditory.config import ExperimentConfig
        form = ConfigForm()
        form.w["rois_number"].setValue(3)
        form.w["binary_threshold"].setValue(99)
        form.set_mode("simple_smooth")
        form.reset_to_defaults()
        assert form.to_config() == ExperimentConfig()


class TestScrollProtection:

    def test_wheel_ignored_without_focus(self, qapp):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QApplication
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        spin = form.w["binary_threshold"]
        spin.clearFocus()
        before = spin.value()
        ev = QWheelEvent(QPoint(5, 5), spin.mapToGlobal(QPoint(5, 5)),
                         QPoint(0, 0), QPoint(0, 120), Qt.NoButton,
                         Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(spin, ev)
        assert spin.value() == before, "scrolling changed an unfocused control"


class TestWaveformPreview:

    def test_preview_returns_one_entry_per_arm(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="simple_smooth", rois_number=4)
        out = preview_stimuli(cfg, duration_s=0.02)
        assert [p.arm for p in out] == ["1", "2", "3", "4"]
        assert all(p.wave.size > 0 for p in out)
        assert not any(p.is_silent for p in out)
        assert "10000" in out[0].label

    def test_preview_marks_silent_arms(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="custom", rois_number=2,
                               custom_stimuli=[{"roi": "1", "kind": "tone",
                                                "frequency": 9000}])
        out = preview_stimuli(cfg, duration_s=0.02)
        assert out[0].is_silent is False
        assert out[1].is_silent is True

    def test_preview_never_touches_the_sound_card(self, monkeypatch):
        import sounddevice as sd
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig

        def boom(*a, **k):
            raise AssertionError("preview configured the sound device")
        monkeypatch.setattr(type(sd.default), "device",
                            property(lambda s: 0, boom), raising=False)
        preview_stimuli(ExperimentConfig(experiment_mode="simple_smooth",
                                         rois_number=2), duration_s=0.02)

    def test_spectrum_peaks_at_the_tone(self):
        import numpy as np
        from amazeing.app.stimulus_preview import spectrum
        fs = 48000
        t = np.arange(fs // 10) / fs
        wave = np.sin(2 * np.pi * 5000 * t)
        freqs, mag = spectrum(wave, fs)
        assert abs(freqs[int(np.argmax(mag))] - 5000) < 50


class TestMainWindow:

    def test_window_builds(self, qapp):
        from amazeing.app.main_window import MainWindow
        win = MainWindow()
        assert win.centralWidget().count() == 5

    def test_preview_never_prompts_on_the_console(self, monkeypatch):
        """A console prompt inside the preview would freeze the window."""
        import builtins
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig

        monkeypatch.setattr(builtins, "input",
                            lambda *a, **k: pytest.fail("preview prompted for input"))
        cfg = ExperimentConfig(experiment_mode="sequences", rois_number=2)
        with pytest.raises(ValueError, match="tone map"):
            preview_stimuli(cfg, duration_s=0.02)

    def test_preview_works_once_the_tone_map_is_filled(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="sequences", rois_number=2,
                               sequence_patterns=["ABAB", "silence"],
                               sequence_tone_map={"A": 8000, "B": 9000})
        out = preview_stimuli(cfg, duration_s=0.02)
        assert len(out) == 2


class TestPaletteAndHelp:

    def test_every_section_has_an_explanation(self, qapp):
        from amazeing.app.config_form import ConfigForm
        from amazeing.app.help_text import SECTION_HELP
        form = ConfigForm()
        assert set(form.boxes) == set(SECTION_HELP), "a section is missing its help text"
        assert set(form.help_widgets) == set(SECTION_HELP)

    def test_help_starts_collapsed_and_toggles(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.show()
        h = form.help_widgets["detection"]
        assert not h.body.isVisible()
        h.button.setChecked(True)
        assert h.body.isVisible()
        h.button.setChecked(False)
        assert not h.body.isVisible()

    def test_help_text_is_substantial(self):
        from amazeing.app.help_text import SECTION_HELP
        for key, text in SECTION_HELP.items():
            assert len(text) > 200, f"{key} help is too thin to be worth a click"

    def test_grammar_categories_match_the_analysis_palette(self):
        from amazeing.app.palette import STIMULUS_COLORS
        from amazeing.auditory.analysis import STIM_COLORS
        for key, colour in STIM_COLORS.items():
            assert STIMULUS_COLORS[key] == colour, f"{key} differs from the figures"

    def test_category_assignment(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="temporal_envelope_modulation",
                               rois_number=8)
        cats = {p.category for p in preview_stimuli(cfg, duration_s=0.02)}
        assert {"smooth", "rough", "rough_complex"} <= cats
        cfg = ExperimentConfig(experiment_mode="custom", rois_number=2,
                               custom_stimuli=[{"roi": "1", "kind": "tone",
                                                "frequency": 9000}])
        out = preview_stimuli(cfg, duration_s=0.02)
        assert out[0].category == "tone"
        assert out[1].category == "silent"

    def test_every_category_has_a_colour(self):
        from amazeing.app.palette import colour_for
        for cat in ("tone", "am_tone", "wav", "smooth", "rough", "rough_complex",
                    "consonant", "dissonant", "silent", "vocalisation",
                    "dominant EE", "rare SC", "pattern", "interval", "nonsense"):
            assert colour_for(cat).startswith("#")

    def test_form_reports_the_stimulus_count(self, qapp):
        from amazeing.app.config_form import ConfigForm
        form = ConfigForm()
        form.set_mode("temporal_envelope_modulation")
        form.w["rois_number"].setValue(8)
        assert "8 stimuli for 8 arms" in form.stimulus_label.text()
        form.w["rois_number"].setValue(4)
        assert "will not start" in form.stimulus_label.text()
        with pytest.raises(ValueError, match="arms"):
            form.check_ready()
        form.w["rois_number"].setValue(8)
        form.check_ready()
