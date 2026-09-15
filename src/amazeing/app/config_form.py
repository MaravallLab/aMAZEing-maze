"""Form widget that edits an ``ExperimentConfig``.

``to_config()`` builds the dataclass from the widgets (raising ValueError
with a readable message when something does not parse) and
``from_config()`` fills the widgets from an existing one, so the form and
the YAML session file always describe the same thing.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget, QLabel)

from amazeing.auditory.config import ExperimentConfig

EXPERIMENT_MODES = ["custom", "grammar", "simple_smooth", "simple_intervals",
                    "temporal_envelope_modulation", "complex_intervals",
                    "sequences", "vocalisation"]
WAVEFORMS = ["sine", "square", "sawtooth", "triangle", "pulse wave", "white noise"]
STIMULUS_KINDS = ["tone", "am_tone", "wav", "silent"]
STIM_COLUMNS = ["roi", "kind", "frequency", "waveform", "path", "mod_freq", "depth", "label"]


class PathPicker(QWidget):
    """Line edit plus a Browse button (file or directory)."""

    def __init__(self, mode: str = "file", filter_: str = "", parent=None):
        super().__init__(parent)
        self.mode, self.filter = mode, filter_
        self.edit = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, s: str) -> None:
        self.edit.setText(s or "")

    def _browse(self) -> None:
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Choose folder", self.text())
        elif self.mode == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Choose file", self.text(), self.filter)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", self.text(), self.filter)
        if p:
            self.edit.setText(p)


def _list_audio_devices() -> List[str]:
    try:
        import sounddevice as sd
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                out.append(f"{i}: {d['name']}")
        return out
    except Exception:
        return []


def _list_serial_ports() -> List[str]:
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


class ConfigForm(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.w: Dict[str, Any] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.inner_layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._build_general()
        self._build_devices()
        self._build_detection()
        self._build_sounds()
        self._build_custom_stimuli()
        self._build_grammar()
        self._build_paths()
        self.inner_layout.addStretch(1)
        self.from_config(ExperimentConfig())
        self.w["experiment_mode"].currentTextChanged.connect(self._mode_changed)
        self._mode_changed(self.w["experiment_mode"].currentText())

    # -- builders ----------------------------------------------------------
    def _group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.inner_layout.addWidget(box)
        box.form = form
        setattr(self, "_box_" + title.lower().replace(" ", "_").replace("&", "and"), box)
        return form

    def _build_general(self):
        f = self._group("Experiment")
        self.w["experiment_mode"] = QComboBox(); self.w["experiment_mode"].addItems(EXPERIMENT_MODES)
        f.addRow("Experiment mode", self.w["experiment_mode"])
        self.w["rois_number"] = QSpinBox(); self.w["rois_number"].setRange(1, 32)
        f.addRow("Number of arm ROIs", self.w["rois_number"])
        self.w["experiment_day"] = QLineEdit()
        self.w["experiment_day"].setPlaceholderText("optional folder label, e.g. day_1")
        f.addRow("Day label", self.w["experiment_day"])
        self.w["base_output_path"] = PathPicker("dir")
        f.addRow("Recordings folder", self.w["base_output_path"])
        self.w["record_video"] = QCheckBox("Save the video")
        f.addRow("", self.w["record_video"])
        self.w["show_binary_view"] = QCheckBox("Show the binary (thresholded) view during the session")
        f.addRow("", self.w["show_binary_view"])
        self.w["testing"] = QCheckBox("Testing mode (very short blocks)")
        f.addRow("", self.w["testing"])
        self.w["longer_middle_silence"] = QCheckBox("Longer middle silent block (legacy schedule)")
        f.addRow("", self.w["longer_middle_silence"])

    def _build_devices(self):
        f = self._group("Devices")
        self.w["video_input"] = QSpinBox(); self.w["video_input"].setRange(0, 16)
        f.addRow("Camera index", self.w["video_input"])
        self.w["channel_id"] = QComboBox(); self.w["channel_id"].setEditable(True)
        self.w["channel_id"].addItems(_list_audio_devices())
        f.addRow("Audio output device", self.w["channel_id"])
        self.w["samplerate"] = QComboBox(); self.w["samplerate"].setEditable(True)
        self.w["samplerate"].addItems(["44100", "48000", "96000", "192000"])
        f.addRow("Sample rate (Hz)", self.w["samplerate"])
        self.w["use_microcontroller"] = QCheckBox("Send TTL pulses through an Arduino")
        f.addRow("", self.w["use_microcontroller"])
        self.w["arduino_port"] = QComboBox(); self.w["arduino_port"].setEditable(True)
        self.w["arduino_port"].addItems(_list_serial_ports())
        f.addRow("Arduino port", self.w["arduino_port"])
        self.w["arduino_baud"] = QSpinBox(); self.w["arduino_baud"].setRange(300, 2000000)
        f.addRow("Arduino baud rate", self.w["arduino_baud"])

    def _build_detection(self):
        f = self._group("Mouse detection")
        self.w["binary_threshold"] = QSpinBox(); self.w["binary_threshold"].setRange(0, 255)
        f.addRow("Binary threshold (0-255)", self.w["binary_threshold"])
        self.w["detection_sensitivity"] = QDoubleSpinBox()
        self.w["detection_sensitivity"].setRange(0.01, 0.99); self.w["detection_sensitivity"].setSingleStep(0.05)
        f.addRow("Detection sensitivity (fraction of baseline)", self.w["detection_sensitivity"])
        self.w["roi_csv_path"] = PathPicker("save", "CSV (*.csv)")
        f.addRow("ROI file", self.w["roi_csv_path"])
        f.addRow("", QLabel("Leave the ROI file empty to use <recordings folder>/rois1.csv. "
                            "Use the Draw ROIs button to create or replace it."))

    def _build_sounds(self):
        f = self._group("Sound defaults")
        self.w["default_sound_duration"] = QDoubleSpinBox(); self.w["default_sound_duration"].setRange(0.1, 600)
        f.addRow("Sound duration (s)", self.w["default_sound_duration"])
        self.w["default_volume"] = QDoubleSpinBox(); self.w["default_volume"].setRange(0.0, 1.0); self.w["default_volume"].setSingleStep(0.05)
        f.addRow("Volume (0-1)", self.w["default_volume"])
        self.w["default_ramp_length_s"] = QDoubleSpinBox(); self.w["default_ramp_length_s"].setRange(0.0, 5.0); self.w["default_ramp_length_s"].setDecimals(3)
        f.addRow("Onset ramp (s)", self.w["default_ramp_length_s"])
        self.w["default_waveform"] = QComboBox(); self.w["default_waveform"].addItems(WAVEFORMS)
        f.addRow("Waveform", self.w["default_waveform"])
        self.w["calibration_gain_path"] = PathPicker("file", "CSV (*.csv)")
        f.addRow("Speaker calibration CSV", self.w["calibration_gain_path"])
        self.w["grammar_apply_speaker_gain"] = QCheckBox("Apply speaker compensation to grammar tones")
        f.addRow("", self.w["grammar_apply_speaker_gain"])

    def _build_custom_stimuli(self):
        f = self._group("Custom stimuli (experiment mode: custom)")
        self.stim_table = QTableWidget(0, len(STIM_COLUMNS))
        self.stim_table.setHorizontalHeaderLabels(STIM_COLUMNS)
        self.stim_table.horizontalHeader().setStretchLastSection(True)
        self.stim_table.setMinimumHeight(180)
        f.addRow(self.stim_table)
        btns = QHBoxLayout()
        b_fill = QPushButton("One row per ROI"); b_fill.clicked.connect(self._fill_rows)
        b_add = QPushButton("Add row"); b_add.clicked.connect(lambda: self._add_stim_row({}))
        b_del = QPushButton("Remove selected"); b_del.clicked.connect(self._remove_stim_row)
        for b in (b_fill, b_add, b_del):
            btns.addWidget(b)
        btns.addStretch(1)
        f.addRow(btns)
        f.addRow("", QLabel("kind: tone (needs frequency), am_tone (frequency, mod_freq, depth), "
                            "wav (path), silent. Arms without a row are silent."))
        self.w["custom_block_minutes"] = QLineEdit()
        self.w["custom_block_minutes"].setPlaceholderText("9 block durations in minutes, e.g. 2,15,2,15,2,15,2,15,2 (blank = standard)")
        f.addRow("Block schedule (min)", self.w["custom_block_minutes"])

    def _build_grammar(self):
        f = self._group("Grammar experiment")
        self.w["grammar_mode"] = QComboBox(); self.w["grammar_mode"].addItems(["silent_baseline", "test", "training"])
        f.addRow("Grammar mode", self.w["grammar_mode"])
        self.w["enriched_grammar"] = QComboBox(); self.w["enriched_grammar"].addItems(["A", "B"])
        f.addRow("Grammar heard in the enriched cage", self.w["enriched_grammar"])
        self.w["grammar_seed"] = QLineEdit(); self.w["grammar_seed"].setPlaceholderText("blank = random")
        f.addRow("Random seed", self.w["grammar_seed"])
        self.w["grammar_silent_baseline_minutes"] = QDoubleSpinBox(); self.w["grammar_silent_baseline_minutes"].setRange(0.1, 600)
        f.addRow("Silent baseline length (min)", self.w["grammar_silent_baseline_minutes"])
        self.w["grammar_test_block_minutes"] = QLineEdit()
        f.addRow("Test block schedule (9 values, min)", self.w["grammar_test_block_minutes"])
        self.w["complex_interval_day"] = QComboBox(); self.w["complex_interval_day"].addItems(["w1day2", "w1day3", "w1day4", "another_day"])
        f.addRow("Complex-intervals day", self.w["complex_interval_day"])

    def _build_paths(self):
        f = self._group("Vocalisation files")
        self.w["path_to_vocalisation_folder"] = PathPicker("dir")
        f.addRow("Folder of .wav files (vocalisation mode)", self.w["path_to_vocalisation_folder"])
        self.w["path_to_vocalisation_control"] = PathPicker("file", "WAV (*.wav)")
        f.addRow("Control vocalisation .wav (mixed modes)", self.w["path_to_vocalisation_control"])

    # -- custom stimuli table -----------------------------------------------
    def _add_stim_row(self, spec: Dict[str, Any]) -> None:
        r = self.stim_table.rowCount()
        self.stim_table.insertRow(r)
        for c, col in enumerate(STIM_COLUMNS):
            if col == "kind":
                combo = QComboBox(); combo.addItems(STIMULUS_KINDS)
                combo.setCurrentText(str(spec.get("kind", "tone")))
                self.stim_table.setCellWidget(r, c, combo)
            elif col == "waveform":
                combo = QComboBox(); combo.addItems([""] + WAVEFORMS)
                combo.setCurrentText(str(spec.get("waveform", "")))
                self.stim_table.setCellWidget(r, c, combo)
            else:
                val = spec.get(col, "")
                self.stim_table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))

    def _remove_stim_row(self) -> None:
        rows = sorted({i.row() for i in self.stim_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.stim_table.removeRow(r)

    def _fill_rows(self) -> None:
        self.stim_table.setRowCount(0)
        for i in range(self.w["rois_number"].value()):
            self._add_stim_row({"roi": str(i + 1), "kind": "tone"})

    def _stim_rows(self) -> List[Dict[str, Any]]:
        out = []
        for r in range(self.stim_table.rowCount()):
            spec: Dict[str, Any] = {}
            for c, col in enumerate(STIM_COLUMNS):
                widget = self.stim_table.cellWidget(r, c)
                if widget is not None:
                    val = widget.currentText().strip()
                else:
                    item = self.stim_table.item(r, c)
                    val = item.text().strip() if item else ""
                if val == "":
                    continue
                if col in ("frequency", "mod_freq", "depth"):
                    try:
                        val = float(val)
                    except ValueError:
                        raise ValueError(f"Custom stimuli row {r + 1}: {col} must be a number, got {val!r}")
                spec[col] = val
            if not spec or "roi" not in spec:
                if spec:
                    raise ValueError(f"Custom stimuli row {r + 1}: the roi column is required")
                continue
            spec["roi"] = str(spec["roi"])
            out.append(spec)
        return out

    # -- mode visibility -----------------------------------------------------
    def _mode_changed(self, mode: str) -> None:
        for name, box in self.__dict__.items():
            if not name.startswith("_box_"):
                continue
            if "custom" in name:
                box.setVisible(mode == "custom")
            elif "grammar" in name:
                box.setVisible(mode in ("grammar", "complex_intervals"))

    # -- conversion ------------------------------------------------------------
    def to_config(self) -> ExperimentConfig:
        w = self.w
        cfg = ExperimentConfig(
            experiment_mode=w["experiment_mode"].currentText(),
            rois_number=w["rois_number"].value(),
            experiment_day=w["experiment_day"].text().strip(),
            base_output_path=w["base_output_path"].text() or ExperimentConfig().base_output_path,
            record_video=w["record_video"].isChecked(),
            show_binary_view=w["show_binary_view"].isChecked(),
            testing=w["testing"].isChecked(),
            longer_middle_silence=w["longer_middle_silence"].isChecked(),
            video_input=w["video_input"].value(),
            channel_id=_leading_int(w["channel_id"].currentText(), "Audio output device"),
            samplerate=_leading_int(w["samplerate"].currentText(), "Sample rate"),
            use_microcontroller=w["use_microcontroller"].isChecked(),
            arduino_port=w["arduino_port"].currentText().strip(),
            arduino_baud=w["arduino_baud"].value(),
            binary_threshold=w["binary_threshold"].value(),
            detection_sensitivity=w["detection_sensitivity"].value(),
            roi_csv_path=w["roi_csv_path"].text(),
            default_sound_duration=w["default_sound_duration"].value(),
            default_volume=w["default_volume"].value(),
            default_ramp_length_s=w["default_ramp_length_s"].value(),
            default_waveform=w["default_waveform"].currentText(),
            calibration_gain_path=w["calibration_gain_path"].text() or ExperimentConfig().calibration_gain_path,
            grammar_apply_speaker_gain=w["grammar_apply_speaker_gain"].isChecked(),
            grammar_mode=w["grammar_mode"].currentText(),
            enriched_grammar=w["enriched_grammar"].currentText(),
            grammar_seed=_optional_int(w["grammar_seed"].text(), "Random seed"),
            grammar_silent_baseline_minutes=w["grammar_silent_baseline_minutes"].value(),
            grammar_test_block_minutes=_float_list(w["grammar_test_block_minutes"].text(), "Test block schedule") or ExperimentConfig().grammar_test_block_minutes,
            complex_interval_day=w["complex_interval_day"].currentText(),
            path_to_vocalisation_folder=w["path_to_vocalisation_folder"].text(),
            path_to_vocalisation_control=w["path_to_vocalisation_control"].text(),
            custom_stimuli=self._stim_rows(),
            custom_block_minutes=_float_list(w["custom_block_minutes"].text(), "Block schedule") or None,
        )
        # Validate the schedule early so the user sees it before a session starts.
        cfg.get_trial_lengths()
        return cfg

    def from_config(self, cfg: ExperimentConfig) -> None:
        w = self.w
        w["experiment_mode"].setCurrentText(cfg.experiment_mode)
        w["rois_number"].setValue(cfg.rois_number)
        w["experiment_day"].setText(cfg.experiment_day)
        w["base_output_path"].setText(cfg.base_output_path)
        w["record_video"].setChecked(cfg.record_video)
        w["show_binary_view"].setChecked(cfg.show_binary_view)
        w["testing"].setChecked(cfg.testing)
        w["longer_middle_silence"].setChecked(cfg.longer_middle_silence)
        w["video_input"].setValue(cfg.video_input)
        _set_combo_by_index_prefix(w["channel_id"], cfg.channel_id)
        w["samplerate"].setCurrentText(str(cfg.samplerate))
        w["use_microcontroller"].setChecked(cfg.use_microcontroller)
        w["arduino_port"].setCurrentText(cfg.arduino_port)
        w["arduino_baud"].setValue(cfg.arduino_baud)
        w["binary_threshold"].setValue(cfg.binary_threshold)
        w["detection_sensitivity"].setValue(cfg.detection_sensitivity)
        w["roi_csv_path"].setText(cfg.roi_csv_path)
        w["default_sound_duration"].setValue(cfg.default_sound_duration)
        w["default_volume"].setValue(cfg.default_volume)
        w["default_ramp_length_s"].setValue(cfg.default_ramp_length_s)
        w["default_waveform"].setCurrentText(cfg.default_waveform)
        w["calibration_gain_path"].setText(cfg.calibration_gain_path)
        w["grammar_apply_speaker_gain"].setChecked(cfg.grammar_apply_speaker_gain)
        w["grammar_mode"].setCurrentText(cfg.grammar_mode)
        w["enriched_grammar"].setCurrentText(cfg.enriched_grammar)
        w["grammar_seed"].setText("" if cfg.grammar_seed is None else str(cfg.grammar_seed))
        w["grammar_silent_baseline_minutes"].setValue(cfg.grammar_silent_baseline_minutes)
        w["grammar_test_block_minutes"].setText(",".join(_fmt(x) for x in cfg.grammar_test_block_minutes))
        w["complex_interval_day"].setCurrentText(cfg.complex_interval_day)
        w["path_to_vocalisation_folder"].setText(cfg.path_to_vocalisation_folder)
        w["path_to_vocalisation_control"].setText(cfg.path_to_vocalisation_control)
        self.stim_table.setRowCount(0)
        for spec in cfg.custom_stimuli:
            self._add_stim_row(spec)
        w["custom_block_minutes"].setText(
            "" if cfg.custom_block_minutes is None else ",".join(_fmt(x) for x in cfg.custom_block_minutes))


# -- parsing helpers -------------------------------------------------------------
def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _leading_int(text: str, label: str) -> int:
    head = text.strip().split(":")[0].strip()
    try:
        return int(head)
    except ValueError:
        raise ValueError(f"{label}: expected a number (or 'index: name'), got {text!r}")


def _optional_int(text: str, label: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{label}: expected a whole number, got {text!r}")


def _float_list(text: str, label: str) -> List[float]:
    text = text.strip().strip("[]")
    if not text:
        return []
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            raise ValueError(f"{label}: {tok!r} is not a number")
    return out


def _set_combo_by_index_prefix(combo: QComboBox, index: int) -> None:
    for i in range(combo.count()):
        if combo.itemText(i).split(":")[0].strip() == str(index):
            combo.setCurrentIndex(i)
            return
    combo.setCurrentText(str(index))
