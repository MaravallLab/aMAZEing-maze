"""Form widget that edits an ``ExperimentConfig``.

``to_config()`` builds the dataclass from the widgets (raising ValueError with
a readable message when something does not parse) and ``from_config()`` fills
the widgets from an existing one, so the form and the YAML session file always
describe the same thing.

Only the panel belonging to the selected experiment mode is shown, and rows
that depend on a switch (the Arduino port, for instance) appear only when that
switch is on, so the form never offers a setting that would be ignored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from amazeing.app.help_text import SECTION_HELP
from amazeing.app.widgets import (CheckableList, CollapsibleHelp, PathPicker,
                                  format_numbers, parse_float_list,
                                  parse_str_list, protect_from_scroll)
from amazeing.auditory.config import (COMPLEX_INTERVAL_DAYS, INTERVAL_NAMES,
                                      ExperimentConfig)

EXPERIMENT_MODES = ["custom", "grammar", "simple_smooth", "simple_intervals",
                    "temporal_envelope_modulation", "complex_intervals",
                    "sequences", "vocalisation"]

MODE_LABELS = {
    "custom": "Custom - one stimulus per arm, defined by you",
    "grammar": "Grammar learning (Markov tone grammars)",
    "simple_smooth": "Pure tones - one frequency per arm",
    "simple_intervals": "Musical intervals - two-tone chords",
    "temporal_envelope_modulation": "Temporal envelope modulation (AM)",
    "complex_intervals": "Consonant vs dissonant intervals",
    "sequences": "Tone sequences (patterns)",
    "vocalisation": "Vocalisation recordings",
}

# Which parameter panel belongs to which mode.
MODE_BOX = {
    "simple_smooth": "simple_smooth",
    "simple_intervals": "simple_intervals",
    "temporal_envelope_modulation": "tem",
    "complex_intervals": "complex_intervals",
    "sequences": "sequences",
    "vocalisation": "vocalisation",
    "grammar": "grammar",
    "custom": "custom",
}
# Modes that can play a .wav, so the vocalisation file box is relevant.
MODES_USING_WAVS = {"temporal_envelope_modulation", "complex_intervals",
                    "sequences", "vocalisation", "grammar", "custom"}

WAVEFORMS = ["sine", "square", "sawtooth", "triangle", "pulse wave", "white noise"]
STIMULUS_KINDS = ["tone", "am_tone", "wav", "silent"]
STIM_COLUMNS = ["roi", "kind", "frequency", "waveform", "path", "mod_freq", "depth", "label"]


def _list_audio_devices() -> List[str]:
    try:
        import sounddevice as sd
        return [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices())
                if d.get("max_output_channels", 0) > 0]
    except Exception:
        return []


def _list_serial_ports() -> List[str]:
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("hint", True)
    return lab


class ConfigForm(QWidget):
    changed = Signal()          # any edit that could alter the stimuli

    def __init__(self, parent=None):
        super().__init__(parent)
        self.w: Dict[str, Any] = {}
        self.boxes: Dict[str, QGroupBox] = {}
        self.help_widgets: Dict[str, CollapsibleHelp] = {}
        self._rows: Dict[str, List[QWidget]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.reset_button = QPushButton("Reset all to defaults")
        self.reset_button.setToolTip(
            "Restore every setting on this form to the values the package ships with.")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        bar.addWidget(self.reset_button)
        bar.addStretch(1)
        outer.addLayout(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.inner_layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self._build_general()
        self._build_devices()
        self._build_detection()
        self._build_sounds()
        self._build_simple_smooth()
        self._build_simple_intervals()
        self._build_tem()
        self._build_complex_intervals()
        self._build_sequences()
        self._build_vocalisation()
        self._build_grammar()
        self._build_custom_stimuli()
        self._build_voc_files()
        self.inner_layout.addStretch(1)

        self.from_config(ExperimentConfig())
        protect_from_scroll(self)
        self._connect_change_signals()
        self.w["experiment_mode"].currentTextChanged.connect(self._mode_changed)
        self.w["use_microcontroller"].toggled.connect(self._arduino_toggled)
        self.w["complex_use_preset"].toggled.connect(self._preset_toggled)
        self.w["complex_interval_day"].currentTextChanged.connect(self._load_day_preset)
        self._mode_changed(self.w["experiment_mode"].currentText())
        self._arduino_toggled(self.w["use_microcontroller"].isChecked())
        self._preset_toggled(self.w["complex_use_preset"].isChecked())

    # ----------------------------------------------------------------- build
    def _group(self, key: str, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        help_text = SECTION_HELP.get(key)
        if help_text:
            help_widget = CollapsibleHelp(help_text)
            form.addRow(help_widget)
            self.help_widgets[key] = help_widget
        self.inner_layout.addWidget(box)
        self.boxes[key] = box
        return form

    def _build_general(self):
        f = self._group("experiment", "Experiment")
        self.w["experiment_mode"] = QComboBox()
        for m in EXPERIMENT_MODES:
            self.w["experiment_mode"].addItem(MODE_LABELS[m], userData=m)
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
        self.w["block_minutes"] = QLineEdit()
        self.w["block_minutes"].setPlaceholderText(
            "blank = this mode's default, e.g. 2, 15, 2, 15, 2, 15, 2, 15, 2")
        f.addRow("Block lengths (min)", self.w["block_minutes"])
        f.addRow("", _hint(
            "Nine values: positions 1, 3, 5, 7, 9 are silent blocks and 2, 4, 6, 8 "
            "are active blocks. Use 0 to skip a block. Leave blank to keep the "
            "default schedule for the selected mode."))
        self.schedule_label = QLabel()
        self.schedule_label.setWordWrap(True)
        f.addRow("Block schedule", self.schedule_label)

    def _build_devices(self):
        f = self._group("devices", "Devices")
        self.w["video_input"] = QSpinBox(); self.w["video_input"].setRange(0, 16)
        f.addRow("Camera index", self.w["video_input"])
        self.w["channel_id"] = QComboBox(); self.w["channel_id"].setEditable(True)
        self.w["channel_id"].addItems(_list_audio_devices())
        f.addRow("Audio output device", self.w["channel_id"])
        self.w["samplerate"] = QComboBox(); self.w["samplerate"].setEditable(True)
        self.w["samplerate"].addItems(["44100", "48000", "96000", "192000"])
        f.addRow("Sample rate (Hz)", self.w["samplerate"])
        self.w["use_microcontroller"] = QCheckBox(
            "Send TTL pulses through an Arduino (photometry sync)")
        f.addRow("", self.w["use_microcontroller"])
        self.w["arduino_port"] = QComboBox(); self.w["arduino_port"].setEditable(True)
        self.w["arduino_port"].addItems(_list_serial_ports())
        self.w["arduino_baud"] = QSpinBox(); self.w["arduino_baud"].setRange(300, 2000000)
        self._add_dependent_row(f, "arduino", "Arduino port", self.w["arduino_port"])
        self._add_dependent_row(f, "arduino", "Arduino baud rate", self.w["arduino_baud"])
        self._add_dependent_row(f, "arduino", "", _hint(
            "With TTL on, the legacy block schedule becomes 15/10/2 minutes instead "
            "of 15/15/2. The resulting schedule is shown under Experiment."))

    def _add_dependent_row(self, form: QFormLayout, key: str, label: str, widget: QWidget):
        """Add a row that can be hidden as a unit (label included)."""
        form.addRow(label, widget)
        lab = form.labelForField(widget)
        self._rows.setdefault(key, []).extend([w for w in (lab, widget) if w is not None])

    def _build_detection(self):
        f = self._group("detection", "Mouse detection")
        self.w["binary_threshold"] = QSpinBox(); self.w["binary_threshold"].setRange(0, 255)
        f.addRow("Binary threshold (0-255)", self.w["binary_threshold"])
        self.w["detection_sensitivity"] = QDoubleSpinBox()
        self.w["detection_sensitivity"].setRange(0.01, 0.99)
        self.w["detection_sensitivity"].setSingleStep(0.05)
        f.addRow("Detection sensitivity (fraction of baseline)", self.w["detection_sensitivity"])
        self.w["roi_csv_path"] = PathPicker("save", "CSV (*.csv)")
        f.addRow("ROI file", self.w["roi_csv_path"])
        f.addRow("", _hint("Leave the ROI file empty to use <recordings folder>/rois1.csv. "
                           "Use Draw ROIs to create or replace it."))

    def _build_sounds(self):
        f = self._group("sound", "Sound defaults")
        self.w["default_sound_duration"] = QDoubleSpinBox()
        self.w["default_sound_duration"].setRange(0.1, 600)
        f.addRow("Sound duration (s)", self.w["default_sound_duration"])
        self.w["default_volume"] = QDoubleSpinBox()
        self.w["default_volume"].setRange(0.0, 1.0); self.w["default_volume"].setSingleStep(0.05)
        f.addRow("Volume (0-1)", self.w["default_volume"])
        self.w["default_ramp_length_s"] = QDoubleSpinBox()
        self.w["default_ramp_length_s"].setRange(0.0, 5.0)
        self.w["default_ramp_length_s"].setDecimals(3)
        f.addRow("Onset ramp (s)", self.w["default_ramp_length_s"])
        self.w["default_waveform"] = QComboBox(); self.w["default_waveform"].addItems(WAVEFORMS)
        f.addRow("Waveform", self.w["default_waveform"])
        self.w["calibration_gain_path"] = PathPicker("file", "CSV (*.csv)")
        f.addRow("Speaker calibration CSV", self.w["calibration_gain_path"])
        self.w["grammar_apply_speaker_gain"] = QCheckBox(
            "Apply speaker compensation to grammar tones")
        f.addRow("", self.w["grammar_apply_speaker_gain"])

    # ------------------------------------------------------------ mode boxes
    def _build_simple_smooth(self):
        f = self._group("simple_smooth", "Pure tones")
        self.w["smooth_frequencies"] = QLineEdit()
        f.addRow("Frequencies (Hz), one per arm", self.w["smooth_frequencies"])
        f.addRow("", _hint("Comma separated. If you give fewer frequencies than arms "
                           "the list is recycled."))

    def _build_simple_intervals(self):
        f = self._group("simple_intervals", "Musical intervals")
        self.w["simple_interval_tonal_centre"] = QDoubleSpinBox()
        self.w["simple_interval_tonal_centre"].setRange(20, 100000)
        self.w["simple_interval_tonal_centre"].setSingleStep(500)
        self.w["simple_interval_tonal_centre"].setSuffix(" Hz")
        f.addRow("Tonal centre", self.w["simple_interval_tonal_centre"])
        self.w["simple_intervals_list"] = CheckableList(INTERVAL_NAMES)
        f.addRow("Intervals", self.w["simple_intervals_list"])
        f.addRow("", _hint("Each arm plays the tonal centre together with one interval "
                           "above it. One unison arm and one silent arm are added "
                           "automatically, so tick (number of arms - 2) intervals."))

    def _build_tem(self):
        f = self._group("tem", "Temporal envelope modulation")
        self.w["tem_smooth_freqs"] = QLineEdit()
        f.addRow("Smooth carrier frequencies (Hz)", self.w["tem_smooth_freqs"])
        self.w["tem_constant_rough_freqs"] = QLineEdit()
        f.addRow("Constant-AM carrier frequencies (Hz)", self.w["tem_constant_rough_freqs"])
        self.w["tem_constant_mod_freq"] = QDoubleSpinBox()
        self.w["tem_constant_mod_freq"].setRange(0.1, 5000); self.w["tem_constant_mod_freq"].setSuffix(" Hz")
        f.addRow("Constant AM rate", self.w["tem_constant_mod_freq"])
        self.w["tem_complex_rough_freqs"] = QLineEdit()
        f.addRow("Complex-AM carrier frequencies (Hz)", self.w["tem_complex_rough_freqs"])
        self.w["tem_complex_mod_freqs"] = QLineEdit()
        f.addRow("Complex AM rates (Hz)", self.w["tem_complex_mod_freqs"])
        self.w["tem_mod_depth"] = QDoubleSpinBox()
        self.w["tem_mod_depth"].setRange(0.0, 1.0); self.w["tem_mod_depth"].setSingleStep(0.05)
        f.addRow("AM depth (0-1)", self.w["tem_mod_depth"])
        row = QHBoxLayout()
        self.w["tem_control_voc"] = QCheckBox("Vocalisation arm")
        self.w["tem_control_silent"] = QCheckBox("Silent arm")
        row.addWidget(self.w["tem_control_voc"]); row.addWidget(self.w["tem_control_silent"])
        row.addStretch(1)
        f.addRow("Control arms", row)
        f.addRow("", _hint("Arms are assigned in this order: control arms, smooth, "
                           "constant AM, complex AM. The totals must add up to the "
                           "number of arms."))

    def _build_complex_intervals(self):
        f = self._group("complex_intervals", "Consonant vs dissonant intervals")
        self.w["complex_interval_day"] = QComboBox()
        self.w["complex_interval_day"].addItems(list(COMPLEX_INTERVAL_DAYS))
        f.addRow("Protocol day", self.w["complex_interval_day"])
        self.w["complex_use_preset"] = QCheckBox("Use this day's published stimulus set")
        f.addRow("", self.w["complex_use_preset"])
        self.w["complex_interval_tonal_centre"] = QDoubleSpinBox()
        self.w["complex_interval_tonal_centre"].setRange(20, 100000)
        self.w["complex_interval_tonal_centre"].setSingleStep(500)
        self.w["complex_interval_tonal_centre"].setSuffix(" Hz")
        f.addRow("Tonal centre", self.w["complex_interval_tonal_centre"])
        self.w["complex_consonant_intervals"] = CheckableList(INTERVAL_NAMES)
        f.addRow("Consonant intervals", self.w["complex_consonant_intervals"])
        self.w["complex_dissonant_intervals"] = CheckableList(INTERVAL_NAMES)
        f.addRow("Dissonant intervals", self.w["complex_dissonant_intervals"])
        row = QHBoxLayout()
        self.w["complex_control_voc"] = QCheckBox("Vocalisation arm")
        self.w["complex_control_silent"] = QCheckBox("Silent arm")
        row.addWidget(self.w["complex_control_voc"]); row.addWidget(self.w["complex_control_silent"])
        row.addStretch(1)
        f.addRow("Control arms", row)
        self.w["complex_include_smooth"] = QCheckBox("Smooth unison arm")
        self.w["complex_include_rough"] = QCheckBox("Rough (AM) unison arm")
        row2 = QHBoxLayout()
        row2.addWidget(self.w["complex_include_smooth"]); row2.addWidget(self.w["complex_include_rough"])
        row2.addStretch(1)
        f.addRow("Extra arms", row2)
        f.addRow("", _hint("Untick 'use this day's published stimulus set' to edit the "
                           "stimuli. The day name still labels the output folder."))

    def _build_sequences(self):
        f = self._group("sequences", "Tone sequences")
        self.w["sequence_patterns"] = QLineEdit()
        f.addRow("Patterns, one per arm", self.w["sequence_patterns"])
        self.tone_table = QTableWidget(0, 2)
        self.tone_table.setHorizontalHeaderLabels(["letter", "frequency (Hz)"])
        self.tone_table.horizontalHeader().setStretchLastSection(True)
        self.tone_table.setMaximumHeight(150)
        f.addRow("Tone map", self.tone_table)
        row = QHBoxLayout()
        b_add = QPushButton("Add tone"); b_add.clicked.connect(lambda: self._add_tone_row("", ""))
        b_del = QPushButton("Remove selected"); b_del.clicked.connect(self._remove_tone_row)
        b_auto = QPushButton("Add the letters used"); b_auto.clicked.connect(self._autofill_tones)
        for b in (b_add, b_del, b_auto):
            row.addWidget(b)
        row.addStretch(1)
        f.addRow("", row)
        self.w["sequence_repetitions"] = QSpinBox()
        self.w["sequence_repetitions"].setRange(1, 1000)
        f.addRow("Pattern repetitions", self.w["sequence_repetitions"])
        f.addRow("", _hint("Patterns are letters, plus the special names silence, "
                           "vocalisation and random. 'o' is a silent slot. Give every "
                           "other letter a frequency in the tone map; leave the map "
                           "empty to be asked on the console instead."))

    def _build_vocalisation(self):
        f = self._group("vocalisation", "Vocalisation recordings")
        self.w["vocalisation_include_silent_arm"] = QCheckBox("Include a silent control arm")
        f.addRow("", self.w["vocalisation_include_silent_arm"])
        f.addRow("", _hint("Every .wav in the folder below becomes one arm, in the "
                           "order the folder lists them."))

    def _build_grammar(self):
        f = self._group("grammar", "Grammar experiment")
        self.w["grammar_mode"] = QComboBox()
        self.w["grammar_mode"].addItems(["silent_baseline", "test", "training"])
        f.addRow("Grammar mode", self.w["grammar_mode"])
        self.w["enriched_grammar"] = QComboBox(); self.w["enriched_grammar"].addItems(["A", "B"])
        f.addRow("Grammar heard in the enriched cage", self.w["enriched_grammar"])
        self.w["grammar_seed"] = QLineEdit(); self.w["grammar_seed"].setPlaceholderText("blank = random")
        f.addRow("Random seed", self.w["grammar_seed"])
        self.w["grammar_silent_baseline_minutes"] = QDoubleSpinBox()
        self.w["grammar_silent_baseline_minutes"].setRange(0.1, 600)
        f.addRow("Silent baseline length (min)", self.w["grammar_silent_baseline_minutes"])
        self.w["grammar_test_block_minutes"] = QLineEdit()
        f.addRow("Test block schedule (9 values, min)", self.w["grammar_test_block_minutes"])
        f.addRow("", _hint("'training' refuses to run here; use the Grammar training tab."))

    def _build_custom_stimuli(self):
        f = self._group("custom", "Custom stimuli")
        self.stim_table = QTableWidget(0, len(STIM_COLUMNS))
        self.stim_table.setHorizontalHeaderLabels(STIM_COLUMNS)
        self.stim_table.horizontalHeader().setStretchLastSection(True)
        self.stim_table.setMinimumHeight(180)
        f.addRow(self.stim_table)
        btns = QHBoxLayout()
        b_fill = QPushButton("One row per arm"); b_fill.clicked.connect(self._fill_rows)
        b_add = QPushButton("Add row"); b_add.clicked.connect(lambda: self._add_stim_row({}))
        b_del = QPushButton("Remove selected"); b_del.clicked.connect(self._remove_stim_row)
        for b in (b_fill, b_add, b_del):
            btns.addWidget(b)
        btns.addStretch(1)
        f.addRow(btns)
        f.addRow("", _hint("kind: tone (needs frequency), am_tone (frequency, mod_freq, "
                           "depth), wav (path), silent. Arms with no row are silent."))

    def _build_voc_files(self):
        f = self._group("voc_files", "Vocalisation files")
        self.w["path_to_vocalisation_folder"] = PathPicker("dir")
        f.addRow("Folder of .wav files", self.w["path_to_vocalisation_folder"])
        self.w["path_to_vocalisation_control"] = PathPicker("file", "WAV (*.wav)")
        f.addRow("Control vocalisation .wav", self.w["path_to_vocalisation_control"])

    # -------------------------------------------------------------- dynamics
    def _connect_change_signals(self):
        for widget in self.w.values():
            for sig in ("currentTextChanged", "textChanged", "valueChanged", "toggled"):
                if hasattr(widget, sig):
                    getattr(widget, sig).connect(self._on_change)
                    break
            else:
                if isinstance(widget, PathPicker):
                    widget.edit.textChanged.connect(self._on_change)
                elif isinstance(widget, CheckableList):
                    widget.itemChanged.connect(self._on_change)
        for table in (self.stim_table, self.tone_table):
            table.itemChanged.connect(self._on_change)

    def _on_change(self, *_):
        self._update_schedule_label()
        self.changed.emit()

    def _update_schedule_label(self):
        try:
            cfg = self.to_config()
            lengths = cfg.get_trial_lengths()
        except Exception as e:
            self.schedule_label.setText(f"(cannot compute: {e})")
            return
        total = sum(lengths)
        if len(lengths) == 1:
            self.schedule_label.setText(f"one block of {lengths[0]:g} min")
            return
        parts = []
        for i, m in enumerate(lengths):
            if m == 0:
                continue
            parts.append(f"{m:g} {'silent' if i % 2 == 0 else 'active'}")
        self.schedule_label.setText(
            f"{' + '.join(parts)}  =  {total:g} min total")

    def current_mode(self) -> str:
        return self.w["experiment_mode"].currentData() or "custom"

    def set_mode(self, mode: str) -> None:
        """Select an experiment mode by its config value.

        The combo shows friendly labels, so setCurrentText with a raw mode name
        would silently do nothing; always use this.
        """
        i = self.w["experiment_mode"].findData(mode)
        if i < 0:
            raise ValueError(f"Unknown experiment mode {mode!r}; "
                             f"valid: {', '.join(EXPERIMENT_MODES)}")
        self.w["experiment_mode"].setCurrentIndex(i)

    def _mode_changed(self, _text: str = "") -> None:
        mode = self.current_mode()
        wanted = MODE_BOX.get(mode)
        for key in set(MODE_BOX.values()):
            self.boxes[key].setVisible(key == wanted)
        self.boxes["voc_files"].setVisible(mode in MODES_USING_WAVS)
        self._on_change()

    def _arduino_toggled(self, on: bool) -> None:
        for widget in self._rows.get("arduino", []):
            widget.setVisible(on)

    def _preset_toggled(self, on: bool) -> None:
        for key in ("complex_consonant_intervals", "complex_dissonant_intervals",
                    "complex_control_voc", "complex_control_silent",
                    "complex_include_smooth", "complex_include_rough"):
            self.w[key].setEnabled(not on)
        if on:
            self._load_day_preset()

    def _load_day_preset(self, *_):
        if not self.w["complex_use_preset"].isChecked():
            return
        day = COMPLEX_INTERVAL_DAYS.get(self.w["complex_interval_day"].currentText())
        if not day:
            return
        self.w["complex_consonant_intervals"].set_checked(day["consonant"])
        self.w["complex_dissonant_intervals"].set_checked(day["dissonant"])
        self.w["complex_control_voc"].setChecked("vocalisation" in day["controls"])
        self.w["complex_control_silent"].setChecked("silent" in day["controls"])
        self.w["complex_include_smooth"].setChecked(day["smooth"])
        self.w["complex_include_rough"].setChecked(day["rough"])

    def reset_to_defaults(self) -> None:
        self.from_config(ExperimentConfig())

    # ---------------------------------------------------------------- tables
    def _add_tone_row(self, letter, freq) -> None:
        r = self.tone_table.rowCount()
        self.tone_table.insertRow(r)
        self.tone_table.setItem(r, 0, QTableWidgetItem(str(letter)))
        self.tone_table.setItem(r, 1, QTableWidgetItem("" if freq == "" else str(freq)))

    def _remove_tone_row(self) -> None:
        for r in sorted({i.row() for i in self.tone_table.selectedIndexes()}, reverse=True):
            self.tone_table.removeRow(r)

    def _autofill_tones(self) -> None:
        special = ("silence", "vocalisation", "random")
        letters = {ch for p in parse_str_list(self.w["sequence_patterns"].text())
                   if p not in special for ch in p if ch != "o"}
        have = {self.tone_table.item(r, 0).text().strip()
                for r in range(self.tone_table.rowCount())
                if self.tone_table.item(r, 0)}
        for ch in sorted(letters - have):
            self._add_tone_row(ch, "")

    def _tone_map(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in range(self.tone_table.rowCount()):
            k_item = self.tone_table.item(r, 0)
            v_item = self.tone_table.item(r, 1)
            key = k_item.text().strip() if k_item else ""
            val = v_item.text().strip() if v_item else ""
            if not key and not val:
                continue
            if not key or not val:
                raise ValueError(f"Tone map row {r + 1}: both a letter and a frequency are needed")
            try:
                out[key] = float(val)
            except ValueError:
                raise ValueError(f"Tone map row {r + 1}: {val!r} is not a number")
        return out

    def _add_stim_row(self, spec: Dict[str, Any]) -> None:
        r = self.stim_table.rowCount()
        self.stim_table.insertRow(r)
        for c, col in enumerate(STIM_COLUMNS):
            if col == "kind":
                combo = QComboBox(); combo.addItems(STIMULUS_KINDS)
                combo.setCurrentText(str(spec.get("kind", "tone")))
                combo.currentTextChanged.connect(self._on_change)
                self.stim_table.setCellWidget(r, c, combo)
            elif col == "waveform":
                combo = QComboBox(); combo.addItems([""] + WAVEFORMS)
                combo.setCurrentText(str(spec.get("waveform", "")))
                combo.currentTextChanged.connect(self._on_change)
                self.stim_table.setCellWidget(r, c, combo)
            else:
                val = spec.get(col, "")
                self.stim_table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))

    def _remove_stim_row(self) -> None:
        for r in sorted({i.row() for i in self.stim_table.selectedIndexes()}, reverse=True):
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
                        raise ValueError(
                            f"Custom stimuli row {r + 1}: {col} must be a number, got {val!r}")
                spec[col] = val
            if not spec:
                continue
            if "roi" not in spec:
                raise ValueError(f"Custom stimuli row {r + 1}: the roi column is required")
            spec["roi"] = str(spec["roi"])
            out.append(spec)
        return out

    # ------------------------------------------------------------ conversion
    def to_config(self) -> ExperimentConfig:
        w = self.w
        mode = self.current_mode()
        controls_tem = ([c for c, k in (("vocalisation", "tem_control_voc"),
                                        ("silent", "tem_control_silent"))
                         if w[k].isChecked()])
        controls_cx = ([c for c, k in (("vocalisation", "complex_control_voc"),
                                       ("silent", "complex_control_silent"))
                        if w[k].isChecked()])
        use_preset = w["complex_use_preset"].isChecked()

        cfg = ExperimentConfig(
            experiment_mode=mode,
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
            calibration_gain_path=w["calibration_gain_path"].text()
                                  or ExperimentConfig().calibration_gain_path,
            grammar_apply_speaker_gain=w["grammar_apply_speaker_gain"].isChecked(),
            # per-mode
            smooth_frequencies=parse_float_list(w["smooth_frequencies"].text(), "Frequencies"),
            simple_interval_tonal_centre=w["simple_interval_tonal_centre"].value(),
            simple_intervals_list=w["simple_intervals_list"].checked(),
            tem_controls=controls_tem,
            tem_smooth_freqs=parse_float_list(w["tem_smooth_freqs"].text(), "Smooth carriers"),
            tem_constant_rough_freqs=parse_float_list(
                w["tem_constant_rough_freqs"].text(), "Constant-AM carriers"),
            tem_complex_rough_freqs=parse_float_list(
                w["tem_complex_rough_freqs"].text(), "Complex-AM carriers"),
            tem_constant_mod_freq=w["tem_constant_mod_freq"].value(),
            tem_complex_mod_freqs=parse_float_list(
                w["tem_complex_mod_freqs"].text(), "Complex AM rates"),
            tem_mod_depth=w["tem_mod_depth"].value(),
            complex_interval_day=w["complex_interval_day"].currentText(),
            complex_interval_tonal_centre=w["complex_interval_tonal_centre"].value(),
            complex_consonant_intervals=([] if use_preset
                                         else w["complex_consonant_intervals"].checked()),
            complex_dissonant_intervals=([] if use_preset
                                         else w["complex_dissonant_intervals"].checked()),
            complex_controls=(None if use_preset else controls_cx),
            complex_include_smooth=(None if use_preset
                                    else w["complex_include_smooth"].isChecked()),
            complex_include_rough=(None if use_preset
                                   else w["complex_include_rough"].isChecked()),
            sequence_patterns=parse_str_list(w["sequence_patterns"].text()),
            sequence_tone_map=self._tone_map(),
            sequence_repetitions=w["sequence_repetitions"].value(),
            vocalisation_include_silent_arm=w["vocalisation_include_silent_arm"].isChecked(),
            grammar_mode=w["grammar_mode"].currentText(),
            enriched_grammar=w["enriched_grammar"].currentText(),
            grammar_seed=_optional_int(w["grammar_seed"].text(), "Random seed"),
            grammar_silent_baseline_minutes=w["grammar_silent_baseline_minutes"].value(),
            grammar_test_block_minutes=parse_float_list(
                w["grammar_test_block_minutes"].text(), "Test block schedule")
                or ExperimentConfig().grammar_test_block_minutes,
            path_to_vocalisation_folder=w["path_to_vocalisation_folder"].text(),
            path_to_vocalisation_control=w["path_to_vocalisation_control"].text(),
            custom_stimuli=self._stim_rows(),
            block_minutes=parse_float_list(
                w["block_minutes"].text(), "Block lengths") or None,
        )
        cfg.get_trial_lengths()      # fail here rather than at session start
        return cfg

    def from_config(self, cfg: ExperimentConfig) -> None:
        w = self.w
        idx = w["experiment_mode"].findData(cfg.experiment_mode)
        w["experiment_mode"].setCurrentIndex(idx if idx >= 0 else 0)
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

        w["smooth_frequencies"].setText(format_numbers(cfg.smooth_frequencies))
        w["simple_interval_tonal_centre"].setValue(cfg.simple_interval_tonal_centre)
        w["simple_intervals_list"].set_checked(cfg.simple_intervals_list)
        w["tem_control_voc"].setChecked("vocalisation" in cfg.tem_controls)
        w["tem_control_silent"].setChecked("silent" in cfg.tem_controls)
        w["tem_smooth_freqs"].setText(format_numbers(cfg.tem_smooth_freqs))
        w["tem_constant_rough_freqs"].setText(format_numbers(cfg.tem_constant_rough_freqs))
        w["tem_complex_rough_freqs"].setText(format_numbers(cfg.tem_complex_rough_freqs))
        w["tem_constant_mod_freq"].setValue(cfg.tem_constant_mod_freq)
        w["tem_complex_mod_freqs"].setText(format_numbers(cfg.tem_complex_mod_freqs))
        w["tem_mod_depth"].setValue(cfg.tem_mod_depth)

        w["complex_interval_day"].setCurrentText(cfg.complex_interval_day)
        w["complex_interval_tonal_centre"].setValue(cfg.complex_interval_tonal_centre)
        overridden = bool(cfg.complex_consonant_intervals or cfg.complex_dissonant_intervals
                          or cfg.complex_controls is not None
                          or cfg.complex_include_smooth is not None
                          or cfg.complex_include_rough is not None)
        w["complex_use_preset"].setChecked(not overridden)
        resolved = cfg.resolve_complex_interval_day()
        w["complex_consonant_intervals"].set_checked(resolved["consonant"])
        w["complex_dissonant_intervals"].set_checked(resolved["dissonant"])
        w["complex_control_voc"].setChecked("vocalisation" in resolved["controls"])
        w["complex_control_silent"].setChecked("silent" in resolved["controls"])
        w["complex_include_smooth"].setChecked(resolved["smooth"])
        w["complex_include_rough"].setChecked(resolved["rough"])

        w["sequence_patterns"].setText(", ".join(cfg.sequence_patterns))
        self.tone_table.setRowCount(0)
        for k, v in cfg.sequence_tone_map.items():
            self._add_tone_row(k, v)
        w["sequence_repetitions"].setValue(cfg.sequence_repetitions)
        w["vocalisation_include_silent_arm"].setChecked(cfg.vocalisation_include_silent_arm)

        w["grammar_mode"].setCurrentText(cfg.grammar_mode)
        w["enriched_grammar"].setCurrentText(cfg.enriched_grammar)
        w["grammar_seed"].setText("" if cfg.grammar_seed is None else str(cfg.grammar_seed))
        w["grammar_silent_baseline_minutes"].setValue(cfg.grammar_silent_baseline_minutes)
        w["grammar_test_block_minutes"].setText(format_numbers(cfg.grammar_test_block_minutes))
        w["path_to_vocalisation_folder"].setText(cfg.path_to_vocalisation_folder)
        w["path_to_vocalisation_control"].setText(cfg.path_to_vocalisation_control)

        self.stim_table.setRowCount(0)
        for spec in cfg.custom_stimuli:
            self._add_stim_row(spec)
        w["block_minutes"].setText(
            "" if cfg.block_minutes is None else format_numbers(cfg.block_minutes))
        if hasattr(self, "schedule_label"):
            self._update_schedule_label()


# ------------------------------------------------------------------ parsing
def _leading_int(text: str, label: str) -> int:
    head = (text or "").strip().split(":")[0].strip()
    try:
        return int(head)
    except ValueError:
        raise ValueError(f"{label}: expected a number (or 'index: name'), got {text!r}")


def _optional_int(text: str, label: str) -> Optional[int]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{label}: expected a whole number, got {text!r}")


def _set_combo_by_index_prefix(combo: QComboBox, index: int) -> None:
    for i in range(combo.count()):
        if combo.itemText(i).split(":")[0].strip() == str(index):
            combo.setCurrentIndex(i)
            return
    combo.setCurrentText(str(index))
