"""Auditory session tab: edit the config, draw ROIs, start the session."""

from __future__ import annotations

import os
import time

from PySide6.QtWidgets import (QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLineEdit, QMessageBox, QPushButton, QSplitter,
                               QVBoxLayout, QWidget, QLabel)
from PySide6.QtCore import Qt

from amazeing.app.config_form import ConfigForm
from amazeing.app.launcher import command_for
from amazeing.app.process_panel import ProcessPanel
from amazeing.app.waveform_panel import WaveformPanel
from amazeing.auditory.session_config import load_config, save_config


class SessionTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = ConfigForm()
        self.current_config_path: str = ""

        # -- mouse box -----------------------------------------------------
        mouse_box = QGroupBox("Mouse")
        mf = QFormLayout(mouse_box)
        self.mouse_id = QLineEdit(); self.mouse_id.setPlaceholderText("e.g. 6224")
        self.ear_mark = QLineEdit()
        self.birth_date = QLineEdit(); self.birth_date.setPlaceholderText("dd/mm/yyyy")
        self.sex = QLineEdit(); self.sex.setPlaceholderText("m / f")
        mf.addRow("Mouse ID", self.mouse_id)
        mf.addRow("Ear mark", self.ear_mark)
        mf.addRow("Birth date", self.birth_date)
        mf.addRow("Sex", self.sex)

        # -- buttons -------------------------------------------------------
        self.load_btn = QPushButton("Load config...")
        self.save_btn = QPushButton("Save config as...")
        self.check_btn = QPushButton("Check camera")
        self.check_btn.setToolTip(
            "Live camera view with the arm boxes drawn on it. Tune the binary threshold "
            "and detection sensitivity and watch the effect. Nothing is recorded.")
        self.rois_btn = QPushButton("Draw ROIs")
        self.start_btn = QPushButton("Start session")
        self.start_btn.setStyleSheet("font-weight: bold;")
        self.load_btn.clicked.connect(self.load_config_dialog)
        self.save_btn.clicked.connect(self.save_config_dialog)
        self.check_btn.clicked.connect(self.check_camera)
        self.rois_btn.clicked.connect(self.draw_rois)
        self.start_btn.clicked.connect(self.start_session)
        self.config_label = QLabel("No config file loaded (defaults)")
        self.config_label.setWordWrap(True)

        btn_row = QHBoxLayout()
        for b in (self.load_btn, self.save_btn, self.check_btn, self.rois_btn, self.start_btn):
            btn_row.addWidget(b)

        # Set by check_camera so the values tuned in the live view come back to the form.
        self._check_path: str = ""

        self.panel = ProcessPanel()
        self.panel.started.connect(self._process_started)
        self.panel.finished.connect(self._process_finished)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(mouse_box)
        rl.addLayout(btn_row)
        rl.addWidget(self.config_label)
        rl.addWidget(self.panel, 1)

        self.waveform = WaveformPanel()
        self.waveform.bind(self.form.to_config)
        self.form.changed.connect(self.waveform.schedule_refresh)
        # The plots need real estate or matplotlib cannot fit its axis labels.
        self.waveform.setMinimumHeight(360)
        right.setMinimumHeight(260)

        middle = QSplitter(Qt.Vertical)
        middle.addWidget(self.waveform)
        middle.addWidget(right)
        middle.setStretchFactor(0, 3)
        middle.setStretchFactor(1, 2)
        middle.setSizes([520, 340])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.form)
        split.addWidget(middle)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setSizes([820, 700])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

    # -- config files ---------------------------------------------------------
    def load_config_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load session config", "", "YAML (*.yaml *.yml)")
        if p:
            self.load_config_file(p)

    def load_config_file(self, path: str) -> None:
        try:
            cfg = load_config(path)
        except Exception as e:
            QMessageBox.critical(self, "Cannot load config", str(e))
            return
        self.form.from_config(cfg)
        self.current_config_path = path
        self.config_label.setText(f"Config: {path}")

    def save_config_dialog(self) -> None:
        try:
            cfg = self.form.to_config()
        except ValueError as e:
            QMessageBox.warning(self, "Check the form", str(e))
            return
        start = self.current_config_path or os.path.join(cfg.base_output_path, "session_configs", "session.yaml")
        p, _ = QFileDialog.getSaveFileName(self, "Save session config", start, "YAML (*.yaml *.yml)")
        if p:
            save_config(cfg, p)
            self.current_config_path = p
            self.config_label.setText(f"Config: {p}")

    def _write_working_config(self, tag: str, check_stimuli: bool = True) -> str:
        """Validate the form and write it to <recordings>/session_configs/.

        ``check_stimuli`` is off for the camera check, which touches neither
        the sounds nor the stimulus-to-arm plan. Setting the camera up is the
        first thing you do, often before the stimuli are decided, so a
        half-finished Experiment section should not stand in the way of it.
        """
        cfg = self.form.to_config()
        if check_stimuli:
            cfg.check_stimulus_count()
            cfg.check_audio_files()
        folder = os.path.join(cfg.base_output_path, "session_configs")
        stamp = time.strftime("%Y-%m-%d_%H_%M_%S")
        path = os.path.join(folder, f"{stamp}_{tag}.yaml")
        save_config(cfg, path)
        self.config_label.setText(f"Config written: {path}")
        return path

    # -- process state -----------------------------------------------------------
    def _process_started(self) -> None:
        for b in (self.start_btn, self.check_btn, self.rois_btn):
            b.setEnabled(False)

    def _process_finished(self, _code: int) -> None:
        for b in (self.start_btn, self.check_btn, self.rois_btn):
            b.setEnabled(True)
        self._apply_checked_detection()

    def _apply_checked_detection(self) -> None:
        """Bring back any detection values saved from the live camera view."""
        path, self._check_path = self._check_path, ""
        if not path or not os.path.exists(path):
            return
        try:
            cfg = load_config(path)
        except Exception:
            return
        w = self.form.w
        changed = (w["binary_threshold"].value() != cfg.binary_threshold
                   or abs(w["detection_sensitivity"].value() - cfg.detection_sensitivity) > 1e-9)
        if not changed:
            return
        w["binary_threshold"].setValue(cfg.binary_threshold)
        w["detection_sensitivity"].setValue(cfg.detection_sensitivity)
        self.config_label.setText(
            f"From the camera check: binary threshold {cfg.binary_threshold}, "
            f"detection sensitivity {cfg.detection_sensitivity:.2f}. Save the config to keep them.")

    # -- actions -----------------------------------------------------------------
    def check_camera(self) -> None:
        try:
            path = self._write_working_config("check", check_stimuli=False)
        except ValueError as e:
            QMessageBox.warning(self, "Check the form", str(e))
            return
        self._check_path = path
        self.panel.start(command_for("camera-check", ["--config", path]))

    def draw_rois(self) -> None:
        try:
            path = self._write_working_config("rois")
        except ValueError as e:
            QMessageBox.warning(self, "Check the form", str(e))
            return
        self.panel.start(command_for("draw-rois", ["--config", path]))

    def start_session(self) -> None:
        mouse = self.mouse_id.text().strip()
        if not mouse:
            QMessageBox.warning(self, "Mouse ID", "Enter the mouse ID before starting a session.")
            return
        try:
            path = self._write_working_config(f"mouse{mouse}")
        except ValueError as e:
            QMessageBox.warning(self, "Check the form", str(e))
            return
        args = ["--config", path, "--mouse-id", mouse,
                "--ear-mark", self.ear_mark.text().strip(),
                "--birth-date", self.birth_date.text().strip(),
                "--sex", self.sex.text().strip()]
        self.panel.start(command_for("auditory", args))
