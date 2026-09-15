"""Grammar training tab: continuous melody playback in the home cages."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget, QLabel)

from amazeing.app.widgets import PathPicker
from amazeing.app.launcher import command_for
from amazeing.app.process_panel import ProcessPanel


class GrammarTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        intro = QLabel(
            "Training days: play one grammar from one speaker to every cage in the room. "
            "This is the amaze-grammar command; test days in the maze are run from the "
            "Auditory session tab with experiment mode 'grammar'.")
        intro.setWordWrap(True)

        box = QGroupBox("Training session")
        f = QFormLayout(box)
        self.grammar = QComboBox(); self.grammar.addItems(["A", "B"])
        f.addRow("Grammar to play today", self.grammar)
        self.hours = QSpinBox(); self.hours.setRange(1, 24 * 60); self.hours.setValue(240)
        f.addRow("Duration (minutes)", self.hours)
        self.cages = QLineEdit(); self.cages.setPlaceholderText("bookkeeping only, e.g. 6224_EE,6225_SC")
        f.addRow("Cage IDs", self.cages)
        self.device = QSpinBox(); self.device.setRange(0, 64); self.device.setValue(3)
        f.addRow("Audio output device index", self.device)
        self.seed = QLineEdit(); self.seed.setPlaceholderText("blank = random")
        f.addRow("Random seed", self.seed)
        self.out_dir = PathPicker("dir"); self.out_dir.setText("./sessions")
        f.addRow("Log folder", self.out_dir)
        self.dry = QCheckBox("Dry run (no audio, log only)")
        f.addRow("", self.dry)

        b_start = QPushButton("Start training playback")
        b_start.setStyleSheet("font-weight: bold;")
        b_start.clicked.connect(self.start)
        self.panel = ProcessPanel(show_stdin=False)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(box)
        lay.addWidget(b_start)
        lay.addWidget(self.panel, 1)

    def start(self) -> None:
        args = ["--mode", "training", "--grammar", self.grammar.currentText(),
                "--duration-seconds", str(self.hours.value() * 60),
                "--device-id", str(self.device.value()),
                "--output-dir", self.out_dir.text() or "./sessions"]
        if self.cages.text().strip():
            args += ["--cage-ids", self.cages.text().strip()]
        if self.seed.text().strip():
            args += ["--seed", self.seed.text().strip()]
        if self.dry.isChecked():
            args.append("--dry-run")
        self.panel.start(command_for("grammar", args))
