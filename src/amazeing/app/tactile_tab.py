"""Tactile paradigm tab: launch the servo-grating maze script as it is."""

from __future__ import annotations

import os

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from amazeing.app.launcher import command_for
from amazeing.app.process_panel import ProcessPanel


class TactileTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        import amazeing.simplermaze as sm
        self.pkg_dir = os.path.dirname(os.path.abspath(sm.__file__))

        intro = QLabel(
            "The tactile paradigm runs the original script unchanged. Its settings "
            "(serial port, testing flag, video recording) are the flags at the top of "
            "simplerCode.py, and the trial structure comes from grating_maps.csv and "
            "reward_sequences.csv in the same folder. The script asks for the animal ID, "
            "session ID and phase on the console: answer in the line below the output.")
        intro.setWordWrap(True)

        b_open = QPushButton("Open script and CSV folder")
        b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.pkg_dir)))
        b_start = QPushButton("Start tactile session")
        b_start.setStyleSheet("font-weight: bold;")
        b_start.clicked.connect(lambda: self.panel.start(command_for("tactile", [])))
        b_segments = QPushButton("Cut per-trial video segments...")
        b_segments.clicked.connect(lambda: self.panel.start(command_for("tactile-segments", [])))
        row = QHBoxLayout()
        for b in (b_start, b_segments, b_open):
            row.addWidget(b)
        row.addStretch(1)

        self.panel = ProcessPanel()
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(row)
        lay.addWidget(self.panel, 1)
