"""Analysis tab: run the figure and CSV generators on recorded sessions."""

from __future__ import annotations

import glob
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMessageBox,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from amazeing.app.config_form import PathPicker
from amazeing.app.launcher import command_for
from amazeing.app.process_panel import ProcessPanel


class AnalysisTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        intro = QLabel(
            "Pick a folder and run the analysis that matches it. A session folder "
            "(time_..._mouseN) gets per-session figures. A day folder (all mice of one day) "
            "or the experiment folder (all days) gets the cross-session summaries.")
        intro.setWordWrap(True)
        self.folder = PathPicker("dir")

        b_session = QPushButton("Per-session figures")
        b_day = QPushButton("Summary: this day")
        b_all = QPushButton("Summary: all days")
        b_csv = QPushButton("Summary CSVs")
        b_session.clicked.connect(lambda: self._run("analyse-session", [self._folder()]))
        b_day.clicked.connect(lambda: self._run("summary", ["--day", self._folder()]))
        b_all.clicked.connect(lambda: self._run("summary", ["--all", self._folder()]))
        b_csv.clicked.connect(lambda: self._run("summary-csv", ["--all", self._folder()]))
        row = QHBoxLayout()
        for b in (b_session, b_day, b_all, b_csv):
            row.addWidget(b)
        row.addStretch(1)

        self.panel = ProcessPanel(show_stdin=False)
        self.panel.finished.connect(lambda _c: self.refresh_figures())

        self.fig_list = QListWidget()
        self.fig_list.currentTextChanged.connect(self._show_figure)
        self.preview = QLabel("Figures in the folder appear here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        refresh = QPushButton("Refresh figures")
        refresh.clicked.connect(self.refresh_figures)

        figs = QWidget()
        fl = QVBoxLayout(figs)
        fl.addWidget(refresh)
        fl.addWidget(self.fig_list, 1)
        fl.addWidget(self.preview, 3)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.panel)
        split.addWidget(figs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(self.folder)
        lay.addLayout(row)
        lay.addWidget(split, 1)
        self.folder.edit.editingFinished.connect(self.refresh_figures)

    def _folder(self) -> str:
        return self.folder.text()

    def _run(self, entry: str, args) -> None:
        if not self._folder() or not os.path.isdir(self._folder()):
            QMessageBox.warning(self, "Folder", "Choose an existing folder first.")
            return
        self.panel.start(command_for(entry, args))

    def refresh_figures(self) -> None:
        self.fig_list.clear()
        folder = self._folder()
        if not folder or not os.path.isdir(folder):
            return
        pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
        for p in pngs:
            self.fig_list.addItem(p)
        if pngs:
            self.fig_list.setCurrentRow(0)

    def _show_figure(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.preview.setText("Cannot display this file")
            return
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._current = path

    def resizeEvent(self, ev) -> None:  # keep the preview scaled
        super().resizeEvent(ev)
        cur = self.fig_list.currentItem()
        if cur:
            self._show_figure(cur.text())
