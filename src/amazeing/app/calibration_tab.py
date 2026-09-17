"""Speaker calibration tab: edit the frequency / attenuation table.

The table is the CSV that ``Audio`` loads (columns ``Frequency_kHz`` and
``Attenuation_dB``). Enter the speaker's frequency response, either from
the manufacturer's datasheet or from your own microphone measurement, save
it, and point the session config at the saved file.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from amazeing.auditory.config import ExperimentConfig

COLUMNS = ["Frequency_kHz", "Attenuation_dB"]


class CalibrationTab(QWidget):
    saved = Signal(str)     # path of the CSV just saved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = ExperimentConfig().calibration_gain_path

        intro = QLabel(
            "Speaker frequency response. Attenuation is in dB relative to the flattest part "
            "of the curve (0 dB); negative values mean the speaker is quieter at that frequency. "
            "The session boosts each tone by the attenuation at its frequency (interpolated), "
            "so a curve measured on your own speaker gives equal loudness across tones.")
        intro.setWordWrap(True)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)

        b_load = QPushButton("Load CSV..."); b_load.clicked.connect(self.load_dialog)
        b_save = QPushButton("Save"); b_save.clicked.connect(self.save)
        b_save_as = QPushButton("Save as..."); b_save_as.clicked.connect(self.save_as)
        b_add = QPushButton("Add row"); b_add.clicked.connect(lambda: self._add_row("", ""))
        b_del = QPushButton("Remove selected"); b_del.clicked.connect(self._remove_rows)
        b_plot = QPushButton("Plot"); b_plot.clicked.connect(self.plot)
        row = QHBoxLayout()
        for b in (b_load, b_save, b_save_as, b_add, b_del, b_plot):
            row.addWidget(b)
        row.addStretch(1)

        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(row)
        lay.addWidget(self.path_label)
        lay.addWidget(self.table, 2)
        lay.addWidget(self.plot_widget, 2)

        self.load(self.path)

    # -- data -------------------------------------------------------------------
    def rows(self) -> List[Tuple[float, float]]:
        out = []
        for r in range(self.table.rowCount()):
            vals = []
            for c in range(2):
                item = self.table.item(r, c)
                txt = item.text().strip() if item else ""
                if txt == "":
                    vals = None
                    break
                try:
                    vals.append(float(txt))
                except ValueError:
                    raise ValueError(f"Row {r + 1}: {txt!r} is not a number")
            if vals:
                out.append((vals[0], vals[1]))
        out.sort()
        return out

    def _add_row(self, f, a) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(f)))
        self.table.setItem(r, 1, QTableWidgetItem(str(a)))

    def _remove_rows(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def load(self, path: str) -> None:
        try:
            df = pd.read_csv(path)
            if not set(COLUMNS).issubset(df.columns):
                raise ValueError(f"CSV must have columns {COLUMNS}")
        except Exception as e:
            QMessageBox.critical(self, "Cannot load calibration CSV", f"{path}\n\n{e}")
            return
        self.table.setRowCount(0)
        for _, r in df.iterrows():
            self._add_row(r[COLUMNS[0]], r[COLUMNS[1]])
        self.path = path
        self.path_label.setText(f"File: {path}")
        self.plot()

    def load_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load calibration CSV", os.path.dirname(self.path), "CSV (*.csv)")
        if p:
            self.load(p)

    def save(self) -> None:
        self._save_to(self.path)

    def save_as(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save calibration CSV", self.path, "CSV (*.csv)")
        if p:
            self._save_to(p)

    def _save_to(self, path: str) -> None:
        try:
            rows = self.rows()
        except ValueError as e:
            QMessageBox.warning(self, "Check the table", str(e))
            return
        if len(rows) < 2:
            QMessageBox.warning(self, "Check the table", "At least two rows are needed to interpolate.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
        self.path = path
        self.path_label.setText(f"File: {path} (saved)")
        self.saved.emit(path)

    # -- plot --------------------------------------------------------------------
    def plot(self) -> None:
        try:
            rows = self.rows()
        except ValueError as e:
            QMessageBox.warning(self, "Check the table", str(e))
            return
        while self.plot_layout.count():
            w = self.plot_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        if len(rows) < 2:
            return
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as e:  # pragma: no cover - matplotlib without Qt backend
            self.plot_layout.addWidget(QLabel(f"Plot unavailable: {e}"))
            return
        fig = Figure(figsize=(5, 2.6), tight_layout=True)
        ax = fig.add_subplot(111)
        xs, ys = zip(*rows)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Attenuation (dB)")
        ax.grid(True, alpha=0.3)
        self.plot_layout.addWidget(FigureCanvasQTAgg(fig))
