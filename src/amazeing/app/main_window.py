"""Main window: one tab per task."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from amazeing import __version__
from amazeing.app.analysis_tab import AnalysisTab
from amazeing.app.calibration_tab import CalibrationTab
from amazeing.app.grammar_tab import GrammarTab
from amazeing.app.session_tab import SessionTab
from amazeing.app.tactile_tab import TactileTab


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"aMAZEing maze {__version__}")
        self.resize(1280, 820)

        self.session = SessionTab()
        self.calibration = CalibrationTab()
        self.analysis = AnalysisTab()
        self.grammar = GrammarTab()
        self.tactile = TactileTab()

        tabs = QTabWidget()
        tabs.addTab(self.session, "Auditory session")
        tabs.addTab(self.calibration, "Speaker calibration")
        tabs.addTab(self.analysis, "Analysis")
        tabs.addTab(self.grammar, "Grammar training")
        tabs.addTab(self.tactile, "Tactile session")
        self.setCentralWidget(tabs)

        # A calibration file saved in its tab becomes the session's calibration file.
        self.calibration.saved.connect(self.session.form.w["calibration_gain_path"].setText)

        file_menu = self.menuBar().addMenu("&File")
        a_load = QAction("Load session config...", self); a_load.triggered.connect(self.session.load_config_dialog)
        a_save = QAction("Save session config as...", self); a_save.triggered.connect(self.session.save_config_dialog)
        a_quit = QAction("Quit", self); a_quit.triggered.connect(self.close)
        for a in (a_load, a_save):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(a_quit)
        help_menu = self.menuBar().addMenu("&Help")
        a_about = QAction("About", self); a_about.triggered.connect(self.about)
        help_menu.addAction(a_about)

    def about(self) -> None:
        QMessageBox.about(
            self, "aMAZEing maze",
            f"aMAZEing maze {__version__}\n\n"
            "Open-source platform for auditory and tactile rodent maze experiments.\n"
            "Maravall Lab, University of Sussex. GPL-3.0-or-later.\n\n"
            "This interface writes a session config file and runs the same command-line "
            "tools a script user runs; every session it starts is reproducible from that file.")

    def closeEvent(self, ev) -> None:
        running = [t for t in (self.session, self.analysis, self.grammar, self.tactile)
                   if t.panel.is_running()]
        if running:
            ans = QMessageBox.question(self, "Quit", "A process is still running. Stop it and quit?")
            if ans != QMessageBox.Yes:
                ev.ignore()
                return
            for t in running:
                t.panel.stop()
        ev.accept()
