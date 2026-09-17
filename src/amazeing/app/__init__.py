"""Graphical interface for the aMAZEing maze (``amaze-app``).

The app is a thin layer over the command-line tools: it edits the YAML
session config, draws ROIs, starts sessions and analyses as child
processes, and shows their output. Nothing in the experiment loop lives
here, so a session started from the app is identical to one started from a
terminal with the same config file.

The frozen executable also serves as the child-process interpreter:
``amazeing-app.exe --entry auditory --config x.yaml`` runs a session (see
``launcher.py``).
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--entry":
        from amazeing.app.launcher import run_entry
        if len(argv) < 2:
            print("--entry needs a name", file=sys.stderr)
            return 2
        return run_entry(argv[1], argv[2:])

    from PySide6.QtWidgets import QApplication
    from amazeing.app.main_window import MainWindow

    app = QApplication(argv)
    app.setApplicationName("aMAZEing maze")
    win = MainWindow()
    win.show()
    return app.exec()
