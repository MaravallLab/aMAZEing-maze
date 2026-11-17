"""A console panel that runs one child process and shows its output.

Every tab that launches a tool (session, tactile, analysis, grammar
training) embeds one of these. It streams stdout/stderr into a text view,
offers a line to type answers for tools that still prompt on the console
(the sequences mode, for example), and has Start/Stop control.

``CommandQueue`` runs a list of commands in one panel, one after another,
and stops at the first that fails (or is stopped).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from amazeing.app.launcher import child_environment


class ProcessPanel(QWidget):
    started = Signal()
    finished = Signal(int)          # exit code

    def __init__(self, parent: Optional[QWidget] = None, show_stdin: bool = True):
        super().__init__(parent)
        self._proc: Optional[QProcess] = None

        self.console = QPlainTextEdit(readOnly=True)
        self.console.setMaximumBlockCount(5000)
        self.console.setFont(QFont("Consolas" if _is_windows() else "Monospace", 9))

        self.status = QLabel("Idle")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.console.clear)

        top = QHBoxLayout()
        top.addWidget(self.status, 1)
        top.addWidget(self.clear_button)
        top.addWidget(self.stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.console, 1)

        self.stdin_line: Optional[QLineEdit] = None
        if show_stdin:
            self.stdin_line = QLineEdit()
            self.stdin_line.setPlaceholderText("Type an answer for a console prompt and press Enter")
            self.stdin_line.returnPressed.connect(self._send_stdin)
            self.stdin_line.setEnabled(False)
            layout.addWidget(self.stdin_line)

    # -- control ---------------------------------------------------------
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def start(self, command: List[str], cwd: Optional[str] = None) -> None:
        if self.is_running():
            self.append("A process is already running; stop it first.\n")
            return
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment()
        for k, v in child_environment().items():
            env.insert(k, v)
        self._proc.setProcessEnvironment(env)
        if cwd:
            self._proc.setWorkingDirectory(cwd)
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        self.append("$ " + " ".join(_quote(c) for c in command) + "\n")
        proc = self._proc
        proc.start(command[0], command[1:])
        if proc is not self._proc or proc.state() == QProcess.NotRunning:
            return      # it failed to start (reported by _on_error), or a queue moved on
        self.status.setText("Running")
        self.stop_button.setEnabled(True)
        if self.stdin_line is not None:
            self.stdin_line.setEnabled(True)
        self.started.emit()

    def stop(self) -> None:
        if not self.is_running():
            return
        self.append("\n[stopping...]\n")
        self._proc.terminate()
        if not self._proc.waitForFinished(3000):
            self._proc.kill()

    # -- plumbing --------------------------------------------------------
    def append(self, text: str) -> None:
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.End)

    def _read(self) -> None:
        data = bytes(self._proc.readAllStandardOutput())
        self.append(data.decode("utf-8", errors="replace"))

    def _send_stdin(self) -> None:
        if self.stdin_line is None or not self.is_running():
            return
        text = self.stdin_line.text()
        self._proc.write((text + "\n").encode("utf-8"))
        self.append(text + "\n")
        self.stdin_line.clear()

    def _on_finished(self, code: int, _status) -> None:
        self.status.setText(f"Finished (exit code {code})")
        self.stop_button.setEnabled(False)
        if self.stdin_line is not None:
            self.stdin_line.setEnabled(False)
        self.append(f"\n[process finished with exit code {code}]\n")
        self.finished.emit(code)

    def _on_error(self, err) -> None:
        self.append(f"\n[process error: {err}]\n")
        self.status.setText("Error")
        self.stop_button.setEnabled(False)
        # A program that never started emits no finished signal of its own;
        # report it as a failure so whoever waits on finished hears of it.
        if err == QProcess.FailedToStart:
            if self.stdin_line is not None:
                self.stdin_line.setEnabled(False)
            self.finished.emit(-1)


class CommandQueue(QObject):
    """Run commands one after another in a ProcessPanel.

    Each item is ``(label, command, cwd)``. The next starts when the last
    exits with code 0; a non-zero exit (a failure, or Stop) ends the run.
    ``step_finished(index, code)`` follows every command and
    ``all_finished(ok)`` ends the run.
    """

    step_started = Signal(int, str)
    step_finished = Signal(int, int)
    all_finished = Signal(bool)

    def __init__(self, panel: ProcessPanel, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.panel = panel
        self._items: List[Tuple[str, List[str], Optional[str]]] = []
        self._index = -1
        panel.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        return 0 <= self._index < len(self._items)

    def run(self, items: Sequence[Tuple[str, List[str], Optional[str]]]) -> bool:
        """Start the run; False when something is already running."""
        if self.is_running() or self.panel.is_running():
            self.panel.append("A process is already running; stop it first.\n")
            return False
        self._items = list(items)
        self._index = -1
        if not self._items:
            self.all_finished.emit(True)
            return True
        self._next()
        return True

    def _next(self) -> None:
        self._index += 1
        label, command, cwd = self._items[self._index]
        self.panel.append(f"\n=== {self._index + 1} of {len(self._items)}: {label} ===\n")
        self.step_started.emit(self._index, label)
        self.panel.start(command, cwd)

    def _on_finished(self, code: int) -> None:
        if not self.is_running():
            return
        i = self._index
        self.step_finished.emit(i, code)
        if code != 0:
            left = len(self._items) - i - 1
            if left:
                self.panel.append(f"[stopped: {self._items[i][0]} exited with code {code}; "
                                  f"{left} later step(s) not run]\n")
            self._index = -1
            self.all_finished.emit(False)
        elif i + 1 < len(self._items):
            self._next()
        else:
            self._index = -1
            self.panel.append("\n[all steps finished]\n")
            self.all_finished.emit(True)


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s else s
