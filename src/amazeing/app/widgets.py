"""Small shared widgets and helpers for the interface."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (QAbstractSpinBox, QComboBox, QFileDialog,
                               QHBoxLayout, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QWidget)


class NoScrollFilter(QObject):
    """Ignore mouse-wheel events on a control unless it has keyboard focus.

    Without this, scrolling past a form silently changes any spin box or drop
    down the pointer happens to cross, which is an easy way to start a session
    with a value you did not choose.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


_NO_SCROLL = NoScrollFilter()


def protect_from_scroll(widget: QWidget) -> None:
    """Apply the no-scroll rule to every spin box and combo inside ``widget``."""
    for child in widget.findChildren(QAbstractSpinBox):
        child.installEventFilter(_NO_SCROLL)
        child.setFocusPolicy(Qt.StrongFocus)
    for child in widget.findChildren(QComboBox):
        child.installEventFilter(_NO_SCROLL)
        child.setFocusPolicy(Qt.StrongFocus)


class PathPicker(QWidget):
    """Line edit plus a Browse button (file, save-file or directory)."""

    def __init__(self, mode: str = "file", filter_: str = "", parent=None):
        super().__init__(parent)
        self.mode, self.filter = mode, filter_
        self.edit = QLineEdit()
        btn = QPushButton("Browse...")
        btn.setMaximumWidth(90)
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


class CheckableList(QListWidget):
    """A tick-box list that remembers the order things were ticked in.

    Order matters for the interval lists: the stimulus factory assigns
    intervals to arms in list order, so a round trip through this widget must
    not silently re-sort them. Items loaded with ``set_checked`` keep the order
    given; items ticked afterwards are appended.
    """

    def __init__(self, options: List[str], parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.NoSelection)
        for name in options:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.addItem(item)
        self.setMaximumHeight(130)
        self._order: List[str] = []
        self.itemChanged.connect(self._track_order)

    def _track_order(self, item: QListWidgetItem) -> None:
        name = item.text()
        if item.checkState() == Qt.Checked:
            if name not in self._order:
                self._order.append(name)
        elif name in self._order:
            self._order.remove(name)

    def checked(self) -> List[str]:
        ticked = {self.item(i).text() for i in range(self.count())
                  if self.item(i).checkState() == Qt.Checked}
        ordered = [n for n in self._order if n in ticked]
        # anything ticked without passing through _track_order goes last
        ordered += [self.item(i).text() for i in range(self.count())
                    if self.item(i).text() in ticked and self.item(i).text() not in ordered]
        return ordered

    def set_checked(self, names: List[str]) -> None:
        names = list(names or [])
        wanted = set(names)
        self.blockSignals(True)
        for i in range(self.count()):
            item = self.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        self.blockSignals(False)
        self._order = names


def parse_float_list(text: str, label: str) -> List[float]:
    """Parse '10000, 20000' into [10000.0, 20000.0]; empty text gives []."""
    text = (text or "").strip().strip("[]")
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


def parse_str_list(text: str) -> List[str]:
    text = (text or "").strip().strip("[]")
    if not text:
        return []
    return [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]


def format_numbers(values) -> str:
    def fmt(x):
        x = float(x)
        return str(int(x)) if x.is_integer() else str(x)
    return ", ".join(fmt(v) for v in (values or []))
