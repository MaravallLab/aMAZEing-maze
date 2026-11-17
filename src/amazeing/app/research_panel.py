"""The Research figures section of the Analysis tab.

Runs the thesis analyses in ``analysis/auditory/`` (see ``pipelines.py``) on
a data folder: a checklist grouped by experiment in run order, a status mark
per script, the shared options, a queue that runs the selection with the
scripts it depends on, and a browser over the chosen script's output.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Mapping, Optional, Set

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from amazeing.app import pipelines as pl
from amazeing.app.help_text import ANALYSIS_HELP
from amazeing.app.launcher import command_for_script, script_cwd
from amazeing.app.palette import STATUS_COLORS
from amazeing.app.process_panel import CommandQueue, ProcessPanel
from amazeing.app.widgets import CollapsibleHelp, PathPicker

FIGURE_TYPES = ("*.png", "*.pdf", "*.html", "*.txt")
MAX_FILES = 400
ID_ROLE = Qt.UserRole


class ResearchFigures(QWidget):

    def __init__(self, parent=None, env: Optional[Mapping[str, bool]] = None,
                 scripts: Optional[str] = None):
        super().__init__(parent)
        self.env = dict(env) if env is not None else pl.detect_environment()
        self.scripts = scripts if scripts is not None else pl.scripts_dir()
        self.layout_info = pl.Layout()
        self.options: Dict[str, Set[str]] = {}
        self.items: Dict[str, QTreeWidgetItem] = {}
        self._groups: Dict[str, QTreeWidgetItem] = {}
        self._syncing = False
        self._preview_path: Optional[str] = None
        self._out: Optional[str] = None

        intro = QLabel(
            "Run the thesis analyses on a data folder. Pick the data root or one "
            "experiment's folder, tick the scripts, and press Run selected; the scripts "
            "they need run first.")
        intro.setWordWrap(True)
        help_box = CollapsibleHelp(ANALYSIS_HELP["research"])

        self.folder = PathPicker("dir")
        self.folder.edit.setPlaceholderText("Data root, or 8_arms_w_voc, grammar, dopamine recordings")
        self.folder.edit.textChanged.connect(self._folder_changed)
        self.found = QLabel()
        self.found.setWordWrap(True)
        self.found.setTextInteractionFlags(Qt.TextSelectableByMouse)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Data folder"))
        frow.addWidget(self.folder, 1)

        # -- checklist and options --------------------------------------------
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Script", "Status"])
        self.tree.setColumnWidth(0, 270)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._show_details())
        self._build_tree()

        self.shuffled = QCheckBox("Blocks 2 to 4 only")
        self.shuffled.setToolTip("Adds --shuffled-only to exp1_stimulus_pi, exp1_complexity "
                                 "and exp1_lmm; they write to folders ending in _shuffled.")
        self.shuffled.toggled.connect(lambda _on: self.refresh())
        self.thesis_on = QCheckBox("Also write thesis-style figures to")
        self.thesis_dir = PathPicker("dir")
        self.thesis_dir.setEnabled(False)
        self.thesis_on.toggled.connect(self.thesis_dir.setEnabled)
        self.thesis_on.toggled.connect(lambda _on: self._show_details())
        self.thesis_dir.edit.textChanged.connect(lambda _t: self._show_details())
        self.rerun = QCheckBox("Re-run the scripts they need even when already done")
        self.run_button = QPushButton("Run selected")
        self.run_button.clicked.connect(self.run_selected)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)

        opts = QGridLayout()
        opts.addWidget(self.shuffled, 0, 0, 1, 2)
        opts.addWidget(self.thesis_on, 1, 0)
        opts.addWidget(self.thesis_dir, 1, 1)
        opts.addWidget(self.rerun, 2, 0, 1, 2)
        brow = QHBoxLayout()
        brow.addWidget(self.run_button)
        brow.addWidget(refresh)
        brow.addStretch(1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.tree, 1)
        ll.addLayout(opts)
        ll.addLayout(brow)

        # -- details and figures -----------------------------------------------
        self.detail = QLabel("Select a script to see what it does and what it wrote.")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.option_box = QVBoxLayout()
        self.option_box.setContentsMargins(0, 0, 0, 0)

        self.fig_list = QListWidget()
        self.fig_list.currentItemChanged.connect(lambda cur, _prev: self._preview(cur))
        self.fig_list.itemDoubleClicked.connect(self._open_item)
        self.preview = QLabel("Figures appear here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(240, 180)
        self.open_folder = QPushButton("Open folder")
        self.open_folder.clicked.connect(self._open_output)
        self.open_folder.setEnabled(False)

        figs = QSplitter(Qt.Horizontal)
        flist = QWidget()
        fl = QVBoxLayout(flist)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.fig_list, 1)
        fl.addWidget(self.open_folder)
        figs.addWidget(flist)
        figs.addWidget(self.preview)
        figs.setStretchFactor(0, 1)
        figs.setStretchFactor(1, 2)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.detail)
        tl.addLayout(self.option_box)
        tl.addWidget(figs, 1)

        self.panel = ProcessPanel(show_stdin=False)
        self.queue = CommandQueue(self.panel, self)
        self.queue.step_finished.connect(lambda _i, _c: self.refresh())
        self.queue.all_finished.connect(self._run_ended)

        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(self.panel)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(help_box)
        lay.addLayout(frow)
        lay.addWidget(self.found)
        if self.scripts is None:
            warn = QLabel("The research scripts (analysis/auditory) were not found. Run the "
                          "app from a clone of the repository, or set AMAZEING_ANALYSIS_DIR.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {STATUS_COLORS['unavailable']}")
            lay.addWidget(warn)
            self.run_button.setEnabled(False)
        lay.addWidget(split, 1)
        self.refresh()

    # -- the checklist ---------------------------------------------------------
    def _build_tree(self) -> None:
        self.tree.blockSignals(True)
        for group in pl.GROUPS:
            g = QTreeWidgetItem([group, ""])
            g.setFlags(g.flags() | Qt.ItemIsUserCheckable)
            g.setCheckState(0, Qt.Unchecked)
            self.tree.addTopLevelItem(g)
            self._groups[group] = g
            for p in pl.in_group(group):
                it = QTreeWidgetItem([p.title, ""])
                it.setData(0, ID_ROLE, p.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Unchecked)
                g.addChild(it)
                self.items[p.id] = it
            g.setExpanded(True)
        self.tree.blockSignals(False)

    def folders(self) -> Dict[str, str]:
        f = dict(self.layout_info.folders)
        if self.thesis_on.isChecked() and self.thesis_dir.text():
            f["thesis"] = self.thesis_dir.text()
        return f

    def _variant(self, p: pl.Pipeline) -> bool:
        return self.shuffled.isChecked() and p.shuffled

    def unavailable(self, p: pl.Pipeline) -> List[str]:
        reasons = pl.missing(p, self.layout_info.folders, self.env, self.options.get(p.id, ()))
        if self.scripts is not None and not os.path.isfile(pl.script_path(p, self.scripts)):
            reasons.append(f"script not found: {p.script}")
        return reasons

    def status_of(self, p: pl.Pipeline) -> str:
        if self.unavailable(p):
            return "unavailable"
        if pl.is_done(p, self.folders(), self._variant(p)):
            return "done"
        return "missing"

    def refresh(self) -> None:
        """Recompute every status mark and tooltip, then the details pane."""
        self.tree.blockSignals(True)
        for p in pl.PIPELINES:
            it = self.items[p.id]
            reasons = self.unavailable(p)
            status = self.status_of(p)
            if p.output is None and status != "unavailable":
                text = "thesis figures only" if p.thesis_required else "prints only"
            else:
                text = {"done": "done", "missing": "not run",
                        "unavailable": "cannot run"}[status]
            it.setText(1, text)
            colour = QBrush(QColor(STATUS_COLORS[status]))
            it.setForeground(1, colour)
            it.setForeground(0, colour if status == "unavailable" else QBrush())
            tip = f"<b>{p.script}</b><br>{p.question}"
            if p.after:
                tip += "<br>Runs after: " + ", ".join(p.after)
            if reasons:
                tip += "<br><br><b>Cannot run here:</b><br>" + "<br>".join(reasons)
            elif p.note:
                tip += "<br><br>" + p.note
            it.setToolTip(0, tip)
            it.setToolTip(1, tip)
            if status == "unavailable":
                it.setCheckState(0, Qt.Unchecked)
                it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable)
            else:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        for g in self._groups.values():
            self._sync_group(g)
        self.tree.blockSignals(False)
        self._show_details()

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._syncing:
            return
        self._syncing = True
        try:
            if item.parent() is None:          # a group: tick or untick what can run
                state = item.checkState(0)
                if state == Qt.PartiallyChecked:
                    state = Qt.Checked
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child.flags() & Qt.ItemIsUserCheckable:
                        child.setCheckState(0, state)
                self._sync_group(item)
            else:
                self._sync_group(item.parent())
        finally:
            self._syncing = False

    def _sync_group(self, g: QTreeWidgetItem) -> None:
        kids = [g.child(i) for i in range(g.childCount())
                if g.child(i).flags() & Qt.ItemIsUserCheckable]
        ticked = sum(k.checkState(0) == Qt.Checked for k in kids)
        state = (Qt.Unchecked if ticked == 0 else
                 Qt.Checked if ticked == len(kids) else Qt.PartiallyChecked)
        was = self._syncing
        self._syncing = True
        g.setCheckState(0, state)
        self._syncing = was

    def selected(self) -> List[str]:
        return [p.id for p in pl.PIPELINES
                if self.items[p.id].checkState(0) == Qt.Checked]

    def set_selected(self, ids) -> None:
        for p in pl.PIPELINES:
            self.items[p.id].setCheckState(0, Qt.Checked if p.id in set(ids) else Qt.Unchecked)

    def current(self) -> Optional[pl.Pipeline]:
        it = self.tree.currentItem()
        pid = it.data(0, ID_ROLE) if it is not None else None
        return pl.BY_ID.get(pid) if pid else None

    # -- folder ------------------------------------------------------------------
    def _folder_changed(self, text: str) -> None:
        self.layout_info = pl.resolve_layout(text.strip())
        if text.strip():
            kind = {"root": "the data root", "exp1": "Experiment 1", "grammar": "Experiment 2",
                    "photometry": "photometry", "unknown": "not recognised"}[self.layout_info.kind]
            self.found.setText(f"This folder looks like {kind}.\n" + self.layout_info.describe())
        else:
            self.found.setText("")
        self.refresh()

    # -- running -------------------------------------------------------------------
    def build_run(self):
        """(items, problems): the queue items for the selection, or why it cannot run."""
        sel = self.selected()
        if not sel:
            return [], ["Tick at least one script."]
        folders = self.folders()
        thesis = folders.get("thesis")
        problems: List[str] = []
        if self.thesis_on.isChecked() and not (thesis and os.path.isdir(thesis)):
            problems.append("Choose an existing folder for the thesis-style figures.")
        steps = pl.plan(sel, self.shuffled.isChecked(),
                        done=lambda p, s: pl.is_done(p, folders, s),
                        rerun_dependencies=self.rerun.isChecked())
        items = []
        for st in steps:
            p = st.pipeline
            reasons = self.unavailable(p)
            if p.thesis_required and not thesis:
                reasons.append("writes thesis figures only: tick Also write thesis-style "
                               "figures and choose the folder")
            if reasons:
                problems.append(f"{st.label()}: " + "; ".join(reasons))
                continue
            path = pl.script_path(p, self.scripts)
            args = pl.build_args(p, folders, st.shuffled, thesis, self.options.get(p.id, ()))
            items.append((st.label(), command_for_script(path, args), script_cwd(path)))
        return items, problems

    def run_selected(self) -> None:
        if self.queue.is_running() or self.panel.is_running():
            QMessageBox.information(self, "Running", "A run is in progress; stop it first.")
            return
        items, problems = self.build_run()
        if problems:
            QMessageBox.warning(self, "Cannot run", "\n\n".join(problems))
            return
        self.panel.append("Run plan:\n" + "".join(f"  {i + 1}. {label}\n"
                                                  for i, (label, _c, _w) in enumerate(items)))
        self.run_button.setEnabled(False)
        self.queue.run(items)

    def _run_ended(self, _ok: bool) -> None:
        self.run_button.setEnabled(self.scripts is not None)
        self.refresh()

    # -- details and figures ---------------------------------------------------------
    def _show_details(self) -> None:
        p = self.current()
        self._clear_options()
        if p is None:
            self.detail.setText("Select a script to see what it does and what it wrote.")
            self._list_figures(None)
            return
        out = pl.output_dir(p, self.folders(), self._variant(p))
        lines = [f"<b>{p.title}</b> &nbsp; <code>{p.script}</code>", p.question]
        if p.after:
            lines.append("Runs after: " + ", ".join(p.after))
        if p.needs:
            lines.append("Needs: " + ", ".join(pl.NEED_LABELS[n] for n in p.needs))
        if p.output is not None:
            where = out or "(choose a data folder)"
        elif p.thesis_required:
            where = out or "the thesis figures folder"
        else:
            where = "the console only"
        lines.append("Writes to: " + where)
        reasons = self.unavailable(p)
        if reasons:
            lines.append(f"<span style='color:{STATUS_COLORS['unavailable']}'>Cannot run here: "
                         + "; ".join(reasons) + "</span>")
        elif p.note:
            lines.append(p.note)
        self.detail.setText("<br>".join(lines))
        for o in p.options:
            box = QCheckBox(o.label)
            box.setToolTip(o.tip)
            box.setChecked(o.id in self.options.get(p.id, set()))
            box.toggled.connect(lambda on, pid=p.id, oid=o.id: self._set_option(pid, oid, on))
            self.option_box.addWidget(box)
        self._list_figures(out)

    def _clear_options(self) -> None:
        while self.option_box.count():
            w = self.option_box.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _set_option(self, pid: str, oid: str, on: bool) -> None:
        chosen = self.options.setdefault(pid, set())
        if on:
            chosen.add(oid)
        else:
            chosen.discard(oid)
        self.refresh()

    def _list_figures(self, folder: Optional[str]) -> None:
        self._out = folder if folder and os.path.isdir(folder) else None
        self.open_folder.setEnabled(self._out is not None)
        keep = self.fig_list.currentItem().data(ID_ROLE) if self.fig_list.currentItem() else None
        self.fig_list.blockSignals(True)
        self.fig_list.clear()
        files: List[str] = []
        if self._out:
            for pattern in FIGURE_TYPES:
                files += glob.glob(os.path.join(glob.escape(self._out), "**", pattern),
                                   recursive=True)
            files = sorted(files)[:MAX_FILES]
        for f in files:
            item = QListWidgetItem(os.path.relpath(f, self._out))
            item.setData(ID_ROLE, f)
            item.setToolTip(f if f.lower().endswith(".png")
                            else f + "\nDouble-click to open it in its own program")
            self.fig_list.addItem(item)
        self.fig_list.blockSignals(False)
        row = next((i for i, f in enumerate(files) if f == keep), None)
        if row is None:
            row = next((i for i, f in enumerate(files) if f.lower().endswith(".png")), None)
        if row is not None:
            self.fig_list.setCurrentRow(row)
        else:
            self._preview(None)

    def _preview(self, item: Optional[QListWidgetItem]) -> None:
        path = item.data(ID_ROLE) if item is not None else None
        self._preview_path = path
        if not path:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No figures yet" if self._out else "Figures appear here")
            return
        if not path.lower().endswith(".png"):
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Double-click to open this file in its own program")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.preview.setText("Cannot display this file")
            return
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation))

    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(ID_ROLE)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_output(self) -> None:
        if self._out:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._out))

    def resizeEvent(self, ev) -> None:  # keep the preview scaled
        super().resizeEvent(ev)
        if self._preview_path:
            self._preview(self.fig_list.currentItem())
