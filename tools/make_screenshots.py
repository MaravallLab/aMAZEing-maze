#!/usr/bin/env python3
"""Regenerate the application screenshots used in the documentation.

Run this whenever the interface changes, so the pictures in the tutorials
match what a reader will actually see:

    python tools/make_screenshots.py

Images are written to docs/images/. The script drives the real application
offscreen, so it needs no display and starts no experiment: it only builds
windows, sets a few fields, and grabs the result.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "docs" / "images"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output folder (default: {DEFAULT_OUT})")
    # Rendered larger than the site shows them, so the documentation displays a
    # downscaled copy and the interface labels stay sharp on a high-resolution
    # monitor. Below about 2000 wide the small print goes soft.
    ap.add_argument("--width", type=int, default=2200)
    ap.add_argument("--height", type=int, default=1360)
    ap.add_argument("--show", action="store_true",
                    help="use a real window instead of the offscreen renderer")
    args = ap.parse_args(argv)

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        # Qt no longer ships fonts, and the offscreen renderer finds none on its
        # own: without this every label is grabbed as a row of empty boxes.
        if "QT_QPA_FONTDIR" not in os.environ:
            for candidate in (r"C:\Windows\Fonts", "/usr/share/fonts",
                              "/System/Library/Fonts"):
                if os.path.isdir(candidate):
                    os.environ["QT_QPA_FONTDIR"] = candidate
                    break
            else:
                print("No font directory found; the text in these images will be "
                      "boxes. Set QT_QPA_FONTDIR, or pass --show.", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from amazeing.app.main_window import MainWindow

    app = QApplication([])
    win = MainWindow()
    win.resize(args.width, args.height)
    win.show()
    tabs = win.centralWidget()
    session, waveform, form = win.session, win.session.waveform, win.session.form

    def grab(name: str) -> None:
        app.processEvents()
        path = args.out / f"{name}.png"
        win.grab().save(str(path))
        print(f"  {path.relative_to(REPO)}")

    # Each step runs after a pause so plots and layouts have settled.
    steps = []

    def add(fn):
        steps.append(fn)
        return fn

    @add
    def _session_tones():
        # Pure tones, eight arms: the configuration the first-session tutorial
        # walks through, and one that previews without anything left to fill in.
        tabs.setCurrentIndex(0)
        form.set_mode("simple_smooth")
        form.w["rois_number"].setValue(8)
        session.mouse_id.setText("6224")
        waveform.refresh()

    @add
    def _shot_session():
        grab("app-session")

    @add
    def _detail_view():
        form.set_mode("temporal_envelope_modulation")
        waveform.view_combo.setCurrentText("One arm in detail")
        waveform.refresh()

    @add
    def _shot_detail():
        grab("app-waveform-detail")

    @add
    def _compare_view():
        waveform.view_combo.setCurrentText("Compare all arms")
        waveform.refresh()

    @add
    def _shot_compare():
        grab("app-waveform-compare")

    @add
    def _grammar():
        form.set_mode("grammar")
        form.w["grammar_mode"].setCurrentText("test")
        waveform.refresh()

    @add
    def _shot_grammar():
        grab("app-grammar")

    @add
    def _help_open():
        form.set_mode("simple_intervals")
        waveform.view_combo.setCurrentText("One arm in detail")
        form.help_widgets["experiment"].button.setChecked(True)
        waveform.refresh()

    @add
    def _shot_help():
        grab("app-help")

    @add
    def _calibration():
        form.help_widgets["experiment"].button.setChecked(False)
        tabs.setCurrentIndex(1)

    @add
    def _shot_calibration():
        grab("app-calibration")

    @add
    def _analysis():
        tabs.setCurrentIndex(2)

    @add
    def _shot_analysis():
        grab("app-analysis")

    state = {"i": 0}

    def tick():
        i = state["i"]
        if i >= len(steps):
            app.quit()
            return
        steps[i]()
        state["i"] += 1
        # Screen grabs need less settling time than a redraw.
        QTimer.singleShot(120 if steps[i].__name__.startswith("_shot") else 900, tick)

    print(f"Writing screenshots to {args.out}")
    QTimer.singleShot(600, tick)
    app.exec()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
