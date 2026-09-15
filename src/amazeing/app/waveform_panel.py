"""Live view of the stimulus being designed: waveform, envelope and spectrum.

The sounds shown here come from the real trial factory with a shortened
duration, so the shape you see is the shape that will play. Nothing touches
the sound card until you press Play.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from amazeing.app.stimulus_preview import StimulusPreview, preview_stimuli, spectrum

# How much of the waveform to draw, in milliseconds. Long enough to show a
# few cycles of an amplitude envelope, short enough that individual carrier
# cycles are still visible when zoomed.
WAVE_WINDOW_MS = 60.0
PREVIEW_SECONDS = 0.3


class WaveformPanel(QWidget):
    """Shows one arm's stimulus; refreshed on demand from a ConfigForm."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._previews: List[StimulusPreview] = []
        self._get_config = None

        self.arm_combo = QComboBox()
        self.arm_combo.currentIndexChanged.connect(self._draw)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._play)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(QLabel("Arm"))
        top.addWidget(self.arm_combo, 1)
        top.addWidget(self.play_button)
        top.addWidget(self.refresh_button)

        self.status = QLabel()
        self.status.setWordWrap(True)

        self.canvas_holder = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_holder)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.status)
        lay.addWidget(self.canvas_holder, 1)

        # Coalesce bursts of edits into one redraw.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)

    def bind(self, get_config) -> None:
        """Give the panel a callable returning the current ExperimentConfig."""
        self._get_config = get_config

    def schedule_refresh(self) -> None:
        self._timer.start()

    # ------------------------------------------------------------------ data
    def refresh(self) -> None:
        if self._get_config is None:
            return
        try:
            cfg = self._get_config()
            self._previews = preview_stimuli(cfg, duration_s=PREVIEW_SECONDS)
        except Exception as e:
            self._previews = []
            self.status.setText(f"Preview unavailable: {e}")
            self._clear_canvas()
            self.arm_combo.blockSignals(True)
            self.arm_combo.clear()
            self.arm_combo.blockSignals(False)
            return

        keep = self.arm_combo.currentText()
        self.arm_combo.blockSignals(True)
        self.arm_combo.clear()
        for p in self._previews:
            self.arm_combo.addItem(f"Arm {p.arm}: {p.label}")
        self.arm_combo.blockSignals(False)
        i = self.arm_combo.findText(keep) if keep else -1
        if i < 0:
            # Land on something audible rather than on a silent control arm.
            i = next((j for j, p in enumerate(self._previews) if not p.is_silent), 0)
        self.arm_combo.setCurrentIndex(i)
        self.status.setText(f"{len(self._previews)} arms in the first active block")
        self._draw()

    def _current(self) -> Optional[StimulusPreview]:
        i = self.arm_combo.currentIndex()
        if 0 <= i < len(self._previews):
            return self._previews[i]
        return None

    # ------------------------------------------------------------------ draw
    def _clear_canvas(self) -> None:
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _draw(self) -> None:
        self._clear_canvas()
        p = self._current()
        if p is None:
            return
        if p.is_silent:
            lab = QLabel("This arm is silent.")
            lab.setAlignment(Qt.AlignCenter)
            self.canvas_layout.addWidget(lab)
            return
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as e:
            self.canvas_layout.addWidget(QLabel(f"Plot unavailable: {e}"))
            return

        fig = Figure(figsize=(5, 4.2), tight_layout=True)
        ax_w = fig.add_subplot(211)
        ax_s = fig.add_subplot(212)

        n = min(int(p.samplerate * WAVE_WINDOW_MS / 1000.0), p.wave.size)
        seg = p.wave[:n]
        t_ms = np.arange(seg.size) / p.samplerate * 1000.0
        ax_w.plot(t_ms, seg, linewidth=0.8)
        # Envelope, so amplitude modulation is visible at a glance.
        env = np.abs(seg)
        win = max(1, int(p.samplerate * 0.001))
        if env.size > win * 3:
            smooth = np.convolve(env, np.ones(win) / win, mode="same")
            ax_w.plot(t_ms, smooth, linewidth=1.2, alpha=0.9)
            ax_w.plot(t_ms, -smooth, linewidth=1.2, alpha=0.9,
                      color=ax_w.lines[-1].get_color())
        ax_w.set_xlabel("Time (ms)")
        ax_w.set_ylabel("Amplitude")
        ax_w.set_title(f"Arm {p.arm}: {p.label}", fontsize=9)
        ax_w.grid(True, alpha=0.3)
        peak = float(np.max(np.abs(p.wave)))
        ax_w.set_ylim(-max(peak, 1e-3) * 1.15, max(peak, 1e-3) * 1.15)

        freqs, mag = spectrum(p.wave, p.samplerate, max_hz=min(p.samplerate / 2, 60000))
        ax_s.plot(freqs / 1000.0, mag, linewidth=0.9)
        ax_s.set_xlabel("Frequency (kHz)")
        ax_s.set_ylabel("Relative level")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim(0, 1.05)

        self.canvas_layout.addWidget(FigureCanvasQTAgg(fig))

    # ------------------------------------------------------------------ play
    def _play(self) -> None:
        p = self._current()
        if p is None:
            return
        if p.is_silent:
            QMessageBox.information(self, "Play", "This arm is silent.")
            return
        try:
            import sounddevice as sd
            cfg = self._get_config() if self._get_config else None
            if cfg is not None:
                sd.play(p.wave, p.samplerate, device=cfg.channel_id)
            else:
                sd.play(p.wave, p.samplerate)
        except Exception as e:
            QMessageBox.warning(
                self, "Cannot play",
                f"{e}\n\nCheck the audio output device and sample rate in the form. "
                f"Ultrasonic stimuli need an interface that supports the chosen rate.")
