"""Live view of the stimulus being designed.

Two views: one arm in detail (waveform with its envelope, spectrum and
spectrogram) or every arm side by side. Colours come from
``amazeing.app.palette``, so a stimulus category looks the same here as it
does in the session figures.

The sounds come from the real trial factory with a shortened duration, so the
shape you see is the shape that will play. Nothing touches the sound card
until you press Play.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from amazeing.app.palette import colour_for, figure_style
from amazeing.app.stimulus_preview import StimulusPreview, preview_stimuli, spectrum

# How much of the waveform to draw, in milliseconds: long enough to show a few
# cycles of an amplitude envelope, short enough to still see carrier cycles.
WAVE_WINDOW_MS = 60.0
PREVIEW_SECONDS = 0.3
VIEW_SINGLE = "One arm in detail"
VIEW_ALL = "Compare all arms"


class WaveformPanel(QWidget):
    """Shows the stimuli of the current config; refreshed from a ConfigForm."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._previews: List[StimulusPreview] = []
        self._get_config = None

        self.view_combo = QComboBox()
        self.view_combo.addItems([VIEW_SINGLE, VIEW_ALL])
        self.view_combo.currentTextChanged.connect(self._view_changed)
        self.arm_combo = QComboBox()
        self.arm_combo.currentIndexChanged.connect(self._draw)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._play)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_combo)
        self.arm_label = QLabel("Arm")
        top.addWidget(self.arm_label)
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
        n_silent = sum(1 for p in self._previews if p.is_silent)
        self.status.setText(
            f"{len(self._previews)} arms in the first active block"
            + (f", {n_silent} silent" if n_silent else ""))
        self._draw()

    def _current(self) -> Optional[StimulusPreview]:
        i = self.arm_combo.currentIndex()
        if 0 <= i < len(self._previews):
            return self._previews[i]
        return None

    def _view_changed(self, view: str) -> None:
        single = view == VIEW_SINGLE
        self.arm_combo.setVisible(single)
        self.arm_label.setVisible(single)
        self.play_button.setEnabled(single)
        self._draw()

    # ------------------------------------------------------------------ draw
    def _clear_canvas(self) -> None:
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _figure(self, height: float):
        """Create a themed figure and its Qt canvas, or None if unavailable."""
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as e:
            self.canvas_layout.addWidget(QLabel(f"Plot unavailable: {e}"))
            return None, None
        import matplotlib as mpl
        mpl.rcParams.update(figure_style())
        fig = Figure(figsize=(5, height), tight_layout=True)
        return fig, FigureCanvasQTAgg

    def _draw(self) -> None:
        self._clear_canvas()
        if not self._previews:
            return
        if self.view_combo.currentText() == VIEW_ALL:
            self._draw_all()
        else:
            self._draw_single()

    def _draw_single(self) -> None:
        p = self._current()
        if p is None:
            return
        if p.is_silent:
            lab = QLabel("This arm is silent.")
            lab.setAlignment(Qt.AlignCenter)
            self.canvas_layout.addWidget(lab)
            return
        fig, Canvas = self._figure(5.4)
        if fig is None:
            return
        colour = colour_for(p.category)

        ax_w = fig.add_subplot(311)
        n = min(int(p.samplerate * WAVE_WINDOW_MS / 1000.0), p.wave.size)
        seg = p.wave[:n]
        t_ms = np.arange(seg.size) / p.samplerate * 1000.0
        ax_w.plot(t_ms, seg, linewidth=0.7, color=colour)
        env = np.abs(seg)
        win = max(1, int(p.samplerate * 0.001))
        if env.size > win * 3:
            smooth = np.convolve(env, np.ones(win) / win, mode="same")
            ax_w.plot(t_ms, smooth, linewidth=1.3, color=colour, alpha=0.55)
            ax_w.plot(t_ms, -smooth, linewidth=1.3, color=colour, alpha=0.55)
        ax_w.set_xlabel("Time (ms)")
        ax_w.set_ylabel("Amplitude")
        ax_w.set_title(f"Arm {p.arm}: {p.label}  [{p.category}]", fontsize=9)
        ax_w.grid(True, alpha=0.3)
        peak = max(float(np.max(np.abs(p.wave))), 1e-3)
        ax_w.set_ylim(-peak * 1.15, peak * 1.15)

        ax_s = fig.add_subplot(312)
        max_hz = min(p.samplerate / 2, 60000)
        freqs, mag = spectrum(p.wave, p.samplerate, max_hz=max_hz)
        ax_s.plot(freqs / 1000.0, mag, linewidth=0.9, color=colour)
        ax_s.set_xlabel("Frequency (kHz)")
        ax_s.set_ylabel("Level")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim(0, 1.05)

        ax_g = fig.add_subplot(313)
        self._spectrogram(ax_g, p, max_hz)

        self.canvas_layout.addWidget(Canvas(fig))

    def _spectrogram(self, ax, p: StimulusPreview, max_hz: float) -> None:
        """Time by frequency view; shows modulation and sequence structure."""
        nfft = 1024
        overlap = nfft // 2
        if p.wave.size < nfft * 2:
            ax.text(0.5, 0.5, "too short for a spectrogram",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax.specgram(p.wave, NFFT=nfft, Fs=p.samplerate, noverlap=overlap,
                    cmap="magma")
        ax.set_ylim(0, max_hz)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    def _draw_all(self) -> None:
        audible = [p for p in self._previews if not p.is_silent]
        if not audible:
            lab = QLabel("Every arm is silent.")
            lab.setAlignment(Qt.AlignCenter)
            self.canvas_layout.addWidget(lab)
            return
        fig, Canvas = self._figure(max(3.0, 0.75 * len(self._previews)))
        if fig is None:
            return
        ax_w = fig.add_subplot(121)
        ax_s = fig.add_subplot(122)
        max_hz = min(self._previews[0].samplerate / 2, 60000)

        # Stack the arms vertically, one lane each, so the set is comparable.
        ticks, tick_labels, tick_colours = [], [], []
        for i, p in enumerate(self._previews):
            offset = -i * 2.4
            colour = colour_for(p.category)
            if p.is_silent:
                ax_w.plot([0, WAVE_WINDOW_MS], [offset, offset], linewidth=1.0,
                          color=colour, linestyle=":")
            else:
                n = min(int(p.samplerate * WAVE_WINDOW_MS / 1000.0), p.wave.size)
                seg = p.wave[:n]
                peak = max(float(np.max(np.abs(seg))), 1e-9)
                t_ms = np.arange(seg.size) / p.samplerate * 1000.0
                ax_w.plot(t_ms, seg / peak + offset, linewidth=0.6, color=colour)
                freqs, mag = spectrum(p.wave, p.samplerate, max_hz=max_hz)
                ax_s.plot(freqs / 1000.0, mag, linewidth=0.9,
                          color=colour, alpha=0.85, label=f"Arm {p.arm}")
            ticks.append(offset)
            tick_labels.append(f"Arm {p.arm}: {p.label}")
            tick_colours.append(colour)

        ax_w.set_xlabel("Time (ms)")
        ax_w.set_yticks(ticks)
        ax_w.set_yticklabels(tick_labels, fontsize=7)
        for tick, colour in zip(ax_w.get_yticklabels(), tick_colours):
            tick.set_color(colour)
        ax_w.set_title("Waveforms (normalised)", fontsize=9)
        ax_w.grid(True, alpha=0.25, axis="x")

        ax_s.set_xlabel("Frequency (kHz)")
        ax_s.set_ylabel("Level")
        ax_s.set_ylim(0, 1.05)
        ax_s.set_title("Spectra", fontsize=9)
        ax_s.grid(True, alpha=0.25)
        if audible:
            ax_s.legend(fontsize=6, loc="upper right", framealpha=0.3)

        self.canvas_layout.addWidget(Canvas(fig))

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
