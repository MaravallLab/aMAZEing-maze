"""Render the stimuli a config would play, for display in the interface.

The preview calls the real trial factory with a shortened sound duration, so
what you see is produced by exactly the same code path that runs the session.
Nothing here touches the sound card unless you ask it to play.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from amazeing.auditory.config import ExperimentConfig


@dataclass
class StimulusPreview:
    arm: str                 # ROI name, e.g. "1"
    label: str               # human-readable stimulus description
    category: str            # colour key, see amazeing.app.palette
    samplerate: int
    wave: np.ndarray         # mono float waveform
    is_silent: bool


def _label_for(row) -> str:
    """Describe a trials-table row in one short phrase."""
    for key, fmt in (("stimulus_label", "{}"),
                     ("interval_name", "interval {}"),
                     ("pattern", "pattern {}")):
        val = row.get(key, None)
        if isinstance(val, str) and val and val not in ("0", "none"):
            base = fmt.format(val)
            break
    else:
        base = ""
    freq = row.get("frequency", None)
    if freq == "grammar":
        return f"grammar {row.get('tier','')} {row.get('environment_association','')}".strip()
    if isinstance(freq, str) and freq:
        return base or freq
    if isinstance(freq, (list, tuple)) and len(freq) == 2:
        pair = f"{float(freq[0]):.0f} + {float(freq[1]):.0f} Hz"
        return f"{base} ({pair})" if base else pair
    if isinstance(freq, (int, float)):
        if freq == 0:
            return base or "silent"
        return f"{base} ({freq:.0f} Hz)" if base else f"{freq:.0f} Hz"
    return base or "stimulus"


def _category_for(row) -> str:
    """Classify a stimulus so it can be coloured consistently.

    Grammar arms use the tier and environment keys the session figures use;
    the other modes fall back to the sound_type / interval_type column the
    trial factory already writes.
    """
    freq = row.get("frequency", None)
    if freq == "grammar":
        tier = str(row.get("tier", "")).strip()
        env = str(row.get("environment_association", "")).strip()
        if tier and env:
            return f"{tier} {env}"
        return "unknown"
    if freq == "vocalisation" or (isinstance(freq, str) and freq.lower().endswith(".wav")):
        return "vocalisation"
    for col in ("sound_type", "interval_type"):
        val = row.get(col, None)
        if isinstance(val, str) and val:
            if val in ("silent_trial", "silent"):
                return "silent"
            return val
    if isinstance(row.get("pattern", None), str) and row.get("pattern") not in ("", "0"):
        pat = row["pattern"]
        return "silent" if pat == "silence" else ("vocalisation" if pat == "vocalisation" else "pattern")
    if isinstance(row.get("interval_name", None), str) and row.get("interval_name") not in ("", "0"):
        return "interval"
    if isinstance(freq, (int, float)) and freq == 0:
        return "silent"
    return "tone"


def _to_wave(clip, audio, arm: str) -> Optional[np.ndarray]:
    """Turn whatever the factory stored for an arm into one waveform."""
    from amazeing.auditory.experiments import GrammarStimulus
    if isinstance(clip, GrammarStimulus):
        return clip.render(audio, roi=arm, trial_id=2, n_repeats=1)
    if isinstance(clip, tuple) and len(clip) == 2:
        a, b = clip
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return None                      # the (0, 0) silent sentinel
        return audio.mix_sounds(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if isinstance(clip, (list, np.ndarray)):
        arr = np.asarray(clip, dtype=float)
        return arr if arr.ndim == 1 else np.mean(arr, axis=0)
    return None


def preview_stimuli(cfg: ExperimentConfig, duration_s: float = 0.3) -> List[StimulusPreview]:
    """Return one preview per arm for the first active block of ``cfg``.

    ``duration_s`` shortens the generated sounds so the preview is quick; the
    waveform shape, frequencies and modulation are otherwise identical to the
    session. Raises whatever the factory raises if the config is invalid, so
    the caller can show the message.
    """
    import builtins

    from amazeing.auditory.audio import Audio
    from amazeing.auditory.experiments import ExperimentFactory

    c = copy.deepcopy(cfg)
    c.default_sound_duration = float(duration_s)

    if c.experiment_mode == "sequences" and not c.sequence_tone_map:
        raise ValueError(
            "fill in the tone map (a frequency for each letter used in the "
            "patterns) to preview this mode")

    audio = Audio(c, calibration_gain_path=c.calibration_gain_path,
                  configure_device=False)

    def _no_prompts(*_a, **_k):
        # A console prompt here would block the interface forever, so turn any
        # remaining interactive path into a message the caller can show.
        raise ValueError("this mode asks for values on the console; "
                         "set them in the form to preview it")

    real_input, builtins.input = builtins.input, _no_prompts
    try:
        df, _ = ExperimentFactory.generate_trials(c, audio)
    finally:
        builtins.input = real_input
    active = df[df["trial_ID"] == 2] if (df["trial_ID"] == 2).any() else df
    out: List[StimulusPreview] = []
    for _, row in active.iterrows():
        arm = str(row["ROIs"])
        wave = _to_wave(row.get("wave_arrays"), audio, arm)
        silent = wave is None or not np.any(wave)
        if wave is None:
            wave = np.zeros(int(c.samplerate * duration_s))
        out.append(StimulusPreview(
            arm=arm, label=_label_for(row),
            category="silent" if silent else _category_for(row),
            samplerate=c.samplerate, wave=np.asarray(wave, dtype=float),
            is_silent=silent))
    return out


def spectrum(wave: np.ndarray, samplerate: int, max_hz: Optional[float] = None):
    """Single-sided amplitude spectrum, normalised to its peak."""
    if wave.size == 0 or not np.any(wave):
        return np.array([0.0]), np.array([0.0])
    n = int(2 ** np.ceil(np.log2(min(wave.size, 1 << 16))))
    seg = wave[:n] * np.hanning(min(n, wave.size))[:n] if wave.size >= n else wave
    mag = np.abs(np.fft.rfft(seg, n=n))
    freqs = np.fft.rfftfreq(n, 1.0 / samplerate)
    peak = mag.max()
    if peak > 0:
        mag = mag / peak
    if max_hz:
        keep = freqs <= max_hz
        freqs, mag = freqs[keep], mag[keep]
    return freqs, mag
