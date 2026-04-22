"""Pure-tone synthesis for grammar melodies.

All functions return ``float32`` numpy arrays at ``config.SAMPLE_RATE``.
Tones are ramped with a cosine-squared envelope (``RAMP_MS`` in/out) to
avoid spectral splatter at onset/offset.

Deliberately minimal: no speaker-response compensation here. If calibration
is required, wrap the output of :func:`generate_tone` with the same gain
curve used in ``code/auditory/updated_version/modules/audio.py``.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from . import config as cfg


def _ms_to_samples(duration_ms: float, sample_rate: int) -> int:
    return int(round(duration_ms * 1e-3 * sample_rate))


def _cos2_ramp(n_samples: int) -> np.ndarray:
    """Cosine-squared onset ramp of length ``n_samples``, rising 0 -> 1."""
    if n_samples <= 0:
        return np.empty(0, dtype=np.float32)
    # sin(pi/2 * x)^2 over x in [0, 1] = cos^2(pi/2 * (1-x)): smooth 0->1.
    x = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=np.float64)
    ramp = np.sin(0.5 * np.pi * x) ** 2
    return ramp.astype(np.float32)


def generate_tone(
    frequency_hz: float,
    duration_ms: float = cfg.TONE_DURATION_MS,
    sample_rate: int = cfg.SAMPLE_RATE,
    amplitude: float = cfg.AMPLITUDE,
    ramp_ms: float = cfg.RAMP_MS,
) -> np.ndarray:
    """Synthesise a ramped pure tone.

    The output is a 1-D ``float32`` array of exactly
    ``round(duration_ms/1000 * sample_rate)`` samples.
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")

    n = _ms_to_samples(duration_ms, sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)

    n_ramp = _ms_to_samples(ramp_ms, sample_rate)
    if n_ramp > 0 and 2 * n_ramp <= n:
        env = np.ones(n, dtype=np.float64)
        ramp_in = _cos2_ramp(n_ramp).astype(np.float64)
        env[:n_ramp] = ramp_in
        env[-n_ramp:] = ramp_in[::-1]
        wave *= env

    return wave.astype(cfg.DTYPE)


def generate_silence(
    duration_ms: float,
    sample_rate: int = cfg.SAMPLE_RATE,
) -> np.ndarray:
    """Return ``float32`` zeros of the requested duration."""
    if duration_ms < 0:
        raise ValueError(f"duration_ms must be >= 0, got {duration_ms}")
    return np.zeros(_ms_to_samples(duration_ms, sample_rate), dtype=cfg.DTYPE)


def generate_tone_unit(
    symbol: str,
    sample_rate: int = cfg.SAMPLE_RATE,
    amplitude: float = cfg.AMPLITUDE,
) -> np.ndarray:
    """Generate one "tone unit": the tone plus the inter-tone silent gap.

    A melody is the concatenation of ``MELODY_LENGTH`` tone units.
    """
    if symbol not in cfg.TONES:
        raise KeyError(f"Unknown tone symbol {symbol!r}. Known: {list(cfg.TONES)}")
    tone = generate_tone(
        cfg.TONES[symbol],
        duration_ms=cfg.TONE_DURATION_MS,
        sample_rate=sample_rate,
        amplitude=amplitude,
        ramp_ms=cfg.RAMP_MS,
    )
    gap = generate_silence(cfg.INTER_TONE_GAP_MS, sample_rate=sample_rate)
    return np.concatenate([tone, gap])


def generate_melody(
    symbols: Iterable[str],
    sample_rate: int = cfg.SAMPLE_RATE,
    amplitude: float = cfg.AMPLITUDE,
) -> np.ndarray:
    """Concatenate tone units for the given sequence of tone symbols."""
    units: List[np.ndarray] = [
        generate_tone_unit(s, sample_rate=sample_rate, amplitude=amplitude)
        for s in symbols
    ]
    if not units:
        return np.zeros(0, dtype=cfg.DTYPE)
    return np.concatenate(units)


def generate_silence_gap(sample_rate: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """The inter-melody gap (2 s by default)."""
    return generate_silence(cfg.INTER_MELODY_GAP_MS, sample_rate=sample_rate)
