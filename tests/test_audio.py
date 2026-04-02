"""Tests for the Audio class — sound generation, ramping, calibration."""

import numpy as np
import pytest
from unittest.mock import patch


class TestAudioGeneration:
    """Test sound waveform generation."""

    def test_sine_wave_shape(self, mock_audio):
        """Sine wave has correct length for given duration and sample rate."""
        sound = mock_audio.generate_sound_data(1000, waveform="sine", duration_s=1.0)
        expected_samples = int(mock_audio.fs * 1.0)
        assert len(sound) == expected_samples

    def test_sine_wave_frequency_content(self, mock_audio):
        """Generated sine has dominant frequency at the requested value."""
        freq = 10000
        duration = 1.0
        sound = mock_audio.generate_sound_data(freq, waveform="sine", duration_s=duration, ramp_duration_s=0.0)

        # FFT to find dominant frequency
        fft_vals = np.abs(np.fft.rfft(sound))
        freqs = np.fft.rfftfreq(len(sound), 1.0 / mock_audio.fs)
        dominant_freq = freqs[np.argmax(fft_vals)]

        # Allow 10 Hz tolerance (FFT bin resolution)
        assert abs(dominant_freq - freq) < 10, f"Expected ~{freq} Hz, got {dominant_freq} Hz"

    def test_zero_frequency_is_silence(self, mock_audio):
        """Frequency 0 produces an all-zeros array."""
        sound = mock_audio.generate_sound_data(0, duration_s=1.0)
        assert np.all(sound == 0)

    def test_square_wave_values(self, mock_audio):
        """Square wave only contains values near +volume and -volume."""
        sound = mock_audio.generate_sound_data(1000, waveform="square", duration_s=0.1, ramp_duration_s=0.0)
        # After removing potential edge effects, values should be ~+1 or ~-1
        unique_abs = np.unique(np.abs(np.round(sound, 1)))
        # Should have values near 0 (at transitions) and near volume
        assert len(unique_abs) <= 3  # 0, ~volume, maybe intermediate

    def test_white_noise_not_constant(self, mock_audio):
        """White noise has non-zero standard deviation."""
        sound = mock_audio.generate_sound_data(1000, waveform="white noise", duration_s=0.1)
        assert np.std(sound) > 0

    def test_all_waveforms_produce_output(self, mock_audio):
        """Every supported waveform returns a non-empty array."""
        waveforms = ["sine", "square", "sawtooth", "triangle", "pulse wave", "white noise"]
        for wf in waveforms:
            sound = mock_audio.generate_sound_data(5000, waveform=wf, duration_s=0.01)
            assert len(sound) > 0, f"Waveform '{wf}' produced empty output"


class TestRamp:
    """Test fade-in ramp application."""

    def test_ramp_starts_at_zero(self, mock_audio):
        """Sound with ramp starts at zero amplitude."""
        sound = mock_audio.generate_sound_data(10000, waveform="sine", duration_s=0.1, ramp_duration_s=0.01)
        assert abs(sound[0]) < 0.001

    def test_no_ramp_starts_nonzero(self, mock_audio):
        """Sound without ramp starts at non-zero amplitude (for non-zero phase)."""
        sound = mock_audio.generate_sound_data(10000, waveform="sine", duration_s=0.1, ramp_duration_s=0.0)
        # The very first sample of sin(2*pi*f*0) = 0, so check a few samples in
        assert np.max(np.abs(sound[:100])) > 0

    def test_ramp_preserves_length(self, mock_audio):
        """Ramp does not change array length."""
        dur = 0.5
        sound_with_ramp = mock_audio.generate_sound_data(5000, duration_s=dur, ramp_duration_s=0.02)
        sound_no_ramp = mock_audio.generate_sound_data(5000, duration_s=dur, ramp_duration_s=0.0)
        assert len(sound_with_ramp) == len(sound_no_ramp)


class TestTemporalModulation:
    """Test AM-modulated sound generation."""

    def test_simple_tem_shape(self, mock_audio):
        """Simple TEM sound has correct length."""
        sound = mock_audio.generate_simple_tem_sound_data(10000, duration_s=1.0)
        assert len(sound) == int(mock_audio.fs * 1.0)

    def test_complex_tem_shape(self, mock_audio):
        """Complex TEM sound has correct length."""
        sound = mock_audio.generate_complex_tem_sound_data(10000, duration_s=1.0)
        assert len(sound) == int(mock_audio.fs * 1.0)

    def test_complex_tem_normalized(self, mock_audio):
        """Complex TEM sound stays within [-1, 1]."""
        sound = mock_audio.generate_complex_tem_sound_data(10000, duration_s=1.0, volume=1.0)
        assert np.max(np.abs(sound)) <= 1.0 + 1e-6


class TestMixSounds:
    """Test sound mixing."""

    def test_mix_equal_length(self, mock_audio):
        """Mixing two equal-length arrays returns same length."""
        a = np.ones(100)
        b = np.ones(100) * 0.5
        mixed = mock_audio.mix_sounds(a, b)
        assert len(mixed) == 100

    def test_mix_different_length(self, mock_audio):
        """Mixing different-length arrays returns shorter length."""
        a = np.ones(100)
        b = np.ones(50) * 0.5
        mixed = mock_audio.mix_sounds(a, b)
        assert len(mixed) == 50

    def test_mix_normalizes(self, mock_audio):
        """Mixed result is normalized so max absolute value <= 1."""
        a = np.ones(100) * 0.8
        b = np.ones(100) * 0.8
        mixed = mock_audio.mix_sounds(a, b)
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6


class TestCalibration:
    """Test speaker calibration gain computation."""

    def test_no_calibration_returns_unity(self, mock_audio):
        """Without calibration CSV, gain is always 1.0."""
        # mock_audio was created with calibration_gain_path=None
        assert mock_audio.compute_gain(10000) == 1.0
        assert mock_audio.compute_gain(20000) == 1.0

    def test_defaults_resolve_correctly(self, mock_audio):
        """_resolve_params falls back to defaults when None is passed."""
        wf, dur, vol, ramp = mock_audio._resolve_params(None, None, None, None)
        assert wf == mock_audio.default_waveform
        assert dur == mock_audio.default_duration
        assert vol == mock_audio.default_volume
        assert ramp == mock_audio.default_ramp

    def test_overrides_resolve_correctly(self, mock_audio):
        """_resolve_params uses provided values over defaults."""
        wf, dur, vol, ramp = mock_audio._resolve_params("square", 2.0, 0.5, 0.05)
        assert wf == "square"
        assert dur == 2.0
        assert vol == 0.5
        assert ramp == 0.05
