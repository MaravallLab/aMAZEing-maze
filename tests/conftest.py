"""Shared fixtures and mocks for the aMAZEing-maze test suite.

The package is imported as ``amazeing`` (install with ``pip install -e .``),
so no sys.path manipulation is needed here. Hardware-facing third-party
modules are stubbed before anything imports them, so the suite runs on a
machine with no sound card, camera or Arduino.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── stub out hardware modules BEFORE any imports that touch them ────
# sounddevice needs PortAudio and an output device
_sd_mock = MagicMock()
_sd_mock.default = MagicMock()
sys.modules.setdefault("sounddevice", _sd_mock)

# soundfile: stub with a minimal read() that returns silence
_sf_mock = MagicMock()
_sf_mock.read = MagicMock(return_value=(np.zeros(192000), 192000))
sys.modules.setdefault("soundfile", _sf_mock)

# serial (pyserial): stub to avoid hardware
_serial_mock = MagicMock()
_serial_mock.Serial = MagicMock()
sys.modules.setdefault("serial", _serial_mock)


# ── fixtures: directories ──────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


# ── fixtures: Audio mock ───────────────────────────────────────────
@pytest.fixture
def mock_audio():
    """A lightweight Audio-like object that generates real waveforms without hardware."""
    from amazeing.auditory.config import ExperimentConfig
    cfg = ExperimentConfig()

    # Patch the calibration CSV path so Audio.__init__ doesn't fail
    with patch("os.path.exists", return_value=False):
        from amazeing.auditory.audio import Audio
        audio = Audio(cfg, calibration_gain_path=None)
    return audio


@pytest.fixture
def experiment_config():
    """Return a default ExperimentConfig for testing."""
    from amazeing.auditory.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.testing = True
    return cfg


# ── fixtures: sample data ─────────────────────────────────────────
@pytest.fixture
def sample_rois_csv(tmp_path):
    """Create a minimal ROIs CSV file."""
    rois = {
        "": ["xstart", "ystart", "xlen", "ylen"],
        "entrance1": [10, 10, 50, 50],
        "entrance2": [70, 10, 50, 50],
        "ROI1": [10, 80, 50, 50],
        "ROI2": [70, 80, 50, 50],
        "ROI3": [130, 80, 50, 50],
        "ROI4": [190, 80, 50, 50],
        "ROI5": [250, 80, 50, 50],
        "ROI6": [310, 80, 50, 50],
        "ROI7": [370, 80, 50, 50],
        "ROI8": [430, 80, 50, 50],
    }
    csv_path = tmp_path / "rois1.csv"
    df = pd.DataFrame(rois)
    # Write with the index column name as first header
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_binary_frame():
    """A synthetic 480x640 binary frame (all white = empty maze)."""
    return np.full((480, 640), 255, dtype=np.uint8)


@pytest.fixture
def sample_binary_frame_with_mouse():
    """Binary frame with a dark region in ROI1 area (10:60, 80:130)."""
    frame = np.full((480, 640), 255, dtype=np.uint8)
    # Place a dark blob in ROI1 (x=10..60, y=80..130) -> frame[y:y+h, x:x+w]
    frame[80:130, 10:60] = 0
    return frame


@pytest.fixture
def sample_trials_df():
    """A minimal trials DataFrame matching the expected schema."""
    return pd.DataFrame({
        "trial_ID": [1, 1, 1, 1, 2, 2, 2, 2],
        "ROIs": ["ROI1", "ROI2", "ROI3", "ROI4"] * 2,
        "frequency": [10000, 12000, 0, 14000, 0, 0, 0, 0],
        "wave_arrays": [np.zeros(100)] * 8,
        "time_spent": [None] * 8,
        "visitation_count": [None] * 8,
        "time_in_maze_ms": [0] * 8,
        "trial_start_time": [None] * 8,
        "end_trial_time": [None] * 8,
    })


# ── fixtures: simplermaze support ─────────────────────────────────
@pytest.fixture
def sample_grating_maps_csv(tmp_path):
    """Create a minimal grating_maps.csv for supFun tests."""
    csv_path = tmp_path / "grating_maps.csv"
    df = pd.DataFrame({
        "grating_id": [1, 2, 3, 4],
        "servo_channel": [0, 1, 2, 3],
        "open_angle": [90, 90, 90, 90],
        "close_angle": [0, 0, 0, 0],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
