"""Shared fixtures and mocks for the mice-maze test suite."""

import sys
import os
import types
import csv
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── path setup ──────────────────────────────────────────────────────
# Make the source directories importable without installing the package
ROOT = Path(__file__).resolve().parent.parent
AUDITORY_SRC = ROOT / "src" / "auditory"
SIMPLERMAZE_SRC = ROOT / "src" / "simplermaze"
ANALYSIS_SRC = ROOT / "analysis" / "simplermaze"

# We need to add paths so that `from config import ExperimentConfig` etc. work
# The modules/ dir must come FIRST so `from modules.audio import Audio` works
# when experiments.py is imported from the modules/ directory itself.
sys.path.insert(0, str(AUDITORY_SRC))
sys.path.insert(0, str(AUDITORY_SRC / "modules"))
sys.path.insert(0, str(SIMPLERMAZE_SRC))

# Also create a 'modules' package alias so that
# `from modules.audio import Audio` works when the CWD is updated_version
_modules_dir = AUDITORY_SRC / "modules"
if "modules" not in sys.modules:
    import importlib
    spec = importlib.util.spec_from_file_location(
        "modules", str(_modules_dir / "__init__.py"),
        submodule_search_locations=[str(_modules_dir)]
    )
    if spec:
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["modules"] = _mod
        # Don't exec — just need the search path registered

# ── stub out hardware modules BEFORE any imports that touch them ────
# sounddevice is not installed in CI and requires audio hardware
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
    from config import ExperimentConfig
    cfg = ExperimentConfig()

    # Patch the calibration CSV path so Audio.__init__ doesn't fail
    with patch("os.path.exists", return_value=False):
        from audio import Audio
        audio = Audio(cfg, calibration_gain_path=None)
    return audio


@pytest.fixture
def experiment_config():
    """Return a default ExperimentConfig for testing."""
    from config import ExperimentConfig
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
