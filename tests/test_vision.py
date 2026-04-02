"""Tests for the ROIMonitor vision module — ROI tracking and debouncing."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def roi_monitor(tmp_path):
    """Create an ROIMonitor with a known CSV, patching out camera access."""
    # Write a valid rois CSV
    rois_data = {
        "entrance1": [10, 10, 50, 50],
        "entrance2": [70, 10, 50, 50],
        "ROI1": [10, 80, 50, 50],
        "ROI2": [70, 80, 50, 50],
    }
    df = pd.DataFrame(rois_data, index=["xstart", "ystart", "xlen", "ylen"])
    csv_path = tmp_path / "rois.csv"
    df.to_csv(csv_path)

    from vision import ROIMonitor
    roi_names = ["entrance1", "entrance2", "ROI1", "ROI2"]

    # Patch out camera access for define_rois
    monitor = ROIMonitor(
        roi_csv_path=str(csv_path),
        video_input=0,
        roiNames=roi_names,
        enter_frames=1,
        exit_frames=3,
    )
    return monitor


class TestROIMonitorCalibration:
    """Test baseline calibration."""

    def test_calibrate_sets_thresholds(self, roi_monitor):
        """Calibration on a white frame sets non-zero thresholds."""
        white_frame = np.full((200, 200), 255, dtype=np.uint8)
        roi_monitor.calibrate(white_frame)

        # All monitored ROIs should have a threshold
        for name in roi_monitor.roiNames:
            if name in roi_monitor.thresholds:
                assert roi_monitor.thresholds[name] > 0


class TestROIMonitorUpdate:
    """Test frame-by-frame occupancy detection and debouncing."""

    def _calibrate_with_white(self, monitor):
        white = np.full((200, 200), 255, dtype=np.uint8)
        monitor.calibrate(white)

    def test_empty_frame_no_entries(self, roi_monitor):
        """White frame (no mouse) should produce no entries."""
        self._calibrate_with_white(roi_monitor)
        white_frame = np.full((200, 200), 255, dtype=np.uint8)
        entered = roi_monitor.update(white_frame)
        assert entered == []

    def test_dark_roi_triggers_entry(self, roi_monitor):
        """Darkening an ROI region triggers an entry event."""
        self._calibrate_with_white(roi_monitor)

        # Create frame with ROI1 (x=10:60, y=80:130) darkened
        frame = np.full((200, 200), 255, dtype=np.uint8)
        frame[80:130, 10:60] = 0  # ROI1 region

        entered = roi_monitor.update(frame)
        assert "ROI1" in entered

    def test_debounce_exit_requires_multiple_frames(self, roi_monitor):
        """Mouse must be absent for exit_frames consecutive frames to register exit."""
        self._calibrate_with_white(roi_monitor)

        # Enter ROI1
        dark_frame = np.full((200, 200), 255, dtype=np.uint8)
        dark_frame[80:130, 10:60] = 0
        roi_monitor.update(dark_frame)
        assert roi_monitor.is_occupied["ROI1"] is True

        # Send 2 empty frames (less than exit_frames=3)
        white_frame = np.full((200, 200), 255, dtype=np.uint8)
        roi_monitor.update(white_frame)
        roi_monitor.update(white_frame)
        # Should still be occupied (need 3 frames to exit)
        assert roi_monitor.is_occupied["ROI1"] is True

        # Third empty frame triggers exit
        roi_monitor.update(white_frame)
        assert roi_monitor.is_occupied["ROI1"] is False

    def test_entry_not_re_reported(self, roi_monitor):
        """Staying in ROI doesn't re-trigger entry event after first frame."""
        self._calibrate_with_white(roi_monitor)

        dark_frame = np.full((200, 200), 255, dtype=np.uint8)
        dark_frame[80:130, 10:60] = 0

        entered1 = roi_monitor.update(dark_frame)
        assert "ROI1" in entered1

        # Second frame in same ROI
        entered2 = roi_monitor.update(dark_frame)
        assert "ROI1" not in entered2


class TestROICrop:
    """Test the internal _crop_roi helper."""

    def test_crop_returns_correct_region(self, roi_monitor):
        """Crop extracts the expected slice of the frame."""
        frame = np.arange(200 * 200, dtype=np.uint8).reshape(200, 200)
        # ROI1 is at x=10, y=80, w=50, h=50 → frame[80:130, 10:60]
        crop = roi_monitor._crop_roi(frame, "ROI1")
        expected = frame[80:130, 10:60]
        np.testing.assert_array_equal(crop, expected)
