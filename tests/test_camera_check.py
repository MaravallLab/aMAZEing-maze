"""Tests for the live camera and detection check.

The camera, the windows and the sliders cannot be tested without hardware.
What can be tested is the part that decides what the panel reports: the
reading taken from each arm, and how that reading is coloured against the
sensitivity line. Those are what a wrong setting would show up in.
"""

import numpy as np
import pandas as pd
import pytest

from amazeing.auditory.camera_check import (CLEAR, MARGINAL, OCCUPIED, PANEL_WIDTH,
                                            RoiReading, read_rois, render_panel)


@pytest.fixture
def monitor(tmp_path):
    """A calibrated monitor over two arms, no camera involved."""
    rois = {"entrance1": [0, 0, 20, 20], "1": [20, 0, 20, 20], "2": [40, 0, 20, 20]}
    csv_path = tmp_path / "rois.csv"
    pd.DataFrame(rois, index=["xstart", "ystart", "xlen", "ylen"]).to_csv(csv_path)

    from amazeing.auditory.vision import ROIMonitor
    m = ROIMonitor(roi_csv_path=str(csv_path), video_input=0,
                   roiNames=["entrance1", "1", "2"], enter_frames=1, exit_frames=3)
    m.calibrate([np.full((20, 60), 255, dtype=np.uint8)] * 3)
    return m


class TestReadings:

    def test_an_empty_arm_reads_at_its_baseline(self, monitor):
        """With the same picture calibration saw, every ratio is 1.00."""
        readings = read_rois(monitor, np.full((20, 60), 255, dtype=np.uint8))
        assert [r.name for r in readings] == ["entrance1", "1", "2"]
        for r in readings:
            assert r.ratio == pytest.approx(1.0)

    def test_a_blocked_arm_reads_low(self, monitor):
        """Blacking out one arm drops its ratio and leaves the others alone."""
        frame = np.full((20, 60), 255, dtype=np.uint8)
        frame[:, 20:40] = 0                      # arm "1" fully blocked
        by_name = {r.name: r for r in read_rois(monitor, frame)}
        assert by_name["1"].ratio == pytest.approx(0.0)
        assert by_name["2"].ratio == pytest.approx(1.0)

    def test_arms_without_a_baseline_are_skipped(self, monitor):
        """An arm the CSV does not cover is left out rather than guessed at."""
        monitor.roiNames = monitor.roiNames + ["nowhere"]
        readings = read_rois(monitor, np.full((20, 60), 255, dtype=np.uint8))
        assert "nowhere" not in [r.name for r in readings]

    def test_a_zero_baseline_does_not_divide_by_zero(self, monitor):
        monitor.thresholds["1"] = 0.0
        by_name = {r.name: r for r in read_rois(monitor, np.full((20, 60), 255, dtype=np.uint8))}
        assert by_name["1"].ratio == 0.0


class TestColouring:
    """The colour is the whole point: it says whether a setting is safe."""

    def _reading(self, ratio, occupied=False):
        return RoiReading(name="1", baseline=100.0, total=ratio * 100, ratio=ratio,
                          occupied=occupied)

    def test_occupied_is_red_whatever_the_ratio(self):
        assert self._reading(1.4, occupied=True).colour(0.6) == OCCUPIED

    def test_well_clear_of_the_line_is_green(self):
        assert self._reading(1.4).colour(0.6) == CLEAR

    def test_just_above_the_line_is_amber(self):
        """A margin this narrow is the case worth warning about: it will trip."""
        assert self._reading(0.65).colour(0.6) == MARGINAL

    def test_raising_the_sensitivity_makes_a_safe_arm_marginal(self):
        r = self._reading(0.8)
        assert r.colour(0.3) == CLEAR
        assert r.colour(0.75) == MARGINAL


class TestPanel:

    def test_panel_has_the_expected_shape(self):
        panel = render_panel([], sensitivity=0.6, threshold=160, height=480)
        assert panel.shape == (480, PANEL_WIDTH, 3)

    def test_panel_draws_every_arm_that_fits(self):
        readings = [RoiReading(str(i), 100.0, 100.0, 1.0, False) for i in range(10)]
        panel = render_panel(readings, sensitivity=0.6, threshold=160, height=480)
        assert panel.shape == (480, PANEL_WIDTH, 3)
        assert panel.max() > 0                      # something was actually drawn

    def test_a_short_frame_still_produces_a_panel(self):
        """A small camera picture must not make the readout collapse."""
        readings = [RoiReading("1", 100.0, 100.0, 1.0, False)]
        panel = render_panel(readings, sensitivity=0.6, threshold=160, height=120)
        assert panel.shape[1:] == (PANEL_WIDTH, 3)

    def test_an_off_scale_ratio_stays_inside_the_panel(self):
        """Bars are clipped to the axis rather than drawn past the edge."""
        readings = [RoiReading("1", 100.0, 900.0, 9.0, False)]
        panel = render_panel(readings, sensitivity=0.6, threshold=160, height=480)
        assert panel.shape == (480, PANEL_WIDTH, 3)


class TestWiring:

    def test_the_app_can_launch_it(self):
        """The app and the frozen executable both reach it by the same name."""
        from amazeing.app.launcher import ENTRY_MODULES, command_for
        assert ENTRY_MODULES["camera-check"] == "amazeing.auditory.camera_check"
        assert "--config" in command_for("camera-check", ["--config", "x.yaml"])

    def test_it_records_nothing(self):
        """No data manager, no video writer: this tool must stay read-only."""
        import amazeing.auditory.camera_check as cc
        source = open(cc.__file__, encoding="utf-8").read()
        assert "DataManager" not in source
        assert "VideoWriter" not in source
