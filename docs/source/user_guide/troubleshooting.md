# Troubleshooting

This page collects fixes for common problems when running a live experiment —
camera, tracking, audio, and microcontroller issues — plus how to run the test
suite. Reach for it when something does not work during
{doc}`auditory_experiment` setup or a session.

## Camera not detected

- Check the USB connection and reinstall the camera driver if necessary.
- Confirm `video_input` in `config.py` matches the camera's device index.

## Mouse not detected / tracking unstable

- Tune `binary_threshold` (default 160) to your IR lighting.
- Adjust `detection_sensitivity` (default 0.5) — the mouse is detected when the
  binary pixel sum drops below this fraction of the empty-maze baseline.
- Re-run background calibration with the maze **empty**, or force ROI redraw
  with `--draw-rois`.

## No audio / wrong output device

- Set `channel_id` to your audio interface's output device index.
- Confirm the interface supports the configured `samplerate` (192 kHz).

## Arduino / servos not responding

- Verify the Arduino connection and that the correct sketch is uploaded.
- Set `arduino_port` to the right COM port, or set `use_microcontroller = False`
  to run without the microcontroller.

## Tests

The test suite mocks all hardware dependencies, so it runs without a camera,
audio interface, or Arduino:

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/auditory/modules --cov-report=term-missing
```
