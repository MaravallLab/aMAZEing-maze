# Auditory experiment

This is the main guide for running auditory experiments — the day-to-day
reference once the software is installed (and, optionally, the rig is built). It
covers launching a session, the available experiment modes, the configuration
options, and where session data is written. New to the project? Read
{doc}`overview` first.

The auditory maze plays a stimulus when the tracked mouse enters a configured
region of interest (ROI). The experiment is driven by `src/auditory/main.py` and
configured through the `ExperimentConfig` dataclass in `src/auditory/config.py`.

## Running

```bash
cd src/auditory
python main.py
```

### Command-line flags

```{list-table}
:header-rows: 1
:widths: 35 65

* - Flag
  - Effect
* - `--grammar-mode {silent_baseline,test}`
  - Which day of the test protocol (overrides `cfg.grammar_mode`)
* - `--enriched-grammar {A,B}`
  - Which grammar the mouse heard in the EE cage during training
* - `--day LABEL`
  - Parent folder label in the output path (e.g. `habituation`, `day_1`)
* - `--seed N`
  - RNG seed for reproducible melody draws
* - `--draw-rois`
  - Force interactive ROI re-drawing even if `rois1.csv` exists
```

## Experiment modes

Set `experiment_mode` in `config.py` to one of:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Mode
  - Description
* - `grammar`
  - Grammar-learning test — two Markov grammars × three predictability tiers
    (dominant / secondary / rare) + vocalisation + silent control, shuffled
    across four active 15-min blocks
* - `simple_smooth`
  - One pure tone per ROI arm
* - `simple_intervals`
  - Two-tone chords (musical intervals) per ROI
* - `temporal_envelope_modulation`
  - Smooth, constant-AM, and complex-AM sounds
* - `complex_intervals`
  - Multi-day interval protocol with consonant/dissonant contrasts
* - `sequences`
  - Tone-pattern sequences (ABAB, AoAo, …)
* - `vocalisation`
  - Each ROI plays a different vocalisation recording
```

## Configuration

All experiment parameters live in a single dataclass:

```python
# src/auditory/config.py
@dataclass
class ExperimentConfig:
    samplerate: int = 192000
    channel_id: int = 3
    default_sound_duration: float = 10.0
    default_waveform: str = "sine"
    experiment_mode: str = "complex_intervals"
    # ... see config.py for all options
```

Settings you will most often adjust:

- `channel_id` — audio output device index
- `arduino_port` — COM port for the Arduino (e.g. `"COM4"`); set
  `use_microcontroller = False` to disable
- `video_input` — camera device index
- `base_output_path` — where session data is saved
- `binary_threshold` — pixel threshold for IR detection (default 160; tune to
  your lighting)
- `detection_sensitivity` — mouse detected when the binary pixel sum drops below
  this fraction of the raw baseline (default 0.5)
- `grammar_test_block_minutes` — nine block durations; even indices are silent,
  odd are active; set silent entries to `0` to skip
- `path_to_vocalisation_control` — `.wav` played on the vocalisation control arm

The speaker frequency-response calibration CSV
(`analysis/calibration/frequency_response_speaker.csv`) is loaded automatically.

## Output layout

Sessions are saved under:

```text
base_output_path / experiment_mode / [--day label] / time_<timestamp>_<mouseID> /
```

For example:

```text
maze_recordings/grammar/habituation/time_2026-05-23_10_00_00_mouse1/
maze_recordings/grammar/day_1/time_2026-05-23_14_30_00_mouse1/
```

## Grammar-learning stimuli

The grammar-learning protocol is implemented in the `grammar_stimuli` package:
two first-order Markov grammars over six pure tones, with complexity controlled
by restricting sampling to probability tiers (dominant / secondary / rare)
*without ever modifying the stored transition matrix*. See the
{doc}`API reference <../api/index>` (`grammar_stimuli`) and the package
`QUICKSTART.md` for the day-by-day protocol.
