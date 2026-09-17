# Session configuration

Every setting of a session lives in one place: an `ExperimentConfig`, which can be written to and read from a YAML file. The application edits it through a form; the command line reads it with `--config`. Both describe the same thing.

Every field of `ExperimentConfig` can be set from a YAML file instead of editing `config.py`. This is the format the graphical interface reads and writes, so a session started from the app and one started from the command line are identical.

```bash
# 1. Get a template containing every setting and its current default
amaze-auditory --write-config my_session.yaml

# 2. Edit it, then run
amaze-auditory --config my_session.yaml

# Command-line flags still win over the file, e.g. per-mouse values:
amaze-auditory --config my_session.yaml --enriched-grammar B --day day_2
```

Unknown field names are rejected (a typo cannot silently fall back to a default), and the file carries a `schema_version` so old files can be migrated later.

**Mapping your own stimuli to ROIs** uses `experiment_mode: custom`:

```yaml
schema_version: 1
experiment_mode: custom
rois_number: 4
block_minutes: [2, 15, 2, 15, 2, 15, 2, 15, 2]   # optional; positions 1,3,5,7,9 are silent
custom_stimuli:
  - roi: "1"
    kind: tone          # pure tone, speaker-compensated
    frequency: 10000
    label: low_tone
  - roi: "2"
    kind: am_tone       # amplitude-modulated tone
    frequency: 20000
    mod_freq: 50
    depth: 0.5
  - roi: "3"
    kind: wav           # any recording; resampled to the session rate
    path: C:/data/vocalisations/call.wav
  - roi: "4"
    kind: silent        # explicit silent control arm
```

ROIs without an entry are silent. The 9-block structure and per-block shuffling are the same as in every other mode, so the existing analysis scripts work unchanged; the trials CSV gains `sound_type` and `stimulus_label` columns.

**Block lengths** are set with `block_minutes` in any mode. Nine values: positions 1, 3, 5, 7 and 9 are silent blocks, positions 2, 4, 6 and 8 are active blocks, and a `0` skips that block. Leave it out to keep the mode's default schedule. A grammar silent-baseline day ignores it and runs one continuous block of `grammar_silent_baseline_minutes`.

**Every mode's stimuli are configurable too.** The frequencies, intervals, modulation rates and patterns that used to be literals inside `experiments.py` are now config fields, with the original values as defaults:

| Mode | Fields |
|---|---|
| `simple_smooth` | `smooth_frequencies` |
| `simple_intervals` | `simple_interval_tonal_centre`, `simple_intervals_list` |
| `temporal_envelope_modulation` | `tem_smooth_freqs`, `tem_constant_rough_freqs`, `tem_complex_rough_freqs`, `tem_constant_mod_freq`, `tem_complex_mod_freqs`, `tem_mod_depth`, `tem_controls` |
| `complex_intervals` | `complex_interval_day` (preset), `complex_interval_tonal_centre`, and the overrides `complex_consonant_intervals`, `complex_dissonant_intervals`, `complex_controls`, `complex_include_smooth`, `complex_include_rough` |
| `sequences` | `sequence_patterns`, `sequence_tone_map`, `sequence_repetitions` |
| `vocalisation` | `vocalisation_include_silent_arm`, `path_to_vocalisation_folder` |

Leaving a field at its default reproduces the published protocol exactly; `tests/test_mode_parameters.py` pins those defaults so they cannot drift.

**Session manifest.** Every session folder also gets a `session_manifest.json` recording the package version, the full configuration used, the files written, and the units of every column (for example that `time_spent` is in seconds in this version, whereas v1 recordings stored milliseconds). Analysis tools should read units from there rather than assume them.

All experiment parameters are defined in a single dataclass:

```python
# src/amazeing/auditory/config.py
@dataclass
class ExperimentConfig:
    samplerate: int = 192000
    channel_id: int = 3
    default_sound_duration: float = 10.0
    default_waveform: str = "sine"
    experiment_mode: str = "complex_intervals"
    complex_interval_day: str = "w1day3"
    # ... see config.py for all options
```

Key settings to adjust for your setup:
- `channel_id` - your audio output device index
- `arduino_port` - COM port for the Arduino (e.g. `"COM4"`); set `use_microcontroller = False` to disable
- `video_input` - camera device index
- `base_output_path` - where session data is saved (defaults to `~/Desktop/auditory_maze_experiments/maze_recordings`)
- `binary_threshold` - pixel threshold for IR camera detection (default 160; tune to your lighting)
- `detection_sensitivity` - mouse detected when binary pixel sum drops below this fraction of the raw baseline (default 0.5)
- `grammar_test_block_minutes` - list of 9 durations (min) for the 9-block cycle; even indices are silent blocks, odd are active; set silent entries to `0` to skip them
- `grammar_apply_speaker_gain` - equalise the six grammar tones for the calibrated speaker (default `True`; set `False` to reproduce sessions recorded before this option existed)
- `roi_csv_path` - where the ROI rectangles are stored; defaults to `<base_output_path>/rois1.csv` because ROIs belong to a rig, not to the code
- `path_to_vocalisation_folder` - folder of `.wav` files for the all-vocalisation mode (defaults to a `vocalisations/` folder next to the recordings folder, i.e. `~/Desktop/auditory_maze_experiments/vocalisations`)
- `path_to_vocalisation_control` - the single `.wav` played on the vocalisation control arm of the mixed modes; leave empty for a silent arm

The speaker frequency-response calibration CSV (`src/amazeing/auditory/data/frequency_response_speaker.csv`) is loaded automatically - no path configuration needed.

**Visits that straddle a block boundary** are closed at the block end, written to the visit log, and counted in `time_spent`. Data recorded before this fix either dropped such visits (this code base) or inflated them (the archived v1 script); the auditory analysis loader in `analysis/auditory/preference_analysis_config.py` documents the caps it applies to v1 data.

---

| Flag | Effect |
|---|---|
| `--grammar-mode {silent_baseline,test}` | Which day of the test protocol (overrides `cfg.grammar_mode`) |
| `--enriched-grammar {A,B}` | Which grammar this mouse heard in the EE cage during training |
| `--day LABEL` | Parent folder label in the output path (e.g. `habituation`, `day_1`, `day_2`) |
| `--seed N` | RNG seed for reproducible melody draws |
| `--draw-rois` | Force interactive ROI re-drawing even if `rois1.csv` already exists |
| `--config FILE.yaml` | Load every setting from a session config file (described above); other flags still override |
| `--write-config FILE.yaml` | Write the effective configuration to a YAML file and exit (the easiest way to get a template) |
