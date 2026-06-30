# Overview

This page orients a reader who has never seen the project: what the aMAZEing
maze is, the major parts that make it up, how they relate, and where to go next.
If you only want to install and run the code, you can skip ahead to
{doc}`../getting_started/installation` — but the map below is worth two minutes
first.

## What is the aMAZEing maze?

The aMAZEing maze is a modular, automated, open-source platform for studying how
sensory cues shape active exploration in rodents. A single mouse explores a
reconfigurable maze while the system tracks its position in real time, delivers
sensory stimuli, and records where the animal spends its time.

It supports two experimental paradigms:

- an **auditory maze** — a multi-arm maze where real-time region-of-interest
  (ROI) tracking triggers sound playback (pure tones, musical intervals,
  temporal-envelope modulation, tone sequences, vocalisations, and a
  Markov-grammar learning protocol); and
- the **SimplerMaze** — a 2-level binary decision tree with servo-controlled
  moveable walls for reward-based navigation studies.

## Major components

The repository is organised around four pieces — a physical rig, the
experiment software that runs on it, the firmware on its microcontrollers, and
the analysis code that processes what it records.

**The hardware rig** (`hardware/`) is the physical maze: laser-cut acrylic wall
panels on a MakerBeam frame over a baseplate, with a camera and IR illuminator
mounted below the maze for tracking. The same base rig — acrylic panels,
MakerBeam posts, camera, and IR — is shared across both paradigms and
reconfigured for each: the tactile (SimplerMaze) setup adds the 3D-printed
servo-controlled moveable walls (gratings) and a reward-delivery chute, while
the auditory setup instead fits a speaker over the maze. See
{doc}`../getting_started/hardware_setup`.

**The auditory-experiment software** (`src/auditory/`) is the main Python
application. `main.py` runs the experiment loop, configured by a single
`ExperimentConfig` dataclass, and delegates to the modules in
`src/auditory/modules/`: `vision` tracks the mouse from the camera feed,
`audio` plays stimuli when it enters an arm, `hardware` optionally drives an
Arduino (TTL sync) and the camera, `data_manager` logs visits, and `analysis`
generates figures. The grammar-learning stimuli live in
`src/auditory/grammar_stimuli/`. See {doc}`auditory_experiment`.

**The SimplerMaze** (`src/simplermaze/`) is a separate paradigm on the same rig:
the tactile, reward-based decision task driven by `simplerCode.py`, using the
servo-controlled moveable walls (gratings) and reward delivery in place of the
auditory speaker. See {doc}`simplermaze`.

**The firmware** (`firmware/`) is the microcontroller code. In the auditory
experiments, an Arduino provides TTL/BNC synchronisation (e.g. for photometry
alignment); in the tactile (SimplerMaze) experiments, an Adafruit PCA9685 PWM
driver (MicroPython) controls the servos that move the walls.

**The analysis pipeline** (`analysis/`) processes recorded data offline. It
includes the auditory post-session and cross-session figure tools
(`src/auditory/run_analysis.py`, `run_summary_analysis.py`), the SimplerMaze
first-paper behavioural analysis (`analysis/simplermaze/`), a SLEAP-NN
pose-estimation pipeline that turns raw maze video into per-frame keypoints, and
the `model_validation` package. The pose pipeline was built to batch-process the
tactile-setup videos; it applies equally to the auditory videos but has not been
needed for that data so far (see the repository QuickGuide). See
{doc}`analysis_pipeline`.

How they relate: the experiment software runs a session and writes recordings —
per-trial CSV and NPY logs, plus the video that the tactile setup feeds to pose
estimation. The analysis code then turns those recordings into figures and
statistics. These steps are run separately rather than as one automated
pipeline, and the analysis workflow is still being actively developed.

## Key features

- **Real-time ROI tracking** via OpenCV binary thresholding with temporal
  debouncing
- **High-fidelity sound generation** at 192 kHz: sine, square, sawtooth,
  triangle, pulse, and white-noise waveforms
- **Speaker frequency-response compensation** loaded automatically from the
  repository calibration CSV
- **Musical-interval system** using just-intonation ratios (consonant vs
  dissonant)
- **Temporal-envelope modulation** (constant AM and complex multi-frequency AM)
- **Grammar-learning experiment** — two first-order Markov grammars (A/B) over
  six pure tones, with three predictability tiers (dominant / secondary / rare),
  EE/SC environment counterbalancing, and a 4-block shuffled test protocol
- **Trial state machine** with 9-block silent/active alternation and
  unique-shuffle constraints
- **Arduino TTL synchronisation** for photometry alignment
- **MicroPython servo control** via the PCA9685 PWM driver
- **Automatic post-session analysis** — per-session figures at the end of every
  session and cross-session summaries via a single CLI command
- **Fully configurable** from one dataclass (`ExperimentConfig`) with per-session
  CLI overrides

## How the pieces fit together

```{list-table}
:header-rows: 1
:widths: 25 75

* - Layer
  - Where it lives
* - Experiment entry point & main loop
  - `src/auditory/main.py`
* - Configuration
  - `src/auditory/config.py` (`ExperimentConfig` dataclass)
* - Sound, vision, hardware, data, analysis
  - `src/auditory/modules/`
* - Grammar-learning stimuli
  - `src/auditory/grammar_stimuli/`
* - Tactile paradigm
  - `src/simplermaze/`
* - Firmware (Arduino / MicroPython)
  - `firmware/`
* - Offline analysis & model validation
  - `analysis/`
```

## Suggested reading order

For a newcomer, the path that ends with a working setup:

1. **This overview** — what the system is and how the parts relate.
2. {doc}`../getting_started/hardware_setup` — if you are building or modifying
   the physical maze (skip if you are only working with already-recorded data).
3. {doc}`../getting_started/installation` — install the software and get a
   working environment. **This is where you start running things.**

From there, follow the page for your task: {doc}`auditory_experiment`,
{doc}`simplermaze`, or {doc}`analysis_pipeline`, with {doc}`troubleshooting` and
the {doc}`API reference <../api/index>` as references.

The {doc}`API reference <../api/index>` documents the modules above in detail,
generated directly from their docstrings.
