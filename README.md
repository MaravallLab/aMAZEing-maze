# The aMAZEing maze 

**A modular automated sensory engaging open-source platform for studying how sensory cues shape active exploration in rodents.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen.svg)](#testing)

---

## Table of Contents

- [The aMAZEing maze](#the-amazeing-maze)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running an Auditory Experiment](#running-an-auditory-experiment)
    - [Experiment Modes](#experiment-modes)
    - [Running the SimplerMaze](#running-the-simplermaze)
  - [Configuration](#configuration)
  - [Repository Structure](#repository-structure)
  - [Testing](#testing)
  - [Hardware Build](#hardware-build)
  - [Analysis Pipeline](#analysis-pipeline)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Expected data layout](#expected-data-layout)
    - [Running the pipeline](#running-the-pipeline)
      - [Step 1: Choice accuracy across sessions](#step-1-choice-accuracy-across-sessions)
      - [Step 2: P1/P2 metrics and statistical models](#step-2-p1p2-metrics-and-statistical-models)
      - [Step 3: Transition probabilities](#step-3-transition-probabilities)
    - [Interpreting the outputs](#interpreting-the-outputs)
    - [Auditory Analysis Pipeline](#auditory-analysis-pipeline)
      - [Key analyses](#key-analyses)
      - [Fiber Photometry Alignment (`fibpho_alignment.py`)](#fiber-photometry-alignment-fibpho_alignmentpy)
  - [Contributing](#contributing)
  - [Contributors](#contributors)
  - [License](#license)

---

## Overview

The aMAZEing Maze is a modular, reconfigurable maze system for behavioural neuroscience experiments. It combines real-time video tracking, auditory stimulus delivery, and optional hardware control (servos, TTL synchronisation for photometry) into a single Python-driven platform.

The system supports two main experimental paradigms:

| Paradigm | Description |
|---|---|
| **Auditory Maze** | Multi-arm maze with auditory stimuli (pure tones, musical intervals, temporal envelope modulation, tone sequences, vocalisations). Real-time ROI tracking triggers sound playback when the mouse enters specific arms. |
| **SimplerMaze** | Y-maze with servo-controlled moveable walls for reward-based navigation studies. |

### Key Features

- **Real-time ROI tracking** via OpenCV binary thresholding with temporal debouncing
- **High-fidelity sound generation** at 192 kHz: sine, square, sawtooth, triangle, pulse, white noise
- **Speaker frequency-response compensation** automatically loaded from the repo calibration CSV
- **Musical interval system** using just-intonation ratios (consonant vs dissonant)
- **Temporal envelope modulation** (constant AM and complex multi-frequency AM)
- **Grammar learning experiment** — two first-order Markov grammars (A/B) over six pure tones, with three predictability tiers (dominant / secondary / rare), EE/SC environment counterbalancing, and a 4-block shuffled test protocol
- **Trial state machine** with 9-block silent/active alternation and unique-shuffle constraints
- **Arduino TTL synchronisation** for photometry alignment
- **MicroPython servo control** via PCA9685 PWM driver
- **Automatic post-session analysis** — per-session figures generated at the end of every session; cross-session summary figures (per mouse, per day, group) via a single CLI command
- **Fully configurable** from a single dataclass (`ExperimentConfig`) with per-session CLI overrides

---

## System Requirements

- **Python** 3.10 or later
- **OS**: Windows 10/11 (tested), Linux/macOS (should work with minor path adjustments)
- **Hardware** (optional, for live experiments):
  - USB camera (any OpenCV-compatible webcam)
  - Audio interface capable of 192 kHz output
  - Arduino Uno/Nano (for TTL sync)
  - Raspberry Pi Pico with PCA9685 (for servo control)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aMAZEing-maze.git
cd aMAZEing-maze 

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# For development (includes pytest)
pip install -r requirements-dev.txt
```

---

## Usage

### Running an Auditory Experiment

1. Edit `src/auditory/config.py` to set your experiment parameters (mode, sample rate, device IDs, paths).
2. Run the main script:

```bash
cd src/auditory
python main.py
```

For the grammar experiment, pass per-session flags on the command line:

```bash
# Silent baseline (Day 1 — no audio, establishes location preference baseline)
python main.py --grammar-mode silent_baseline --enriched-grammar A --day habituation

# Audio test (Day 2 / Day 3)
python main.py --grammar-mode test --enriched-grammar A --day day_1
python main.py --grammar-mode test --enriched-grammar A --day day_2
```

3. The system will:
   - Prompt for mouse ID and create a timestamped session folder
   - Generate the trial structure based on your chosen experiment mode
   - Calibrate the background using raw camera frames (keep the maze empty during calibration)
   - Run the experiment loop: track the mouse, play sounds on ROI entry, log every visit
   - Save trial data (CSV + NPY) after every trial
   - Generate post-session analysis figures automatically when the session ends

### CLI flags

| Flag | Effect |
|---|---|
| `--grammar-mode {silent_baseline,test}` | Which day of the test protocol (overrides `cfg.grammar_mode`) |
| `--enriched-grammar {A,B}` | Which grammar this mouse heard in the EE cage during training |
| `--day LABEL` | Parent folder label in the output path (e.g. `habituation`, `day_1`, `day_2`) |
| `--seed N` | RNG seed for reproducible melody draws |
| `--draw-rois` | Force interactive ROI re-drawing even if `rois1.csv` already exists |

### Experiment Modes

Set `experiment_mode` in `config.py` to one of:

| Mode | Description |
|---|---|
| `grammar` | Grammar learning test — two Markov grammars × three predictability tiers (dominant/secondary/rare) + vocalisation + silent control, shuffled across 4 active 15-min blocks |
| `simple_smooth` | One pure tone per ROI arm |
| `simple_intervals` | Two-tone chords (musical intervals) per ROI |
| `temporal_envelope_modulation` | Smooth, constant-AM, and complex-AM sounds |
| `complex_intervals` | Multi-day interval protocol with consonant/dissonant contrasts |
| `sequences` | Tone-pattern sequences (ABAB, AoAo, etc.) |
| `vocalisation` | Each ROI plays a different vocalisation recording |

### Running the SimplerMaze

```bash
cd src/simplermaze
python simplerCode.py
```

This runs the Y-maze paradigm with servo-controlled walls.

---

## Configuration

All experiment parameters are defined in a single dataclass:

```python
# src/auditory/config.py
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
- `channel_id` — your audio output device index
- `arduino_port` — COM port for the Arduino (e.g. `"COM4"`); set `use_microcontroller = False` to disable
- `video_input` — camera device index
- `base_output_path` — where session data is saved (defaults to `~/Desktop/auditory_maze_experiments/maze_recordings`)
- `binary_threshold` — pixel threshold for IR camera detection (default 160; tune to your lighting)
- `detection_sensitivity` — mouse detected when binary pixel sum drops below this fraction of the raw baseline (default 0.5)
- `grammar_test_block_minutes` — list of 9 durations (min) for the 9-block cycle; even indices are silent blocks, odd are active; set silent entries to `0` to skip them
- `path_to_vocalisation_control` — path to the `.wav` file played on the vocalisation control arm

The speaker frequency-response calibration CSV (`analysis/calibration/frequency_response_speaker.csv`) is loaded automatically — no path configuration needed.

---

## Repository Structure

```
aMAZEing-maze/
├── src/
│   ├── auditory/               # Auditory maze experiment
│   │   ├── config.py           #   Experiment configuration dataclass
│   │   ├── main.py             #   Main experiment loop
│   │   ├── run_analysis.py     #   Standalone per-session analysis CLI
│   │   ├── run_summary_analysis.py # Cross-session summary analysis CLI (--day / --all)
│   │   ├── rois1.csv           #   ROI coordinates (auto-created on first run)
│   │   ├── modules/
│   │   │   ├── audio.py        #   Sound generation, playback & speaker compensation
│   │   │   ├── experiments.py  #   Trial structure factory (all experiment modes)
│   │   │   ├── vision.py       #   ROI tracking (OpenCV binary threshold + debounce)
│   │   │   ├── data_manager.py #   Session setup, visit & maze-entry logging
│   │   │   ├── hardware.py     #   Arduino TTL & camera control
│   │   │   ├── analysis.py     #   Per-session figure generation (SessionAnalyzer)
│   │   │   └── summary_analysis.py # Cross-session figure generation (SummaryAnalyzer)
│   │   └── grammar_stimuli/    #   Grammar learning stimulus package
│   │       ├── config.py       #     Tone inventory, transition matrices, arm plan
│   │       ├── sequence_sampler.py # Markov sampler with complexity tiers
│   │       ├── tone_generator.py   # Pure-tone melody synthesis
│   │       ├── run.py          #     Training-day playback CLI
│   │       └── QUICKSTART.md   #     Step-by-step grammar experiment guide
│   └── simplermaze/            # Y-maze with servos
│       ├── simplerCode.py      #   Main script
│       └── supFun.py           #   Support functions
│
├── firmware/
│   ├── ttl_bnc/                # Arduino TTL synchronisation sketch
│   ├── arduino/                # Servo control sketches
│   └── micropython/            # Pico PCA9685 servo driver
│
├── analysis/
│   ├── auditory/               # Auditory experiment analysis notebooks
│   ├── simplermaze/            # SimplerMaze analysis & DLC pipeline
│   │   ├── first_paper_exploratory_analysis/
│   │   └── trials_segmentation/
│   └── calibration/            # Speaker frequency-response calibration
│
├── hardware/
│   ├── 3dmodels/               # FreeCAD & STL files for maze parts
│   ├── drawings/               # Schematics and task design diagrams
│   └── photos/                 # Construction photos
│
├── archive/                    # Legacy code (preserved for reference)
│   ├── auditory_v1/            #   Original monolithic auditory script
│   ├── abandoned/              #   Abandoned experimental approaches
│   └── legacy/                 #   Bonsai workflow & old segmentation
│
├── docs/                       # Sphinx documentation source
├── tests/                      # pytest test suite (63 tests)
├── requirements.txt
├── requirements-dev.txt
└── LICENSE                     # GPLv3
```

---

## Post-Session Analysis

### Output path structure

Sessions are saved under:

```
base_output_path / experiment_mode / [--day label] / time_<timestamp>_<mouseID> /
```

Examples:

```
maze_recordings/grammar/habituation/time_2026-05-23_10_00_00_mouse1/
maze_recordings/grammar/day_1/time_2026-05-23_14_30_00_mouse1/
maze_recordings/grammar/day_2/time_2026-05-24_10_00_00_mouse1/
```

### Per-session figures

Figures are generated **automatically at the end of every session** and saved inside the session folder alongside the CSVs. To regenerate them for sessions already collected:

```bash
cd src/auditory

# Single session
python run_analysis.py "C:\path\to\session_folder"

# Multiple sessions at once
python run_analysis.py "C:\path\to\session1" "C:\path\to\session2"
```

| File | What it shows |
|------|--------------|
| `fig1_arm_totals.png` | Total time (min) and visit count per arm, coloured by stimulus type |
| `fig2_ee_vs_sc.png` | EE vs SC arms grouped by predictability tier — time and visits |
| `fig3_block_evolution.png` | Time per stimulus category across active blocks — checks whether preference shifts |
| `fig4_visit_duration.png` | Boxplot of individual visit durations per stimulus type |
| `fig5_maze_time.png` | Total time inside the maze per trial block |
| `fig6_location_preference.png` | Heatmap of time per arm per block with stimulus labels — distinguishes location bias from stimulus preference |

### Cross-session summary figures

After running all mice for a day (or across multiple days), generate summary figures with `run_summary_analysis.py`. Saved into the folder you pass.

```bash
cd src/auditory

# All mice on one day
python run_summary_analysis.py --day "C:\...\maze_recordings\grammar\day_1"

# All mice across all days collected so far
python run_summary_analysis.py --all "C:\...\maze_recordings\grammar"
```

Silent-baseline sessions are automatically excluded — only active test-day sessions contribute.

**EE vs SC preference:**

| File | What it shows |
|------|--------------|
| `summary_A_ee_sc_per_mouse.png` | EE vs SC total time — one pair of bars per mouse, one panel per day |
| `summary_B_preference_index.png` | EE preference index (−1 to +1) per mouse; positive = EE preference |
| `summary_C_group_summary.png` | Group mean ± SEM time and visit count on EE vs SC arms per day |
| `summary_D_cross_day_pi.png` | PI trajectory per mouse + group mean ± SEM across days *(multi-day only)* |

**Predictive complexity (dominant / secondary / rare):**

| File | What it shows |
|------|--------------|
| `summary_E_tier_breakdown_per_mouse.png` | Stacked bars per mouse: EE bar and SC bar each split by tier (dark → light = dominant → rare) |
| `summary_F_group_tier_breakdown.png` | Group mean ± SEM for all 6 tier × environment combinations |
| `summary_G_cross_day_tiers.png` | Per-tier preference across days — group mean ± SEM for each complexity level, EE and SC panels *(multi-day only)* |

---

## Testing

The test suite covers audio generation, trial structure, ROI tracking, data management, configuration, and integration scenarios. All hardware dependencies are mocked.

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/auditory/modules --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_audio.py -v
```

---

## Hardware Build

The maze is built from laser-cut acrylic and 3D-printed components. All CAD files are in `hardware/3dmodels/`.

Key components:
- **Maze base plate** with reconfigurable arm slots
- **Moveable walls** with servo-driven cog mechanism
- **Camera holder** mounted above the maze
- **Electronics enclosure** for Arduino and driver boards
- **Reward delivery chute** with servo-actuated gate

See `hardware/drawings/` for schematics and `hardware/photos/` for assembly reference.

![Maze with tuneable walls](hardware/drawings/model.png)

---

## Analysis Pipeline

The SimplerMaze behavioural analysis pipeline lives in `analysis/simplermaze/first_paper_exploratory_analysis/`. It processes DeepLabCut (or SLEAP) pose-estimation data alongside trial CSVs to produce publication-ready figures and statistics.

### Prerequisites

In addition to `requirements.txt`, the analysis pipeline needs:

```bash
pip install rpy2          # optional: for GLMM via R's lme4
pip install statsmodels   # already in requirements.txt
```

If using the GLMM features, you also need R installed with the `lme4` package:

```r
install.packages("lme4")
```

### Setup

Edit `analysis/simplermaze/first_paper_exploratory_analysis/config.py`:

```python
MOUSE_ID = "6357"
BASE_PATH = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data", "simplermaze"
)
```

The pipeline auto-discovers all sessions for the mouse. Verify with:

```bash
cd analysis/simplermaze/first_paper_exploratory_analysis
python config.py
```

This prints every detected session, whether it has DLC tracking data, and which trial CSV it found.

### Expected data layout

```
<BASE_PATH>/mouse <MOUSE_ID>/
├── habituation/
│   ├── mouse<ID>_session1.1_trial_info.csv
│   └── rois1.csv
├── <timestamp><ID>session<X.Y>/          # e.g. 2024-08-29_10_23_026357session3.7
│   ├── new_session<X.Y>_trials.csv       # or clean_mouse<ID>_session<X.Y>_trial_info.csv
│   └── rois1.csv
└── deeplabcut/                           # DLC tracking (only some sessions)
    └── .../<ID>_<timestamp>s<X.Y>DLC_*.csv
```

**Trial CSV columns used:**

| Column | Description |
|---|---|
| `rew_location` | Correct arm letter (A/B/C/D) |
| `first_reward_area_visited` | ROI the mouse visited first (e.g. `rewB`) |
| `rewA`, `rewB`, `rewC`, `rewD` | Time spent in each arm (ms), empty if not visited |
| `hit`, `miss`, `incorrect` | Original classifications (may have misdetections) |
| `start_trial_frame`, `end_trial_frame` | Frame boundaries for DLC alignment |

### Running the pipeline

All scripts are run from the `first_paper_exploratory_analysis/` directory.

#### Step 1: Choice accuracy across sessions

```bash
python 01_choice_accuracy.py
```

**What it does:** Loads trial CSVs from all sessions (habituation through 3.8). Recomputes trial outcomes from `first_reward_area_visited[-1] == rew_location` as a sanity check against the `hit`/`miss`/`incorrect` columns. Excludes trials where the mouse never entered any reward arm. Fits a binomial GLMM (`correct ~ session + (1|mouse_id)`) to test whether choice accuracy changes across sessions.

**Outputs** (saved to `mouse <ID>/MOUSE_<ID>_TOTAL_ANALYSIS/`):

| File | Description |
|---|---|
| `choice_accuracy_across_sessions.png/.pdf` | Grouped bar chart: correct / incorrect / no-choice per session |
| `choice_accuracy_summary.csv` | Per-session counts and percentage correct |
| Terminal | GLMM coefficients and p-values |

#### Step 2: P1/P2 metrics and statistical models

```bash
python 02_metrics_and_models.py
```

**What it does:** For sessions with DLC data (3.6, 3.7, 3.8), splits each trial into Phase 1 (maze entry to first ROI reached) and Phase 2 (ROI to trial end). Computes per-phase duration, mean speed (cm/s, Savitzky-Golay smoothed), and spatial entropy. Runs three statistical tests comparing Hit vs Miss: Mann-Whitney U, Linear Mixed Model (statsmodels), and Gamma GLMM (rpy2/lme4) for the positively-skewed speed and duration data.

**Outputs:**

| File | Description |
|---|---|
| `master_behavioural_data.csv` | Per-trial metrics: session, status, P1/P2 duration, speed, entropy |
| `violin_plots.png/.pdf` | 2x3 grid of violin plots (P1/P2 x duration/speed/entropy) with MWU and LMM p-values |
| `stats_report.csv` | All p-values in one table (MWU, LMM, Gamma GLMM) |
| Terminal | Gamma GLMM summaries from R |

#### Step 3: Transition probabilities

```bash
python 03_transition_probabilities.py
```

**What it does:** For sessions with DLC data, assigns each frame to an ROI (entrance1/2, rewA-D) or "corridor" using bounding-box checks. Collapses consecutive identical states to extract the sequence of compartment transitions. Builds per-trial transition matrices, aggregates separately for Hit and Miss trials, and computes derived metrics: perseveration rate (how often the mouse returns to the same arm), exploration entropy (how evenly it visits different arms), and number of unique ROIs visited.

**Outputs:**

| File | Description |
|---|---|
| `transition_combined.png/.pdf` | Side-by-side heatmaps: Hit transitions, Miss transitions, difference (Hit - Miss) |
| `exploration_metrics.png/.pdf` | Violin plots comparing perseveration, exploration entropy, and unique ROIs (Hit vs Miss) |
| `transition_summary.csv` | Per-trial: state sequence, perseveration rate, exploration entropy, unique ROIs |

### Interpreting the outputs

**Choice accuracy plot:** A learning curve. If the bars shift from mostly red (incorrect) to mostly green (correct) across sessions, the mouse is learning the task. The GLMM p-value for the session coefficient tells you whether this trend is statistically significant.

**Violin plots:** Each panel shows the distribution of a metric for Hit vs Miss trials. Key comparisons:
- P1 speed: do mice run faster on trials where they find the reward?
- P1 entropy: do successful trials show more directed (lower entropy) trajectories?
- P2 duration: do mice spend more time in the reward zone on Hit trials?

**Transition heatmaps:** Read row-by-row: "given the mouse is in row-ROI, what is the probability it goes to column-ROI next?" The difference map highlights where Hit and Miss trials diverge -- e.g., Hit trials may show stronger corridor-to-correct-arm transitions.

**Exploration metrics:** Perseveration rate > 0 means the mouse tends to revisit the same arm after leaving it. Higher exploration entropy means more uniform visiting across arms.

### Auditory Analysis Pipeline

The auditory analysis pipeline lives under `analysis/auditory/`. It processes visit data from the 8-arm radial maze across 6 experiment days, producing publication-ready figures, interactive visualisations, and comprehensive statistical reports.

For a detailed technical report on the computational modelling component, see [`analysis/auditory/REPORT_aesthetic_value_model.md`](analysis/auditory/REPORT_aesthetic_value_model.md).

**Prerequisites:**

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels plotly
```

**Expected data layout:**

```
8_arms_w_voc/
  w1_d1/                            # Day 1: temporal envelope modulation
    time_2025-06-04_14_22_30mouse10049/
      trials_time_2025-06-04_14_22_30.csv
      mouseXXXXX_..._detailed_visits.csv   # optional ground-truth log
    ...
  w1_d2/                            # Day 2: consonant/dissonant intervals
  w1_d3/                            # Day 3: consonant/dissonant intervals
  w1_d4/                            # Day 4: intervals (no silent control)
  w2_sequences/                     # Week 2: tone sequences
  w2_vocalisations/                 # Week 2: mouse vocalisations
```

**Configuration:**

Edit `preference_analysis_config.py`:
- `BASE_PATH` -- root folder containing `w1_d1/`, `w1_d2/`, etc.
- Or set the `MAZE_DATA_DIR` environment variable to override.
- `VISIT_CLIP_MS` env var -- per-visit duration cap in ms (default 10000).

**Running:**

```bash
cd analysis/auditory
# Verify session discovery first:
python preference_analysis_config.py

# Option A: standalone single-pipeline run (no batch summary CSV):
python 01_preference_analysis.py

# Option B (recommended): full batch analysis (generates all CSVs + figures + stats,
# automatically calls 02_within_trial_preference.py at the end):
python run_batch_preference.py

# Optional: completers-only linear mixed model on the batch CSV
# (run AFTER run_batch_preference.py so preference_data.csv exists):
python 03_completers_lmm.py /path/to/8_arms_w_voc

# Optionally check visit-duration outliers:
python check_visit_outliers.py

# Run the computational model:
python aesthetic_value_model_4D.py
```

**Scripts overview:**

| Script | Description |
|--------|-------------|
| `preference_analysis_config.py` | Shared configuration, session discovery, data loading with DV-first corruption handling, and `compute_first_minute_re` helper |
| `01_preference_analysis.py` | Standalone single-pipeline driver for PI + RE + first-minute RE + voc-vs-other-sounds analysis (mirrors `run_batch_preference` outputs without the batch summary CSV) |
| `run_batch_preference.py` | Main batch pipeline: per-mouse/day PI computation, 8+ static figures, interactive Plotly figures, enhanced statistics; auto-invokes `02_within_trial_preference.py` |
| `02_within_trial_preference.py` | Within-trial scatter plots: sound vs silent-arm visit duration per mouse per day |
| `03_completers_lmm.py` | Completers-only linear mixed-model analysis: filters mice present on all required days and fits four nested LMMs (null / day fixed / linear trend / random slopes) on PI, voc PI, other-sounds PI, RE, and first-minute RE |
| `check_visit_outliers.py` | Diagnostic tool for identifying and reporting visit-duration outliers |
| `aesthetic_value_model_4D.py` | Brielmann & Dayan (2022) aesthetic value model -- 4D extension for mouse acoustic preference |

**Outputs** (saved to `BATCH_ANALYSIS/` inside the data folder):

| File | Description |
|------|-------------|
| `preference_data.csv` | Per-mouse, per-session PI + visit metrics. Includes `preference_index`, `voc_pi`, `other_sounds_pi`, `avg_voc_dur_ms`, `avg_other_sounds_dur_ms`, `roaming_entropy`, and `re_first_min` |
| `stimulus_breakdown.csv` | Per-stimulus-type visit duration |
| `within_trial_preference.csv` | Per-mouse, per-day sound vs silent-arm durations |
| `voc_vs_other_sounds_pi.csv` | Per-session voc PI and other-sounds PI side-by-side, including the average voc / other-sounds / silent durations used to compute each index |
| `fig1_pi_trajectories.png/pdf` | Individual mouse PI trajectories across days |
| `fig2_pi_by_day.png/pdf` | Mean PI per day with 95% CI |
| `fig3_pi_violins.png/pdf` | Violin plots of PI distribution by day |
| `fig4_complexity_heatmap.png/pdf` | Visit duration by stimulus type per day |
| `fig5_vocalisation_contrast.png/pdf` | Paired comparison: vocalisation vs other days |
| `fig6_re_vs_pi.png/pdf` | Roaming entropy vs preference (within & between mouse) |
| `fig6b_re_firstmin_vs_pi.png/pdf` | First-minute roaming entropy (first 60 s of habituation) vs PI, within- and between-mouse panels |
| `fig7_icc_summary.png/pdf` | Variance decomposition (ICC) + Kruskal-Wallis |
| `fig8_*.png/pdf` | Additional analysis panels |
| `fig_voc_vs_other_sounds_pi.png/pdf` | Per-day scatter of voc PI vs other-sounds PI plus pooled panel |
| `fig_within_trial_preference.png/pdf/html` | Within-trial preference scatter (interactive Plotly version with hover) |
| `fig1_*.html`, `fig3_*.html`, etc. | Interactive Plotly versions of main figures (hover to identify mice) |
| `stats_report.txt` | Full statistical report (descriptive, inferential, enhanced analyses, plus sections 12b voc vs other-sounds and 12c first-minute RE) |
| `aesthetic_value_model_4D.png/pdf` | 6-panel computational model figure |
| `aesthetic_model_4D_predictions.csv` | Per-stimulus, per-day model predictions |
| `aesthetic_model_4D_params.csv` | Best-fit model parameters |

**Completers LMM outputs** (produced by `03_completers_lmm.py`, saved to `BATCH_ANALYSIS/completers/` by default):

| File | Description |
|------|-------------|
| `completers_summary.csv` | Filtered subset (mice present on all required days) with PI, voc PI, other-sounds PI, RE, and first-minute RE per session |
| `completers_lmm_fixed_effects.csv` | Fixed-effects estimates (coef, SE, z, p, 95% CI) for every fitted model and outcome |
| `completers_lmm_variance.csv` | Variance components (between-mouse, residual, ICC, marginal R-squared, conditional R-squared) per model |
| `completers_lmm_random_intercepts.csv` | Per-mouse BLUPs (random intercepts) with 95% CI from the day-fixed model |
| `completers_stats_report.txt` | Human-readable summary including LRT comparisons (M0 vs M1, M2 vs M3) and Nakagawa & Schielzeth (2013) R-squared |
| `fig_completers_caterpillar.png/pdf` | Per-mouse random-intercept caterpillar plot (BLUPs ranked with 95% CI) |
| `fig_completers_day_estimates.png/pdf` | Estimated marginal means by day from the day-fixed LMM |
| `fig_completers_spaghetti.png/pdf` | Per-mouse trajectories across sessions (one line per completer) |

#### Key analyses

**1. Preference Index (PI):**

PI is computed using within-trial comparison: sound-playing ROIs vs the silent-control ROI, both measured during sound trials (2, 4, 6, 8) only. This avoids bias from comparing 15-min sound trials against 2-min silent trials.

```
PI = (Avg_Sound_Duration - Avg_Silent_Duration) / (Avg_Sound_Duration + Avg_Silent_Duration)
```

Ranges from -1 (prefer silence) to +1 (prefer sound).

**2. Data integrity:**

The pipeline handles a known trial-boundary bug in the experiment control code where visits were not closed at trial end, producing inflated duration values in `trials.csv`. The loader (`load_session_visits`) uses `detailed_visits.csv` as ground truth where available, with a 10 s per-visit clip and sanity caps for fallback to `trials.csv`.

**3. Enhanced statistical analyses:**

| Analysis | Method | Description |
|----------|--------|-------------|
| Per-day PI test | One-sample Wilcoxon + rank-biserial effect size | Tests whether PI differs from zero on each day, with Holm-Bonferroni correction |
| Across-day comparison | Kruskal-Wallis + epsilon-squared | Tests whether PI differs across experiment days |
| Post-hoc pairwise | Dunn's test (Holm-corrected) | Identifies which day pairs differ significantly |
| Complexity gradient | Jonckheere-Terpstra trend test | Tests monotone trend in dwell time along the complexity ordering |
| Vocalisation contrast | Paired Wilcoxon + rank-biserial | Compares vocalisation PI vs overall PI per mouse |
| Mixed-effects model | LMM with day contrasts + random intercept | `PI ~ day + (1\|mouse)`, reports Nakagawa marginal/conditional R-squared |
| Model comparison | AIC, BIC, log-likelihood ratio test | Compares null, day-only, and day+RE models |
| Sensitivity | Beta regression (binomial GLM on scaled PI) | Checks robustness of day effects to distributional assumption |
| Voc PI vs other-sounds PI | Per-day paired Wilcoxon + rank-biserial (section 12b) | Tests whether vocalisations elicit a different preference than the other-sounds aggregate, per session |
| First-minute RE | Pearson + Spearman correlation, within- and between-mouse (section 12c) | Tests whether exploration during the first 60 s of habituation predicts subsequent PI |

**4. Completers LMM (`03_completers_lmm.py`):**

A standalone follow-up that restricts the dataset to mice present on every required experimental day (`PI_DAYS` by default: `w1_d1`, `w1_d2`, `w1_d3`, `w2_sequences`, `w2_vocalisations`) and fits four nested mixed models per outcome (PI, voc PI, other-sounds PI, RE, first-minute RE):

| Model | Formula | Purpose |
|-------|---------|---------|
| M0 (null) | `y ~ 1 + (1\|mouse)` | Baseline; quantifies between-mouse variance and ICC |
| M1 (day fixed) | `y ~ C(day) + (1\|mouse)` | Day differences with random intercepts (the user's primary request) |
| M2 (linear trend) | `y ~ session_num + (1\|mouse)` | Linear change across sessions |
| M3 (random slopes) | `y ~ session_num + (1+session_num\|mouse)` | Heterogeneous per-mouse slopes |

The script reports REML estimates for inference and refits with ML for likelihood-ratio tests (M0 vs M1; M2 vs M3). It produces Nakagawa & Schielzeth (2013) marginal/conditional R-squared, ICC, BLUPs (per-mouse random intercepts) with 95% CIs, and three figures (caterpillar plot, day estimated marginal means, per-mouse spaghetti).

CLI:

```bash
python 03_completers_lmm.py /path/to/8_arms_w_voc \
    --csv BATCH_ANALYSIS/preference_data.csv \
    --output-dir BATCH_ANALYSIS/completers \
    --required-days w1_d1 w1_d2 w1_d3 w2_sequences w2_vocalisations
```

**5. Computational model (Brielmann & Dayan 2022):**

A 4-dimensional extension of the aesthetic value model, where stimuli are represented as vectors in a feature space of [location_familiarity, spectral_complexity, biological_relevance, temporal_predictability]. The model simulates an agent traversing the full experiment and predicts PI from the difference in aesthetic value between sound and silence. Fitted via 2000 random initialisations of SLSQP. Includes lesioned model comparisons, dimension dropout analysis, and location vs acoustic decomposition. See the [detailed report](analysis/auditory/REPORT_aesthetic_value_model.md) for full documentation.

#### Fiber Photometry Alignment (`fibpho_alignment.py`)

Aligns Tucker-Davis Technologies (TDT) fiber photometry recordings with auditory maze visit CSV timestamps using TTL pulse matching.

**Prerequisites:**

```bash
pip install numpy pandas matplotlib scipy tdt
```

**Configuration:**

Edit paths at the top of `fibpho_alignment.py`:
- `TANK_PATH` -- path to the TDT tank folder (contains `.Tbk`, `.Tdx`, `.tev`, `.tsq` files)
- `VISIT_CSV` -- path to the `mouseXXXXX_vocalisations_detailed_visits.csv`
- `TRIALS_CSV` -- path to the `trials_time_YYYY-MM-DD_HH_MM_SS.csv`

**Running:**

```bash
cd analysis/auditory
python fibpho_alignment.py
```

**Outputs** (saved to `fibpho_analysis/` inside the session folder):

| File | Description |
|------|-------------|
| `alignment_report.csv` | Event-by-event TTL-to-CSV mapping with residuals |
| `fibpho_aligned_overview.png/pdf` | Full-session dF/F with colour-coded TTLs |
| `fibpho_trial_panels.png/pdf` | Per-trial zoomed dF/F panels |
| `fibpho_peri_event.png/pdf` | Peri-event average dF/F by vocalisation type |
| `fibpho_peri_event_pooled.png/pdf` | Pooled peri-event dF/F (all stimuli) |

**How alignment works:**

1. The TDT recording starts before the maze experiment (different clocks)
2. The script reads TTL onset times from the TDT `MTL_` epoc store
3. It matches inter-event intervals to `sound_on_time` entries in the visit CSV
4. The first TTL is identified as a test pulse and excluded
5. A precise clock offset is computed (typical alignment: <70ms residual)
6. Delta F/F is calculated using 405nm isosbestic correction of the 465nm GCaMP signal

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and ensure tests pass (`python -m pytest tests/ -v`)
4. Commit and push
5. Open a pull request

Please keep the test suite green and add tests for new functionality.

---

## Contributors



- Miguel Maravall
- Alejandra Carriero
- Shahd Al Balushi
- Andre Maia Chagas
- Isobel Parkes
- Oluwaseyi Jesusanmi
- Isabel Maranhao
- Maja Nowak
- Narcus Burnell-Spetcor
- Yuri Elias Rodrigues
  





---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
