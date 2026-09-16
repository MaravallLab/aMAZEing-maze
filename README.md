# The aMAZEing maze

**A modular, automated, sensory-engaging open-source platform for studying how sensory cues shape active exploration in rodents.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-219%20passing-brightgreen.svg)](#testing)
[![Documentation](https://img.shields.io/badge/docs-website-blue.svg)](https://maravalllab.github.io/aMAZEing-maze/)

![The maze with tuneable walls](hardware/drawings/model.png)

The maze is a reconfigurable arena filmed under infrared light, built on a drilled floor plate so the walls move and the same rig becomes a different maze. Software tracks which arm the animal is in and drives the stimuli in real time.

It runs as **two paradigms**, equally developed, sharing the hardware, the tracking and the analysis tooling.

### Auditory maze

Sound triggered by arm entry: pure tones, musical intervals, amplitude modulation, tone sequences, artificial grammars, recorded vocalisations. Nothing is rewarded and nothing is required. The animal explores, the stimulus-to-arm mapping is reshuffled between blocks, and **where it chooses to spend its time is the measurement**.

### Tactile maze

A two-level binary decision tree with a right answer. The cue is texture, read by the whiskers.

**Three pairs of servo-driven 3D-printed gratings**, one pair before every decision point, face each other across the corridor. **The vertical grating marks the correct way**; the other is horizontal. If the reward is in the bottom right arm, the right grating is vertical at the first junction and again at the second, so the animal follows vertical twice and arrives at the reward.

Reaching the correct arm is detected by the camera and releases a food pellet from an **automated reward port**: a servo rocks the dispenser while an **infrared beam watches the chute**, and stops the moment a pellet is detected falling through. The loop is closed, so a trial recorded as rewarded really was rewarded, rather than the mechanism turning by a fixed amount and hoping.

A trial runs from the animal entering the maze to leaving it, and is scored as a hit, an incorrect choice, or a miss. Training proceeds through four stages of increasing difficulty.

The electronics are an Arduino driving an Adafruit PCA9685 sixteen-channel PWM controller for the ten servos (six gratings, four reward ports) and four infrared detectors, alongside a BeeHive board developed at the University of Sussex.

## Documentation

**[maravalllab.github.io/aMAZEing-maze](https://maravalllab.github.io/aMAZEing-maze/)** has the tutorials, the guide and the reference.

| If you want to | Start at |
|---|---|
| Run an auditory session | [Run your first session](https://maravalllab.github.io/aMAZEing-maze/getting-started/first-session/) |
| Run the tactile task | [The tactile paradigm](https://maravalllab.github.io/aMAZEing-maze/guide/tactile/) |
| Build a rig | [Build and set up the rig](https://maravalllab.github.io/aMAZEing-maze/hardware/build/) |
| Analyse recordings | [Analyse a set of sessions](https://maravalllab.github.io/aMAZEing-maze/analysis/tutorial/) |

## Install

Download `amazeing-app.zip` from the [releases page](https://github.com/MaravallLab/aMAZEing-maze/releases), unzip it and run `amazeing-app.exe`. No Python needed.

Or from source:

```bash
git clone https://github.com/MaravallLab/aMAZEing-maze.git
cd aMAZEing-maze
pip install -e ".[gui,analysis]"
```

## Run

```bash
amaze-app
```

The application has a tab per task: auditory session, speaker calibration, analysis, grammar training, tactile session.

Or from a terminal, which runs exactly the same code:

```bash
# auditory
amaze-auditory --write-config my_protocol.yaml   # get a template to edit
amaze-auditory --config my_protocol.yaml         # run it

# tactile
amaze-tactile
```

| Command | What it does |
|---|---|
| `amaze-app` | The graphical application |
| `amaze-auditory` | Run one auditory session |
| `amaze-camera-check` | Live camera and detection check, before anything is recorded |
| `amaze-tactile` | Run one tactile session |
| `amaze-tactile-segments` | Cut a tactile session's video into per-trial clips |
| `amaze-grammar` | Continuous grammar playback for training days |
| `amaze-analyse-session` | Figures for one recorded session |
| `amaze-summary` | Figures across sessions |
| `amaze-summary-csv` | Summary tables across sessions |

## Repository layout

```
├── src/amazeing/       Installable package
│   ├── auditory/         Auditory paradigm: config, session loop, audio, vision, analysis
│   ├── simplermaze/      Tactile paradigm: session loop, grating maps, reward stages
│   └── app/              Graphical interface
├── analysis/           Post-hoc research pipelines (preference index, pose estimation, models)
├── hardware/           CAD sources, STLs, drawings, construction photos
├── firmware/           Arduino and MicroPython sketches: grating and reward servos, TTL sync
├── docs/               Documentation site sources
├── packaging/          Application build: PyInstaller spec, build script, pinned versions
├── tools/              Utilities, including documentation screenshot generation
├── tests/              Test suite
└── archive/            Superseded code, kept for reference
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest
```

## Contributing

Issues and pull requests are welcome. The [architecture page](https://maravalllab.github.io/aMAZEing-maze/development/architecture/) explains how the pieces fit together and the design rules the code follows.

To work on the documentation:

```bash
pip install -e ".[docs]"
mkdocs serve                      # preview at http://127.0.0.1:8000
python tools/make_screenshots.py  # regenerate the application screenshots
```

## Citing

The code used for the original behavioural experiments is tagged [`v1.0-thesis`](https://github.com/MaravallLab/aMAZEing-maze/releases/tag/v1.0-thesis).

## Contributors

Alejandra Carriero, Shahd Al Balushi, Andre Maia Chagas, Miguel Maravall, Oluwaseyi Jesusanmi, Yuri Elias Rodrigues, Maja Nowak, Marcus Burnell-Spector, Isabel Maranhao, Moira Eley.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
