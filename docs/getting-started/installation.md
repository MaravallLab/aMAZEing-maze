# Installation

There are two ways to install, depending on whether you want to read and change the code.

## The standalone application

For running experiments on a Windows rig computer with no Python installed. Download `amazeing-app.zip`, extract it, and run `amazeing-app.exe`.

[:material-download: Download and first launch, step by step](download.md){ .md-button .md-button--primary }

## From source

For running the analysis scripts, changing the code, or working on a machine the application has not been built for.

=== "With uv (recommended)"

    [uv](https://docs.astral.sh/uv/getting-started/installation/) sets everything up in one command. It downloads Python 3.13 if you do not have it (the version is fixed in `.python-version`), creates a `.venv` folder inside the repository, and installs the exact library versions recorded in `uv.lock`, the ones the code is tested with.

    ```bash
    git clone https://github.com/MaravallLab/aMAZEing-maze.git
    cd aMAZEing-maze
    uv sync --extra all
    ```

    Then either put `uv run` in front of every command, which always uses the right environment,

    ```bash
    uv run amaze-app
    uv run python analysis/auditory/exp1_stimulus_pi.py D:/AuditoryExperiments/8_arms_w_voc
    ```

    or activate the environment once per terminal and type the commands as they appear in the rest of this site:

    ```bash
    .venv\Scripts\activate        # Windows
    source .venv/bin/activate     # Linux and macOS
    amaze-app
    ```

    Run `uv sync` again after pulling new code, so the environment follows any change in the libraries.

=== "With pip"

    Create and activate an environment with Python 3.10 or later (3.13 is the version tested), then install the package from the repository folder:

    ```bash
    git clone https://github.com/MaravallLab/aMAZEing-maze.git
    cd aMAZEing-maze

    python -m venv .venv
    .venv\Scripts\activate        # Windows
    source .venv/bin/activate     # Linux and macOS

    pip install -e ".[all]"
    ```

    `pip install -r requirements.txt` does the same, as long as it is run from the repository folder.

The extras choose what is installed alongside the experiment code:

| Extra | Adds | With uv | With pip |
|---|---|---|---|
| `all` | The application window and the analysis pipelines; what most people want | `uv sync --extra all` | `pip install -e ".[all]"` |
| `gui` | The application window only | `uv sync --extra gui` | `pip install -e ".[gui]"` |
| `analysis` | Plotting, statistics and photometry reading for the pipelines under `analysis/` | `uv sync --extra analysis` | `pip install -e ".[analysis]"` |
| `bayes` | PyMC and ArviZ, for the Bayesian fits of the value model | add `--extra bayes` | add `bayes` to the list |
| `dev`, `docs` | The test suite; the documentation site | add `--extra dev` or `--extra docs` | add `dev` or `docs` |

The mixed models of the thesis analyses also need R 4, installed separately ([Reproduce the thesis](../analysis/reproduce-thesis.md)).

Either way you get both the `amaze-app` window and the command-line tools:

| Command | What it does |
|---|---|
| `amaze-app` | The graphical application |
| `amaze-auditory` | Run one auditory session |
| `amaze-tactile` | Run one tactile session |
| `amaze-grammar` | Continuous grammar playback for training days |
| `amaze-analyse-session` | Figures for one recorded session |
| `amaze-summary` | Figures across sessions |
| `amaze-summary-csv` | Summary tables across sessions |
| `amaze-tactile-segments` | Cut per-trial video segments |

## What you need

### Software

- **Python 3.10 or later**, only if installing from source. The code is tested with 3.13, which uv installs for you.
- **Windows 10 or 11** is what the system is tested on. Linux and macOS work for the analysis, and should work for sessions, but no one has run animals on them.

### Hardware, for live experiments

| Part | Requirement | Why |
|---|---|---|
| Camera | Any camera OpenCV can open, with infrared sensitivity | Arm occupancy is detected from an infrared silhouette |
| Infrared illuminator | Mounted below the plate beside the camera, even coverage | Detection compares each arm against an empty-maze baseline, so uneven light causes false entries. Sessions run with the room dark |
| Audio interface | Capable of 192 kHz output | Mouse-relevant sounds run well above human hearing; an ordinary sound card cannot produce them |
| Speaker | Ultrasonic, with a known frequency response | Used for stimulus delivery and for the calibration curve |
| Arduino | Uno or Nano, optional | Only needed to send TTL pulses for photometry synchronisation |
| Servo driver | Adafruit PCA9685, tactile paradigm only | Drives the gratings and reward delivery |

## Checking it works

```bash
amaze-auditory --write-config check.yaml
```

This writes a configuration file containing every setting and its current value, then exits without touching the camera or the sound card. If you get a file, the installation is sound.

To list the audio devices your machine offers, so you can find the right index:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Next

[Run your first session](first-session.md), which walks through a short recording from start to finish.
