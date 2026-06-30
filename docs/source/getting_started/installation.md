# Installation

This page gets the software installed and a working environment ready — the
prerequisite for nearly everything else in these docs. Start here if you want to
run the code or build the documentation. If you are also assembling the physical
maze, read {doc}`hardware_setup` alongside it; if you are new to the project,
{doc}`../user_guide/overview` explains how the parts fit together first.

## System requirements

- **Python** 3.10 or later
- **OS**: Windows 10/11 (tested); Linux/macOS should work with minor path
  adjustments
- **Hardware** (optional, only for live experiments):
  - USB camera (any OpenCV-compatible webcam)
  - Audio interface capable of 192 kHz output
  - Arduino Uno/Nano (for TTL synchronisation)
  - Adafruit PCA9685 (for servo control)

## Setup

```bash
# Clone the repository
git clone https://github.com/MaravallLab/aMAZEing-maze.git
cd aMAZEing-maze

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# For development (adds pytest + coverage)
pip install -r requirements-dev.txt
```

## Your first experiment

Once the dependencies are installed, the shortest path to a running auditory
session is:

```bash
cd src/auditory
python main.py
```

When launched, the experiment will:

1. Prompt for a mouse ID and create a timestamped session folder.
2. Generate the trial structure for the configured experiment mode.
3. Calibrate the background from raw camera frames — **keep the maze empty
   during calibration**.
4. Run the experiment loop: track the mouse, play sounds on ROI entry, and log
   every visit.
5. Save trial data (CSV + NPY) after every trial.
6. Generate post-session analysis figures automatically when the session ends.

For experiment modes, the grammar-learning protocol, and all command-line flags,
see {doc}`../user_guide/auditory_experiment`.

## Building the documentation

The docs are built with [Sphinx](https://www.sphinx-doc.org/). Install the docs
toolchain into the **same** environment that has the runtime dependencies (so
the API reference can import the modules):

```bash
pip install -r docs/requirements-docs.txt
```

Then build the HTML site from the `docs/` directory:

```bash
cd docs
make html          # Linux/macOS
make.bat html      # Windows
```

The rendered site is written to `docs/build/html/index.html`.
