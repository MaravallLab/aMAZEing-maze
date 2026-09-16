# Installation

There are two ways to install, depending on whether you want to read and change the code.

## The standalone application

For running experiments on a rig computer with no Python installed.

1. Download `amazeing-app.zip` from the [releases page](https://github.com/MaravallLab/aMAZEing-maze/releases).
2. Unzip it somewhere permanent, such as `C:\amazeing-app`.
3. Run `amazeing-app.exe`.

!!! warning "Keep the whole folder"
    The executable needs the `_internal` folder beside it, which holds Python and every library. Copying the `.exe` on its own will not work. Move the whole folder.

## From source

For running the analysis scripts, changing the code, or working on a machine the application has not been built for.

```bash
git clone https://github.com/MaravallLab/aMAZEing-maze.git
cd aMAZEing-maze

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux and macOS

pip install -e ".[gui,analysis]"
```

This gives you both the `amaze-app` window and the command-line tools:

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

The `[analysis]` extra adds the plotting and statistics libraries the post-hoc pipelines need. Leave it out if you only want to run experiments.

## What you need

### Software

- **Python 3.10 or later**, only if installing from source.
- **Windows 10 or 11** is what the system is tested on. Linux and macOS work for the analysis, and should work for sessions, but no one has run animals on them.

### Hardware, for live experiments

| Part | Requirement | Why |
|---|---|---|
| Camera | Any camera OpenCV can open, with infrared sensitivity | Arm occupancy is detected from an infrared silhouette |
| Infrared illuminator | Even coverage of the maze floor | The detection compares each arm against an empty-maze baseline, so uneven light causes false entries |
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
