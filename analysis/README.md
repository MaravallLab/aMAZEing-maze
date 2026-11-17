# analysis/

Research pipelines that run after the experiments. They are standalone scripts, not part of the installed package: each reads a folder of recorded sessions and writes its results to a folder of its own.

Full documentation: **https://maravalllab.github.io/aMAZEing-maze/analysis/**

The thesis appendix cites these files by path, so they are not moved or renamed, even where a better place exists.

## What is where

| Path | What it is | Documentation |
|---|---|---|
| `auditory/` | The auditory maze: the thesis analyses of Experiments 1 and 2, the four-arm pilot, photometry, the value model, and the earlier batch pipeline. See [`auditory/README.md`](auditory/README.md). | [Reproduce the thesis](https://maravalllab.github.io/aMAZEing-maze/analysis/reproduce-thesis/) |
| `simplermaze/` | The tactile maze: choice accuracy, trajectory metrics, transition probabilities, trial segmentation | [Tactile pipeline](https://maravalllab.github.io/aMAZEing-maze/analysis/tactile/) |
| `calibration/` | Notebooks and scripts used to measure the speaker's frequency response. See [`calibration/README.md`](calibration/README.md). | [Speaker calibration](https://maravalllab.github.io/aMAZEing-maze/guide/speaker-calibration/) |

## The pose estimation pipeline (files at this level)

| File | Stage | What it does |
|---|---|---|
| `crop_and_align_maze.py` | 1 | Warps every video so the maze sits in one fixed place |
| `sleap_batch_processing.py` | 2 | Predicts the mouse keypoints in every frame (sleap-nn, GPU) |
| `slp_export.py` | 3 | Exports predictions to a CSV per video, and optionally an overlay video |
| `filter_in_maze.py` | 4 | Drops detections that fall outside the maze |
| `run_pipeline.py`, `run_pipeline.ps1` | 2 and 4 | Runs stages 2 and 4 in one command. Both were written to sit at the repository root and look for `analysis/` beside themselves, so from this folder they stop with "missing script". Run the stages one by one until that is fixed. |
| `kpms_split_bouts.py` | after 4 | Trims the filtered keypoints to the stretches when the mouse is on the maze, for keypoint-MoSeq |
| `make_3d_dlc_plot_trajectories.py` | | Empty file |

Documentation: [Pose estimation pipeline](https://maravalllab.github.io/aMAZEing-maze/analysis/pose/) and [Cropping and aligning the videos](https://maravalllab.github.io/aMAZEing-maze/analysis/crop-align/).

## Before you run anything

```bash
pip install -e ".[analysis]"
```

from the repository root. Some pipelines need more (R for the mixed models, PyMC for the Bayesian fits, sleap-nn for pose estimation); the documentation page of each says what.
