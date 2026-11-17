# analysis/auditory/

The analyses of the auditory maze. Each script takes its data folder as its first argument and writes to `BATCH_ANALYSIS/<its name>/` inside it, never into a session folder.

Full documentation:

- **[Reproduce the thesis](https://maravalllab.github.io/aMAZEing-maze/analysis/reproduce-thesis/)**: every thesis figure and result, with its script, command and output folder
- **[The thesis scripts](https://maravalllab.github.io/aMAZEing-maze/analysis/auditory/)**: what each script asks, and its options
- **[Earlier batch pipeline (legacy)](https://maravalllab.github.io/aMAZEing-maze/analysis/auditory-legacy/)**

The thesis appendix cites these files by path, so none is moved or renamed.

## The thesis scripts

Use these for anything you report.

| Files | What they cover |
|---|---|
| `exp1_*.py`, `exp1_*.R` | Experiment 1: eight arms, six stimulus sets (Chapters 4 and 5) |
| `exp2_*.py`, `exp2_lmm.R` | Experiment 2: the grammar experiment (Chapter 6) |
| `model_validation/` | The value model of Experiment 2 and its Bayesian fits. See [`model_validation/README.md`](model_validation/README.md). |
| `survey_standalone_sessions.py`, `pilot_four_arm.py`, `pilot_lmm.R` | The four-arm pilot (Chapter 3) |
| `fibpho_*.py` | Photometry: dopamine recordings aligned with maze sessions (Section 4.12) |
| `stimulus_complexity.py`, `complexity_3d.py` | Spectral and temporal complexity of the stimuli |
| `pi_fisher_figure.py`, `bayes_principle_figure.py` | Figures of the general methods (Chapter 2) |
| `fisher_z_check.py`, `fisher_z_check.R` | A check that the Fisher z scale changes no result |

## Shared by both generations

| File | What it is |
|---|---|
| `preference_analysis_config.py` | The session loader. The thesis scripts of Experiment 1 and the pilot read every visit through it. |
| `run_batch_preference.py` | The driver of the earlier pipeline, also imported by the thesis scripts for the classification of the stimuli. |

## The earlier batch pipeline (legacy)

Kept so that earlier results can be reproduced. It caps every visit at 10 s, and its 4D value model is not the model of the thesis.

`01_preference_analysis.py`, `02_within_trial_preference.py`, `03_completers_lmm.py`, `04_random_slopes_presentation.py`, `check_visit_outliers.py`, `aesthetic_value_model_4D.py`, `REPORT_aesthetic_value_model.md`

## Tools and helpers

| File | What it is |
|---|---|
| `automatic_roi_selection_posthoc_time_in_maze.py` | Measures time in the maze from the session videos, after the fact. It writes into the session files, so run it on a copy. |
| `calibration_volume.py` | Plays test tones to check a speaker, from the first version of the software. It imports `supfun_sequences`, which is now in `archive/auditory_v1/`, so it does not run from this folder as it stands. |
| `functions.py`, `RE_functions.py` | Helper functions from the first analyses (file search, roaming entropy) |

## Notebooks and scratch

Exploratory work, kept as a record. Nothing in the thesis depends on running them.

| File | What it is |
|---|---|
| `analysis.ipynb`, `analysis_toplevel.ipynb` | Early exploratory analysis |
| `calibrating_sounds.ipynb`, `generate_sounds.ipynb`, `sequences.ipynb` | Building and checking stimuli |
| `dopamine_analysis.ipynb`, `dopamine_analysis_toplevel.ipynb` | Early photometry analysis, before the `fibpho_*` scripts |
| `delta_f_plot.html` | A saved interactive plot of a delta F/F trace (plotly, about 7 MB) |
| `fitting_aesthetic_model.ipynb` | Early fitting of the value model |
| `fix_corrupted_videos.ipynb` | Repairing damaged video files |
| `poseestimation_analysis.ipynb`, `usv_analysis.ipynb` | Early pose and vocalisation analysis |
| `prova.ipynb` | Scratch ("prova" is Italian for "test") |
