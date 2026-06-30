# Analysis pipeline

This page is about turning recorded sessions into figures and statistics — the
offline counterpart to running experiments. It covers the auditory post-session
figures, the SimplerMaze behavioural pipeline, and the model-validation package.
Pose estimation (turning raw maze video into per-frame keypoints) is a separate
upstream step documented in the repository QuickGuide.

## Auditory: post-session figures

Per-session figures are generated **automatically at the end of every session**
and saved into the session folder. To regenerate them for already-collected
sessions:

```bash
cd src/auditory

# Single session
python run_analysis.py "C:\path\to\session_folder"

# Multiple sessions at once
python run_analysis.py "C:\path\to\session1" "C:\path\to\session2"
```

Cross-session summaries are produced by `run_summary_analysis.py`:

```bash
cd src/auditory

# All mice on one day
python run_summary_analysis.py --day "C:\...\maze_recordings\grammar\day_1"

# All mice across all days collected so far
python run_summary_analysis.py --all "C:\...\maze_recordings\grammar"
```

Silent-baseline sessions are excluded automatically; only active test-day
sessions contribute. The figure-generation code is documented in the
{doc}`API reference <../api/index>` (`modules.analysis`,
`modules.summary_analysis`).

## SimplerMaze: behavioural pipeline

The SimplerMaze pipeline lives in
`analysis/simplermaze/first_paper_exploratory_analysis/`. It processes
DeepLabCut (or SLEAP) pose-estimation data alongside trial CSVs to produce
figures and statistics.

### Prerequisites

In addition to `requirements.txt`:

```bash
pip install rpy2          # optional: GLMM via R's lme4
pip install statsmodels   # already in requirements.txt
```

The GLMM features also need R with the `lme4` package installed.

### Setup

Edit `analysis/simplermaze/first_paper_exploratory_analysis/config.py` to set
`MOUSE_ID` and `BASE_PATH`, then verify session discovery:

```bash
cd analysis/simplermaze/first_paper_exploratory_analysis
python config.py
```

This prints every detected session, whether it has DLC tracking data, and which
trial CSV it found.

The numbered scripts (`01_choice_accuracy.py`, `02_metrics_and_models.py`,
`03_transition_probabilities.py`) run the stages in order.

## Model validation

`analysis/auditory/model_validation/` is a read-only package that tests whether
a semantic-value term (an HMM context-belief signal over the EE/SC grammar
association) explains maze preference beyond a fluency-only baseline. Phase 1
(data loading, sanity checks, design analysis, latent regressors, collinearity,
parameter recovery) runs on the standard scientific stack; Phase 2 (PyMC nested
models, LOO, posterior-predictive checks) additionally requires `pymc` and
`arviz` and is imported lazily. It is documented in the
{doc}`API reference <../api/index>` (`model_validation`).
```
