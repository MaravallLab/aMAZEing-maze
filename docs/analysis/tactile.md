# Tactile analysis pipeline

!!! info "At a glance"
    **Time:** a few minutes per mouse, once the sessions are in place.
    **You need:** one mouse's tactile sessions (the trial tables, and DeepLabCut tracking for some sessions), Python with the `[analysis]` extra.
    **You get:** a learning curve, phase-by-phase speed, duration and path entropy, and transition maps between compartments, each comparing hit and miss trials.

## What it does

The [tactile task](../guide/tactile.md) asks a mouse to follow the vertical grating at two junctions to reach a rewarded arm. These three scripts ask whether the mouse learned, and how its movement differs between trials it got right and trials it got wrong.

| Script | Question | Needs tracking |
|---|---|---|
| `01_choice_accuracy.py` | Does the mouse choose the correct arm more often as training goes on? | No |
| `02_metrics_and_models.py` | On hit trials, does the mouse run faster and more directly than on misses? | Yes |
| `03_transition_probabilities.py` | Which compartment does the mouse go to next, and does that differ between hits and misses? | Yes |

They live in `analysis/simplermaze/first_paper_exploratory_analysis/` and analyse one mouse at a time.

## What you need

- The analysis extra: `pip install -e ".[analysis]"` from the repository root.
- For the Gamma mixed models of `02_metrics_and_models.py` (optional): R with the `lme4` package (`install.packages("lme4")` in R) and `pip install rpy2`. Without them the script runs its other tests.
- One mouse's sessions in this layout:

    ```
    <BASE_PATH>/mouse <MOUSE_ID>/
      habituation/
        mouse<ID>_session1.1_trial_info.csv
        rois1.csv
      <timestamp><ID>session<X.Y>/              for example 2024-08-29_10_23_026357session3.7
        new_session<X.Y>_trials.csv             or clean_mouse<ID>_session<X.Y>_trial_info.csv
        rois1.csv
      deeplabcut/                               tracking, for some sessions
        .../<ID>_<timestamp>s<X.Y>DLC_*.csv
    ```

## Quick start

1. The scripts take no command-line arguments. Open `analysis/simplermaze/first_paper_exploratory_analysis/config.py` and set the two lines at the top:

    ```python
    MOUSE_ID = "6357"
    BASE_PATH = os.path.join(os.path.expanduser("~"), "Box", "Awake Project", "Maze data", "simplermaze")
    ```

2. Check that the sessions are found, then run the three steps:

    ```bash
    cd analysis/simplermaze/first_paper_exploratory_analysis
    python config.py
    python 01_choice_accuracy.py
    python 02_metrics_and_models.py
    python 03_transition_probabilities.py
    ```

    `python config.py` lists every session it found, whether it has tracking, and which trial table it will read.

Everything is written to `<BASE_PATH>/mouse <MOUSE_ID>/MOUSE_<MOUSE_ID>_TOTAL_ANALYSIS/`.

## Details

### Step 1: choice accuracy

Loads the trial tables of every session, from habituation to the last. It recomputes each trial's outcome from whether the reward area the mouse visited first (`first_reward_area_visited`, whose last letter is the arm) was the rewarded arm (`rew_location`), as a check on the `hit`, `miss` and `incorrect` columns, and leaves out trials where the mouse entered no reward arm. A binomial mixed model, `correct ~ session + (1|mouse_id)`, tests whether accuracy changes across sessions.

| Output | Contents |
|---|---|
| `choice_accuracy_across_sessions.png`, `.pdf` | Correct, incorrect and no choice per session |
| `choice_accuracy_summary.csv` | Counts and percentage correct per session |
| printed | The mixed model's coefficients and p-values |

**Reading it.** A learning curve: if the bars move from mostly incorrect to mostly correct across sessions, the mouse is learning, and the p-value of the session coefficient says whether the trend is significant.

### Step 2: metrics by phase, and models

For the sessions with tracking (3.6, 3.7 and 3.8), splits each trial into phase 1 (entering the maze to reaching the first reward area) and phase 2 (from there to the end of the trial). For each phase it measures duration, mean speed (cm/s, Savitzky-Golay smoothed) and spatial entropy, and compares hit with miss trials by a Mann-Whitney U test, a linear mixed model (statsmodels) and a Gamma mixed model (R, through rpy2), as speed and duration are skewed.

| Output | Contents |
|---|---|
| `master_behavioural_data.csv` | One row per trial: session, outcome, duration, speed and entropy of each phase |
| `violin_plots.png`, `.pdf` | Six panels, phase by measure, with the p-values |
| `stats_report.csv` | Every p-value in one table |
| printed | The Gamma mixed models from R |

**Reading it.** Do mice run faster in phase 1 on hit trials? Is their path more direct (lower entropy)? Do they stay longer in the reward area on hits?

### Step 3: transition probabilities

For the sessions with tracking, assigns each frame to a compartment (`entrance1`, `entrance2`, `rewA` to `rewD`, or corridor), collapses repeats into a sequence of moves, and builds a transition matrix per trial, pooled separately for hits and misses. It also measures perseveration (how often the mouse goes back to the arm it just left), exploration entropy (how evenly it visits the arms) and the number of compartments visited.

| Output | Contents |
|---|---|
| `transition_combined.png`, `.pdf` | Transition maps for hits, for misses, and their difference |
| `exploration_metrics.png`, `.pdf` | Perseveration, exploration entropy and compartments visited, hits against misses |
| `transition_summary.csv` | One row per trial: the sequence and the three measures |

**Reading it.** Read a transition map row by row: given the mouse is in the row's compartment, how likely is it to go to the column's next? The difference map shows where hits and misses part ways.

??? note "Columns of the trial table the scripts use"
    | Column | Meaning |
    |---|---|
    | `rew_location` | The rewarded arm (A, B, C or D) |
    | `first_reward_area_visited` | The reward area the mouse visited first (for example `rewB`) |
    | `rewA`, `rewB`, `rewC`, `rewD` | Time in each arm, in ms; empty if not visited |
    | `hit`, `miss`, `incorrect` | The outcome recorded during the session |
    | `start_trial_frame`, `end_trial_frame` | Video frames of the trial, to align the tracking |

??? note "Settings in `config.py`"
    Besides `MOUSE_ID` and `BASE_PATH`, `config.py` sets `FPS = 30`, the tracked body part (`BODYPART = "mid"`), the minimum tracking likelihood (`LIKELIHOOD_THRESH = 0.8`) and the scale (`PX_PER_CM = 7.5`).

## Troubleshooting

??? question "`python config.py` finds no sessions"
    Check `BASE_PATH` and `MOUSE_ID`: the scripts look in `<BASE_PATH>/mouse <MOUSE_ID>/` (with a space). Session folders must contain `session<X.Y>` in their name and a trial table named as in the layout above.

??? question "Steps 2 and 3 skip most sessions"
    They need tracking, and only the sessions with a DeepLabCut CSV under `deeplabcut/` have it. `python config.py` says which.

??? question "The Gamma models are missing from the output"
    R, `lme4` or `rpy2` is not installed. The other tests still run.
