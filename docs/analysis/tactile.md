# Tactile analysis pipeline

Processes DeepLabCut or SLEAP pose data alongside the trial CSVs to produce figures and statistics for the tactile paradigm.

In addition to `requirements.txt`, the analysis pipeline needs:

```bash
pip install rpy2          # optional: for GLMM via R's lme4
pip install statsmodels   # already in requirements.txt
```

If using the GLMM features, you also need R installed with the `lme4` package:

```r
install.packages("lme4")
```

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

**Choice accuracy plot:** A learning curve. If the bars shift from mostly red (incorrect) to mostly green (correct) across sessions, the mouse is learning the task. The GLMM p-value for the session coefficient tells you whether this trend is statistically significant.

**Violin plots:** Each panel shows the distribution of a metric for Hit vs Miss trials. Key comparisons:
- P1 speed: do mice run faster on trials where they find the reward?
- P1 entropy: do successful trials show more directed (lower entropy) trajectories?
- P2 duration: do mice spend more time in the reward zone on Hit trials?

**Transition heatmaps:** Read row-by-row: "given the mouse is in row-ROI, what is the probability it goes to column-ROI next?" The difference map highlights where Hit and Miss trials diverge -- e.g., Hit trials may show stronger corridor-to-correct-arm transitions.

**Exploration metrics:** Perseveration rate > 0 means the mouse tends to revisit the same arm after leaving it. Higher exploration entropy means more uniform visiting across arms.
