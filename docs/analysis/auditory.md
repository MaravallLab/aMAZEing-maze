# Auditory pipeline: the thesis scripts

The scripts in `analysis/auditory/` named by experiment (`exp1_*`, `exp2_*`, `fibpho_*`, `pilot_four_arm.py`) produce every number and figure in the thesis. Use them for anything you report. To go from a thesis figure to its command, see [Reproduce the thesis](reproduce-thesis.md).

The earlier scripts in the same folder (`01_preference_analysis.py`, `run_batch_preference.py` and the rest) are described on [Earlier batch pipeline (legacy)](auditory-legacy.md).

## What you need

- The analysis extra, from the repository root: `uv sync --extra all`, or `pip install -e ".[all]"` ([Installation](../getting-started/installation.md#from-source)).
- For the mixed models, R 4 with `lme4`, `lmerTest` and `emmeans`. The scripts look for `Rscript` on the PATH, then in the newest R under `C:/Program Files/R`.
- For the Bayesian fits of Experiment 2, PyMC and ArviZ, as the [value model quick guide](model-validation.md) describes.
- For photometry, the `tdt` package, to read Tucker-Davis tanks. The `analysis` extra installs it, and the downloadable application includes it.

## Quick start

```bash
cd analysis/auditory
python exp1_stimulus_pi.py D:/AuditoryExperiments/8_arms_w_voc
```

Every script takes its data folder as the first argument and writes to `<data folder>/BATCH_ANALYSIS/<its name>/`. Nothing is written into the session folders. Exclusions are printed as a script runs, with the reason for each.

??? note "Running without an argument"
    The Experiment 1 scripts then use the `MAZE_DATA_DIR` environment variable, or the default in `preference_analysis_config.py`. The photometry scripts use `MAZE_PHOTOMETRY_DIR`. The Experiment 2 scripts default to `D:/AuditoryExperiments/grammar`.

??? note "The data layout"
    ```
    <data root>/
      8_arms_w_voc/            Experiment 1, one folder per stimulus set
        w1_d1/                   day 1: temporal envelope modulation
        w1_d2/ w1_d3/ w1_d4/     days 2 to 4: musical intervals
        w2_sequences/            day 5: tone sequences
        w2_vocalisations/        day 6: recorded vocalisations
      grammar/                 Experiment 2: habituation, day_1, day_2
      dopamine recordings/     photometry: TDT tanks, and the maze sessions recorded with them
      vocalisationzzzzzz/      the vocalisation recordings the maze played
      time_..._mouseN/         the four-arm pilot sessions, loose at the top
    ```

## Conventions every script shares

**The preference index** compares the mean visit length in two kinds of arm, both taken from the same sound blocks:

```
PI = (mean visit with a - mean visit with b) / (mean visit with a + mean visit with b)
```

It runs from -1 (only b) to +1 (only a). Visits are not capped. See the [glossary](../reference/glossary.md#preference-index).

**The first sound block keeps the listed order.** By design, the first sound block of a session places each stimulus in the arm its place in the list gives it; only the later blocks are reshuffled. The Experiment 1 scripts that compare stimuli take `--shuffled-only`, which uses blocks 2 to 4 alone and writes to `<name>_shuffled/`. The Experiment 2 scripts report blocks 2 to 4, all four blocks and block 1 alone.

**Thesis figures.** A script with `--thesis-figures DIR` also writes its figures, in the thesis fonts and sizes, into `DIR`.

## Experiment 1: eight arms, six stimulus sets

| Script | Question | Output in `BATCH_ANALYSIS/` |
|---|---|---|
| `exp1_vocalisation_pi.py` | Do mice prefer the vocalisation to silence and to the synthetic sounds? | `vocalisation_pi/` |
| `exp1_stimulus_pi.py` | Is each stimulus preferred to the silent arm? | `stimulus_pi/` |
| `exp1_familiarity.py` | Is the recording a mouse heard on days 1 to 5 preferred on the vocalisation day? | `familiarity/` |
| `exp1_lmm.py` | Does visit length depend on the kind of stimulus? | `lmm/` |
| `exp1_individuals.py` | Do the averages hold mouse by mouse? | `individuals/` |
| `exp1_consistency_checks.py` | Are some mice consistent about consonance, and does sound magnify place preference? | `consistency/` |
| `exp1_complexity.py` | Does preference follow how complex a stimulus is? | `complexity/` |
| `exp1_exploration.py` | Does exploring more in silence go with engaging more with sounds? | `exploration/` |
| `exp1_order_place.py` | Do day order or place explain the preferences? | `order_place/` |
| `exp1_approach.py` | Is the effect in entering an arm, or in staying? | `approach/` |
| `exp1_bd_model.py` | Does the model of Brielmann and Dayan account for every index? | `bd_model/` |
| `pi_fisher_figure.py` | How does the index relate to its Fisher z? (Figure 2.7) | `pi_fisher/` |
| `fisher_z_check.py` | Does analysing the index on the Fisher z scale change any result? | `fisher_z_check/` |

```bash
cd analysis/auditory
python exp1_vocalisation_pi.py    D:/AuditoryExperiments/8_arms_w_voc
python exp1_stimulus_pi.py        D:/AuditoryExperiments/8_arms_w_voc
python exp1_lmm.py                D:/AuditoryExperiments/8_arms_w_voc
python exp1_complexity.py         D:/AuditoryExperiments/8_arms_w_voc
python exp1_order_place.py        D:/AuditoryExperiments/8_arms_w_voc
python exp1_approach.py           D:/AuditoryExperiments/8_arms_w_voc
python exp1_exploration.py        D:/AuditoryExperiments/8_arms_w_voc
python exp1_individuals.py        D:/AuditoryExperiments/8_arms_w_voc
python exp1_consistency_checks.py D:/AuditoryExperiments/8_arms_w_voc
python exp1_bd_model.py           D:/AuditoryExperiments/8_arms_w_voc
```

Run them in this order: later scripts read the tables of earlier ones. [Reproduce the thesis](reproduce-thesis.md#the-order-to-run-them-in) has the diagram.

??? note "`exp1_vocalisation_pi.py`"
    Three indices per day: the vocalisation against silence, the synthetic sounds against silence, and the vocalisation against the synthetic sounds. Each is computed with and without the 10 s cap of the earlier pipeline, so the effect of the cap can be seen.

??? note "`exp1_stimulus_pi.py`"
    Each stimulus against the silent arm, then two contrasts: consonant against dissonant intervals, and rough against smooth envelopes. Writes `stimulus_pi.csv` and `contrasts.csv`, which later scripts read. `--shuffled-only` uses blocks 2 to 4 and writes to `stimulus_pi_shuffled/`.

??? note "`exp1_lmm.py` (calls `exp1_lmm.R`)"
    Mixed models of visit length: class by set, consonance, roughness, modulation. Also writes `arm_blocks.csv`, the arm by block table that `exp1_approach.py`, `exp1_individuals.py`, `exp1_bd_model.py` and `complexity_3d.py` read. `--shuffled-only` writes its results to `lmm_shuffled/`.

??? note "`exp1_individuals.py` (calls `exp1_individuals.R`)"
    Each stimulus against silence in a mixed model beside the mice; every mouse's index for every stimulus and day; the vocalisation day by day, mouse by mouse; how the variance of the index divides between day, stimulus, mouse and session; principal components of what each session did. Run `exp1_stimulus_pi.py` and `exp1_lmm.py` first.

??? note "`exp1_consistency_checks.py`"
    Two questions raised on Chapter 4.

    1. **Consonance in single mice.** Over the group, consonant intervals were neither preferred nor avoided. Were some mice consistent all the same? The script correlates each mouse's consonant against dissonant index between the three interval sets, tests the spread of the mouse means by permutation, and counts the mice with the same sign on all three sets.
    2. **Does sound magnify the place preference?** It compares the share of time in the favoured arms (1, 2, 7 and 8) in sound blocks against silent blocks of the same session, with mice as the unit (Wilcoxon signed rank). Three versions: arms 1, 2, 7 and 8; each session's four favoured arms in its habituation block; and the reshuffled sound blocks alone.

    Reads `stimulus_pi/contrasts.csv`, so run `exp1_stimulus_pi.py` first.

    | Option | Meaning |
    |---|---|
    | `data_dir` | The `8_arms_w_voc` folder |
    | `--perms N` | Permutations for the test of the mouse means (default 10000) |
    | `--thesis-figures DIR` | Also write `fig_exp1_consonance_mice` (Figure B.3) into `DIR` |

    Writes `stats.txt`, `consonance_by_mouse.csv`, `place_sound_silent.csv` and `fig_consonance_by_mouse` to `consistency/`.

??? note "`exp1_complexity.py`"
    Does preference follow spectral or temporal complexity, in a line or an inverted U? Also within the seven recordings of the vocalisation day, by spectral and temporal entropy. Run `exp1_stimulus_pi.py` first. The second part reads the recordings themselves from the folders in `VOCALISATION_FOLDERS` at the top of the script, and skips that part if none exists. `--shuffled-only` writes to `complexity_shuffled/`.

??? note "`exp1_order_place.py`"
    Day order against habituation, and place bias: three indices block by block, arm shares in habituation and sound blocks, the usual silent arm, first entries. Writes `arm_trials.csv`, which `exp1_approach.py` reads.

??? note "`exp1_approach.py` (calls `exp1_approach.R`)"
    Entries per arm against mean visit length, with a Poisson mixed model of the entries. Run `exp1_lmm.py` and `exp1_order_place.py` first.

??? note "`exp1_bd_model.py`"
    Nested models (intercept, vocalisation only, fluency, Brielmann and Dayan, the same with the expected world fixed at the centre, and with an extra value for a vocalisation), each stimulus described only by its measured spectral and temporal complexity. Fitted by least squares; compared by AIC and by predicting each left out day. Run `exp1_stimulus_pi.py`, `exp1_complexity.py` and `exp1_lmm.py` first.

??? note "`fisher_z_check.py` (calls `fisher_z_check.R`)"
    Refits every model fitted to the index itself with artanh(PI) in place of PI and sets the answers side by side. Run it last, as it reads the tables of most other scripts and of the pilot. `--survey DIR` points at the pilot survey (default `D:/AuditoryExperiments_survey`); `--skip-model` leaves out the slow refit of the model of Brielmann and Dayan.

## Experiment 2: the grammar experiment

| Script | Question | Output in `BATCH_ANALYSIS/` |
|---|---|---|
| `exp2_lmm.py` | Do mice prefer the grammar of their enriched cage, with place controlled? | `exp2/` |
| `model_validation/` | Does a semantic value term explain the preference? | the folder you choose |
| `exp2_model_compare.py` | Which of the four nested value models fits best by AIC and BIC? | `exp2_model/` |
| `exp2_model_check.py` | Could place alone produce the semantic weight? | `exp2_model/` |
| `exp2_model_explainer.py` | How is the model fitted, step by step? (Figure B.4) | `exp2_model/` |
| `exp2_figures.py` | The thesis figures of Chapter 6 | the folder given to `--thesis-figures` |
| `exp2_power.py` | How many mice does the main test need? | `exp2/power.txt` |
| `bayes_principle_figure.py` | What is Bayesian fitting? (Figure 2.8, reads no data) | `bayes_principle/` |

```bash
cd analysis/auditory
python exp2_lmm.py             D:/AuditoryExperiments/grammar
python exp2_model_compare.py   D:/AuditoryExperiments/grammar
python exp2_model_check.py     D:/AuditoryExperiments/grammar
python exp2_model_explainer.py D:/AuditoryExperiments/grammar
python exp2_figures.py         D:/AuditoryExperiments/grammar --thesis-figures <thesis>/figures
python exp2_power.py           D:/AuditoryExperiments/grammar
python bayes_principle_figure.py --thesis-figures <thesis>/figures
```

`exp2_model_explainer.py` and `exp2_figures.py` also read the Bayesian fits in `grammar/model_validation_output/`. [Reproduce the thesis](reproduce-thesis.md#running-the-value-model) has the commands that write them.

??? note "Options"
    | Script | Option | Meaning |
    |---|---|---|
    | `exp2_model_check.py` | `--boot N` | Bootstrap resamples (default 500) |
    | `exp2_model_check.py` | `--sims N` | Simulations (default 200) |
    | `exp2_power.py` | `--dz D`, `--dz-day2 D` | Effect sizes to use when the data are absent (defaults 0.69 and 0.57) |
    | `exp2_power.py` | `--sims N` | Simulations (default 20000) |
    | `bayes_principle_figure.py` | `out_dir` | Output folder (default `D:/AuditoryExperiments/grammar/BATCH_ANALYSIS/bayes_principle`) |

    `exp2_model_compare.py` takes only the grammar folder.

??? note "`exp2_model_compare.py`"
    AIC and BIC for the four nested models (intercept, fluency, Brielmann and Dayan, full), at the grain of the session-level preference on day 1: each mouse's share of its time in each of the seven kinds of arm over the whole session. Two likelihoods, Dirichlet and Gaussian, so that the comparison does not rest on one choice. Writes `information_criteria.csv` and `information_criteria.txt`.

The `exp2_*` scripts read `model_validation/` (its data loading, exclusions and features) but never change it or its outputs.

### Exploratory, viva version only

These two scripts are exploratory. They are not part of the preregistered or confirmatory analyses and are left out of the submitted thesis. Their results appear only in the viva version (`future_main.pdf`, the passages inside `\ifdefined\FUTUREVERSION` in `chapters/discussion.tex`).

| Script | Question | Output |
|---|---|---|
| `exp2_listener_check.py` | Which simple listener reads a secondary melody as its own grammar? (no mouse data) | `exp2_listener_check.txt`, and with `ddm`, `exp2_listener_ddm.txt`, next to the script |
| `exp2_evidence_accumulation.py` | How many tone transitions does a listener, and a mouse, need to tell which cage a melody belongs to, at each level? | `exp2_evidence_accumulation/` next to the script |

```bash
cd analysis/auditory
python exp2_evidence_accumulation.py D:/AuditoryExperiments/grammar
```

??? note "`exp2_evidence_accumulation.py`"
    The question in plain words: if you hear a piano melody, how many notes do you need before you can say whether it is classical or jazz? Here the two styles are the grammar of the enriched cage (EE) and the grammar of the standard cage (SC). Many single steps are shared by the two grammars, so one transition is rarely enough.

    **Part 1, simulation (reads no data).** A listener hears the melodies of one arm and after each transition adds that step's evidence (the log of how much more likely the step is under the EE grammar than under the SC grammar) to a running total. The total moves like a ball rolling towards one of two walls, a drift diffusion process; the walls are at 95 and 5 percent belief, and the listener decides when the ball first touches one. The listener keeps only its last W transitions in mind, W from 1 to a whole melody (11), two melodies (22) and the whole visit (5 melodies, 55 transitions). A leaky accumulator, which forgets a little of the old evidence at every step, behaves much like a short window, so the two cannot be told apart. Two listeners, both from `exp2_listener_check.py`: the step by step listener (each step weighed against the full grammars) and the listener that also infers the level of the melody (each cage a mixture of its three level restricted chains). For each level it gives the share of visits reaching the right wall, the wrong wall or neither, and the transitions to a decision, against transitions heard and against W. Averaged over both cages and both counterbalancing groups (15 and 17 mice), seed 20261119, 4000 simulated visits per group, cage and level.

    **Part 2, mouse data (test day 1).** Each arm entry starts a fresh melody (melody, 2 s gap, repeat), so a visit's length gives the transitions the mouse heard: one per 0.2 s within a melody, counted once the second tone has ended, nothing new in the gap. Reads the per visit tables (`*_grammar_detailed_visits.csv`) of the day 1 sessions through `model_validation.data_loading` (same exclusions as the other `exp2_*` scripts) and checks that their times add up to the trials tables. Two simple measures per level, both pooled over mice with 95 percent intervals from a bootstrap over mice (2000 resamples): the EE share of the visits that lasted at least n transitions, and the leaving rate (of the visits still going after n transitions, the share that ended before the next) in EE and SC arms. Each is given for all visits and for the first visit to each arm in each block. No p values and no correction: read the onsets as a rough range.

    Writes to `analysis/auditory/exp2_evidence_accumulation/`: `simulation.txt`, `simulation_summary.csv`, `simulation_curves.csv` (Part 1); `mice.txt`, `mice_survivor_share.csv`, `mice_leave_rates.csv` (Part 2, summaries only); `evidence_accumulation.png` and `.pdf` (the figure, copied into the thesis as `figures/fig_exp2_evidence_accumulation.pdf`). Without the data folder only Part 1 runs and the figure has no mouse row. Takes about a minute. The repository ignores CSV files, so only the two text files and the figure are committed; every number the viva version quotes is in the text files.

## Stimulus measures

`stimulus_complexity.py` holds the measures of how complex a stimulus is. Run on its own, it prints a worked table.

| Measure | What it is |
|---|---|
| Spectral complexity | The normalised entropy of the power spectrum, 0 to 96 kHz: about 0.10 for a pure tone, near 1 for noise |
| Temporal complexity | The cycles of the lowest frequency before the waveform repeats, in bits: 0 for a pure tone, log2 q for an interval p/q |
| Temporal entropy | How evenly a recording's energy is spread over time, in 5 ms frames (after Sueur et al. 2008) |

`complexity_3d.py` draws the complexity figures in three dimensions (Figures 3.5 and 4.17), as pages that can be turned with the mouse and open without a connection.

??? note "`complexity_3d.py` options"
    Run `pilot_four_arm.py`, `exp1_complexity.py` and `exp1_lmm.py` first.

    | Option | Meaning |
    |---|---|
    | `--survey DIR` | The pilot survey folder (default `D:/AuditoryExperiments_survey`) |
    | `--exp1 DIR` | The Experiment 1 folder (default `D:/AuditoryExperiments/8_arms_w_voc`) |
    | `--gallery DIR` | Also write each view as JSON for a page that loads plotly itself |
    | `--thesis-figures DIR` | Also write still images through `kaleido` (`pip install kaleido`, which drives an installed Chrome) |

    It writes `pilot/pilot_complexity_3d.html`, `complexity/exp1_complexity_3d.html` and the numbers the thesis quotes to `complexity/stats_3d.txt`.

## The four-arm pilot

The sessions recorded before the eight-arm protocol sit loose at the top of the data folder, written by several versions of the code.

```bash
python analysis/auditory/survey_standalone_sessions.py --root D:/AuditoryExperiments
python analysis/auditory/pilot_four_arm.py --survey D:/AuditoryExperiments_survey
```

`survey_standalone_sessions.py` reads every session on its own terms and writes `sessions.csv` and `blocks.csv` to a folder beside the data (`--out` to change it). `pilot_four_arm.py` describes the three cohorts from those tables and tests them as the thesis does for Experiment 1; the numbers go to `pilot/stats.txt` and `pilot/lmm_results.txt`.

??? note "What the pilot tests"
    Wilcoxon signed rank tests across mice (Holm within a family) cover sound against silent blocks, the arm each mouse favoured before any sound, the indices of the interval sessions and consonant against dissonant pairs. A mixed model relates the index to temporal complexity. The mixed models of entries and visit length run in R (`pilot_lmm.R`). With two to five mice per cohort the cohorts are pooled, and the tests describe these animals rather than mice in general.

## Photometry

Dopamine recordings (Tucker-Davis) made during maze sessions. `fibpho_alignment.py` runs first; the rest read what it saves.

| Script | Question | Output in `FIBPHO_ANALYSIS/` |
|---|---|---|
| `fibpho_alignment.py` | Which recording belongs to which session, and what is its delta F/F? | `<day>/<mouse>/`, `alignment_summary.csv` |
| `fibpho_checks.py` | Is the response to arm entries real? | `checks/` |
| `fibpho_arms.py` | Does dopamine in an arm follow the preference for it? | `arms/` |
| `fibpho_dynamics.py` | How does the response change within and across blocks? | `dynamics/` |
| `fibpho_behaviour.py` | How does dopamine depend on where the mouse was, and on repeated exposure? | `behaviour/` |
| `fibpho_dwell.py` | How does dopamine change before, during and after a visit? | `dwell/` |

```bash
cd analysis/auditory
python fibpho_alignment.py "D:/AuditoryExperiments/dopamine recordings"
python fibpho_checks.py    "D:/AuditoryExperiments/dopamine recordings"
python fibpho_dwell.py     "D:/AuditoryExperiments/dopamine recordings/FIBPHO_ANALYSIS"
```

`fibpho_arms.py`, `fibpho_dynamics.py` and `fibpho_behaviour.py` take the same `FIBPHO_ANALYSIS` folder as `fibpho_dwell.py`. `fibpho_alignment.py --out DIR` writes somewhere else. With two mice, every photometry test is within these two animals.

??? note "Expected layout"
    ```
    dopamine recordings/
      dopaminergic recordings/        one TDT block (tank) per recording
      temporal_envelope_modulation/   one folder per experiment day, holding the maze sessions
      intervals/
      sequences/
      vocalisations/
    ```

??? note "How the alignment works"
    The TDT system and the maze keep separate clocks. The maze sends a TTL pulse on every sound, and every difference between a logged sound onset and a TTL onset is a candidate offset between the clocks. The true offset is the one most of them share, so stray pulses (tests sent before the session, a restarted session) do not move it. The block with the most matched pulses is the session's own.

    Delta F/F divides each channel by a double exponential fitted to remove bleaching, then corrects the 465 nm signal with the 405 nm isosbestic channel, fitted separately within each block. Every arm entry becomes an event, timed by its TTL where one matched and by its logged onset otherwise.

## Other files in the folder

The [README in `analysis/auditory/`](https://github.com/MaravallLab/aMAZEing-maze/blob/main/analysis/auditory/README.md) sorts every file into thesis scripts, legacy pipeline, helpers and notebooks.

??? note "`calibration_volume.py`, a speaker check from the first version"
    Plays test tones through the sound card at 192 kHz so the output can be checked with a microphone and a sound meter. It has no command-line options. With `found_channel = False` (the setting in the file) it tries the audio devices 0 to 29 in turn and plays 10, 16 and 40 kHz tones on each, so you can hear which device is the speaker; set it to `True` and set `sd.default.device` to play six test tones at set volumes instead.

    It imports `supfun_sequences`, which is not in `analysis/auditory/` but in `archive/auditory_v1/`, so it does not run from this folder as it stands. To calibrate a speaker today, use the [speaker calibration](../guide/speaker-calibration.md) tab of the application.

??? note "`automatic_roi_selection_posthoc_time_in_maze.py`"
    Measures, after the fact, how long the mouse was in the maze in each trial, from the session video. You draw the two entrance regions once on a reference session; the script carries them to every other session by homography and asks you to review them. It takes a top-level folder, a day folder or a single session folder.

    !!! warning
        It writes `time_in_maze_ms` back into each session's trials CSV, and `rois1.csv` into the reference session. Work on a copy of the data.

    | Option | Meaning |
    |---|---|
    | `--threshold F` | Mouse-present factor against the baseline (default 0.5) |
    | `--manual-offset-secs S` | Shift video time against the trial clock (default 0) |
    | `--draw-scale F` | Zoom of the drawing window (default 2.0) |
    | `--review-rois`, `--no-review-rois` | Review each session's regions by hand (on by default) |
    | `--auto-scale F` | Scale images before matching, for speed (default 1.0) |
    | `--no-manual-fallback` | Mark a session as failed, rather than ask you to draw, when matching fails |
    | `--save-overlays` | Save a PNG of the regions in each session, for checking |
