# Earlier batch pipeline (legacy)

!!! warning "Not the thesis analyses"
    These scripts came before the thesis analyses and are kept so that earlier results can be reproduced. For anything you report, use the [thesis scripts](auditory.md). Two differences matter: these scripts cap every visit at 10 s, and the 4D aesthetic value model below is not the model of the thesis.

All of them live in `analysis/auditory/` and read Experiment 1 (the folder `8_arms_w_voc/`). A technical report on the 4D model is in [`analysis/auditory/REPORT_aesthetic_value_model.md`](https://github.com/MaravallLab/aMAZEing-maze/blob/main/analysis/auditory/REPORT_aesthetic_value_model.md).

## Quick start

```bash
cd analysis/auditory
python preference_analysis_config.py                        # check which sessions are found
python run_batch_preference.py D:/AuditoryExperiments/8_arms_w_voc
python 03_completers_lmm.py D:/AuditoryExperiments/8_arms_w_voc
python aesthetic_value_model_4D.py
```

`run_batch_preference.py` takes the data folder as its argument. Without it, it falls back on the `MAZE_DATA_DIR` environment variable, then on `BASE_PATH` in `preference_analysis_config.py`. It calls `02_within_trial_preference.py` itself at the end.

## The scripts

| Script | Question |
|---|---|
| `preference_analysis_config.py` | Which sessions are there, and what did each mouse do in each arm? |
| `run_batch_preference.py` | What is each mouse's preference index on each day, and does it differ between days? |
| `01_preference_analysis.py` | The same as `run_batch_preference.py`, without the batch summary table. |
| `02_within_trial_preference.py` | Within each session, how long did a mouse stay in sound arms against the silent arm? |
| `03_completers_lmm.py` | Among the mice present on every day, does the index change across days? |
| `04_random_slopes_presentation.py` | Does each completer have its own trajectory across sessions? |
| `check_visit_outliers.py` | Which sessions have implausibly long visits? |
| `aesthetic_value_model_4D.py` | Does a four-dimensional value model predict the index of each stimulus on each day? |

??? note "Settings and environment variables"
    `preference_analysis_config.py` holds the shared settings and the session loader.

    - `BASE_PATH`: the folder holding `w1_d1/`, `w1_d2/` and the rest. The `MAZE_DATA_DIR` environment variable overrides it.
    - `VISIT_CLIP_MS` (environment variable): the cap on each visit, in milliseconds, default 10000.

    Running `python preference_analysis_config.py` on its own prints the sessions it finds, which is the quickest check that the data folder is right.

??? note "Data layout"
    ```
    8_arms_w_voc/
      w1_d1/                        day 1: temporal envelope modulation
        time_2025-06-04_14_22_30mouse10049/
          trials_time_2025-06-04_14_22_30.csv
          mouseXXXXX_..._detailed_visits.csv   optional per-visit log
      w1_d2/                        day 2: consonant and dissonant intervals
      w1_d3/                        day 3: consonant and dissonant intervals
      w1_d4/                        day 4: intervals, no silent control
      w2_sequences/                 week 2: tone sequences
      w2_vocalisations/             week 2: mouse vocalisations
    ```

??? note "The preference index of this pipeline"
    The index compares sound arms with the silent arm, both measured in the sound blocks (trials 2, 4, 6 and 8) only, so a 15 minute sound block is never compared with a 2 minute silent block:

    ```
    PI = (mean sound visit - mean silent visit) / (mean sound visit + mean silent visit)
    ```

    It runs from -1 (only silence) to +1 (only sound). Each visit is capped at 10 s.

??? note "Why visits are capped"
    The 10 s cap was added against a suspected fault in which a visit still open at the end of a block ran on past it. The maze rules that out, because a new block starts only once the mouse has left the maze and entered again, and in the per-visit logs no visit over 10 s ends at its block's end. The cap therefore only shortens genuine long stays, which is why the thesis scripts do not use it; `exp1_vocalisation_pi.py` computes the index both ways so the effect can be seen. Separately, some rows of the aggregate table `trials.csv` hold a mean visit longer than any real visit, so the loader (`load_session_visits`) takes `detailed_visits.csv` as the ground truth where it exists and zeroes those rows when it falls back on `trials.csv`.

## Command-line options

??? note "`03_completers_lmm.py` and `04_random_slopes_presentation.py`"
    Both take the same options.

    | Option | Meaning |
    |---|---|
    | `data_path` | The `8_arms_w_voc` folder; `BATCH_ANALYSIS/` must exist inside it |
    | `--csv PATH` | Read this `preference_data.csv` instead (overrides `data_path`) |
    | `--required-days LIST` | Comma-separated days that define a completer; default `w1_d1,w1_d2,w1_d3,w2_sequences,w2_vocalisations` |
    | `--output-dir DIR` | Where to write; default beside the input CSV |

    ```bash
    python 03_completers_lmm.py D:/AuditoryExperiments/8_arms_w_voc --required-days w1_d1,w1_d2,w1_d3,w2_sequences,w2_vocalisations --output-dir D:/AuditoryExperiments/8_arms_w_voc/BATCH_ANALYSIS/completers
    ```

??? note "`check_visit_outliers.py`"
    | Option | Meaning |
    |---|---|
    | `--top N` | Show the N worst sessions (default 15) |
    | `--threshold MS` | Show every row with a mean visit longer than MS milliseconds (overrides `--top`) |
    | `--export PATH` | Write every flagged row to this CSV |

## Outputs

Everything goes to `BATCH_ANALYSIS/` inside the data folder.

??? note "Tables"
    | File | Contents |
    |---|---|
    | `preference_data.csv` | One row per mouse per session: the index, visit measures, roaming entropy, first-minute roaming entropy |
    | `stimulus_breakdown.csv` | Visit length by kind of stimulus |
    | `within_trial_preference.csv` | Sound and silent arm visit length, per mouse per day |
    | `voc_vs_other_sounds_pi.csv` | The vocalisation index beside the index of the other sounds, per session |
    | `stats_report.txt` | The full statistical report |
    | `aesthetic_model_4D_predictions.csv`, `aesthetic_model_4D_params.csv` | The 4D model's predictions and fitted parameters |

??? note "Figures"
    | File | Contents |
    |---|---|
    | `fig1_pi_trajectories` | Each mouse's index across days |
    | `fig2_pi_by_day` | Mean index per day with 95% confidence interval |
    | `fig3_pi_violins` | The distribution of the index per day |
    | `fig4_complexity_heatmap` | Visit length by kind of stimulus per day |
    | `fig5_vocalisation_contrast` | Vocalisation day against the other days |
    | `fig6_re_vs_pi`, `fig6b_re_firstmin_vs_pi` | Roaming entropy against the index |
    | `fig7_icc_summary` | Division of variance and Kruskal-Wallis test |
    | `fig_voc_vs_other_sounds_pi` | Vocalisation index against the index of the other sounds |
    | `fig_within_trial_preference` | Sound against silent arm, per session |
    | `aesthetic_value_model_4D` | The 4D model, six panels |

    Most are written as PNG and PDF; several also as interactive HTML (hover to identify a mouse).

??? note "Completers outputs (`03_completers_lmm.py`)"
    | File | Contents |
    |---|---|
    | `completers_summary.csv` | The completers' sessions |
    | `completers_lmm_fixed_effects.csv` | Fixed effects of every model and outcome |
    | `completers_lmm_variance.csv` | Variance components, ICC and R-squared per model |
    | `completers_lmm_random_intercepts.csv` | Each mouse's random intercept with 95% confidence interval |
    | `completers_stats_report.txt` | Likelihood ratio tests and R-squared |
    | `fig_completers_caterpillar`, `fig_completers_day_estimates`, `fig_completers_spaghetti` | The figures |

## Methods

??? note "Statistical tests of `run_batch_preference.py`"
    | Analysis | Method |
    |---|---|
    | Index per day against zero | One-sample Wilcoxon, rank-biserial effect size, Holm correction |
    | Index across days | Kruskal-Wallis, epsilon-squared |
    | Pairs of days | Dunn's test, Holm correction |
    | Complexity gradient | Jonckheere-Terpstra trend test |
    | Vocalisation against the overall index | Paired Wilcoxon |
    | Mixed model | `PI ~ day + (1|mouse)`, Nakagawa R-squared |
    | Model comparison | AIC, BIC, likelihood ratio test |
    | Sensitivity | Binomial GLM on the rescaled index |
    | Vocalisation against other sounds | Paired Wilcoxon per day (report section 12b) |
    | First-minute roaming entropy | Pearson and Spearman correlations (report section 12c) |

??? note "The completers models"
    `03_completers_lmm.py` keeps the mice present on every required day and fits four nested mixed models to each outcome (the index, the vocalisation index, the index of the other sounds, roaming entropy and its first minute):

    | Model | Formula |
    |---|---|
    | M0, null | `y ~ 1 + (1|mouse)` |
    | M1, day fixed | `y ~ C(day) + (1|mouse)` |
    | M2, linear trend | `y ~ session_num + (1|mouse)` |
    | M3, random slopes | `y ~ session_num + (1 + session_num|mouse)` |

    It reports REML estimates, refits with ML for the likelihood ratio tests (M0 against M1, M2 against M3), and gives the Nakagawa and Schielzeth (2013) R-squared.

??? note "The 4D aesthetic value model"
    A four-dimensional extension of the model of Brielmann and Dayan (2022), where each stimulus is a point in a space of location familiarity, spectral complexity, biological relevance and temporal predictability. The model walks an agent through the whole experiment and predicts the index from the difference in value between sound and silence. It is fitted from 2000 random starts of SLSQP, with lesioned models, dimension dropout and a split of location against acoustic value. See the [report](https://github.com/MaravallLab/aMAZEing-maze/blob/main/analysis/auditory/REPORT_aesthetic_value_model.md).
