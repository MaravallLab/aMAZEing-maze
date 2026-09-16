# Auditory analysis pipeline

The auditory analysis pipeline lives under `analysis/auditory/`. It processes visit data from the 8-arm radial maze across 6 experiment days, producing publication-ready figures, interactive visualisations, and comprehensive statistical reports.

For a detailed technical report on the computational modelling component, see [`analysis/auditory/REPORT_aesthetic_value_model.md`](https://github.com/MaravallLab/aMAZEing-maze/blob/main/analysis/auditory/REPORT_aesthetic_value_model.md).

**Prerequisites:**

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels plotly
```

**Expected data layout:**

```
8_arms_w_voc/
  w1_d1/                            # Day 1: temporal envelope modulation
    time_2025-06-04_14_22_30mouse10049/
      trials_time_2025-06-04_14_22_30.csv
      mouseXXXXX_..._detailed_visits.csv   # optional ground-truth log
    ...
  w1_d2/                            # Day 2: consonant/dissonant intervals
  w1_d3/                            # Day 3: consonant/dissonant intervals
  w1_d4/                            # Day 4: intervals (no silent control)
  w2_sequences/                     # Week 2: tone sequences
  w2_vocalisations/                 # Week 2: mouse vocalisations
```

**Configuration:**

Edit `preference_analysis_config.py`:
- `BASE_PATH` -- root folder containing `w1_d1/`, `w1_d2/`, etc.
- Or set the `MAZE_DATA_DIR` environment variable to override.
- `VISIT_CLIP_MS` env var -- per-visit duration cap in ms (default 10000).

**Running:**

```bash
cd analysis/auditory
# Verify session discovery first:
python preference_analysis_config.py

# Option A: standalone single-pipeline run (no batch summary CSV):
python 01_preference_analysis.py

# Option B (recommended): full batch analysis (generates all CSVs + figures + stats,
# automatically calls 02_within_trial_preference.py at the end):
python run_batch_preference.py

# Optional: completers-only linear mixed model on the batch CSV
# (run AFTER run_batch_preference.py so preference_data.csv exists):
python 03_completers_lmm.py /path/to/8_arms_w_voc

# Optionally check visit-duration outliers:
python check_visit_outliers.py

# Run the computational model:
python aesthetic_value_model_4D.py
```

**Scripts overview:**

| Script | Description |
|--------|-------------|
| `preference_analysis_config.py` | Shared configuration, session discovery, data loading with DV-first corruption handling, and `compute_first_minute_re` helper |
| `01_preference_analysis.py` | Standalone single-pipeline driver for PI + RE + first-minute RE + voc-vs-other-sounds analysis (mirrors `run_batch_preference` outputs without the batch summary CSV) |
| `run_batch_preference.py` | Main batch pipeline: per-mouse/day PI computation, 8+ static figures, interactive Plotly figures, enhanced statistics; auto-invokes `02_within_trial_preference.py` |
| `02_within_trial_preference.py` | Within-trial scatter plots: sound vs silent-arm visit duration per mouse per day |
| `03_completers_lmm.py` | Completers-only linear mixed-model analysis: filters mice present on all required days and fits four nested LMMs (null / day fixed / linear trend / random slopes) on PI, voc PI, other-sounds PI, RE, and first-minute RE |
| `check_visit_outliers.py` | Diagnostic tool for identifying and reporting visit-duration outliers |
| `aesthetic_value_model_4D.py` | Brielmann & Dayan (2022) aesthetic value model -- 4D extension for mouse acoustic preference |

**Outputs** (saved to `BATCH_ANALYSIS/` inside the data folder):

| File | Description |
|------|-------------|
| `preference_data.csv` | Per-mouse, per-session PI + visit metrics. Includes `preference_index`, `voc_pi`, `other_sounds_pi`, `avg_voc_dur_ms`, `avg_other_sounds_dur_ms`, `roaming_entropy`, and `re_first_min` |
| `stimulus_breakdown.csv` | Per-stimulus-type visit duration |
| `within_trial_preference.csv` | Per-mouse, per-day sound vs silent-arm durations |
| `voc_vs_other_sounds_pi.csv` | Per-session voc PI and other-sounds PI side-by-side, including the average voc / other-sounds / silent durations used to compute each index |
| `fig1_pi_trajectories.png/pdf` | Individual mouse PI trajectories across days |
| `fig2_pi_by_day.png/pdf` | Mean PI per day with 95% CI |
| `fig3_pi_violins.png/pdf` | Violin plots of PI distribution by day |
| `fig4_complexity_heatmap.png/pdf` | Visit duration by stimulus type per day |
| `fig5_vocalisation_contrast.png/pdf` | Paired comparison: vocalisation vs other days |
| `fig6_re_vs_pi.png/pdf` | Roaming entropy vs preference (within & between mouse) |
| `fig6b_re_firstmin_vs_pi.png/pdf` | First-minute roaming entropy (first 60 s of habituation) vs PI, within- and between-mouse panels |
| `fig7_icc_summary.png/pdf` | Variance decomposition (ICC) + Kruskal-Wallis |
| `fig8_*.png/pdf` | Additional analysis panels |
| `fig_voc_vs_other_sounds_pi.png/pdf` | Per-day scatter of voc PI vs other-sounds PI plus pooled panel |
| `fig_within_trial_preference.png/pdf/html` | Within-trial preference scatter (interactive Plotly version with hover) |
| `fig1_*.html`, `fig3_*.html`, etc. | Interactive Plotly versions of main figures (hover to identify mice) |
| `stats_report.txt` | Full statistical report (descriptive, inferential, enhanced analyses, plus sections 12b voc vs other-sounds and 12c first-minute RE) |
| `aesthetic_value_model_4D.png/pdf` | 6-panel computational model figure |
| `aesthetic_model_4D_predictions.csv` | Per-stimulus, per-day model predictions |
| `aesthetic_model_4D_params.csv` | Best-fit model parameters |

**Completers LMM outputs** (produced by `03_completers_lmm.py`, saved to `BATCH_ANALYSIS/completers/` by default):

| File | Description |
|------|-------------|
| `completers_summary.csv` | Filtered subset (mice present on all required days) with PI, voc PI, other-sounds PI, RE, and first-minute RE per session |
| `completers_lmm_fixed_effects.csv` | Fixed-effects estimates (coef, SE, z, p, 95% CI) for every fitted model and outcome |
| `completers_lmm_variance.csv` | Variance components (between-mouse, residual, ICC, marginal R-squared, conditional R-squared) per model |
| `completers_lmm_random_intercepts.csv` | Per-mouse BLUPs (random intercepts) with 95% CI from the day-fixed model |
| `completers_stats_report.txt` | Human-readable summary including LRT comparisons (M0 vs M1, M2 vs M3) and Nakagawa & Schielzeth (2013) R-squared |
| `fig_completers_caterpillar.png/pdf` | Per-mouse random-intercept caterpillar plot (BLUPs ranked with 95% CI) |
| `fig_completers_day_estimates.png/pdf` | Estimated marginal means by day from the day-fixed LMM |
| `fig_completers_spaghetti.png/pdf` | Per-mouse trajectories across sessions (one line per completer) |

#### Key analyses

**1. Preference Index (PI):**

PI is computed using within-trial comparison: sound-playing ROIs vs the silent-control ROI, both measured during sound trials (2, 4, 6, 8) only. This avoids bias from comparing 15-min sound trials against 2-min silent trials.

```
PI = (Avg_Sound_Duration - Avg_Silent_Duration) / (Avg_Sound_Duration + Avg_Silent_Duration)
```

Ranges from -1 (prefer silence) to +1 (prefer sound).

**2. Data integrity:**

The pipeline handles a known trial-boundary bug in the experiment control code where visits were not closed at trial end, producing inflated duration values in `trials.csv`. The loader (`load_session_visits`) uses `detailed_visits.csv` as ground truth where available, with a 10 s per-visit clip and sanity caps for fallback to `trials.csv`.

**3. Enhanced statistical analyses:**

| Analysis | Method | Description |
|----------|--------|-------------|
| Per-day PI test | One-sample Wilcoxon + rank-biserial effect size | Tests whether PI differs from zero on each day, with Holm-Bonferroni correction |
| Across-day comparison | Kruskal-Wallis + epsilon-squared | Tests whether PI differs across experiment days |
| Post-hoc pairwise | Dunn's test (Holm-corrected) | Identifies which day pairs differ significantly |
| Complexity gradient | Jonckheere-Terpstra trend test | Tests monotone trend in dwell time along the complexity ordering |
| Vocalisation contrast | Paired Wilcoxon + rank-biserial | Compares vocalisation PI vs overall PI per mouse |
| Mixed-effects model | LMM with day contrasts + random intercept | `PI ~ day + (1\|mouse)`, reports Nakagawa marginal/conditional R-squared |
| Model comparison | AIC, BIC, log-likelihood ratio test | Compares null, day-only, and day+RE models |
| Sensitivity | Beta regression (binomial GLM on scaled PI) | Checks robustness of day effects to distributional assumption |
| Voc PI vs other-sounds PI | Per-day paired Wilcoxon + rank-biserial (section 12b) | Tests whether vocalisations elicit a different preference than the other-sounds aggregate, per session |
| First-minute RE | Pearson + Spearman correlation, within- and between-mouse (section 12c) | Tests whether exploration during the first 60 s of habituation predicts subsequent PI |

**4. Completers LMM (`03_completers_lmm.py`):**

A standalone follow-up that restricts the dataset to mice present on every required experimental day (`PI_DAYS` by default: `w1_d1`, `w1_d2`, `w1_d3`, `w2_sequences`, `w2_vocalisations`) and fits four nested mixed models per outcome (PI, voc PI, other-sounds PI, RE, first-minute RE):

| Model | Formula | Purpose |
|-------|---------|---------|
| M0 (null) | `y ~ 1 + (1\|mouse)` | Baseline; quantifies between-mouse variance and ICC |
| M1 (day fixed) | `y ~ C(day) + (1\|mouse)` | Day differences with random intercepts (the user's primary request) |
| M2 (linear trend) | `y ~ session_num + (1\|mouse)` | Linear change across sessions |
| M3 (random slopes) | `y ~ session_num + (1+session_num\|mouse)` | Heterogeneous per-mouse slopes |

The script reports REML estimates for inference and refits with ML for likelihood-ratio tests (M0 vs M1; M2 vs M3). It produces Nakagawa & Schielzeth (2013) marginal/conditional R-squared, ICC, BLUPs (per-mouse random intercepts) with 95% CIs, and three figures (caterpillar plot, day estimated marginal means, per-mouse spaghetti).

CLI:

```bash
python 03_completers_lmm.py /path/to/8_arms_w_voc \
    --csv BATCH_ANALYSIS/preference_data.csv \
    --output-dir BATCH_ANALYSIS/completers \
    --required-days w1_d1 w1_d2 w1_d3 w2_sequences w2_vocalisations
```

**5. Computational model (Brielmann & Dayan 2022):**

A 4-dimensional extension of the aesthetic value model, where stimuli are represented as vectors in a feature space of [location_familiarity, spectral_complexity, biological_relevance, temporal_predictability]. The model simulates an agent traversing the full experiment and predicts PI from the difference in aesthetic value between sound and silence. Fitted via 2000 random initialisations of SLSQP. Includes lesioned model comparisons, dimension dropout analysis, and location vs acoustic decomposition. See the [detailed report](https://github.com/MaravallLab/aMAZEing-maze/blob/main/analysis/auditory/REPORT_aesthetic_value_model.md) for full documentation.

#### Fiber Photometry Alignment (`fibpho_alignment.py`)

Aligns Tucker-Davis Technologies (TDT) fiber photometry recordings with auditory maze visit CSV timestamps using TTL pulse matching.

**Prerequisites:**

```bash
pip install numpy pandas matplotlib scipy tdt
```

**Configuration:**

Edit paths at the top of `fibpho_alignment.py`:
- `TANK_PATH` -- path to the TDT tank folder (contains `.Tbk`, `.Tdx`, `.tev`, `.tsq` files)
- `VISIT_CSV` -- path to the `mouseXXXXX_vocalisations_detailed_visits.csv`
- `TRIALS_CSV` -- path to the `trials_time_YYYY-MM-DD_HH_MM_SS.csv`

**Running:**

```bash
cd analysis/auditory
python fibpho_alignment.py
```

**Outputs** (saved to `fibpho_analysis/` inside the session folder):

| File | Description |
|------|-------------|
| `alignment_report.csv` | Event-by-event TTL-to-CSV mapping with residuals |
| `fibpho_aligned_overview.png/pdf` | Full-session dF/F with colour-coded TTLs |
| `fibpho_trial_panels.png/pdf` | Per-trial zoomed dF/F panels |
| `fibpho_peri_event.png/pdf` | Peri-event average dF/F by vocalisation type |
| `fibpho_peri_event_pooled.png/pdf` | Pooled peri-event dF/F (all stimuli) |

**How alignment works:**

1. The TDT recording starts before the maze experiment (different clocks)
2. The script reads TTL onset times from the TDT `MTL_` epoc store
3. It matches inter-event intervals to `sound_on_time` entries in the visit CSV
4. The first TTL is identified as a test pulse and excluded
5. A precise clock offset is computed (typical alignment: <70ms residual)
6. Delta F/F is calculated using 405nm isosbestic correction of the 465nm GCaMP signal

---
