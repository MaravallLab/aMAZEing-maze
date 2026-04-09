# Aesthetic Value Model for Mouse Acoustic Preference

**A 4-dimensional extension of Brielmann & Dayan (2022) applied to approach/avoidance behaviour in an 8-arm radial maze.**

> **Reference:** Brielmann, A. A., & Dayan, P. (2022). A computational model of aesthetic value. *Psychological Review*, 129(6), 1319--1337. https://doi.org/10.1037/rev0000337

> **Script:** [`aesthetic_value_model_4D.py`](aesthetic_value_model_4D.py)

---

## Table of Contents

- [Motivation](#motivation)
- [Experimental Context](#experimental-context)
- [The Original Model](#the-original-model)
- [4D Extension for Acoustic Stimuli](#4d-extension-for-acoustic-stimuli)
  - [Dimension 0: Location Familiarity](#dimension-0-location-familiarity)
  - [Dimension 1: Spectral Complexity](#dimension-1-spectral-complexity)
  - [Dimension 2: Biological Relevance](#dimension-2-biological-relevance)
  - [Dimension 3: Temporal Predictability](#dimension-3-temporal-predictability)
- [Model Equations](#model-equations)
  - [System State](#system-state)
  - [Expected True Distribution](#expected-true-distribution)
  - [Immediate Sensory Reward r(t)](#immediate-sensory-reward-rt)
  - [System State Value V(X(t))](#system-state-value-vxt)
  - [Learning Update](#learning-update)
  - [Aesthetic Value A(t)](#aesthetic-value-at)
  - [PI Prediction](#pi-prediction)
- [Experiment Simulation](#experiment-simulation)
- [Model Fitting](#model-fitting)
  - [Free Parameters](#free-parameters)
  - [Objective Function](#objective-function)
  - [Fitting Procedure](#fitting-procedure)
- [Decomposition Analyses](#decomposition-analyses)
  - [Location vs Acoustic Decomposition](#location-vs-acoustic-decomposition)
  - [Dimension Dropout](#dimension-dropout)
  - [Lesioned Models](#lesioned-models)
- [Secondary Evaluations](#secondary-evaluations)
- [Outputs](#outputs)
- [Interpreting the Results](#interpreting-the-results)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Running the Script](#running-the-script)

---

## Motivation

The auditory maze experiment produces a rich dataset: mice freely explore an 8-arm radial maze where 7 arms play sounds and 1 arm is silent, with all arm assignments reshuffling every 15 minutes. The primary behavioural readout -- the Preference Index (PI) -- captures whether mice spend more time near sounds than near silence.

Two striking patterns emerge from the data:

1. **Day 1 avoidance with a vocalisation anomaly:** On the first day, mice show overall avoidance of sound arms (PI < 0), but vocalisations attract disproportionately long dwell times (906 ms vs 511 ms for smooth tones) despite the negative overall PI.

2. **Habituation arc:** PI evolves from strongly negative on Day 1 to near-zero by Week 2, suggesting adaptation to the acoustic environment over days.

Standard statistical models (Wilcoxon, Kruskal-Wallis, mixed-effects) describe *what* happens but not *why*. A computational model can formalise hypotheses about the underlying cognitive processes -- immediate processing fluency, learning-driven adaptation, and innate biological priors -- and generate testable predictions about how these components interact.

The Brielmann & Dayan (2022) aesthetic value model provides exactly this framework. Originally developed for human aesthetic pleasure, it decomposes the value of a stimulus into an immediate component (how well the stimulus matches current expectations) and a learning component (how much encountering the stimulus improves the system's future readiness). Here we extend it to 4 dimensions relevant to mouse acoustic processing.

---

## Experimental Context

### Design

- **Maze:** 8-arm radial, 7 sound arms + 1 silent arm
- **Reshuffling:** All arm assignments (including silence) randomise every 15 minutes
- **Sessions:** ~60 minutes per day (approximately 4 reshuffling blocks)
- **Days:** 6 experiment days across 2 weeks, each with different sound categories:

| Day | Sound categories | Mode |
|-----|-----------------|------|
| D1 (w1_d1) | smooth, rough, rough_complex, vocalisation | Temporal envelope modulation |
| D2 (w1_d2) | smooth, rough, consonant, dissonant, vocalisation | Complex intervals |
| D3 (w1_d3) | smooth, rough, consonant, dissonant, vocalisation | Complex intervals |
| D4 (w1_d4) | smooth, rough, consonant, dissonant (no silent control) | Complex intervals |
| W2_Seq (w2_sequences) | AAAAA, AoAo, ABAB, ABCABC, BABA, ABBA | Tone sequences |
| W2_Voc (w2_vocalisations) | vocalisation x 7 arms | Mouse vocalisations |

### Key design features

The 15-minute reshuffling is a critical design strength:

- **No arm has a stable acoustic identity** -- spatial learning of sound locations is impossible by design.
- **No arm has a stable silence identity** -- spatial learning of silence locations is equally impossible.
- **PI reflects real-time acoustic processing only**, free of place-learning confounds.
- Sounds change category every day, so the system state never fully converges on any single stimulus.

### Preference Index

PI is computed using within-trial comparison: sound-playing ROIs vs the silent-control ROI, measured during sound trials only.

```
PI = (avg_sound_dur - avg_silent_dur) / (avg_sound_dur + avg_silent_dur)
```

- PI = -1: complete avoidance of sound
- PI = 0: no preference
- PI = +1: complete preference for sound

### Observed data

The model is fitted to the following corrected observations (from `stats_report.txt`):

| Day | Voc PI (mean) | Overall PI | Within-trial median diff (ms) |
|-----|--------------|------------|-------------------------------|
| D1 | -0.037 | -0.142 | -425 |
| D2 | +0.010 | +0.024 | -329 |
| D3 | -0.024 | -0.057 | -214 |
| W2_Seq | +0.048 | +0.005 | +32 |
| W2_Voc | +0.044 | +0.031 | +74 |

---

## The Original Model

Brielmann & Dayan (2022) propose that aesthetic value A(t) at time t has two components:

```
A(t) = w0 + wr * r(t) + wV * Delta_V(t)
```

where:

- **r(t)** is the *immediate sensory reward*: how well the stimulus matches the observer's current internal state. A familiar, fluently processed stimulus yields higher r(t).
- **Delta_V(t)** is the *change in system state value*: how much encountering the stimulus moves the internal state closer to the expected true distribution p_T. A stimulus that teaches the system something useful yields positive Delta_V.
- **w0** is a bias term.
- **wr, wV >= 0** weight the two components.

The system state X(t) is a multivariate Gaussian that updates toward encountered stimuli via a learning rule. The expected true distribution p_T represents what the observer "expects the world to be like" in the long run.

The original model was validated on human self-report ratings of visual artworks and faces. This implementation extends it to mouse acoustic behaviour -- a novel application that adds an untested decision layer between aesthetic value and approach/avoidance.

---

## 4D Extension for Acoustic Stimuli

Each stimulus (including silence) is represented as a 4-dimensional feature vector:

```
s = [location_familiarity, spectral_complexity, biological_relevance, temporal_predictability]
```

### Dimension 0: Location Familiarity

**What it captures:** Global comfort with the physical maze environment -- the smell, floor texture, lighting, spatial layout. This does NOT represent familiarity with any specific arm or sound location (impossible by design due to reshuffling).

**Properties:**
- Updates every day regardless of which sounds are played, with its own faster learning rate `alpha_loc`
- Affects ALL stimuli equally, including silence
- Therefore PARTIALLY CANCELS in PI (which is a relative measure of sound vs silence)
- Does NOT cancel in absolute dwell time
- Captures the baseline anxiety/arousal state: a more familiar maze = calmer mouse = more fluent processing of everything

**Fixed values by day** (based on cumulative maze exposure):

| Day | Location Familiarity |
|-----|---------------------|
| D1 | 0.10 (first exposure, highly novel) |
| D2 | 0.40 (some familiarity) |
| D3 | 0.60 (moderately familiar) |
| W2_Seq | 0.85 (well habituated) |
| W2_Voc | 0.90 (very familiar) |

Silence shares the same location familiarity as sounds on any given day.

### Dimension 1: Spectral Complexity

**What it captures:** The spectral richness of the sound. Vocalisations are spectrally complex but structured, scoring lower than broadband noise on raw spectral entropy despite being perceptually rich. Silence scores 0 (no spectral content).

| Stimulus | Spectral Complexity |
|----------|-------------------|
| silence | 0.0 |
| smooth | 0.0 |
| rough | 1.0 |
| rough_complex | 1.5 |
| consonant | 0.8 |
| dissonant | 1.2 |
| vocalisation | 1.8 |
| AAAAA | 0.5 |
| AoAo | 0.6 |
| ABAB | 0.8 |
| ABCABC | 1.3 |
| BABA | 0.8 |
| ABBA | 0.8 |

Note: sequence stimuli (AAAAA through ABBA) use the same or similar tones -- their differences are captured by temporal predictability (dim 3), not spectral complexity.

### Dimension 2: Biological Relevance

**What it captures:** How much the sound resembles an ecologically important signal for mice. This dimension reflects a **partially innate** component of p_T: mice are born with an evolutionary prior that conspecific calls matter, even without prior lab exposure.

This is the **key theoretical assumption** distinguishing this implementation from the original paper. The expected true distribution p_T has its mode near "vocalisation" on this dimension for naive animals, independent of experience. This explains why vocalisations attract longer dwell times on D1 despite overall avoidance -- they shift the system state toward p_T even before any acoustic learning has occurred.

| Stimulus | Biological Relevance |
|----------|---------------------|
| silence | 0.0 |
| smooth, rough, rough_complex | 0.0 |
| consonant, dissonant | 0.1 |
| vocalisation | 1.0 |
| All sequences (AAAAA, etc.) | 0.0 |

### Dimension 3: Temporal Predictability

**What it captures:** How predictable the sound is moment-to-moment within a single stimulus. This is DISTINCT from spectral complexity:

- AAAAA: spectrally simple AND temporally predictable
- Noise: spectrally complex AND temporally unpredictable
- Vocalisation: spectrally complex BUT temporally structured
- ABBA: spectrally simple BUT temporally surprising (violates ABAB expectation)

Silence is maximally temporally predictable (no moment-to-moment variation).

| Stimulus | Temporal Predictability |
|----------|----------------------|
| silence | 1.0 |
| smooth | 0.9 |
| rough | 0.1 |
| rough_complex | 0.3 |
| consonant | 0.8 |
| dissonant | 0.5 |
| vocalisation | 0.6 |
| AAAAA | 1.0 |
| AoAo | 0.8 |
| ABAB | 0.7 |
| ABCABC | 0.5 |
| BABA | 0.6 |
| ABBA | 0.4 |

---

## Model Equations

### System State

The system state X(t) is a multivariate Gaussian with:
- Mean: **mu(t)**, shape (4,), initialised as mu_0 = [0.0, 0.0, 0.0, 0.5]
- Covariance: **Sigma** = sigma_sq * I (diagonal, scalar variance, fitted)

Initial state interpretation:
- dim 0 = 0.0: maze is completely novel
- dim 1 = 0.0: no acoustic experience (tuned to silence)
- dim 2 = 0.0: no biological sound exposure yet
- dim 3 = 0.5: moderate baseline temporal expectation

### Expected True Distribution

p_T is a fixed Gaussian with:
- Mean: **mu_T** (4D, partially free -- see [Free Parameters](#free-parameters))
- Covariance: **Sigma_T** = sigma_T_sq * I (diagonal, scalar, fitted)

mu_T represents what the mouse "expects the world to be like" in the long run. Crucially, mu_T[2] (biological relevance) is constrained >= 0.4 to encode the innate evolutionary prior for conspecific calls.

### Immediate Sensory Reward r(t)

The log-likelihood of stimulus s(t) under the current system state (omitting the constant normalisation term):

```
r(t) = -1 / (2 * sigma_sq) * sum_j[ (s_j(t) - mu_j(t))^2 ]
```

Higher r(t) means the stimulus is closer to the current system state mean -- more fluently processed.

### System State Value V(X(t))

The negative KL divergence from p_T to X(t):

```
V(X(t)) = -KL(p_T || X(t))
```

Using the closed-form for two multivariate Gaussians with diagonal scalar covariances:

```
KL(p_T || X(t)) = 0.5 * [
    n_dims * (sigma_T_sq / sigma_sq)
    + (1 / sigma_sq) * sum_j[ (mu_j(t) - mu_T_j)^2 ]
    - n_dims
    + n_dims * ln(sigma_sq / sigma_T_sq)
]
```

Higher V means the system state is closer to p_T -- better positioned for future processing.

### Learning Update

Applied AFTER computing r(t) and BEFORE computing Delta_V:

**Acoustic dimensions** (j = 1, 2, 3):
```
mu_j(t+1) = mu_j(t) + alpha * (s_j(t) - mu_j(t))
```

**Location dimension** (j = 0):
```
mu_0(t+1) = mu_0(t) + alpha_loc * (loc_day - mu_0(t))
```

where `loc_day` is the day-level location familiarity value. The location dimension updates toward `loc_day` on every trial regardless of the stimulus, because maze familiarity accumulates from being in the maze, not from hearing any particular sound.

**Reshuffling penalty:** Every 15 minutes all arm assignments reshuffle, generating background acoustic prediction error. This is modelled as a fixed penalty:

```
Delta_V_effective(t) = Delta_V(t) - reshuffle_penalty
```

where `reshuffle_penalty = 0.05` (fixed, not fitted). This prevents the model from overestimating learning benefits within a session.

### Aesthetic Value A(t)

```
A(t) = w0 + wr * r(t) + wV * Delta_V_effective(t)
```

with constraints wr >= 0, wV >= 0.

### PI Prediction

For each sound stimulus s on day d:

```
PI_predicted(s, d) = A(s, d) - A(silence, d)
```

Both are computed at the same system state mu(t) and the same location familiarity for that day. Because location familiarity is identical for sound and silence, its contribution to the PI subtraction partially cancels.

---

## Experiment Simulation

The script simulates a single agent across all 5 days in sequence, carrying mu(t) forward from one day to the next -- learning accumulates across the full experiment.

Each day consists of 4 reshuffling blocks. Within each block, the agent encounters each of that day's 7 sound stimuli once, plus one encounter with silence:

```
4 blocks x (7 sounds + 1 silence) = 32 stimulus encounters per day
5 days x 32 = 160 total encounters
```

Within each block, the order of the 8 stimuli is randomised (seed = 42 for reproducibility).

**Special case -- W2_Voc:** All 7 sound arms play vocalisations, giving the agent 7x more exposure to vocalisation features per block than any other stimulus type on other days. This asymmetry is noted in the output and may cause the model to overpredict W2_Voc PI relative to W2_Seq.

On each trial:
1. Get stimulus s(t)
2. Compute A(silence) at current state (without updating)
3. Compute r(t), V_before, apply learning update, compute V_after, Delta_V_eff, A(t)
4. Store all quantities for summarisation

---

## Model Fitting

### Free Parameters

| Parameter | Description | Init | Bounds | Notes |
|-----------|------------|------|--------|-------|
| alpha | Acoustic learning rate | 0.05 | [0.001, 1] | |
| alpha_loc | Location learning rate | 0.30 | [0.001, 1] | Expected > alpha |
| wr | Weight on r(t) | 1.0 | [0, 50] | |
| wV | Weight on Delta_V | 5.0 | [0, 200] | |
| w0 | Bias | 0.0 | [-5, 5] | |
| sigma_sq | System state variance | 1.0 | [0.01, 20] | |
| mu_T_0 | p_T mean: location | 0.75 | [0, 1] | |
| mu_T_1 | p_T mean: spectral | 1.5 | [0, 3] | |
| mu_T_2 | p_T mean: biological | 0.6 | **[0.4, 1]** | Innate prior constraint |
| mu_T_3 | p_T mean: temporal | 0.6 | [0, 1] | |
| sigma_T_sq | p_T variance | 2.0 | [0.1, 20] | |

**11 free parameters total.** The constraint on mu_T_2 >= 0.4 is theoretically motivated: biological relevance in p_T cannot be zero for mice, reflecting the innate evolutionary prior for conspecific vocalisation processing.

### Objective Function

Minimise RMSE between linearly-scaled predicted Voc PI across days and observed mean Voc PI:

1. Simulate the full experiment with candidate parameters
2. Extract predicted Voc PI for each day
3. Find optimal linear scaling analytically: `PI_scaled = a * Voc_PI_pred + b` (solved via `np.polyfit`)
4. Compute RMSE between `PI_scaled` and observed Voc PI

The linear scaling absorbs the unknown mapping between model A(t) units and the empirical PI scale, without adding parameters to the optimiser.

### Fitting Procedure

- **Optimiser:** `scipy.optimize.minimize` with method SLSQP
- **Initialisations:** 2000 random starts, sampled uniformly within bounds (seed = 42)
- **Convergence:** `maxiter=500`, `ftol=1e-10`
- **Output:** Best fit (lowest RMSE) + top 20 fits for robustness assessment

> **Note:** With 11 free parameters and 5 primary observations, the model is overparameterised for formal statistical inference. This is a proof-of-concept intended as a theoretical framework contribution, not a predictive model. Near-zero RMSE on the primary target is expected and does not indicate good generalisation.

---

## Decomposition Analyses

### Location vs Acoustic Decomposition

A key question: how much of the D1 avoidance is driven by maze novelty (location familiarity) vs acoustic novelty?

**Counterfactual A** -- "Familiar maze from day 1":
- Set location familiarity to 0.9 on D1 (as if the maze were already well known)
- Rerun D1 simulation
- Difference from actual = location novelty contribution to D1 avoidance

**Counterfactual B** -- "No acoustic learning":
- Set alpha = 0.0 (system state never updates on acoustic dimensions)
- Rerun the full experiment
- Reveals how much of the D1-to-W2 habituation arc is driven by acoustic learning vs location habituation

### Dimension Dropout

For each dimension d in {0, 1, 2, 3}:
1. Set s_d(t) = mu_d(t) for all t (dimension contributes zero to r(t) and Delta_V)
2. Rerun full simulation with best-fit parameters
3. Compute RMSE of Voc PI predictions
4. Report delta RMSE relative to the full model

The resulting table ranks dimensions by their importance to the model fit.

### Lesioned Models

Three lesioned versions are run using best-fit parameters (except the lesioned weight), mirroring Figure 3 from Brielmann & Dayan (2022):

| Lesion | Description | What it tests |
|--------|------------|---------------|
| A: wr = 0 | Learning only | Can the model explain PI from Delta_V alone? |
| B: wV = 0 | Fluency only | Can the model explain PI from r(t) alone? |
| C: wr = wV = 1 | Equal weights | How does fixed equal weighting compare? |

---

## Secondary Evaluations

Using the best-fit parameters (no refitting), the script computes:

1. **Pearson r: predicted A vs D1 dwell times** (4 stimuli: smooth, rough, rough_complex, vocalisation). Tests whether the model's absolute aesthetic value tracks absolute dwell time.

2. **Pearson r: predicted A vs D2 dwell times** (5 stimuli).

3. **Pearson r: predicted A vs W2_Seq dwell times** (6 sequence stimuli, ordered by temporal predictability). Tests whether the temporal predictability dimension does independent work.

4. **Sign accuracy:** Does the model correctly predict the direction of PI on each day? (D1 negative, D2 near zero, D3 slightly negative, W2 near zero/positive.)

5. **Vocalisation anomaly:** Is A(vocalisation) > A(smooth) on D1? Tests whether the biological relevance dimension captures the dwell-time anomaly.

6. **Temporal structure:** Is A(AAAAA) > A(ABBA) in W2_Seq? Tests whether temporal predictability contributes independently.

---

## Outputs

All outputs are saved to `BATCH_ANALYSIS/` alongside the statistical analysis outputs.

### Files

| File | Description |
|------|-------------|
| `aesthetic_value_model_4D.png` | 6-panel publication figure (300 dpi) |
| `aesthetic_value_model_4D.pdf` | Vector version of the figure |
| `aesthetic_model_4D_predictions.csv` | Per-stimulus, per-day: mean A, mean r, mean Delta_V, predicted PI |
| `aesthetic_model_4D_params.csv` | Best-fit parameter values |

### Figure panels

| Panel | Content |
|-------|---------|
| A | Primary Voc PI fit: predicted vs observed across days, with robustness band from top 20 fits |
| B | D1 complexity gradient: predicted A(t) vs observed dwell time per stimulus type, with Pearson r |
| C | W2 sequence predictions: predicted A(t) vs observed dwell time, ordered by temporal predictability |
| D | System state trajectory: mu(t) across all 160 trials, 4 subplots (one per dimension), with day boundaries and stimulus feature reference lines |
| E | Lesioned model comparison: full, wr=0, wV=0, equal weights -- all overlaid with observed data |
| F | Dimension dropout (left) and location decomposition (right) |

### Printed report

The script prints a structured report with:
- Best-fit parameters with interpretations and diagnostic flags
- Fit quality metrics (RMSE, correlations, sign accuracy, anomaly tests)
- Decomposition results (dimension dropout ranking, location vs acoustic contributions)
- W2_Voc special case note
- Five key theoretical statements with computed values filled in

---

## Interpreting the Results

### What the model can tell you

1. **Decomposition:** The relative contributions of immediate fluency vs learning to acoustic preference, which is not measurable from behavioural data alone.

2. **Location vs acoustic:** What fraction of D1 avoidance is attributable to maze novelty (a non-acoustic confound) vs genuine acoustic novelty. Because location familiarity affects sound and silence equally, it partially cancels in PI -- but the residual may still be substantial.

3. **Innate priors:** Whether a biological relevance dimension with an innate prior (mu_T_2 >= 0.4) is necessary to explain the vocalisation dwell-time anomaly on D1.

4. **Feature space structure:** Whether the 4 proposed dimensions are jointly sufficient to capture the observed patterns, and which dimensions carry the most explanatory weight.

### What the model cannot tell you

1. **Causal mechanism:** The model describes a computational-level account ("what computation is being performed?"), not an algorithmic or implementational one. It does not specify which neural circuits compute r(t) or Delta_V.

2. **Unique parameter estimates:** With 11 parameters and 5 observations, the model is overparameterised. Many different parameter combinations can produce equally good fits. The robustness analysis (top 20 fits) partially addresses this.

3. **Generalisability:** The model was fitted to aggregate group-level PI values. It does not model individual mouse variability, and its predictions for new stimuli or new experiment designs are untested.

---

## Assumptions and Limitations

1. **Feature vectors are manually assigned,** not empirically derived. Results are sensitive to these values. A formal sensitivity analysis varying feature assignments is recommended.

2. **mu_T dim 2 (biological relevance) reflects a partially innate evolutionary prior** for conspecific calls. This is the primary theoretical assumption. Supporting literature on innate vocalisation processing in mice should be cited (e.g., auditory cortex selectivity for ultrasonic vocalisations in naive mice).

3. **PI is treated as a monotone linear proxy for A(sound) - A(silence).** The model does not include an explicit decision or action layer between aesthetic value and approach/avoidance.

4. **The original model was validated on human self-report.** Application to mouse behaviour is novel and adds an untested decision layer. Interpret cautiously.

5. **Diagonal covariance with scalar variance** assumes feature dimensions are independent and equally weighted. This is a strong simplifying assumption.

6. **reshuffle_penalty = 0.05 is fixed and not fitted.** Sensitivity to this value (0 to 0.2) should be tested.

7. **11 parameters, 5 observations.** Formal model comparison against simpler alternatives (e.g., 2D or 3D models) is needed to justify the additional dimensions.

8. **The within-day structure (4 blocks x 8 encounters) is an approximation.** Actual visit counts depend on individual mouse locomotion.

9. **No feedback loop between A(t) and exposure.** In reality, preferred stimuli receive more exposure, strengthening learning. This simplification may underestimate learning for preferred stimuli.

10. **Sounds change every day.** Unlike the paper's mere-exposure simulations, the system state never fully converges on any single stimulus. The observed habituation arc reflects cumulative cross-stimulus learning, not single-stimulus overexposure.

11. **Random seed = 42** is used for full reproducibility.

---

## Running the Script

### Quick start

```bash
cd analysis/auditory
python aesthetic_value_model_4D.py
```

The full 2000-initialisation fitting takes approximately 10--30 minutes depending on hardware.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAZE_DATA_DIR` | `~/Box/Awake Project/Maze data/Auditory experiments/8_arms_w_voc` | Root data directory |
| `VISIT_CLIP_MS` | `10000` | Per-visit duration cap (ms), used by the data loading pipeline |

### Dependencies

- numpy
- scipy
- matplotlib
- seaborn
- pandas

All standard in a scientific Python environment. No additional packages required beyond the base analysis pipeline.

### Output location

All files are saved to `BATCH_ANALYSIS/` inside the data directory (same location as the statistical analysis outputs from `run_batch_preference.py`).

---

*This report documents the implementation in `aesthetic_value_model_4D.py`. For the broader auditory analysis pipeline (PI computation, statistical tests, interactive figures), see the [main README](../../README.md#auditory-analysis-pipeline).*
