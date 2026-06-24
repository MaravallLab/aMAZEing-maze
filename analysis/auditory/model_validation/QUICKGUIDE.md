# model_validation — Quick Guide

A from-scratch guide to *what* this analysis is, *why* it is built the way it is,
and *how* to run and tweak it yourself. For the broader experiment see the repo
README; this document is only about the `model_validation/` package.

Everything here is **read-only** with respect to the experiment code and the
recordings: it reads logged sessions from a results tree and writes only into an
output directory you choose. It never regenerates a stimulus and never edits
`src/auditory/` or the recordings.

---

## 1. The scientific question (in one paragraph)

During training, each mouse heard one grammar (A or B) in its **enriched (EE)**
cage and the other grammar in its **standard (SC)** cage. On the test day the
maze plays *both* grammars, at three predictability tiers, on different arms. The
behavioural readout is the **EE−SC preference index**

```
PI = (time on EE-grammar arms − time on SC-grammar arms) / (sum of the two)
```

The narrow question this package answers is:

> **Does a semantic value term S(t) — a signal that the mouse is recognising the
> grammar associated with its enriched environment — explain the EE−SC
> preference, over and above a fluency-only Brielmann & Dayan (B–D) baseline, out
> of sample, with the predicted EE > SC sign?**

Not "do mice like sound" (that is silent-vs-sound, the *old* experiment). This is
specifically about the **EE/SC association**.

---

## 2. The three regressors, and the one idea that makes them work

We extend the Brielmann & Dayan aesthetic-value model. Each arm's melody is
scored by an ideal observer, producing three time series that we summarise to one
number per arm:

- **r(t) — fluency / predictability.** How expected is each tone transition under
  the grammar the mouse learned? Predictable (dominant-tier) transitions are
  high-r; surprising (rare-tier) transitions are low-r. This is the *complexity*
  axis. Computed as the log-likelihood (negative surprise) of each transition.
- **ΔV(t) — learning signal.** How much does hearing this melody move the B–D
  "system state" toward its long-run target? Within a steady-state arm this is
  near zero (learning has plateaued) — expected, and reported, not hidden.
- **S(t) — semantic value.** The novel term. A hidden-context filter tracks the
  mouse's belief that it is in an "EE-grammar world" vs an "SC-grammar world".
  Hearing a melody that follows the EE grammar pushes that belief up; `S` is the
  belief shift weighted by context value (V_EE = 1, V_SC = 0). So **S > 0 on EE
  arms, S < 0 on SC arms**, and the size depends on how diagnostic the melody is.

The per-arm aesthetic value is the linear combination

```
A = w0 + wr·r + wV·ΔV + wS·S
```

and `wS` is the thing we care about. `wS > 0` means semantic recognition adds
preference beyond fluency.

### The load-bearing idea: emissions are BIGRAM, not single-tone

Both grammars are **doubly stochastic** — every tone is, on average, equally
likely under A and under B. So if the observer asked "how likely is this *tone*
under EE vs SC?", the answer is always "equally likely", the belief never moves,
and **S(t) would be identically zero**. The grammars differ only in their
*transitions* (A's dominant step is `i→i+1`, an ascending sweep; B's is `i→i+3`,
an octave trill). So the observer must score **transitions**:

```
P(tone_t | tone_{t-1}, context = EE or SC) = (that context's grammar matrix entry)
```

This is why the model is a *Markov-switching* observer: the hidden context picks
which grammar governs the next tone. Get this wrong and the analysis would
"prove" S is inestimable for the wrong reason.

### Complexity is recomputed, not read from the log

The logged `mean_bits` column collapses the secondary and rare tiers (it is
computed under the restricted sampling distribution, where both are "1 of 2
choices"). Under the **full** grammar the mouse actually learned, the three tiers
are genuinely different surprise levels (≈0.74 / 3.06 / 3.84 bits). We recompute
surprise under the full grammar from the `symbols` string — we never trust the
logged bits.

---

## 3. Why the counterbalancing GROUP matters (the confound)

At the dominant tier, Grammar A is an **ascending sweep** and Grammar B is an
**octave trill** — perceptually very different *regardless* of any learned
association. So a preference could be "semantic" (the mouse prefers its
EE-associated grammar) or "intrinsic" (everyone just prefers the sweep, or the
trill).

The two counterbalancing groups dissociate these:
- **Group 1**: EE = A, SC = B. **Group 2**: EE = B, SC = A.
- If the effect is **semantic**, EE−SC is positive in **both** groups (it tracks
  the *association*, which is consistent by construction).
- If it is an **intrinsic-grammar** preference, EE−SC **flips sign** between
  groups (it tracks the *physical* grammar, which maps to EE/SC oppositely).

So the model-free test includes `group`, and the model-based recovery includes an
"intrinsic-grammar" generator and a grammar-identity nuisance weight, to make sure
`wS` cannot be a relabelled intrinsic preference. **The dominant-tier result stays
provisional until this group split is confirmed clean.**

---

## 4. The pipeline order (and why it is fixed)

The ordering is deliberate — do not reorder:

1. **Cohort filter** — apply named, auditable exclusions; report N and group
   balance.
2. **Sanity** — is the rig sane? The gate is the arm-index residual (physical ROI
   must not predict dwell after the 15-min reshuffle). Silent-vs-sound is
   *reported but not gated* (test-day sound avoidance is a real possibility here).
3. **Model-free design analysis** — `dwell ~ environment × tier × group +
   (1|mouse)`. Establishes whether an EE−SC effect exists *without* trusting the
   model. If there's nothing here, no model fit is worth interpreting.
4. **Latent regressors + collinearity** — build r, ΔV, S; check they are actually
   separable in the delivered sequences (if `r` and `S` are collinear, `wS` can't
   be identified).
5. **Recovery (the gate)** — on synthetic data built from the *real* sequences:
   can we recover a known `wS`? Do we get a false positive when `wS = 0`? Does the
   comparison pick the right model? **No fitted `wS` is interpreted until this
   passes.** If recovery fails, "the design cannot separate semantics from
   fluency" is the real result, not a bug.
6. **Phase 2 (Bayesian)** — only if recovery passes: fit the four nested models,
   compare out-of-sample (LOO), report the `wS` posterior and the
   posterior-predictive arm pattern.

Day 1 is the **primary** analysis. Day 2 is a **separate secondary** section
(model-free + a block time-course `PI ~ block`, where monotonic decay across the
four 15-min blocks is the extinction signature). The two days are never pooled.

---

## 5. How to run it

From the repo root (or anywhere — the script bootstraps its own path).

### Phase 1 (no extra dependencies beyond the standard scientific stack)

```bash
python analysis/auditory/model_validation/run_validation.py \
    --results_dir "C:\Users\labuser\Desktop\auditory_maze_experiments\maze_recordings\grammar" \
    --out_dir     "C:\path\to\an\output\folder" \
    --day primary
```

Outputs into `--out_dir`:
- `report.md` — the human-readable report (cohort, gates, effects, verdict).
- `results.json` — every raw number (machine-readable).
- `day1_arm_block_features.csv` — the tidy per-arm-block table with r/ΔV/S.

Add `--day both` to also run the day-2 secondary section.

### Phase 2 (Bayesian LOO — needs PyMC)

PyMC + ArviZ must be installed (pinned to versions that match the env):

```bash
python -m pip install --only-binary=:all: "pymc>=5.15,<5.20" "arviz>=0.18,<0.20" \
    "pandas==2.2.2" "numpy==1.26.4"
```

Then:

```bash
# Windows: force single-threaded BLAS or pytensor's Python-mode sampler thrashes.
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:MKL_NUM_THREADS=1

python analysis/auditory/model_validation/run_validation.py \
    --results_dir "...\grammar" --out_dir "...\out" \
    --day primary --skip-recovery --phase2 \
    --draws 500 --tune 500 --chains 2
```

`--skip-recovery` skips the (slow) Phase-1 recovery sims when you only want the
Bayesian fit. If you have a working C compiler for pytensor (`conda install -c
conda-forge m2w64-toolchain`), NUTS runs in seconds instead of minutes.

### Run the tests

```bash
python -m pytest analysis/auditory/model_validation/tests -q
```

The tests lock the central claims: S = 0 for a non-diagnostic sequence, S > 0 for
a Grammar-A sequence and ≤ 0 for Grammar B, belief stays normalised; the K=1 / wS=0
reduction to plain B–D; and that recovery recovers a known `wS` without flagging
`wS = 0` as positive.

---

## 6. How to read the report

`report.md` sections, top to bottom:

- **Cohort & exclusions** — sessions found, day-1 N (should be 33 right now),
  group balance, every excluded session *with its reason*, duplicate filings
  dropped, and the `trials ↔ grammar_samples` label cross-check.
- **Sanity gate** — GO/NO-GO on the arm-index residual; control arms reported as
  information.
- **Model-free design analysis** — the headline. `group_consistent: True` means
  the EE−SC sign agrees across counterbalancing groups (semantic). The per-tier
  simple effects show whether the effect is concentrated at the predictable
  (dominant) tier and attenuates at high complexity.
- **Feature collinearity** — `corr(r,S)` and VIFs. Near-zero correlation means the
  design separated fluency from semantics.
- **Recovery gate** — GO/NO-GO. This is what licenses interpreting `wS`.
- **Day-2 secondary** — the block time-course (decay = extinction).
- **Phase 2** — LOO ranking of the four models, the `wS` posterior (mean, 95% HDI,
  P(wS>0)), leave-one-mouse-out `wS` stability, and the posterior-predictive arm
  pattern.
- **Verdict** — a plain-language synthesis.

---

## 7. How to tweak it

| You want to… | Change |
|---|---|
| Reinstate the excluded mouse once its tracking is cleaned | remove `"13533"` from `EXCLUDED_ANIMALS` in `data_loading.py` and rerun (that's the whole change) |
| Exclude another animal | add `"id": "reason"` to `EXCLUDED_ANIMALS` |
| Change how stable the context belief is | `ctx_self_transition` in `validated_config()` (default 0.99) |
| Change the B–D learning rate / target / prior | `alpha`, `p_T_*`, `prior` args of `validated_config()` |
| Use the softmax link instead of the matching law | `--link softmax` |
| Use accumulated instead of mean features | `--summary sum` (means are primary; sums are over-logging-sensitive) |
| Make recovery faster / more thorough | `--recovery-sims`, `--confusion-sims` |
| Change MCMC effort | `--draws`, `--tune`, `--chains` |
| Loosen/tighten the sanity gate | `arm_idx_r2_max` in `run_sanity` (default 0.10) |
| Run day 2 | `--day secondary` or `--day both` |
| Point at a different results tree | `--results_dir` |

**Structural parameters are fixed by design and never fitted** — the grammar
matrices, the context transition matrix, the B–D learning rate and target, and
`V_EE = 1 / V_SC = 0`. Only the linear weights `(w0, wr, wV, wS)` (and the link
temperature) are free. This is a hard commitment: fitting the generative
structure would destroy identifiability.

---

## 8. Module map

```
config.py              structural constants (imported from grammar_stimuli),
                       bigram emission matrices, V_EE/V_SC, prior, A_ctx, p_T
data_loading.py        discover sessions, parse folder names, derive group,
                       named exclusions, cross-session integrity, EE-SC PI
latent_regressors.py   bigram HMM filter (r, S) + B-D system-state (ΔV),
                       canonical per-cell features, collinearity/VIF
link_function.py       A = w0 + wr·r + wV·ΔV + wS·S; matching / softmax; → PI
sanity_checks.py       exploration, control arms (informational), arm-index gate
design_analysis.py     PI/dwell mixed models incl. group; block time-course
recovery.py            parameter recovery + model-confusion (incl. intrinsic-
                       grammar) + leave-one-mouse-out CV  ← the gate
models.py              PyMC nested models (intercept/fluency/bd_baseline/full),
                       vectorised batched-Dirichlet observation
model_comparison.py    LOO via ArviZ + frequentist leave-one-mouse-out
posterior_predictive.py predicted six-arm pattern vs observed
individual_diffs.py    per-mouse wS, leave-one-mouse-out wS stability
reporting.py           report.md + results.json
run_validation.py      CLI orchestrating the ordered pipeline
tests/                 reduction, latent-regressor, and recovery tests
```

---

## 9. Assumptions & caveats (read before quoting any number)

- **Latent regressors are properties of the stimulus, not the trajectory.** We
  filter each arm's delivered melodies from a fixed prior and summarise to one
  number per (group, environment, tier) cell. This deliberately avoids belief
  depending on dwell depending on belief. Consequence: the regressors take only
  ~6 distinct values, so the model-based fit is close to a structured
  re-expression of the model-free `grammar × tier` analysis — which is exactly why
  **recovery is the binding test**.
- **The logged melody count over-states what was delivered** (the player
  pre-renders 20 melodies per arm entry and is cut off when the mouse leaves). We
  therefore use per-cell *means* (unbiased to over-logging) and assign them by
  arm label, so unvisited arms still get correct features.
- **ΔV has almost no between-arm variance** — within-arm learning plateaus. So
  `wV` does little work here; the real contrast is `r` vs `S`.
- **EE−SC PI matches the existing `summary_analysis` convention exactly**, so this
  pipeline is consistent with the figures you already produce.
- **Behavioural validity only.** A positive `wS` says the brain is computing
  something S(t)-shaped. Neural/mechanistic validity (belief-state decoding, a
  dopaminergic signal tracking S) is the next tier and is *not* established here.
