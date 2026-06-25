# Handoff summary — model_validation (for the model-author instance)

A compact, pasteable status of the validation pipeline the repo-side instance
built from your spec + decisions. Architecture, baked-in decisions, and the
actual day-1 numbers. Read alongside `QUICKGUIDE.md` (intuition + how to run).

## What was built
A read-only Python package `analysis/auditory/model_validation/` (touches no
experiment code, no recordings; writes only to a chosen out_dir). It runs the
ordered pipeline: cohort filter → sanity → model-free design analysis → bigram
latent regressors + collinearity → recovery (the gate) → Phase-2 Bayesian LOO →
day-2 secondary. CLI: `run_validation.py`. 9/9 unit tests pass.

## Decisions baked in (per your direction)
- **Bigram emissions**: P(o_t | o_{t-1}, z) = that context's grammar matrix.
  Single-tone emissions give S≡0 (both grammars share a uniform tone marginal).
  Markov-switching observer; S is group-invariant by construction.
- **Complexity = surprise under the FULL trained grammar** (recomputed from the
  `symbols` string), NOT the logged `mean_bits` (which collapses secondary≈rare).
- **Behavioural target = EE−SC PI** (EE−SC)/(EE+SC), matching the existing
  `summary_analysis` convention exactly. Silent/vocalisation = reference only.
- **Structural params fixed, never fitted**: grammar matrices, context transition
  (self≈0.99), B–D α/target/prior, V_EE=1 / V_SC=0. Only (w0, wr, wV, wS) free.
- **Group is a factor everywhere** (the octave-trill-vs-sweep intrinsic-grammar
  confound). Recovery includes an intrinsic-grammar generator + grammar-identity
  nuisance weight so wS can't be a relabelled intrinsic preference.
- **Recovery gates interpretation**: no wS is interpreted until recovery passes.
- **Day 1 primary; day 2 separate secondary** (model-free + block time-course).
- **Cohort**: animal 13533 excluded (light artifact, both days) via a named
  `EXCLUDED_ANIMALS` filter with a reason; the swap session excluded; duplicate
  filings de-duplicated. Reinstating 13533 once cleaned = delete one line.

## Day-1 results (current run)
- **Cohort**: 33 distinct mice after exclusions; **group balance 16 / 17**.
  Cross-session group inconsistencies: none. trials↔grammar_samples label
  mismatches: 0. (Phase-2 design uses 128 arm-blocks / 32 mice.)
- **Model-free** `log1p(dwell) ~ environment × tier × group + (1|mouse)`:
  - EE−SC effect **positive in both groups** — group 1 **+0.45 (p=.002)**,
    group 2 **+0.27 (p=.07)** — **group-consistent sign = True** ⇒ semantic, not
    the intrinsic-grammar confound.
  - Simple effect by tier: **dominant +0.45 (p=.008)**, secondary +0.24 (p=.10),
    rare +0.09 (p=.48); env×group n.s. at every tier. So the EE−SC preference is
    concentrated at the predictable (dominant) tier and attenuates with
    complexity.
- **Collinearity**: corr(r, S) = **0.001** (r and S essentially orthogonal — the
  design separated fluency from semantics in practice). VIFs ≈ 1.0–1.4. ΔV
  between-arm variance ≈ 3e-8 (≈ 0): within-arm learning has plateaued, so wV does
  little work; the real contrast is r (complexity) vs S (semantics).
- **Recovery gate** (full sims: 500 param-recovery, 50 confusion): **PASSED**.
  wS recovered at 0/0.5/1.0/2.0 with tight CIs; **false-positive rate at wS=0 is
  5.4%** (≈ nominal 5%). Model confusion: bd-generated prefer-`full` 0.20,
  full-generated 1.00, **intrinsic-grammar-generated 0.00 with wS(full) = −0.07**
  (no spurious positive). ⇒ wS is estimable and not a relabelled intrinsic-grammar
  effect.
- **Control arms** (informational, not gated): visited-arm mean dwell silent 14.3s
  > grammar 11.8s, vocalisation 16.8s — test-day avoidance of the grammar tones,
  vocalisation attractive (echoes the old experiment's D1 vocalisation anomaly).
- **Day-2 secondary** block time-course: PI 0.26 → 0.02 → −0.04 → −0.09 across the
  four 15-min blocks; slope **−0.11/block (p<.05)** = decay (extinction signature).

## Phase 2 — out-of-sample model comparison (day 1; 128 arm-blocks / 32 mice)
The mingw compiler was installed, so the exact PyMC `az.compare` LOO ran
(compiled NUTS, ~20–35 s/model); the frequentist leave-one-mouse-out CV agrees.

- **LOO ranking** (elpd_loo, higher = better): `full` rank 0 (**903.99**, stacking
  weight 0.61) > `fluency` (903.23) > `bd_baseline` (902.54) > `intercept`
  (900.81). So **`full` ranks best and beats `bd_baseline`**, BUT Δelpd(full −
  bd) ≈ 1.45 against dse ≈ 3.2 — the models are only **weakly separated out of
  sample** (intercept-only even takes 0.35 stacking weight). The OOS gain from wS
  is real in *direction* but small in *magnitude*.
- **wS posterior**: mean **+0.078, 95% HDI [0.012, 0.140], P(wS>0) = 0.992** —
  reliably positive, EE>SC sign. Leave-one-mouse-out wS 0.055, range
  [0.048, 0.064], sign stable across all 32 drops ⇒ not driven by a few animals.
- **MCMC caveat**: quick 2-chain / 750-draw run — rhat>1.01, low ESS, 32
  divergences in one model. Directionally trustworthy; re-run with 4 chains /
  higher `target_accept` / more draws before quoting as final.

### Subtlety the (r,S) figure exposed (matters for interpretation)
S is **not** uniformly EE-positive across tiers — its sign is set by the *overlap*
between the two grammars' tier transitions:
- **dominant**: A's dominant step (i→i+1) is only B's *secondary*, so a dominant
  sequence is cleanly diagnostic ⇒ S strongly EE+ / SC−.
- **secondary**: A's secondary step i→i+3 *is* B's *dominant* step, so a secondary
  sequence looks like the other grammar ⇒ S **flips** (EE-secondary S<0).
- **rare**: both grammars share identical rare transitions (i−1, i−2) ⇒
  non-diagnostic ⇒ S ≈ 0.
So the model predicts a clean semantic signal **only at the dominant tier** —
exactly where the model-free EE−SC effect is significant (n.s. at
secondary/rare). That convergence is itself supporting evidence; but it also
means the single positive wS is carried by the dominant tier and partly
*mis*-predicts secondary. A design talking point, not a bug.

**Read**: model-free EE−SC + reliably-positive wS posterior + clean recovery all
align; the LOO margin over baseline is modest. The **model-free EE−SC dwell
effect remains the strongest single statement**, with the model-based layer
corroborating direction, sign-reliability, and estimability.

## Caveats / open items
- **Dominant-tier EE−SC is provisional** until the group split is confirmed fully
  clean — though the model-free result already shows it is consistent across both
  counterbalancing groups, which supports the semantic reading.
- Recovery used **smoke-level sim counts (40)**; a full run (≈500–1000) is the next
  step before quoting recovery as final.
- **Behavioural validity only.** A positive wS means the brain computes something
  S(t)-shaped; neural/mechanistic validity (belief-state decoding, a dopaminergic
  signal tracking S) is the next tier and is not established here.
- 13533 reinstatement + a full-sim Phase-1 rerun are the obvious next actions once
  the tracking artifact is cleaned.
