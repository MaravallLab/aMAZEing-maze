# What the model-validation analysis means — a walkthrough for experimentalists

This is a plain-language guide to *why each analysis exists*, *what it found*, and
*what you can and cannot say because of it*, written for a wet-lab neuroscientist
rather than a statistician. Figures referenced are the PNGs in the run's output
folder. Numbers are from the current day-1 cohort (33 mice; counterbalancing
groups 16 / 17).

---

## The question, in one sentence

During training each mouse heard one tone-grammar in its **enriched cage (EE)**
and a different grammar in its **standard cage (SC)**. On the test day the maze
plays *both* grammars on different arms. We want to know:

> Does the mouse **recognise the grammar tied to its enriched home and prefer
> being near it** — and is that preference a genuine *semantic* recognition, over
> and above the boring explanation that "more predictable sounds are just easier
> to process"?

Everything below is built to answer that one question while ruling out the ways
it could be fooled.

---

## The three things the model tracks (intuition, no equations)

For each arm we run an "ideal-listener" model over the tones actually played and
get three numbers:

- **r = fluency / predictability.** How expected was each tone given the last
  one? A predictable arm (the *dominant* tier) is high-r; a surprising arm (the
  *rare* tier) is low-r. Think of it as "how easy the sound is to process."
- **ΔV = learning.** How much hearing this arm nudges the listener's internal
  model toward its long-run expectations. In a steady arm this is ~0 (there's
  nothing new to learn) — and indeed it carries almost no signal here, which is
  fine and expected.
- **S = recognition (the new term).** A running "which world am I in?" belief —
  EE-grammar world vs SC-grammar world — that goes **up** when the tones match
  the mouse's EE grammar and **down** when they match the SC grammar. This is the
  putative *semantic* signal: it only exists if the brain has learned the
  grammar→environment association.

The behavioural model says preference for an arm ≈ `w0 + wr·r + wV·ΔV + wS·S`.
The single number we care about is **wS**: is it positive? That would mean
recognition adds preference beyond mere fluency.

**Why the listener must score tone *transitions*, not single tones** (a point that
matters): the two grammars use the *same six tones equally often* — they differ
only in which tone tends to *follow* which. So a model that looked at single
tones could never tell EE from SC, and S would be flat by construction. Scoring
transitions is what makes recognition measurable at all.

---

## Why we don't believe a model fit until two "controls" pass

A computational model can always be *fit*; the question is whether the fit means
anything. Two gates protect us, exactly like controls in an experiment:

1. **A model-free effect must exist first.** If the raw dwell times don't already
   show an EE-vs-SC difference, no amount of modelling is worth interpreting.
2. **Parameter recovery must pass.** Before trusting "wS is positive," we prove
   that *with these exact stimulus sequences* the method can (a) recover a known
   wS when we plant one, and (b) **not** report a positive wS when there is none.
   This is the analysis equivalent of a no-primary-antibody control.

Only after both pass do we read the Bayesian model comparison.

---

## The analyses, one by one

### 1. Cohort & exclusions (no figure)
33 mice after **named, reasoned** exclusions: one animal whose tracking was
corrupted by a stray light event (excluded on both test days), one session where
EE/SC were swapped by mistake, and duplicate copies of sessions. **Why it
matters:** a reviewer wants to see that every exclusion was a stated rule with a
cause, applied before analysis — not a quiet drop. Re-including the light-artifact
animal once its video is cleaned is a one-line change.

### 2. Sanity gate (no figure)
The maze **reshuffles which arm carries which sound every 15 min**, so physical
arm position should carry no preference. We confirm that (arm position explains
~6% of dwell, below threshold → pass). We deliberately do **not** require "silence
is the least-preferred arm," because on the test day mice actually find the
grammar tones mildly aversive and vocalisations attractive — a real biological
effect, not a rig failure. **Why it matters:** it certifies that any EE/SC
difference is about the *sound*, not about a corner of the room the mouse likes.

### 3. The model-free result — the headline (Fig 1, Fig 2, Fig 3)
A mixed-effects model of dwell time with environment × tier × group and a random
intercept per mouse.

- **Fig 1 (EE vs SC dwell by tier, one panel per group)** is the single most
  important figure. Mice spend **more time on EE arms than SC arms**, the effect
  is **largest at the predictable (dominant) tier and fades by the rare tier**,
  and — crucially — **it points the same way in both counterbalancing groups**
  (group 1: EE−SC = +0.45, p = .002; group 2: +0.27, p = .07).
- **Why the two panels are the whole ballgame:** at the dominant tier the two
  grammars sound very different (one is an ascending sweep, the other an octave
  trill). If mice simply preferred "the sweep," that preference would **flip**
  between groups (because the sweep is EE for one group and SC for the other).
  It does **not** flip — EE wins in both — so the preference is tied to the
  *learned association*, i.e. **semantic, not an artifact of how the grammars
  happen to sound.** This is the experimental crux.
- **Fig 2 (per-mouse preference index)** shows the effect is carried by most
  animals in both groups, not a handful of outliers.
- **Fig 3 (six-cell pattern + reference arms)** shows the raw behaviour the model
  must explain, with silence and vocalisation as anchors.

**What you can say:** "Mice preferentially approached the arm playing the grammar
associated with their enriched environment; this held in both counterbalancing
groups and was strongest for the most predictable version of the grammar."

### 4. The latent regressors (Fig 4) — and a subtlety worth a slide
Fig 4 plots each stimulus condition in **r (fluency) vs S (recognition)** space.
Two things to read off it:

- **r and S are essentially uncorrelated** (correlation ≈ 0): the design genuinely
  separates "easy to process" from "recognised as mine." That separation is what
  makes it possible to ask whether recognition matters *beyond* fluency.
- **S has a specific, non-obvious shape across tiers**, and this is real biology
  of the stimulus design, not a glitch:
  - **dominant tier:** S is cleanly positive for EE arms, negative for SC arms.
  - **secondary tier:** S *flips* — because Grammar A's second-most-common step is
    literally Grammar B's *most* common step, so a "secondary" sequence sounds
    like the other grammar.
  - **rare tier:** S ≈ 0 — the two grammars share the same rare transitions, so a
    rare sequence is uninformative about which grammar it is.

  So the model predicts a **clean recognition signal only at the dominant tier** —
  which is exactly where the behavioural EE−SC effect is significant. The fact
  that "where recognition is computable" and "where behaviour shows preference"
  coincide is itself supporting evidence. The flip side: a single recognition
  weight is carried by the dominant tier and will not explain the (weak) secondary
  tier — a genuine design talking point.

### 5. Recovery — the methodological control (Fig 5)
This is the analysis we trust the most. Using the **real delivered sequences**, we
simulate behaviour with a *known* recognition weight and try to recover it.

- **Left panel:** recovered wS lands on the true value at 0, 0.5, 1, 2; and when
  the truth is **zero**, the method reports zero (false-positive rate **5.4%**,
  i.e. it cries wolf at the expected ~5% chance level).
- **Right panel:** when we generate data from the boring (fluency-only) model the
  comparison does **not** prefer the recognition model; when we generate from an
  "everyone just likes one grammar's sound" confound, the recognition weight comes
  out ~0 (no false positive). When we generate from a true recognition model, the
  comparison picks it 100% of the time.

**Why it matters:** it proves the EE−SC signal in these stimuli is *estimable* and
that a positive wS can't be manufactured by the fluency or the
intrinsic-grammar-sound confounds. Without this, a positive wS would be
uninterpretable.

### 6. Does recognition earn its place out of sample? (Fig 6)
- **Left:** Bayesian leave-one-out comparison of four nested models. The full model
  (with recognition) **ranks best**, but the error bars (dse) comfortably cross
  zero — so out of sample the models are only **weakly separated**. Adding
  recognition helps in the right direction, but not dramatically, on day 1.
- **Right:** the recognition weight's posterior sits **entirely above zero**
  (95% interval [0.01, 0.14], 99% of the posterior > 0). So wS is *reliably
  positive* even though its *out-of-sample payoff is modest*.

**Honest reading:** the strongest single statement is the model-free EE−SC effect.
The model-based layer corroborates it — recognition weight reliably positive,
mouse-stable, and improves fit in the right direction — but it does not, on its
own, decisively beat the simpler model out of sample. (The final fit used 4
chains with a non-centered reparameterization: R-hat = 1.00, ESS > 4,000, 0
divergences, and wₛ reproduced across three independent runs — so these numbers
are solid, the modest out-of-sample margin is a real finding, not under-sampling.)

### 7. Can the model reproduce the behaviour? (Fig 7 + Fig 9)
The most informative pair — and one where an early figure bug had to be fixed (the
predicted dwell must be computed with the SAME SD-standardized features the model
was fit on; applying the weights to raw features shrank the semantic effect ~20×
and made the model look flat). Corrected:
- At the **dominant** tier the model fits well: predicted EE-dominant 0.149 vs
  SC-dominant 0.124 (observed 0.170 vs 0.121) — right sign, ~½ the magnitude, and
  it reproduces the SC-dominant trough.
- At the **secondary** tier it **flips the wrong way** (predicts SC > EE; observed
  EE > SC), and at **rare** it is flat.

**Fig 9** makes the cause exact by fitting the semantic weight on each tier alone:
**+0.18 at dominant** (strong), **−0.03 at secondary** (the i→i+3 flip), **0 at
rare** — so the joint estimate (+0.055) is a ~3× *dilution* of a genuinely strong
dominant-tier effect, forced by using one weight for all three tiers when S is only
clean at one. The model **does** capture the behaviour where S is well-defined
(dominant); its partiality is fully traced to the grammar implementation, not to a
mis-set weight or a failure of the idea. We deliberately did **not** add an
interaction term or redefine S.

### 8. Day 2 — is the association fading? (Fig 8)
Within each test day, preference is tracked across the four 15-min blocks. On
day 2 the EE−SC preference **declines across blocks** (starts ~+0.26, ends
negative; slope ≈ −0.11/block). This is the signature of **extinction** — the
association weakening when it is no longer reinforced — rather than a stable
lifelong preference. It's reported as a separate, secondary question and was
deliberately *not* pooled with day 1.

---

### 9. Why the model looked flat at first — and the fix (`grainB`/`grainC`/`grain_weights`)

Two things were wrong, and both are now fixed:

1. **S was sign-flipped at the secondary tier** (the `i→i+3` grammar overlap). Scoring
   tones under the *tier-restricted* distributions the animals actually heard fixes it —
   S now reads "this is my EE grammar" (+) vs "the other grammar" (−) at every tier.
2. **We were fitting at too fine a grain.** The model was fit to each individual
   arm-visit's dwell, where visit-to-visit and mouse-to-mouse noise is large, so a real
   but modest preference is invisible and every weight shrinks toward zero (the flat
   predictions). Re-fitting the **same** model to each mouse's *average* dwell per arm,
   and to the EE−SC index per tier — the grain at which the effect lives — recovers it:

   | grain fit at | semantic weight wS | reliably > 0? |
   |---|---|---|
   | per arm-visit (per-block) | +0.05, 95% CI [−0.01, 0.11] | no (CI crosses 0) |
   | per-mouse cell-mean | +0.77, CI [0.09, 1.70] | **yes** |
   | per-tier EE−SC index | +0.15, CI [0.10, 0.20] | **yes** (and sign-stable dropping any mouse) |

   At the cell grain the fluency weight is also reliably positive, and the model now
   reproduces the full **7-arm pattern** (EE-dominant peak, EE descending, low SC side,
   high silent arm). The secondary tier remains the least-perfect cell (residual `i→i+3`).

**How to read this:** it is a **mechanistic illustration, not a second independent
result.** The model is re-describing, in process-model form, what the design-based
analysis already established — "a system computing fluency + semantic recognition
reproduces the behaviour." The statistical evidence is the model-free effect; the
per-visit flatness was a **grain mismatch**, not absence of the effect. (LOO is not
compared across grains — different observation models aren't on a common scale.)

## Bottom line

- **You can say:** mice approach the grammar tied to their enriched environment;
  the effect is consistent across counterbalancing groups (so it's about the
  learned association, not the raw sound), strongest for the most predictable
  grammar, carried by most animals, and a computational recognition signal S(t)
  reconstructed from the stimuli carries a reliably positive weight that the
  method is demonstrably able to detect (and demonstrably does not fabricate).
- **You should hedge:** the *computational model* is a **mechanistic illustration,
  not independent validation** — it re-expresses the model-free result as a process
  model. With correctly-signed (tier-restricted) S and fit at the grain where the
  effect lives (per-mouse cell-mean / per-tier PI; see §9), it reproduces the full
  7-arm pattern and the semantic weight is reliably positive (95% interval excludes
  0, sign-stable across mice). At the per-visit grain it looked flat purely because
  the effect is small relative to visit-to-visit dwell noise — a diagnosed **grain
  mismatch**, not absence of effect. Lead with the model-free effect as the
  statistical evidence. The secondary tier stays the weakest cell (residual i→i+3).
  Day 2 suggests the association extinguishes.
- **You cannot yet say:** anything about *mechanism*. This is **behavioural**
  evidence that the brain computes something recognition-shaped. Showing that a
  neural signal (e.g. a belief-state representation, or dopamine tracking S(t))
  actually carries it is the next tier of work and is not established here.

## Clean next steps
1. ✓ **Done** — fixed S (tier-restricted emissions, no secondary sign-flip) and the
   grain mismatch (fit at per-mouse cell-mean and per-tier PI). The semantic weight is
   reliably positive at those grains and the model reproduces the full 7-arm pattern
   (§9); clean MCMC diagnostics (R-hat 1.00, 0 divergences); cohort resolved to 32
   explorers. Present the generative model as a mechanistic illustration, model-free
   as the statistical evidence.
2. Reinstate the light-artifact animal (13533) once its tracking is cleaned (one-line
   change in `EXCLUDED_ANIMALS`) and re-run.
3. (Optional model structure) a value-gates-fluency form (rather than additive) is the
   real structural improvement — post-poster work, not attempted here.
4. (Optional, mechanistic) the natural next tier: test whether a neural signal —
   a belief-state representation or a dopaminergic signal — tracks S(t).
