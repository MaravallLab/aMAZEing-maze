# grammar_stimuli

Acoustic stimulus generation for the mouse-maze grammar-learning experiment.

Two first-order Markov grammars (A and B) are defined over a six-tone
inventory. Mice are trained for several sessions on one grammar (dominant
transitions only), then tested on eight maze arms that cross grammar
(trained vs. novel) with complexity tier (dominant / secondary / rare),
plus a vocalisation arm and a silent arm.

## Design invariants

1. **The grammar matrix is never modified.** Complexity is varied by
   restricting sampling on each transition to the columns of a row whose
   probability equals a chosen *tier* value, then renormalising within that
   subset. The stored 6x6 matrix itself is frozen.
2. Every row has the same mixture `{1 x 0.60, 2 x 0.12, 2 x 0.07, 1 x 0.02}`,
   so both grammars are doubly stochastic with identical per-row entropy.
3. Both grammars have the same entropy rate (~1.78 bits / transition).
   They differ only in *which* transition is dominant: Grammar A cycles
   upwards (A->B->C->...->F->A); Grammar B skips by 3 (A<->D, B<->E, C<->F).

## Files

| File | Role |
|---|---|
| `config.py` | Tone inventory, timing constants, matrices, tiers, test-arm plan, counterbalancing |
| `tone_generator.py` | `generate_tone`, `generate_melody`, `generate_silence_gap` |
| `sequence_sampler.py` | `MarkovSampler`, `get_tier_targets`, `compute_entropy_rate` |
| `session_runner.py` | `SessionRunner` for training and test sessions |
| `run.py` | CLI entry point (`python -m grammar_stimuli.run ...`) |
| `verify_grammars.py` | Standalone 11-step integrity check |
| `tests/test_grammars.py` | pytest suite |

## Quick start

Dry-run (no audio device needed) a 60-second training session for
counterbalance group 1, Enriched-Environment housing:

```bash
python -m grammar_stimuli.run \
    --mode training --group 1 --condition EE \
    --duration-seconds 60 --dry-run --output-dir ./sessions
```

Sanity-check everything:

```bash
python -m grammar_stimuli.verify_grammars
pytest code/grammar_stimuli/tests -q
```

## Counterbalancing

| Group | EE housing trains on | SC housing trains on |
|:---:|:---:|:---:|
| 1 | Grammar A | Grammar B |
| 2 | Grammar B | Grammar A |

Because the two grammars are structurally identical in their tier mixture,
this counterbalancing guarantees that any grammar-specific effect at test
cancels across groups.

## Test-day arm plan

| Arm | Kind | Grammar | Tier |
|:---:|:---|:---:|:---:|
| arm1 | grammar | trained | dominant |
| arm2 | grammar | trained | secondary |
| arm3 | grammar | trained | rare |
| arm4 | grammar | novel   | dominant |
| arm5 | grammar | novel   | secondary |
| arm6 | grammar | novel   | rare |
| arm7 | vocalisation | - | - |
| arm8 | silent | - | - |

## Timing

- Tone: 150 ms + 5 ms cosine-squared ramps (in and out)
- Inter-tone gap: 50 ms
- Melody: 12 tones -> 2400 ms
- Inter-melody gap: 2000 ms
- Cycle: 4400 ms
- Training session: 4 h -> ~3272 melodies -> ~36k transitions

## Information content

Per-tone surprise under the unrestricted grammar:

| Tier | p | -log2 p (bits) |
|---|---|---|
| dominant | 0.60 | 0.74 |
| secondary | 0.12 | 3.06 |
| rare | 0.07 | 3.84 |

Per-transition surprise under the *restricted* tier actually used for
sampling:

| Tier | columns / row | bits / transition |
|---|---|---|
| dominant | 1 | 0.00 (deterministic) |
| secondary | 2 | 1.00 |
| rare | 2 | 1.00 |

The surprise collapses to 0 / 1 / 1 because tier restriction + renormalisation
makes tier choices uniform *within* the restricted subset. The difference
between secondary and rare is therefore not in within-tier entropy but in
their *distance from the trained grammar's dominant transitions*: a rare
transition is by definition less typical than a secondary one.

## Integration with the maze session controller

`SessionRunner` is a standalone driver that plays melodies on a fixed
cycle. For the live behavioural session, the existing
`code/auditory/updated_version/main.py` drives stimulus playback from ROI
entries: it should construct a `MarkovSampler` per arm, call
`sampler.sample_melody(...)` on entry, and pass the resulting waveform
(built with `generate_melody`) to the existing `Audio.play` method.
