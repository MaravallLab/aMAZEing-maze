# grammar_stimuli

Acoustic stimulus generation for the mouse-maze grammar-learning experiment,
integrated with the auditory harness in `src/auditory/`.

Two first-order Markov grammars (A and B) are defined over a six-tone
inventory. Each mouse alternates daily between two cage environments
(enriched, EE; standard, SC), hearing one grammar in each. On test day
the maze probes whether time spent in each arm depends on (a) the
grammar-environment association and (b) the predictability tier of the
transitions in that arm.

---

## 1. Tone inventory and timing

| Symbol | Frequency (Hz) |
|:---:|---:|
| A | 8000 |
| B | 10079 |
| C | 12699 |
| D | 16000 |
| E | 20159 |
| F | 25398 |

Six pure tones, log-spaced (~1.66 octaves, adjacent ratio ~1.26).

- Tone: 150 ms + 5 ms cosine² ramps in/out
- Inter-tone gap: 50 ms
- Melody: 12 tones → 2400 ms
- Inter-melody gap: 2000 ms
- Cycle: 4400 ms
- Training session: 4 h → ~3272 melodies → ~36k transitions

---

## 2. The two grammars

Each row of a 6×6 transition matrix contains a fixed mixture:

| Tier | Probability | Count per row |
|---|:---:|:---:|
| **dominant** | 0.60 | 1 entry |
| **secondary** | 0.12 | 2 entries |
| **rare** | 0.07 | 2 entries |
| **self** | 0.02 | 1 entry |

Rows sum to 1.0. Columns also sum to 1.0 — both matrices are doubly
stochastic, so the stationary distribution is uniform over the six tones.

The grammars differ only in **which** transition is dominant:

- **Grammar A**: ascending cycle, i → (i+1) mod 6 (A→B→C→D→E→F→A)
- **Grammar B**: skip-3, i → (i+3) mod 6 (A↔D, B↔E, C↔F)

Both have the same per-row entropy and the same overall entropy rate
(~1.78 bits/transition). Any grammar-specific effect at test should
therefore cancel across counterbalance groups.

To inspect the matrices and confirm all four probabilities are encoded:

```bash
cd src/auditory
python -m grammar_stimuli.dump_matrices grammars.txt --samples 20
```

The output text file contains the numeric matrices, tier labels, sample
melodies at every tier, and empirical transition counts from 100k draws
(should reproduce 0.60 / 0.12 / 0.07 / 0.02 ±0.005).

---

## 3. Tier system and the special `'all'` tier

The 6×6 matrix is **never modified** at runtime. Complexity is controlled
by which columns of each row the sampler is allowed to draw from.

| Tier | What it samples | Effective per-step surprise |
|---|---|:---:|
| `'all'` | the **full row** (mixture of 0.60/0.12/0.07/0.02) | ~1.78 bits |
| `'dominant'` | the 1 column with p=0.60 | 0.0 bits (deterministic) |
| `'secondary'` | the 2 columns with p=0.12, renormalised to 0.5/0.5 | 1.0 bit |
| `'rare'` | the 2 columns with p=0.07, renormalised to 0.5/0.5 | 1.0 bit |

**Important**: within-tier surprise collapses to 0/1/1 for the test tiers
because tier restriction + renormalisation makes choices uniform inside
the restricted subset. What distinguishes secondary from rare at test is
not within-tier entropy — it's the *prior probability* the mouse has
learned for those transitions during training.

### Why training uses `'all'`

Training samples directly from the unrestricted matrix row. This means
mice are exposed to all four probability tiers during training and form
proper probability estimates over every transition. Otherwise the
secondary/rare tones on test day would be perceived as **novel** rather
than as **statistically rare**, which would change what the experiment
is actually measuring.

In a 4 h training session (~36k transitions) each mouse hears (for the
grammar paired with the day's environment):

| Tier | Approx. count |
|---|---:|
| dominant (0.60) | ~21,600 |
| each secondary (0.12) | ~4,300 (x2 cols/row) |
| each rare (0.07) | ~2,500 (x2 cols/row) |
| self (0.02) | ~720 |

---

## 4. Counterbalancing and the alternate-day training design

Each mouse alternates daily between an enriched (EE) cage and a
standard (SC) cage, while **one grammar plays per day from one speaker**
to the whole room. Mice from different counterbalance groups occupy
different cage-types each day so that, from each mouse's perspective,
each grammar gets paired with one specific environment.

| Group | When in EE cage hears... | When in SC cage hears... |
|:---:|:---:|:---:|
| 1 | Grammar A | Grammar B |
| 2 | Grammar B | Grammar A |

So on any single training day:

| Today's grammar | Group 1 mice are in... | Group 2 mice are in... |
|:---:|:---:|:---:|
| **A** | EE cage | SC cage |
| **B** | SC cage | EE cage |

After several alternating days, each mouse has formed an
**auditory-structure ↔ environment association**: one grammar "belongs"
to the EE cage, the other to the SC cage. Counterbalancing flips which
grammar plays this role across groups so any preference at test cannot
be attributed to grammar-specific properties.

Both grammars have identical row mixtures and identical entropy rates;
only the dominant-cycle structure differs.

---

## 5. Test-day arm plan

8 arms on the maze, presented after training is complete:

| Arm | Kind | Env. association | Tier |
|:---:|:---|:---:|:---:|
| arm1 | grammar | EE | dominant |
| arm2 | grammar | EE | secondary |
| arm3 | grammar | EE | rare |
| arm4 | grammar | SC | dominant |
| arm5 | grammar | SC | secondary |
| arm6 | grammar | SC | rare |
| arm7 | vocalisation | – | – |
| arm8 | silent | – | – |

The 6 grammar arms cross **{EE-paired, SC-paired} × {dominant, secondary, rare}**.
Both grammars are fully familiar to the mouse by test day — there is no
"novel grammar". The contrasts the design measures are:

1. **Association effect**: does the mouse spend more time in EE-paired arms
   than SC-paired arms (or vice versa)? Reflects which environment the
   mouse "prefers" via its auditory association.
2. **Predictability effect**: does the mouse prefer high-predictability
   (dominant), moderately predictable (secondary), or low-predictability
   (rare) transitions? Reflects sensitivity to statistical structure.
3. **Interaction**: does the predictability preference depend on which
   environment the arm is associated with?

`arm7` (vocalisation, if provided) and `arm8` (silent) are control arms
for baseline acoustic preference.

---

## 6. Files

| File | Role |
|---|---|
| `config.py` | Tone inventory, timing, matrices, tiers, test-arm plan, counterbalancing |
| `tone_generator.py` | `generate_tone`, `generate_melody`, `generate_silence_gap` |
| `sequence_sampler.py` | `MarkovSampler`, `get_tier_targets`, `compute_entropy_rate` |
| `session_runner.py` | `SessionRunner` (standalone driver for training days) |
| `run.py` | CLI entry: `python -m grammar_stimuli.run ...` |
| `dump_matrices.py` | Print matrices + sample melodies + empirical counts |

---

## 7. How to run it

### 7a. Training day (no video, just continuous melody playback)

Standalone runner — uses `sounddevice` directly, no camera, no Arduino,
no ROI gating. Run it **once per day**: one grammar plays from one
speaker; every cage in the room (EE-type and SC-type alike) hears the
same audio stream.

```bash
cd src/auditory

# Day 1 (Grammar A): cage 6224 is in its EE-type cage,
# cage 6225 is in its SC-type cage; both in the same room.
python -m grammar_stimuli.run --mode training \
    --grammar A \
    --cage-ids "6224_EE,6225_SC" \
    --duration-seconds 14400 \
    --output-dir ./sessions/2026-05-12_grammarA

# Day 2 (Grammar B): cages swap status.
python -m grammar_stimuli.run --mode training \
    --grammar B \
    --cage-ids "6224_SC,6225_EE" \
    --duration-seconds 14400 \
    --output-dir ./sessions/2026-05-13_grammarB
```

`--cage-ids` is bookkeeping only — it tags the output filename and the
JSON summary; it does not affect the audio.

Optional flags:
- `--seed N` — reproducible RNG (omit for a random seed each run)
- `--dry-run` — no audio output, but still generates and logs symbols
- `--device-id N` — choose sounddevice output (default 3)
- `--group N --condition X` — alternate way to pick the grammar (looks up
  from counterbalance table). Useful when one room contains only cages
  from a single group; less useful in the same-room mixed setup.
- `--duration-seconds 60` — short sanity check

Output (per session): one CSV with melody index, symbols played, per-step
surprise bits, onset/offset times; plus a JSON summary including the
cage-ids list.

### 7b. Test days — 3-day protocol, 1 hour per mouse per day

Each mouse goes through three test sessions:

| Day | `grammar_mode` | What plays | Total duration | Block schedule |
|:---:|---|---|:---:|---|
| Day 1 | `"silent_baseline"` | nothing — pure silence | 60 min | single 60-min trial |
| Day 2 | `"test"` | full grammar test | 60 min | `[4, 12, 2, 12, 2, 12, 2, 12, 2]` min |
| Day 3 | `"test"` | full grammar test (new RNG draw) | 60 min | same as Day 2 |

Driven by `src/auditory/main.py` — the full harness with camera, ROI
monitor, video writer, and trial CSV.

**Stable defaults** live in `src/auditory/config.py` (sample rate, ROI
count, timing schedule, output path, etc.). Set those once.
**Per-mouse and per-day values** come from the command line so you
don't have to edit the config file for each session.

```bash
cd src/auditory

# Day 1, silent baseline, mouse whose EE-grammar is A
python main.py --grammar-mode silent_baseline --enriched-grammar A

# Day 2/3, audio test, same mouse
python main.py --grammar-mode test --enriched-grammar A

# Different mouse whose EE-grammar is B
python main.py --grammar-mode test --enriched-grammar B

# Re-draw the ROIs once at the start of the day
python main.py --grammar-mode silent_baseline --enriched-grammar A --draw-rois
```

CLI flags:

| Flag | Effect |
|---|---|
| `--grammar-mode {silent_baseline,test,training}` | Day of the protocol. `training` will refuse and point you at `grammar_stimuli.run`. |
| `--enriched-grammar {A,B}` | Which grammar this mouse heard in EE. Drives arm assignment on test day. |
| `--seed N` | RNG seed for reproducible melody draws. |
| `--draw-rois` | Force interactive ROI re-drawing (deletes `rois1.csv` first). |

Any flag you don't pass keeps the default in `config.py`. Defaults are
deliberately conservative (`grammar_mode = "training"` so a forgotten
flag raises a clear error instead of silently running the wrong
session).

**Day 1 (silent_baseline)** runs as one continuous 60-minute trial with
silent stimuli on every arm. The ROI monitor still logs every visit so
you get the mouse's baseline arm preference without any acoustic
influence.

**Days 2 and 3 (test)** use the 9-block cycle: four 12-min active
blocks (with the 8 stimuli randomly shuffled across the 8 ROIs each
time) separated by short silent gaps. Every ROI entry samples a
**fresh** 12-tone melody — so the same arm replays the same
grammar/tier but never the exact same sequence. Day 2 and Day 3
produce different melody draws (different RNG state).

#### Customising the durations

The session length is not hardcoded — two `ExperimentConfig` fields
expose them:

```python
grammar_silent_baseline_minutes: float = 60.0    # Day 1 total length

grammar_test_block_minutes: List[float] = [
    4.0, 12.0, 2.0, 12.0, 2.0, 12.0, 2.0, 12.0, 2.0
]                                                # Day 2/3, 9-block cycle
```

The test list must have exactly 9 entries (even indices are silent
blocks, odd indices are active blocks). Zero-minute entries are
allowed if you want to remove a silent gap. Setting it to e.g.
`[3, 15, 0, 1, 0, 15, 0, 15, 0]` yields 3 min of opening silence
followed by four nearly-uninterrupted active blocks.

#### Re-drawing the ROIs

The first time you run `main.py`, you'll be prompted to draw all 10
ROIs (2 entrances + ROIs 1–8) interactively using OpenCV's selectROI.
The layout is saved to `src/auditory/rois1.csv` and reused for every
subsequent session.

To re-draw the ROIs (e.g. you moved the maze), set
`draw_rois: bool = True` in `config.py` for the next run. The existing
`rois1.csv` is deleted and you'll be prompted again. Remember to set
`draw_rois` back to `False` afterwards.

### 7c. Inspect / verify the matrices

```bash
cd src/auditory
python -m grammar_stimuli.dump_matrices grammars.txt --samples 20
```

Open `grammars.txt` — the bottom shows empirical transition counts
under `tier='all'`; you should see all four probabilities reproduced
within ~0.005 of the matrix values.

---

## 8. Output files

### Training (run.py)

`./sessions/<output-dir>/training_g<group>_<condition>_<grammar>_<timestamp>.csv`

| Column | Meaning |
|---|---|
| `melody_index` | 0-indexed melody count |
| `arm_id` | always `"training"` |
| `grammar` | `"A"` or `"B"` (whichever was paired with this day's environment) |
| `tier` | `"all"` |
| `start_symbol` | first tone of this melody |
| `symbols` | 12-character string e.g. `"BCDEFABCDEFA"` |
| `per_step_bits` | semicolon-separated surprise per transition |
| `mean_bits` | mean surprise across the melody (≈1.78 for `'all'`) |
| `onset_s`, `offset_s` | simulated clock times in seconds |

### Test (main.py)

Two CSVs per session, in the standard data_manager output folder:

`trials_<timestamp>.csv` — per (trial × ROI) row:
- `trial_ID`, `ROIs`, `frequency`, `grammar`, `tier`,
  `environment_association`, `wave_arrays`, `time_spent`,
  `visitation_count`, etc.

`grammar_samples_<timestamp>.csv` — one row per rendered melody (only
produced on Day 2 / Day 3; Day 1 has no melodies so the file is absent
or empty):
- `trial_ID`, `ROI`, `grammar`, `tier`, `environment_association`,
  `symbols`, `mean_bits`

`environment_association` is `"EE"`, `"SC"`, or `"-"` (for voc/silent
arms and silent blocks). It directly identifies which cage that grammar
was paired with during training, so you can filter visits by association
without re-deriving from group + grammar.

Plus the usual `visit_log_*.csv`, video file, and metadata that the
harness produces.

---

## 9. Sanity checks

```bash
# Matrix integrity + empirical transition counts
python -m grammar_stimuli.dump_matrices

# Quick dry-run of 60 s of training (no audio device needed)
python -m grammar_stimuli.run --mode training --grammar A \
    --duration-seconds 60 --dry-run --output-dir ./_check
```
