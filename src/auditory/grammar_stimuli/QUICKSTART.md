# Quickstart

Step-by-step operational guide. For the *why*, see `README.md`.

All commands are run from `src/auditory/`.

---

## Before the experiment (one-time setup)

### Step 1. Assign each mouse to a grammar-environment pairing

Decide once per mouse, then never change it. Write it down somewhere.
For each mouse, pick which grammar plays in its EE cage:

- **EE → A** mice: hear Grammar A in EE, Grammar B in SC.
- **EE → B** mice: hear Grammar B in EE, Grammar A in SC.

Split your mice ~half-and-half across these two options (this is the
counterbalance).

On test day you pass that choice as `--enriched-grammar A` or
`--enriched-grammar B` to `python main.py`. The system uses it to
decide which physical grammar plays on the EE-associated arms vs the
SC-associated arms.

### Step 2. Verify the matrices look right

```bash
python -m grammar_stimuli.dump_matrices grammars.txt --samples 10
```

Open `grammars.txt`. At the bottom you should see "Empirical transition
counts" showing all four probabilities (≈0.60 / 0.12 / 0.07 / 0.02).
If yes, the grammar is correctly encoded. You can delete `grammars.txt`.

### Step 3. (Optional) Quick dry-run to confirm audio works

```bash
python -m grammar_stimuli.run --mode training \
    --grammar A \
    --duration-seconds 60 \
    --output-dir ./_test_audio
```

Listen for 60 s. If you hear melodies, the speaker is wired up. Delete
`./_test_audio` afterwards.

---

## Each training day

### Step 1. Decide today's grammar, then place each mouse accordingly

You play **one grammar per day**, alternating A and B. Each mouse goes
into either its EE cage or its SC cage today, depending on its group
and what today's grammar is:

| Today's grammar | Group 1 mice are in... | Group 2 mice are in... |
|---|---|---|
| **A** | EE cages | SC cages |
| **B** | SC cages | EE cages |

(That's how each mouse ends up hearing its group's correct
grammar-environment pairing.) All cages can sit in the same room
sharing one speaker.

### Step 2. Start one training process for the room

All cages in the same room share one speaker, hear the same grammar.
Use `--grammar A|B` to pick today's grammar directly, and list every
cage being exposed (with its EE/SC status as a tag) in `--cage-ids`.

```bash
python -m grammar_stimuli.run --mode training \
    --grammar A \
    --cage-ids "6224_EE,6225_SC" \
    --duration-seconds 14400 \
    --output-dir ./sessions/<date>_grammarA
```

Tomorrow, swap each cage's status and switch the grammar:

```bash
python -m grammar_stimuli.run --mode training \
    --grammar B \
    --cage-ids "6224_SC,6225_EE" \
    --duration-seconds 14400 \
    --output-dir ./sessions/<date>_grammarB
```

`--cage-ids` is bookkeeping only — it tags the output filename and JSON
summary; it does not change the audio. You can use any format for the
cage tags (e.g. `"6224_EE"`, `"6224-EE"`, `"6224:EE"`); the script just
stores them as strings.

> If different cages in the room ever need *different* grammars
> simultaneously, you'd need two speakers and two processes — but your
> schedule (one grammar per day for everyone in the room) avoids that.

### Step 3. Confirm it's running

The console prints melody progress. The output folder fills with one
CSV per session. Leave it alone for 4 hours.

### Step 4. Tomorrow, flip the grammar and reshuffle the cages

If today was Grammar A, tomorrow is Grammar B. Each mouse moves to the
*other* cage-type tomorrow (group 1: EE → SC; group 2: SC → EE). Same
single command structure, just with `--grammar B` and updated
`--cage-ids` tags.

---

## Test days (3-day protocol, 1 hour per mouse per day)

Each mouse goes through **three test days**, one hour each:

| Day | What plays | Purpose |
|:---:|---|---|
| **Day 1** | nothing (pure silence) | Baseline ROI preference without acoustic influence |
| **Day 2** | full grammar test | Effect of audio + grammar associations |
| **Day 3** | full grammar test (different random sequences) | Within-mouse replication |

### Step 1. Move the mouse into the maze

The test happens in the maze, which is a neutral environment (not in
its EE or SC cage).

### Step 2. Pass the per-mouse / per-day flags to `main.py`

Stable settings live in `config.py` and you set them once. The
per-mouse and per-day values come from the command line:

```bash
# Mouse 6224 (EE→A), Day 1 (silent baseline)
python main.py --grammar-mode silent_baseline --enriched-grammar A

# Mouse 6224, Day 2 (audio test)
python main.py --grammar-mode test --enriched-grammar A

# Mouse 6225 (EE→B), Day 1 (silent baseline)
python main.py --grammar-mode silent_baseline --enriched-grammar B
```

All four CLI flags:

| Flag | Effect |
|---|---|
| `--grammar-mode {silent_baseline,test,training}` | Which day of the protocol (overrides `cfg.grammar_mode`) |
| `--enriched-grammar {A,B}` | Which grammar this mouse heard in EE (overrides `cfg.enriched_grammar`) |
| `--seed N` | RNG seed for reproducible melody draws (overrides `cfg.grammar_seed`) |
| `--draw-rois` | Force interactive ROI re-drawing (overrides `cfg.draw_rois`) |

Anything you don't pass keeps the default from `config.py`. Both modes
run for ~60 minutes total (silent_baseline = one 60-min trial; test =
`[4, 12, 2, 12, 2, 12, 2, 12, 2]`-minute 9-block cycle).

#### Customising the schedule

If you want different durations, override either of these fields in
`src/auditory/config.py`:

```python
grammar_silent_baseline_minutes: float = 60.0    # single trial duration

grammar_test_block_minutes: List[float] = field(
    default_factory=lambda: [4.0, 12.0, 2.0, 12.0, 2.0, 12.0, 2.0, 12.0, 2.0]
)
```

The test list **must** have exactly 9 entries. Indices 0/2/4/6/8 are
silent blocks, indices 1/3/5/7 are active blocks. Zero-minute entries
are allowed if you want to skip a silent gap.

Example: `[3, 15, 0, 1, 0, 15, 0, 15, 0]` → 3 min opening silence, then
four active blocks (15 + 1 + 15 + 15 = 46 min active) with no gaps.

### Step 2b. (Optional) Draw or re-draw the ROIs

If `rois1.csv` does not exist in `src/auditory/`, the script will
automatically prompt you to draw the ROIs the first time you run it
(using `cv.selectROI` — drag a rectangle, press Enter, repeat for each
of the 10 ROIs in order: 2 entrances + ROIs 1–8). After that, the same
layout is reused for every subsequent session.

If you want to **re-draw** the ROIs (e.g. you moved the maze), pass
`--draw-rois` once: the existing `rois1.csv` is deleted and you'll get
the drawing prompt again.

```bash
python main.py --grammar-mode silent_baseline --enriched-grammar A --draw-rois
```

### Step 3. Start the maze session

```bash
python main.py --grammar-mode <silent_baseline|test> --enriched-grammar <A|B>
```

### Step 4. After the session ends

Look in the data folder for the new session. CSVs:

- `trials_<timestamp>.csv` — what was assigned to each ROI in each
  block, plus visit counts and time spent.
- `grammar_samples_<timestamp>.csv` — one row per melody actually
  played (only on Day 2 / Day 3; Day 1 has no melodies).
- `visit_log_<timestamp>.csv` — per-visit entries with start, end, and
  duration; same format on all three days.

The `environment_association` column in `trials_<timestamp>.csv` tells
you whether each row was an EE-paired or SC-paired grammar arm (it's
`"-"` for all rows in silent_baseline mode).

### Step 5. Run the next mouse / next day

Just call `python main.py` again with the right flags. No need to edit
`config.py`:

```bash
# Next mouse, same day
python main.py --grammar-mode silent_baseline --enriched-grammar B

# Same mouse, next day
python main.py --grammar-mode test --enriched-grammar A
```

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `python -m grammar_stimuli.run` says "No module named grammar_stimuli" | Make sure you're in `src/auditory/` |
| `main.py` errors complain about rois_number | Set `rois_number: int = 8` in `src/auditory/config.py` |
| No audio | Use `--device-id N` to pick a different sounddevice output |
| Audio plays but I want to verify | Open the session CSV and inspect the `symbols` column |
| Want to re-derive what was played | Re-run with the same `--seed N` and you get the same melodies (otherwise just look at the CSV) |
