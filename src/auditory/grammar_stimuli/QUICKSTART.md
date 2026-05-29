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

## Test days (3-day protocol)

Each mouse goes through **three test days**:

| Day | Label | Duration | What plays | Purpose |
|:---:|---|:---:|---|---|
| **Habituation** | `habituation` | 45 min | nothing (pure silence) | Baseline ROI preference without acoustic influence |
| **Test day 1** | `day_1` | 60 min | full grammar test | Effect of audio + grammar associations |
| **Test day 2** | `day_2` | 60 min | full grammar test (different random sequences) | Within-mouse replication |

### Step 1. Move the mouse into the maze

The test happens in the maze, which is a neutral environment (not in
its EE or SC cage).

### Step 2. Pass the per-mouse / per-day flags to `main.py`

Stable settings live in `config.py` and you set them once. The
per-mouse and per-day values come from the command line:

```bash
# Mouse 6224 (EE→A), habituation day (silent baseline)
python main.py --grammar-mode silent_baseline --enriched-grammar A --day habituation

# Mouse 6224, first audio test day
python main.py --grammar-mode test --enriched-grammar A --day day_1

# Mouse 6224, second audio test day
python main.py --grammar-mode test --enriched-grammar A --day day_2

# Mouse 6225 (EE→B), habituation day
python main.py --grammar-mode silent_baseline --enriched-grammar B --day habituation
```

All CLI flags:

| Flag | Effect |
|---|---|
| `--grammar-mode {silent_baseline,test}` | Which day of the protocol (overrides `cfg.grammar_mode`) |
| `--enriched-grammar {A,B}` | Which grammar this mouse heard in EE (overrides `cfg.enriched_grammar`) |
| `--day LABEL` | Parent folder in the output path — e.g. `habituation`, `day_1`, `day_2` |
| `--seed N` | RNG seed for reproducible melody draws (overrides `cfg.grammar_seed`) |
| `--draw-rois` | Force interactive ROI re-drawing (overrides `cfg.draw_rois`) |

Anything you don't pass keeps the default from `config.py`.

**Current schedule:**
- `silent_baseline` — one 45-minute trial, no audio
- `test` — four 15-minute active blocks, no silent gaps between them (60 min total)

#### Customising the schedule

Override these fields in `src/auditory/config.py` if needed:

```python
grammar_silent_baseline_minutes: float = 45.0    # single trial duration

grammar_test_block_minutes: List[float] = field(
    default_factory=lambda: [0, 15.0, 0, 15.0, 0, 15.0, 0, 15.0, 0]
)
```

The test list **must** have exactly 9 entries. Indices 0/2/4/6/8 are
silent blocks, indices 1/3/5/7 are active blocks. Set a silent-block
entry to `0` to skip it entirely.

Example: `[2, 15, 0, 15, 0, 15, 0, 15, 0]` → 2-minute opening silence,
then four 15-minute active blocks with no gaps between them (62 min total).

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
python main.py --grammar-mode <silent_baseline|test> --enriched-grammar <A|B> --day <habituation|day_1|day_2>
```

### Step 4. After the session ends

The session folder is created at:

```
base_output_path / grammar / <--day label> / time_<timestamp>_<mouseID> /
```

For example:
```
maze_recordings/grammar/day_1/time_2026-05-23_14_30_00_mouse1/
```

**Files saved automatically:**

| File | Contents |
|------|----------|
| `trials_<timestamp>.csv` | One row per (trial block, ROI): stimulus assigned, time spent, visit count |
| `<mouseID>_grammar_detailed_visits.csv` | One row per visit: ROI, stimulus string, entry/exit times, duration |
| `<mouseID>_grammar_maze_entries.csv` | Maze entry and exit events with timestamps |
| `<mouseID>_<timestamp>_metadata.csv` | Mouse metadata (ID, DOB, sex, ear marks) |
| `grammar_samples_<timestamp>.csv` | One row per melody played — grammar, tier, symbol sequence (test days only) |
| `trials_<timestamp>.npy` | Raw audio arrays (for debugging sounds) |
| `fig1_arm_totals.png` … `fig6_location_preference.png` | Per-session analysis figures (auto-generated) |

The `environment_association` column in `trials_<timestamp>.csv` tells
you whether each row was an EE-paired or SC-paired grammar arm (it's
`"-"` on silent_baseline days).

**If you need to regenerate the per-session figures later:**

```bash
cd src/auditory
python run_analysis.py "C:\path\to\session_folder"
```

### Step 4b. After the last mouse of a day — run the day summary

After you've run all mice for a given day, generate cross-mouse summary
figures with:

```bash
cd src/auditory

# Summary for one day (all mice tested that day):
python run_summary_analysis.py --day "C:\...\maze_recordings\grammar\day_1"

# Cross-day summary (all mice, all days collected so far):
python run_summary_analysis.py --all "C:\...\maze_recordings\grammar"
```

Figures are saved into the folder you pass.

| Figure | What it shows |
|--------|---------------|
| `summary_A_ee_sc_per_mouse.png` | EE vs SC total time — one pair of bars per mouse per day |
| `summary_B_preference_index.png` | EE preference index (−1 to +1) per mouse per day |
| `summary_C_group_summary.png` | Group mean ± SEM time and visits on EE vs SC arms |
| `summary_D_cross_day_pi.png` | PI trajectory per mouse + group mean across days *(multi-day only)* |
| `summary_E_tier_breakdown_per_mouse.png` | Stacked bars showing dominant / secondary / rare time on EE and SC arms per mouse |
| `summary_F_group_tier_breakdown.png` | Group mean ± SEM for all 6 tier × environment combinations |
| `summary_G_cross_day_tiers.png` | Per-tier preference across days — group mean ± SEM for each complexity level *(multi-day only)* |

Silent-baseline sessions (no audio) are automatically excluded from
all summary figures — only test-day sessions contribute.

### Step 5. Run the next mouse / next day

Just call `python main.py` again with the right flags. No need to edit
`config.py`:

```bash
# Next mouse, same day
python main.py --grammar-mode silent_baseline --enriched-grammar B --day habituation

# Same mouse, next test day
python main.py --grammar-mode test --enriched-grammar A --day day_2
```

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `python -m grammar_stimuli.run` says "No module named grammar_stimuli" | Make sure you're in `src/auditory/` |
| `main.py` raises `NotImplementedError` about training | You forgot `--grammar-mode test` (or `silent_baseline`) — the config default intentionally forces you to pass it |
| `main.py` errors about `rois_number` | Set `rois_number: int = 8` in `src/auditory/config.py` |
| Baselines all 0 after calibration | The ROI coordinates in `rois1.csv` are likely stale — re-run with `--draw-rois` to redraw them |
| Mouse not detected / no ROI ENTERED messages | Try raising `detection_sensitivity` from `0.5` to `0.7` in `config.py`, or redraw ROIs with `--draw-rois` |
| No audio | Check `channel_id` in `config.py` — run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list available devices |
| "No gain" warning at startup | The calibration CSV was not found — check `calibration_gain_path` in `config.py` resolves to `analysis/calibration/frequency_response_speaker.csv` in the repo |
| Audio plays but I want to verify what was played | Open `grammar_samples_<timestamp>.csv` and inspect the `symbols` and `tier` columns |
| Want to reproduce the same melody draws | Re-run with `--seed N` using the same seed (otherwise check the CSV) |
| Post-session figures not generated | Run `python run_analysis.py <session_folder>` manually; check the terminal for the traceback |
