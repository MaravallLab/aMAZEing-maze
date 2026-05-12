# Quickstart

Step-by-step operational guide. For the *why*, see `README.md`.

All commands are run from `src/auditory/`.

---

## Before the experiment (one-time setup)

### Step 1. Assign each mouse to a counterbalance group

Decide once per mouse, then never change it. Write it down somewhere.

- **Group 1** mice: hear Grammar A when in the EE cage; Grammar B when in the SC cage.
- **Group 2** mice: hear Grammar B when in the EE cage; Grammar A when in the SC cage.

Split your mice ~half-and-half across the two groups.

You won't pass the group to `run.py` (the script only needs to know
which grammar to play today). The group assignment is what tells *you*
which cage-type each mouse should be in on a given day so that the
grammar coming out of the speaker is correctly paired with its
environment.

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

## Test day

### Step 1. Move the mouse into the maze (not into any cage)

The test happens in the maze, which is a neutral environment.

### Step 2. Edit `src/auditory/config.py` for this specific mouse

```python
experiment_mode: str = "grammar"
grammar_mode: str = "test"
grammar_group: int = <THIS-MOUSE's-GROUP, 1 or 2>
rois_number: int = 8
record_video: bool = True
# Optional:
# grammar_seed: int = 42
# path_to_vocalisation_control: str = "/path/to/voc.wav"
```

There's no `grammar_condition` to set — the mouse has been in both
cage-types during training, so the maze just plays both grammars.

### Step 3. Start the maze session

```bash
python main.py
```

The harness runs the standard 9-block cycle
(15-15-2-15-2-15-2-15-2 minutes by default), with the 8 stimuli shuffled
across the 8 ROIs every active block.

### Step 4. After the session ends

Look in the data folder for the new session. Two new CSVs:

- `trials_<timestamp>.csv` — what was assigned to each ROI in each block, plus visit counts and time spent.
- `grammar_samples_<timestamp>.csv` — one row per melody actually played (with the symbol sequence and which arm/tier it was).

The `environment_association` column in both files tells you whether
each row was an EE-paired or SC-paired grammar arm.

### Step 5. Run the next mouse

Edit `grammar_group` in `config.py` again if the next mouse is in a
different group, then `python main.py` again.

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `python -m grammar_stimuli.run` says "No module named grammar_stimuli" | Make sure you're in `src/auditory/` |
| `main.py` errors complain about rois_number | Set `rois_number: int = 8` in `src/auditory/config.py` |
| No audio | Use `--device-id N` to pick a different sounddevice output |
| Audio plays but I want to verify | Open the session CSV and inspect the `symbols` column |
| Want to re-derive what was played | Re-run with the same `--seed N` and you get the same melodies (otherwise just look at the CSV) |
