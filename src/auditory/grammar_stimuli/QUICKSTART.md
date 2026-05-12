# Quickstart

Step-by-step operational guide. For the *why*, see `README.md`.

All commands are run from `src/auditory/`.

---

## Before the experiment (one-time setup)

### Step 1. Assign each mouse to a counterbalance group

Decide once per mouse, then never change it. Write it down somewhere.

- **Group 1** mice: EE-day will play Grammar A, SC-day will play Grammar B.
- **Group 2** mice: EE-day will play Grammar B, SC-day will play Grammar A.

Split your mice ~half-and-half across the two groups.

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
    --group 1 --condition EE \
    --duration-seconds 60 \
    --output-dir ./_test_audio
```

Listen for 60 s. If you hear melodies, the speaker is wired up. Delete
`./_test_audio` afterwards.

---

## Each training day

### Step 1. Decide which cage each mouse is in today

Alternate daily. For example:

| Day | Mouse Alpha (group 1) | Mouse Beta (group 2) |
|---|---|---|
| Mon | EE cage | EE cage |
| Tue | SC cage | SC cage |
| Wed | EE cage | EE cage |
| ... | ... | ... |

(Mice in the same cage today share the same audio stream — only one
`run.py` process is needed per cage per day.)

### Step 2. Start one training process for the room

All cages in the same room share one speaker, hear the same grammar.
Use `--grammar A|B` to pick today's grammar directly, and list every
cage being exposed (with its EE/SC status as a tag) in `--cage-ids`.

```bash
python -m grammar_stimuli.run --mode training \
    --grammar A \
    --cage-ids "6224_EE,6225_SC" \
    --duration-seconds 14400 \
    --seed 12345 \
    --output-dir ./sessions/<date>_grammarA
```

Tomorrow, swap each cage's status and switch the grammar:

```bash
python -m grammar_stimuli.run --mode training \
    --grammar B \
    --cage-ids "6224_SC,6225_EE" \
    --duration-seconds 14400 \
    --seed 12346 \
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

### Step 4. Tomorrow, swap the cages and repeat

Mouse Alpha (group 1) who was in EE today goes to SC tomorrow, and
hears Grammar B. And so on.

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

`grammar_condition` is ignored for test day — leave whatever's there.

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
| Want to re-derive what was played | Re-run with the same `--seed` and you get the same melodies |
