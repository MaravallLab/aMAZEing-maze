# Data dictionary

Every file an auditory session writes, and what each column means. For a shorter tour, see [what a session saves](../guide/outputs.md). The tactile task writes different files, listed in [the tactile paradigm](../guide/tactile.md#what-a-session-saves).

## Where a session goes

```
<recordings folder>/<experiment mode>/[<day label>/]time_<timestamp><mouse ID>/
```

For example `maze_recordings/grammar/day_1/time_2026-05-23_08_56_19mouse13533/`. The timestamp is `YYYY-MM-DD_HH_MM_SS`, the day label is the one given with `--day` (or in the application), and the recordings folder defaults to `Desktop\auditory_maze_experiments\maze_recordings`.

| File | Written | One row per |
|---|---|---|
| [`trials_<timestamp>.csv`](#the-trials-table) | At the start, then after every block | Arm in each block |
| [`<mouse>_<mode>_detailed_visits.csv`](#the-visit-log) | As the session runs | Visit to an arm or entrance zone |
| [`<mouse>_<mode>_maze_entries.csv`](#the-maze-entry-log) | As the session runs | Entry into or exit from the maze |
| [`<mouse>_<timestamp>_metadata.csv`](#the-animal-metadata) | At the start | (one row) |
| [`session_manifest.json`](#the-session-manifest) | At the start and at the end | |
| [`grammar_samples_<timestamp>.csv`](#the-grammar-log) | At the end, grammar mode only | Melody played |
| `trials_<timestamp>.npy` | At the start | The sounds of every arm and block, as arrays, for checking stimuli later |
| `<mouse>_<timestamp>.mp4` | As the session runs, unless recording is off | The video |
| `fig1_arm_totals.png` to `fig6_location_preference.png` | At the end | The [per-session figures](../guide/outputs.md) |

The arm boxes are not in the session folder: they are in `rois1.csv` in the recordings folder, shared by every session, and the manifest names the file used. `rois1.csv` has one column per box (`entrance1`, `entrance2`, then the arms) and four rows, `xstart`, `ystart`, `xlen` and `ylen`, in pixels.

## The trials table

`trials_<timestamp>.csv`: the plan of the session, filled in as it runs. One row per arm per [block](glossary.md#block).

| Column | Meaning |
|---|---|
| `trial_ID` | The block, 1 to 9. Odd blocks are silent, even blocks play sounds. |
| `ROIs` | The arm, 1 to 8 |
| `frequency` | What the arm plays: a frequency in Hz (0 for silence), or a label such as `grammar` or a file path |
| `time_spent` | Seconds in the arm over the whole block, summed over visits |
| `visitation_count` | Visits to the arm in the block |
| `wave_arrays` | A shortened text form of the sound; the sounds themselves are in the `.npy` file |
| `time_in_maze_ms`, `trial_start_time`, `end_trial_time` | Kept for compatibility with the first version; the current software leaves them empty. Time in the maze is in the [maze entry log](#the-maze-entry-log). |

Some modes add columns describing the stimulus:

| Mode | Extra columns |
|---|---|
| Grammar | `grammar` (A or B), `tier` (dominant, secondary, rare), `environment_association` (EE or SC) |
| Musical intervals | `interval`, `interval_ratio` |
| Temporal envelope modulation | `sound_type`, `temporal_modulation` |
| Consonant vs dissonant | `interval_type`, `interval_ratio`, `interval_name` |
| Sequences | `pattern` |
| Custom | `sound_type`, `stimulus_label` |

## The visit log

`<mouse>_<mode>_detailed_visits.csv`: every visit, in order. This is the most detailed record and the one the analysis scripts prefer.

| Column | Meaning |
|---|---|
| `trial_ID` | The block |
| `ROI_visited` | The arm (`1` to `8`) or entrance zone (`entrance1`, `entrance2`) |
| `stimulus` | A text description of what played |
| `sound_on_time` | Start of the visit, in Unix time (seconds) |
| `sound_off_time` | End of the visit, in Unix time (seconds) |
| `time_spent_seconds` | Length of the visit, in seconds |

A visit still going at the end of a block is closed at the block's end and counted in that block.

## The maze entry log

`<mouse>_<mode>_maze_entries.csv`: when the animal came into the maze from the home cage, and when it left. Passing `entrance1` then `entrance2` is an entry; the reverse is an exit.

| Column | Meaning |
|---|---|
| `trial_ID` | The block |
| `event` | `entered`, `exited`, or `session_end_still_inside` |
| `timestamp` | Unix time, in seconds |
| `time_in_maze_seconds` | On exits only: how long that stay in the maze lasted |

## The animal metadata

`<mouse>_<timestamp>_metadata.csv`, one row:

| Column | Meaning |
|---|---|
| `animal ID` | The mouse ID |
| `session_date` | The session timestamp |
| `ear_mark` | Ear mark, as entered |
| `birth_date` | Date of birth, as entered |
| `gender` | Sex, as entered |

## The session manifest

`session_manifest.json` makes the folder explain itself. It is written when the session starts (`status: running`) and again when it ends (`completed`).

| Key | Contents |
|---|---|
| `manifest_version` | The format of this file (1) |
| `amazeing_version` | The version of the software that recorded the session |
| `status` | `running` or `completed`. A session left at `running` stopped without finishing. |
| `written_at` | When the manifest was written |
| `config` | Every setting of the session, as used |
| `files` | The name of each file the session wrote |
| `columns` | The meaning and unit of every column of the three tables above |
| `mouse_id`, `timestamp`, `block_minutes` | The mouse, the timestamp, and the length of each block in minutes |
| `total_time_in_maze_seconds` | Total time in the maze (at the end) |

## The grammar log

`grammar_samples_<timestamp>.csv`, grammar mode only: every melody generated for every arm.

| Column | Meaning |
|---|---|
| `trial_ID` | The block |
| `ROI` | The arm |
| `grammar` | A or B |
| `tier` | dominant, secondary or rare |
| `environment_association` | EE or SC |
| `symbols` | The melody, one letter per tone |
| `mean_bits` | Mean surprise of the melody, in bits, under the distribution it was drawn from |

The player prepares several melodies per entry and stops when the mouse leaves, so this log lists more melodies than were heard in full. `mean_bits` does not separate the secondary and rare tiers; the [value model](../analysis/model-validation.md) recomputes surprise from `symbols`.

## Units and versions

The current software (version 2) records times in **seconds**. The first version recorded `time_spent` in the trials table in **milliseconds**, and did not close visits that ran over the end of a block, so some of its durations are inflated. The manifest came later still: sessions recorded before it, which include all the thesis data, have no `session_manifest.json`, so check which version recorded a session before reading its units. The analysis scripts for the thesis data handle both; see [the earlier batch pipeline](../analysis/auditory-legacy.md) for how.
