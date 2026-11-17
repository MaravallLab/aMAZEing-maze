# Glossary

The words the documentation, the application and the analysis scripts use in a particular sense.

## Arm

One of the eight corridors of the maze, numbered 1 to 8. The software watches a region of interest (ROI) over each arm, so `ROI1` in a data file is arm 1.

## Block

A stretch of a session with one setting. An auditory session runs nine blocks in a row:

| Block | Kind | Default length |
|---|---|---|
| 1 | Silent, the [habituation](#habituation) block | 2 min |
| 2, 4, 6, 8 | Sound blocks: every arm plays its stimulus on entry | 15 min |
| 3, 5, 7, 9 | Silent blocks between the sound blocks | 2 min |

The data files call a block a **trial** (`trial_ID` 1 to 9). The lengths are set with `block_minutes` in the [session configuration](../guide/session-config.md). When the analysis speaks of "sound block 1" or "blocks 2 to 4", it counts the four sound blocks only.

## EE and SC

**Enriched environment** and **standard cage**, the two home cages of Experiment 2. Each mouse heard one grammar in its enriched cage and the other in its standard cage, and on the test day the maze plays both. An **EE arm** plays the grammar the mouse heard in its enriched cage. The EE preference index compares time on EE arms with time on SC arms; above zero means more time with the enriched cage grammar.

## Grain

The level at which the data are summarised before a model is fitted to them. The value model of Experiment 2 can be fitted to each arm in each block (the finest grain), to each mouse's mean for each kind of arm, or to each mouse's session-level preference index (the coarsest). The [quick guide](../analysis/model-validation.md#11-emission-model-the-grain-comparison-the-resolution) explains why the grain changes the answer.

## Habituation

Two different things, by context.

- **The habituation block**: the first block of every session, silent, which lets the animal settle before any sound and shows where it goes by choice.
- **The habituation day** of Experiment 2: a whole session in silence before the test days, saved under the day label `habituation`.

## Preference index

How much longer a mouse stays in one kind of arm than in another, on a scale from -1 to +1:

```
PI = (mean visit with a - mean visit with b) / (mean visit with a + mean visit with b)
```

+1 means it stayed only with a, -1 only with b, 0 equally. In the thesis scripts, both arms are taken from the same sound blocks and visits are not capped. The [earlier batch pipeline](../analysis/auditory-legacy.md) capped every visit at 10 s. The summary figures of the application compute an EE preference index from total time rather than mean visit length.

## Session

One mouse in the maze once, from start to finish, saved in one folder named `time_<date>_<time>mouse<ID>`. The [data dictionary](data-dictionary.md) describes every file in it.

## Shuffled blocks

The sound blocks in which the stimulus played in each arm is drawn afresh, so that a stimulus is not tied to a place. By design, the first sound block of every session keeps the listed order (each stimulus in the arm its place in the list gives it), and the later sound blocks are reshuffled. In the first sound block a stimulus and a place therefore go together, and the analysis scripts can leave it out: `--shuffled-only` in Experiment 1 uses blocks 2 to 4 alone.

## Silent arm

The arm that plays nothing during the sound blocks. It is the control: a preference index against the silent arm says whether a sound draws the mouse more than no sound at all, measured in the same blocks.

## Silent baseline

A grammar session run with `--grammar-mode silent_baseline`: one continuous silent block. It is how the habituation day of Experiment 2 is recorded, and the summary tools leave it out, as it has no sounds to prefer.

## Tier

In the grammar experiment, how predictable a melody is under its grammar: **dominant** (the most likely transitions), **secondary** or **rare**. The test day has one arm per grammar per tier, six in all.
