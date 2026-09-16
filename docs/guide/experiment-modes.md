# Experiment modes

Each mode presents a different kind of stimulus set. Selecting one in the application shows a panel with that mode's parameters and hides the others. Every parameter below is also a field in the [session configuration](session-config.md), and the defaults reproduce the protocols used in the original experiments.

Set `experiment_mode` in `config.py` to one of:

| Mode | Description |
|---|---|
| `grammar` | Grammar learning test - two Markov grammars × three predictability tiers (dominant/secondary/rare) + vocalisation + silent control, shuffled across 4 active 15-min blocks |
| `simple_smooth` | One pure tone per ROI arm |
| `simple_intervals` | Two-tone chords (musical intervals) per ROI |
| `temporal_envelope_modulation` | Smooth, constant-AM, and complex-AM sounds |
| `complex_intervals` | Multi-day interval protocol with consonant/dissonant contrasts |
| `sequences` | Tone-pattern sequences (ABAB, AoAo, etc.) |
| `vocalisation` | Each ROI plays a different vocalisation recording |
| `custom` | Your own stimulus per ROI (tone, AM tone, .wav file or silent), declared in the session config file - see below |


## The stimuli must fill the arms

Some modes build a fixed set of stimuli, and that set has to match the number of arms exactly. The **Stimuli** line under Experiment states the arithmetic as you edit, for example:

```
8 stimuli for 8 arms: 2 control, 2 smooth, 2 constant AM, 2 complex AM
```

If the two disagree the line turns orange and says the session will not start. Starting one anyway is refused with a message naming both numbers, rather than building a trial table that does not match the maze.

| Mode | Arms it fills |
|---|---|
| Musical intervals | one per interval, plus a unison arm and a silent arm |
| Temporal envelope modulation | controls, plus one per smooth, constant-AM and complex-AM carrier |
| Consonant vs dissonant | controls, the optional unison arms, plus one per consonant and dissonant interval |
| Grammar, test day | always 8: six grammar arms, a vocalisation arm and a silent arm |
| Pure tones | any number; a short frequency list is recycled |
| Sequences, vocalisations, custom | any number; unlisted arms are silent |


## Recordings the mode needs

Several modes play a `.wav` file: the vocalisation control arm of the modulation
and interval designs, the vocalisation pattern in sequences, the vocalisation
arm of the grammar test, every arm in vocalisation mode, and any `wav` row in
custom mode.

A missing file does not stop anything on its own. The loader returns silence and
prints a note, so the session would run with a **silent arm where a recording was
intended**, and nothing in the data would say so.

The form therefore checks. A **Sound files** line under Experiment says either
that everything was found, or which file is missing, and Start session refuses
until it is resolved. The stimulus preview says the same thing for the affected
arm, rather than describing it as silent:

> the vocalisation control arm needs a control vocalisation .wav, but none is set

Set the file in the **Vocalisation files** section: the folder supplies one arm
per file in vocalisation mode, and the control file is the single recording used
on the vocalisation arm of the mixed designs. If you do not want a vocalisation
arm at all, untick it in the mode's control arms instead of leaving the file
empty.
