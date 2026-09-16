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
