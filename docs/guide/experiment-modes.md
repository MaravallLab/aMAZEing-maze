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
