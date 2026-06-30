# The aMAZEing maze

```{image} _static/images/happy_mouse.png
:alt: aMAZEing maze
:width: 220px
:align: right
```

**A modular, automated, sensory-engaging open-source platform for studying how
sensory cues shape active exploration in rodents.**

The aMAZEing maze combines real-time video tracking, auditory stimulus delivery,
and optional hardware control (servos, TTL synchronisation for photometry) into
a single Python-driven platform. It supports two experimental paradigms:

- an **auditory maze** — a multi-arm maze where real-time ROI tracking triggers
  sound playback (pure tones, musical intervals, temporal-envelope modulation,
  tone sequences, vocalisations, and a Markov-grammar learning protocol); and
- the **SimplerMaze** — a 2-level binary decision tree with servo-controlled
  moveable walls for reward-based navigation studies.

```{note}
This documentation is being built out. Narrative pages (installation, hardware
setup, usage) are written in Markdown; the {doc}`API reference <api/index>` is
generated automatically from the source docstrings.
```

```{toctree}
:maxdepth: 2
:caption: Getting started

getting_started/installation
getting_started/hardware_setup
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/overview
user_guide/auditory_experiment
user_guide/simplermaze
user_guide/analysis_pipeline
user_guide/troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
references
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
