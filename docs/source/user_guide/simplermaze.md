# SimplerMaze (tactile paradigm)

This page is the entry point for the SimplerMaze paradigm — the tactile,
reward-based decision task, distinct from the auditory experiment. It covers how
to run it, the firmware it relies on, and where its analysis lives. Read it if
you are working with the moveable-wall task rather than the auditory maze.

The SimplerMaze is a 2-level binary decision tree with servo-controlled moveable
walls (gratings) for reward-based navigation studies. It lives in
`src/simplermaze/`.

## Running

```bash
cd src/simplermaze
python simplerCode.py
```

- `simplerCode.py` — main experiment script
- `supFun.py` — support functions
- `post_process_session.py` — post-session processing

## Firmware

The moveable walls and reward delivery are driven by microcontroller firmware in
`firmware/`:

- `firmware/arduino/servo_control/` — servo control sketch
- `firmware/micropython/` — Adafruit PCA9685 PWM driver (`pca9685.py`, `servo.py`)
- `firmware/ttl_bnc/` — Arduino TTL synchronisation sketch for photometry

## Analysis

The behavioural analysis pipeline for the SimplerMaze first paper lives in
`analysis/simplermaze/`. See {doc}`analysis_pipeline`.
