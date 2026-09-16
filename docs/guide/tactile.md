# The tactile paradigm

A two-level binary decision tree with servo-driven gratings and food reward, run by the original script. The application launches it and gives you a line to answer its console prompts.

```bash
amaze-tactile
```

This runs the tactile paradigm with servo-controlled gratings. It reads two
configuration CSVs that ship next to the script:

| File | Contents |
|---|---|
| `grating_maps.csv` | One row per reward location (A - D); each `motor <name>` column holds the `<name> <angle>` command sent to that grating servo (see `firmware/arduino/README.md`). |
| `reward_sequences.csv` | One row per (training stage, reward location): `portprob` (fraction of trials at that port), `rewprob` (probability a trial there is rewarded), `wrongallowed`. |

The values in the repo are a **template**: check them against your rig before running.
ROIs are drawn on first launch and stored as `rois1.csv` in the recordings folder.

---
