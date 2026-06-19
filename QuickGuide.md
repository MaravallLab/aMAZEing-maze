# aMAZEing-maze — QuickGuide

A start-to-finish guide for turning raw ventral maze videos into clean,
analysis-ready mouse pose data. Written to be read top-to-bottom the first
time, then used as a command reference afterwards.

The maze is a white cross/T-shaped decision tree filmed from below; a single
mouse explores it. The pipeline crops/aligns every video to a common maze
frame, runs a trained neural-network pose estimator, and exports per-frame
keypoints — with a spatial filter that removes confident-but-wrong detections
outside the maze.

---

## The pipeline at a glance

```
 raw videos                                                        analysis-ready
 (per mouse/        Stage 1            Stage 2          Stage 3        Stage 4
  session)        crop & align      pose inference    export        in-maze filter
     │                 │                  │              │               │
     ▼                 ▼                  ▼              ▼               ▼
  *.mp4   ──►  *_cropped.mp4   ──►  *.predictions.slp ─► *.keypoints.csv ─► *.keypoints.filtered.csv
                 (+calibration       (sleap-nn,          *.labeled.mp4      *.predictions.filtered.slp
                  per rig)            GPU)               (overlay video)    (+ verification overlay)
```

| Stage | Script | What it does | Engine |
|------|--------|--------------|--------|
| 1 | `analysis/crop_and_align_maze.py` | Warp every video so the maze sits in a fixed place | OpenCV |
| 2 | `analysis/sleap_batch_processing.py` | Predict 10 mouse keypoints per frame | **sleap-nn** (PyTorch, GPU) |
| 3 | `analysis/slp_export.py` | `.slp` → per-frame CSV (+ optional overlay video) | sleap-io / OpenCV |
| 4 | `analysis/filter_in_maze.py` | Drop detections that fall outside the maze | sleap-io / OpenCV |

The 10 keypoints (skeleton nodes) are:
`nose, headbase, midbody, tailbase, midtail, endtail, LF_paw, RF_paw, LH_paw, RH_paw`.

> **Just want to run it?** After one-time setup (Stage 0) and cropping (Stage 1),
> the whole thing is a single command — jump to
> [Run the whole pipeline](#run-the-whole-pipeline-one-command).

---

## ⚠️ Read this first: which Python you run matters

The single most common failure on this project is running a script with the
**wrong Python interpreter**. The models are **sleap-nn** (the PyTorch rewrite
of SLEAP), *not* classic TensorFlow SLEAP — so a plain `pip install sleap`
and a bare `python script.py` will fail even though "sleap" looks installed.

sleap-nn is installed as an isolated **uv tool**. Always call the scripts in
Stages 2–4 with that interpreter. Define it once per terminal:

```powershell
# PowerShell — the sleap-nn interpreter (has sleap_nn, torch, sleap_io, cv2, pandas)
$py = "$env:APPDATA\uv\tools\sleap-nn\Scripts\python.exe"
```

A quick check that it's the right one:

```powershell
& $py -c "import sleap_nn, torch; print(sleap_nn.__version__, '| CUDA', torch.cuda.is_available())"
# -> 0.2.0 | CUDA True
```

> **Why:** a project virtualenv (`.venv`) is often auto-activated in the
> terminal and shadows everything else, so bare `python` resolves to it. Calling
> the sleap-nn interpreter by full path sidesteps that entirely.

---

## Stage 0 — One-time setup

**sleap-nn (Stages 2–4).** Installed with [uv](https://docs.astral.sh/uv/),
which auto-resolves the correct CUDA build of PyTorch:

```powershell
# install uv if you don't have it:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# install sleap-nn with GPU support:
uv tool install --python 3.13 "sleap-nn[torch]" --torch-backend auto

# verify it sees your GPU:
sleap-nn system
```

You need an NVIDIA GPU for practical inference speeds (CPU works but is far
slower). `sleap-nn system` should report your GPU and `PyTorch can use GPU`.

**OpenCV stack (Stage 1).** `crop_and_align_maze.py` only needs
`opencv-python numpy pandas` and can run in any Python — including the sleap-nn
interpreter above (it already has them), so you can use `$py` throughout.

---

## Stage 1 — Crop & align the maze

Full details live in [analysis/README.md](analysis/README.md). The essentials:

1. **Calibrate once per rig** — click the 24 maze corners on one clear frame:
   ```powershell
   & $py analysis\crop_and_align_maze.py --calibrate --input_dir <videos> --output_dir <out>
   ```
   This writes `calibration.json` (and `calibration_rig2.json`, … for extra
   rigs) into `<out>`.

2. **Run the batch** — locates the maze in every video and warps it to a
   canonical frame:
   ```powershell
   & $py analysis\crop_and_align_maze.py --input_dir <videos> --output_dir <out> --calibration_dir <out>
   ```

Output: `<out>/<mouse>/<session>/<name>_cropped.mp4` for every video, plus
`<out>/alignment_summary.csv`.

> **Key idea that the rest of the pipeline relies on:** every video is warped so
> the maze lands at the *same* canonical position. Each rig/calibration has its
> own canonical frame **size**, and those sizes are unique — so later stages can
> identify a video's rig just from its cropped dimensions.

---

## Run the whole pipeline (one command)

Once Stage 0 (setup) and Stage 1 (crop & align) are done, the repo-root runner
does **inference + CSV export, then in-maze filtering**, end to end. There are
two equivalent entry points — use whichever you prefer:

```powershell
python run_pipeline.py        # Python (recommended; runs with any Python)
.\run_pipeline.ps1            # PowerShell equivalent
```

Both call the sleap-nn environment for you. Defaults are baked in for this
dataset; override them as needed:

```powershell
python run_pipeline.py --input_dir D:\simplermaze_output `
    --models_dir "C:\Users\shahd\OneDrive\Desktop\CROPPED_VIDEOS_FOR_SLEAP\models" `
    --calibration_dir D:\simplermaze_output
```

- **This is the multi-day step** — inference dominates (~4–5 days for a few
  hundred ~100k-frame videos). Leave it running; it prints per-video progress.
- **Resumable.** Stop anytime (Ctrl-C) and re-run the same command: finished
  videos are skipped, then the filter runs.
- **Run it detached** so it survives closing the terminal:
  ```powershell
  Start-Process python -ArgumentList (Resolve-Path .\run_pipeline.py) -WindowStyle Hidden
  ```
- Flags: `--skip_inference` (filter only), `--skip_filter` (infer only).
  (PowerShell equivalents: `-SkipInference`, `-SkipFilter`.)
- Keep the machine awake for the duration: `powercfg /change standby-timeout-ac 0`.
- Monitor progress: `(Get-ChildItem <out> -Recurse -Filter *.predictions.slp).Count`

The stages below explain what each part does and how to run them individually.

---

## Stage 2 — Pose inference (sleap-nn)

Predicts the 10 keypoints for every frame of every cropped video, saving a
`.slp` next to each video.

```powershell
$py     = "$env:APPDATA\uv\tools\sleap-nn\Scripts\python.exe"
$script = "<repo>\analysis\sleap_batch_processing.py"

# preview what would run (no GPU work):
& $py $script --input_dir <out> --models_dir <models> --dry_run

# the real run:
& $py $script --input_dir <out> --models_dir <models> --device cuda
```

- `<models>` is the folder holding the trained model subfolders (a
  `*centroid*` and a `*centered_instance*` pair). The script auto-picks the most
  recent pair; the maze is single-animal so `--max_instances` defaults to 1.
- Output per video: `<name>_cropped.predictions.slp` **and**
  `<name>_cropped.keypoints.csv` — the per-frame CSV is written in the same pass
  by default (disable with `--no-write_csv`). A `sleap_inference_summary.csv`
  lands in `<out>` when the run finishes.

**It's resumable.** `--skip_existing` is on by default, so a video that already
has a `.predictions.slp` is skipped. You can stop (`Ctrl-C` / `Stop-Process`)
and re-run the same command to continue.

**Expect it to be slow.** Throughput is bound by video decoding, not the GPU
(~85–100 frames/s regardless of batch size), and the videos are large
(~100k frames each). A full multi-hundred-video dataset can take **days**.
Run it detached and monitor with:

```powershell
# count finished predictions:
(Get-ChildItem <out> -Recurse -Filter "*.predictions.slp").Count
```

For multi-day runs, disable sleep: `powercfg /change standby-timeout-ac 0`.

---

## Stage 3 — Export CSV (+ optional overlay video)

Stage 2 already writes a `keypoints.csv` per video, so you usually **don't need
this step for CSVs**. Use `slp_export.py` when you want to:
- render an **annotated overlay video**, or
- **regenerate** CSVs from existing `.slp` files (e.g. you ran with
  `--no-write_csv`, or you changed the CSV format).

```powershell
$exp = "<repo>\analysis\slp_export.py"

# regenerate all CSVs from existing .slp files:
& $py $exp --input_dir <out>

# one annotated preview video (keypoints drawn on the frames):
& $py $exp --slp "<out>\...\<name>.predictions.slp" --render --render_frames 600
```

**CSV format** (`<name>.keypoints.csv`), one row per video frame, aligned to the
video timeline:

```
frame_idx, instance_score, nose.x, nose.y, nose.score, headbase.x, … , RH_paw.score
```

Undetected points are blank (score `0.0`); frames with no mouse are fully blank,
so the row index always equals the real frame number.

> **Rendering is opt-in and large.** A full ~100k-frame overlay is ~700 MB and
> slow, so use `--render_frames N` for a short preview, or only render specific
> sessions you want to eyeball. CSVs are tiny — generate those for everything.
> Annotated videos use the `mp4v` codec; play them in **VLC**.

---

## Stage 4 — Keep only in-maze detections

**The problem.** Like DeepLabCut's likelihood, sleap-nn's `score` is a
*confidence*, not a *correctness* check. On frames where the mouse is absent,
the model can place a confident, internally-consistent pose on background
clutter (cables, hardware, reflections) and score it **0.9+**. No score
threshold removes these.

**The fix is spatial.** The mouse can only ever be on the maze, so we drop any
detection whose body falls outside the maze outline. The outline is **not**
hand-drawn: the 24 calibration landmarks already define the maze boundary in the
exact coordinate system of the keypoints, and a cropped video's size identifies
its rig → its polygon (this works even for videos missing from
`alignment_summary.csv`).

```powershell
$flt = "<repo>\analysis\filter_in_maze.py"

& $py $flt --input_dir <out> --calibration_dir <out> --save_overlays
```

Per video it writes (non-destructively — raw `.slp` is untouched):

- `<name>.predictions.filtered.slp` — filtered predictions (re-viewable in a GUI)
- `<name>.keypoints.filtered.csv` — same CSV format as Stage 3, off-maze
  detections removed
- `<name>.maze_roi.png` (with `--save_overlays`) — the maze polygon drawn on the
  median frame, so you can confirm the boundary is right

It prints, per video, how many detections were dropped:

```
rig=calibration size=362x322 | anchor=midbody margin=15px | detections=99986  dropped_off_maze=1519 (1.5%)
```

### How the in/out test works (and how to tune it)

A detection is judged by its **body centre** (`--anchor`, default `midbody`;
falls back to the head/mid/tail median). Using the body centre — not the average
of *all* nodes — stops an outstretched nose or tail from dragging the anchor
off-maze at the edges. It is dropped if **either**:

- **clearly outside** — body centre more than `--margin_px` (default **15**)
  beyond the boundary, at *any* confidence. This catches far hallucinations,
  including the occasional confident one.
- **outside + low-confidence** — body centre outside the maze at all *and*
  instance score below `--min_score_outside` (default **0.5**). This catches
  mouse-absent hallucinations hugging the boundary.

Why two rules? Confidence separates the populations: real in-maze poses score
~0.91, hallucinations ~0.41. But there's a ~13px band of *real* edge poses
(score ~0.62) just outside the boundary — the 15px margin keeps those, and the
0.5 score cut-off keeps them too, so real data survives while hallucinations go.

Tuning (**err toward keeping data**):
- Real poses being dropped? Raise `--margin_px` (e.g. 18) and/or lower
  `--min_score_outside` (e.g. 0.4).
- Want more aggressive noise removal? Lower `--margin_px` or raise
  `--min_score_outside` — but watch the "real in-maze" overlay, since the edge
  halo starts to go around 0.6.
- `--min_score_outside 0` disables the confidence rule (pure geometry).
- `--clip_points` (off by default) additionally NaNs *individual* stray points
  outside the maze on otherwise-kept instances.

**Absent-block sweep (on by default).** Mouse-absent periods come in contiguous
blocks. A *confident* detection sitting inside a sustained run of mostly-dropped
frames is itself a hallucination, so it's dropped too. Disable with
`--no_absent_block`; tune with `--absent_block_window` (frames) and
`--absent_block_min_presence`. On the validation clip the model hallucinated
until the mouse entered at ~15.6s — the filter removed 100% of that window and
kept the real mouse. The sweep is deliberately conservative (it won't touch dense
real activity); always eyeball a filtered overlay on a new dataset before
trusting it.

### Validate before trusting the numbers

Always sanity-check on a few videos first:

1. Open a `*.maze_roi.png` — the red polygon should trace the maze.
2. Compare drop rates: a session where the mouse is mostly on-task should drop a
   small percentage. A surprisingly high drop rate usually means either real
   off-maze time (mouse absent) **or** a margin that's too tight — eyeball the
   filtered overlay video to tell which.

---

## Output files, named consistently

For a source video `…/<name>_cropped.mp4`:

| File | Stage | Contents |
|------|-------|----------|
| `<name>_cropped.predictions.slp` | 2 | raw predictions (native SLEAP) |
| `<name>_cropped.keypoints.csv` | 3 | per-frame keypoints, all detections |
| `<name>_cropped.labeled.mp4` | 3 | overlay preview video (opt-in) |
| `<name>_cropped.predictions.filtered.slp` | 4 | in-maze-only predictions |
| `<name>_cropped.keypoints.filtered.csv` | 4 | per-frame keypoints, in-maze only |
| `<name>_cropped.maze_roi.png` | 4 | maze polygon verification image |
| `sleap_inference_summary.csv` | 2 | one row per video (status, runtime, n_frames) |

Raw outputs are never overwritten by later stages — filtering and export are
additive, so you can re-tune and re-run safely.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|--------|--------------|-----|
| `No module named 'sleap_nn'` / `'torch'` | Wrong interpreter (`.venv` shadowing) | Use the `$py` full path (top of this guide) |
| `'sleap' is not installed` from Stage 2 | Models are sleap-nn, not classic SLEAP | Same — run with the sleap-nn interpreter |
| Inference very slow / GPU underused | Decode-bound, not GPU-bound | Expected; bigger `--batch_size` won't help. Plan for days; it's resumable |
| Filter drops a huge fraction | Margin too tight, or real off-maze time | Inspect `*.maze_roi.png` + filtered video; raise `--margin_px` |
| `no rig polygon matches cropped size` | Calibration JSON missing, or non-default `--padding` in Stage 1 | Point `--calibration_dir` at the folder with `calibration*.json`; pass the same `--padding` you cropped with |
| Annotated video won't play | `mp4v` codec | Open in VLC |

---

## Conceptual cheat-sheet

- **`.slp`** — SLEAP's native file bundling predictions + video reference. Read
  it with `sleap-io`; everything else is derived from it.
- **`score`** — per-point confidence (like DLC likelihood). High score ≠
  correct. That's *why* Stage 4 exists.
- **Rig → polygon** — each calibration warps the maze to a unique-size canonical
  frame; cropped size identifies the rig; the 24 landmarks are the maze polygon
  in keypoint coordinates.
- **Coordinate frame** — keypoints, the maze polygon, and the cropped video all
  share the same pixel coordinates, which is what makes spatial filtering exact.
- **Resumability** — Stages 1, 2 skip already-done work; Stages 3, 4 are
  additive. Stop and resume freely.
```
