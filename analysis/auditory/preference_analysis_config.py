"""
Configuration for the auditory maze preference analysis pipeline.

Edit BASE_PATH and EXPERIMENT_DAYS to match your data layout.
Run this file directly to verify session discovery:
    python preference_analysis_config.py
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Sanity caps for visit duration (ms).
# Real visits in detailed_visits never exceed ~25 s, and trial duration
# is 15 min for sound trials and 2 min for silent trials.  Anything
# above these caps in trials.csv is a known data-recording bug
# (the aggregator sometimes records the trial duration instead of
# the actual visit duration).
SOUND_TRIAL_VISIT_CAP_MS = 60_000      # 60 s
SILENT_TRIAL_VISIT_CAP_MS = 130_000    # slightly above 2-min trial duration
SOUND_TRIAL_IDS = {2, 4, 6, 8}
SILENT_TRIAL_IDS = {3, 5, 7, 9}

# Per-visit clip applied to BOTH detailed_visits and trials.csv data.
# This catches residual cases where the "trial-boundary bug" in main.py
# (visits not closed at trial end) bleeds a long visit into DV as well.
# Override with VISIT_CLIP_MS env var.  Set to 0 to disable.
INDIVIDUAL_VISIT_CLIP_MS = int(os.environ.get("VISIT_CLIP_MS", "10000"))  # 10 s

# ── paths ─────────────────────────────────────────────────────────────

# Override with MAZE_DATA_DIR env var if running on a different machine.
# e.g. MAZE_DATA_DIR="D:\data\8_arms_w_voc" python 01_preference_analysis.py
BASE_PATH = os.environ.get("MAZE_DATA_DIR", os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data",
    "Auditory experiments", "8_arms_w_voc"
))

OUTPUT_DIR = os.path.join(BASE_PATH, "BATCH_ANALYSIS")

# ── experiment structure ──────────────────────────────────────────────

# Each day maps to a folder name and an experiment mode.
# Sound types within each day are extracted from the stimulus field.
EXPERIMENT_DAYS = {
    "w1_d1": {
        "folder": "w1_d1",
        "mode": "temporal_envelope_modulation",
        "label": "Day 1: Temporal Envelope",
        "short": "D1 (TEM)",
        # complexity ordering (low -> high)
        "complexity_order": ["smooth", "rough", "rough_complex", "control", "vocalisation"],
    },
    "w1_d2": {
        "folder": "w1_d2",
        "mode": "complex_intervals",
        "label": "Day 2: Intervals (consonant/dissonant)",
        "short": "D2 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w1_d3": {
        "folder": "w1_d3",
        "mode": "complex_intervals",
        "label": "Day 3: Intervals (consonant/dissonant)",
        "short": "D3 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w1_d4": {
        "folder": "w1_d4",
        "mode": "complex_intervals",
        "label": "Day 4: Intervals (no silent control)",
        "short": "D4 (Int)",
        "complexity_order": ["smooth", "rough", "consonant", "dissonant", "control", "vocalisation"],
    },
    "w2_sequences": {
        "folder": "w2_sequences",
        "mode": "sequences",
        "label": "Week 2: Sequences",
        "short": "W2 (Seq)",
        "complexity_order": ["AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA"],
    },
    "w2_vocalisations": {
        "folder": "w2_vocalisations",
        "mode": "vocalisation",
        "label": "Week 2: Vocalisations",
        "short": "W2 (Voc)",
        "complexity_order": [],  # no complexity ordering for vocalisations
    },
}

# Days to include in the standard PI analysis (need silent control trials)
# Day 4 excluded because it has no silent control arm (per report)
PI_DAYS = ["w1_d1", "w1_d2", "w1_d3", "w2_sequences", "w2_vocalisations"]

# Day ordering for plots
DAY_ORDER = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
DAY_SHORT = {d: EXPERIMENT_DAYS[d]["short"] for d in DAY_ORDER}

# ── session discovery ─────────────────────────────────────────────────


@dataclass
class SessionInfo:
    mouse_id: str
    day: str
    session_folder: str
    detailed_visits_csv: Optional[str] = None
    trials_csv: Optional[str] = None


def discover_sessions() -> List[SessionInfo]:
    """Auto-discover all sessions across all experiment days."""
    sessions = []

    for day_key, day_info in EXPERIMENT_DAYS.items():
        day_dir = os.path.join(BASE_PATH, day_info["folder"])
        if not os.path.isdir(day_dir):
            continue

        for entry in os.listdir(day_dir):
            sess_path = os.path.join(day_dir, entry)
            if not os.path.isdir(sess_path):
                continue

            # extract mouse ID from folder name: time_YYYY-MM-DD_HH_MM_SSmouseXXXXX
            mouse_id = _extract_mouse_id(entry)
            if not mouse_id:
                continue

            # find CSVs
            dv_csvs = glob.glob(os.path.join(sess_path, "*detailed_visits*.csv"))
            trials_csvs = glob.glob(os.path.join(sess_path, "trials_*.csv"))

            # also check fixed/ subdirectory
            fixed_dir = os.path.join(sess_path, "fixed")
            if os.path.isdir(fixed_dir):
                dv_csvs += glob.glob(os.path.join(fixed_dir, "*detailed_visits*.csv"))
                trials_csvs += glob.glob(os.path.join(fixed_dir, "trials_*.csv"))

            # prefer non-backup, most recent
            dv_csvs = [f for f in dv_csvs if not f.endswith(".bak")]
            trials_csvs = [f for f in trials_csvs if not f.endswith(".bak")]

            sess = SessionInfo(
                mouse_id=mouse_id,
                day=day_key,
                session_folder=sess_path,
                detailed_visits_csv=dv_csvs[0] if dv_csvs else None,
                trials_csv=trials_csvs[0] if trials_csvs else None,
            )
            sessions.append(sess)

    return sessions


def _extract_mouse_id(folder_name: str) -> Optional[str]:
    """Extract mouse ID from session folder name."""
    import re
    # pattern: ...mouseXXXXX (5-digit ID at end)
    m = re.search(r"mouse(\d{4,6})", folder_name, re.IGNORECASE)
    if m:
        return f"mouse{m.group(1)}"
    return None


def get_mice_with_min_sessions(sessions: List[SessionInfo],
                                min_sessions: int = 2) -> set:
    """Return mouse IDs that appear in at least min_sessions days."""
    from collections import Counter
    day_counts = Counter()
    seen = set()
    for s in sessions:
        key = (s.mouse_id, s.day)
        if key not in seen:
            seen.add(key)
            day_counts[s.mouse_id] += 1
    return {m for m, c in day_counts.items() if c >= min_sessions}


# ── safe IO helpers ───────────────────────────────────────────────────

def _is_file_locally_available(path):
    """Skip Box Drive / OneDrive cloud-only files (would block on read)."""
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        if attrs & 0x00400000 or attrs & 0x00001000:
            return False
        return True
    except Exception:
        try:
            with open(path, "rb") as f:
                f.read(1)
            return True
        except Exception:
            return False


def safe_read_csv(path):
    if path is None or not _is_file_locally_available(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── visit-data loader (DV-first, capped trials.csv fallback) ──────────

def load_session_visits(sess: "SessionInfo") -> Tuple[Optional[pd.DataFrame], str]:
    """Load a session's per-(trial, ROI) visit aggregates with corruption fixes.

    Strategy:
      1. Load trials.csv to get the row layout (one row per trial-ROI)
         plus stimulus metadata columns (sound_type, frequency, etc.).
      2. If detailed_visits.csv exists and has data, REPLACE the
         time_spent and visitation_count columns with values aggregated
         from detailed_visits (ground truth).  Rows not present in DV
         are set to zero (mouse never visited).
      3. Otherwise, apply a sanity cap: zero out any row whose average
         visit duration exceeds the trial-type cap.

    Returns
    -------
    (df, source) : (DataFrame or None, str)
        source is one of:
          'detailed_visits'  -- DV used as ground truth
          'trials_capped'    -- trials.csv used with sanity cap applied
          'trials_raw'       -- trials.csv used unmodified (no bad rows)
    """
    if not sess.trials_csv:
        return None, "no_trials_csv"

    df = safe_read_csv(sess.trials_csv)
    if df is None or "trial_ID" not in df.columns or "ROIs" not in df.columns:
        return None, "bad_trials_csv"

    for col in ["time_spent", "visitation_count", "time_in_maze_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 1. Try DV override ----
    if sess.detailed_visits_csv:
        dv = safe_read_csv(sess.detailed_visits_csv)
        if (dv is not None and len(dv) > 0
                and "ROI_visited" in dv.columns
                and "time_spent_seconds" in dv.columns
                and "trial_ID" in dv.columns):
            dv = dv.copy()
            dv["_dur_ms"] = pd.to_numeric(
                dv["time_spent_seconds"], errors="coerce") * 1000.0
            dv = dv.dropna(subset=["_dur_ms"])
            # Clip individual visits that exceed the per-visit cap
            # (boundary-spanning bug residuals).
            if INDIVIDUAL_VISIT_CLIP_MS > 0:
                dv["_dur_ms"] = dv["_dur_ms"].clip(upper=INDIVIDUAL_VISIT_CLIP_MS)
            agg = (dv.groupby(["trial_ID", "ROI_visited"])
                     .agg(_dv_time=("_dur_ms", "sum"),
                          _dv_count=("_dur_ms", "size"))
                     .reset_index()
                     .rename(columns={"ROI_visited": "ROIs"}))
            df = df.merge(agg, on=["trial_ID", "ROIs"], how="left")
            df["time_spent"] = df["_dv_time"].fillna(0.0)
            df["visitation_count"] = df["_dv_count"].fillna(0).astype(int)
            df = df.drop(columns=["_dv_time", "_dv_count"])
            df["_visit_source"] = "detailed_visits"
            return df, "detailed_visits"

    # ---- 2. Sanity cap on trials.csv ----
    if "time_spent" in df.columns and "visitation_count" in df.columns:
        vc = df["visitation_count"].where(df["visitation_count"] > 0, np.nan)
        avg = df["time_spent"] / vc
        is_sound = df["trial_ID"].isin(SOUND_TRIAL_IDS)
        is_silent = df["trial_ID"].isin(SILENT_TRIAL_IDS)
        # Strong cap: zero-out rows physically impossible (boundary bug)
        bad = (
            (is_sound & (avg > SOUND_TRIAL_VISIT_CAP_MS))
            | (is_silent & (avg > SILENT_TRIAL_VISIT_CAP_MS))
        )
        n_bad = int(bad.sum())
        if n_bad > 0:
            df.loc[bad, "time_spent"] = 0.0
            df.loc[bad, "visitation_count"] = 0
            df["_visit_source"] = "trials_capped"
            df.attrs["n_capped_rows"] = n_bad
            # Also apply per-visit cap to remaining rows
            if INDIVIDUAL_VISIT_CLIP_MS > 0:
                vc2 = df["visitation_count"].where(df["visitation_count"] > 0, np.nan)
                avg2 = df["time_spent"] / vc2
                over = avg2 > INDIVIDUAL_VISIT_CLIP_MS
                if over.any():
                    df.loc[over, "time_spent"] = (
                        df.loc[over, "visitation_count"] * INDIVIDUAL_VISIT_CLIP_MS
                    )
            return df, "trials_capped"
        # Per-visit clip even if no row-level corruption
        if INDIVIDUAL_VISIT_CLIP_MS > 0:
            over = avg > INDIVIDUAL_VISIT_CLIP_MS
            if over.any():
                df.loc[over, "time_spent"] = (
                    df.loc[over, "visitation_count"] * INDIVIDUAL_VISIT_CLIP_MS
                )
                df["_visit_source"] = "trials_clipped"
                return df, "trials_clipped"

    df["_visit_source"] = "trials_raw"
    return df, "trials_raw"


# ── run as standalone to verify ───────────────────────────────────────

if __name__ == "__main__":
    sessions = discover_sessions()
    print(f"Discovered {len(sessions)} sessions")

    # count by day
    from collections import defaultdict
    by_day = defaultdict(list)
    by_mouse = defaultdict(list)
    for s in sessions:
        by_day[s.day].append(s.mouse_id)
        by_mouse[s.mouse_id].append(s.day)

    print("\nSessions per day:")
    for day in DAY_ORDER:
        mice = by_day.get(day, [])
        has_dv = sum(1 for s in sessions
                     if s.day == day and s.detailed_visits_csv)
        print(f"  {day:20s}: {len(mice):3d} mice, "
              f"{has_dv} with detailed_visits")

    print(f"\nUnique mice: {len(by_mouse)}")
    multi = get_mice_with_min_sessions(sessions, 3)
    print(f"Mice with >=3 sessions: {len(multi)}")

    # show a few
    for mouse_id in sorted(list(by_mouse.keys()))[:5]:
        days = by_mouse[mouse_id]
        print(f"  {mouse_id}: {', '.join(sorted(days))}")
