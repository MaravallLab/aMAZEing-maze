"""
Configuration for the auditory maze preference analysis pipeline.

Edit BASE_PATH and EXPERIMENT_DAYS to match your data layout.
Run this file directly to verify session discovery:
    python preference_analysis_config.py
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
