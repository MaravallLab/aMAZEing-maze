"""Load logged grammar sessions into a tidy arm-block dataframe.

Grain: one row per (mouse, day, block, arm/ROI). Day 1 is primary; day 2 is a
separate secondary question and the two are never accidentally pooled
(`load_day`).

REAL SCHEMA (the originally-assumed columns were wrong):

  * animal_id and day live in FOLDER NAMES, not columns. Parsed string-based.
  * EE/SC is logged directly as `environment_association` on each arm row; we do
    NOT re-derive it from counterbalancing group.
  * the counterbalancing GROUP is derived from the EE<->grammar pairing on the
    arms (EE=A => group 1, EE=B => group 2), and validated by within-mouse
    cross-session (day_1 vs day_2) consistency. There is no group field in the
    per-mouse metadata.csv.
  * tier is dominant/secondary/rare; `symbols` is a melody letter string.
  * dwell/PI uses the trials CSV's aggregated `time_spent` (matches the existing
    summary_analysis convention exactly); detailed_visits is the finer per-visit
    grain used by sanity checks.

NAMED, AUDITABLE EXCLUSIONS (logged with a reason, never silent drops):

  * EXCLUDED_ANIMALS: mouse 13533 (light event triggered all ROIs on both test
    days; tracking artifact). Removing the id and re-running is the whole change
    needed once the artifact frames are cleaned.
  * the `wentagainbymistake_swap_SCandEE` session (EE/SC swapped).
  * duplicate filings of the same session (same mouse+day+timestamp under two
    parent folders, e.g. day_1/ and a cage folder) are de-duplicated keeping the
    first.

PI is never computed from a regenerated sequence; if `symbols` are absent for a
grammar arm the row is kept for PI (time-based) but flagged with empty melodies.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Named exclusions (auditable provenance)
# ---------------------------------------------------------------------------
EXCLUDED_ANIMALS: Dict[str, str] = {
    "13533": (
        "Light event spuriously triggered all ROIs on both test days; position "
        "tracking contaminated. Perceived environment unchanged (cage+maze "
        "covered) — tracking artifact, not a perceptual confound. PI untrusted "
        "until artifact frames are removed by hand."
    ),
}

# Folder-name annotations that mark a session as excluded.
_EXCLUDED_SESSION_PATTERNS: List[Tuple[str, str]] = [
    (r"mistake|swap", "EE/SC swapped for this run (logged as a mistake)."),
]

_DAY_PRIMARY = "day_1"
_DAY_SECONDARY = "day_2"

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})")
_MOUSE_RE = re.compile(r"mouse(\d+)")

_TRIALS_USECOLS = ["trial_ID", "ROIs", "frequency", "grammar", "tier",
                   "environment_association", "time_spent", "visitation_count"]


# ---------------------------------------------------------------------------
# Folder / path parsing
# ---------------------------------------------------------------------------

def classify_day(path: str) -> str:
    low = path.lower()
    if "habituat" in low:
        return "habituation"
    if re.search(r"day[_ ]?2", low):
        return _DAY_SECONDARY
    if re.search(r"day[_ ]?1", low):
        return _DAY_PRIMARY
    return "unknown"


def parse_folder(name: str) -> Tuple[Optional[str], str]:
    m = _MOUSE_RE.search(name)
    t = _TS_RE.search(name)
    return (m.group(1) if m else None, t.group(1) if t else "")


@dataclass
class Session:
    mouse: str
    day: str
    timestamp: str
    folder: str
    trials_path: str
    samples_path: Optional[str]
    visits_path: Optional[str]
    excluded: Optional[str] = None       # reason string if excluded
    group: Optional[int] = None
    internal_ok: bool = True
    ee_grammars: Tuple[str, ...] = ()
    sc_grammars: Tuple[str, ...] = ()


def discover_sessions(root: str) -> List[Session]:
    """Walk `root` and build a Session per trials_*.csv found."""
    sessions: List[Session] = []
    for tp in glob.glob(os.path.join(root, "**", "trials_*.csv"), recursive=True):
        folder = os.path.dirname(tp)
        name = os.path.basename(folder)
        rel = os.path.relpath(tp, root)
        mouse, ts = parse_folder(name)
        if mouse is None:
            continue
        day = classify_day(rel)
        samples = _first_glob(os.path.join(folder, "grammar_samples_*.csv"))
        visits = _first_glob(os.path.join(folder, "*_grammar_detailed_visits.csv"))
        sessions.append(Session(mouse, day, ts, name, tp, samples, visits))
    return sessions


def _first_glob(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Group derivation + integrity
# ---------------------------------------------------------------------------

def derive_group(trials_df: pd.DataFrame) -> Tuple[Optional[int], Tuple[str, ...], Tuple[str, ...], bool]:
    """Infer counterbalancing group from the EE<->grammar pairing on the arms."""
    g = trials_df[trials_df["grammar"].isin(["A", "B"])]
    ee = tuple(sorted(g.loc[g["environment_association"] == "EE", "grammar"].unique()))
    sc = tuple(sorted(g.loc[g["environment_association"] == "SC", "grammar"].unique()))
    internal_ok = (len(ee) == 1 and len(sc) == 1 and ee != sc)
    if ee == ("A",) and sc == ("B",):
        return 1, ee, sc, internal_ok
    if ee == ("B",) and sc == ("A",):
        return 2, ee, sc, internal_ok
    return None, ee, sc, internal_ok


def integrity_report(sessions: List[Session]) -> Dict[str, object]:
    """Within-mouse cross-session group consistency (catches unannotated swaps).

    Group is derived from `environment_association`, so it cannot be cross-checked
    against itself within one session; the real check is that a mouse's derived
    group is identical across its (non-excluded) sessions on different days.
    """
    by_mouse: Dict[str, List[Session]] = {}
    for s in sessions:
        if s.excluded or s.group is None:
            continue
        by_mouse.setdefault(s.mouse, []).append(s)

    inconsistent: List[Dict[str, object]] = []
    internal_bad: List[str] = []
    for mouse, ss in by_mouse.items():
        groups = {s.group for s in ss}
        if len(groups) > 1:
            inconsistent.append({
                "mouse": mouse,
                "groups": {s.day: s.group for s in ss},
            })
        for s in ss:
            if not s.internal_ok:
                internal_bad.append(f"{mouse}/{s.day}/{s.timestamp}")
    return {
        "n_mice_checked": len(by_mouse),
        "cross_session_group_inconsistent": inconsistent,
        "internal_mislabel": internal_bad,
    }


# ---------------------------------------------------------------------------
# Arm-block dataframe
# ---------------------------------------------------------------------------

def _arm_type(frequency: object) -> str:
    f = str(frequency).strip().lower()
    if f == "grammar":
        return "grammar"
    if f == "vocalisation":
        return "vocalisation"
    return "silent"


def _load_samples(samples_path: Optional[str]):
    """Map (trial_ID, ROI) -> (melody symbol strings, (grammar, tier, env) label).

    grammar_samples logs grammar/tier/environment_association per melody at play
    time — an independent recording of the same pairing the trials CSV logs at
    block construction. We return both so the caller can attach melodies AND
    cross-check the two sources for mis-joins / logging glitches.
    """
    if not samples_path or not os.path.isfile(samples_path):
        return {}, {}
    df = pd.read_csv(
        samples_path,
        usecols=["trial_ID", "ROI", "grammar", "tier", "environment_association", "symbols"],
    )
    mel: Dict[Tuple[int, str], List[str]] = {}
    lab: Dict[Tuple[int, str], Tuple[str, str, str]] = {}
    for row in df.itertuples(index=False):
        key = (int(row.trial_ID), str(row.ROI))
        mel.setdefault(key, []).append(str(row.symbols))
        if key not in lab:   # constant within an arm-block
            lab[key] = (row.grammar, row.tier, row.environment_association)
    return mel, lab


@dataclass
class LoadResult:
    arm_blocks: pd.DataFrame
    sessions: List[Session]
    report: Dict[str, object] = field(default_factory=dict)


def load_arm_blocks(root: str) -> LoadResult:
    """Discover, exclude, derive group, and build the tidy arm-block dataframe."""
    sessions = discover_sessions(root)

    # Apply named exclusions + derive group per session.
    for s in sessions:
        if s.mouse in EXCLUDED_ANIMALS:
            s.excluded = f"EXCLUDED_ANIMALS[{s.mouse}]: {EXCLUDED_ANIMALS[s.mouse]}"
        for pat, reason in _EXCLUDED_SESSION_PATTERNS:
            if re.search(pat, s.folder, flags=re.IGNORECASE):
                s.excluded = reason
        try:
            tdf = pd.read_csv(s.trials_path, usecols=_TRIALS_USECOLS)
        except Exception as e:  # pragma: no cover - corrupt file
            s.excluded = s.excluded or f"unreadable trials csv: {e}"
            continue
        s._trials = tdf  # type: ignore[attr-defined]  (cache for reuse)
        grp, ee, sc, ok = derive_group(tdf)
        s.group, s.ee_grammars, s.sc_grammars, s.internal_ok = grp, ee, sc, ok

    # De-duplicate identical filings: same (mouse, day, timestamp).
    seen: set = set()
    dup_dropped: List[str] = []
    for s in sessions:
        key = (s.mouse, s.day, s.timestamp)
        if key in seen and not s.excluded:
            s.excluded = "duplicate filing of an already-loaded session (kept first)"
            dup_dropped.append(f"{s.mouse}/{s.day}/{s.timestamp}")
        elif not s.excluded:
            seen.add(key)

    integ = integrity_report(sessions)

    rows: List[Dict[str, object]] = []
    missing_melodies = 0
    label_mismatches = 0
    for s in sessions:
        if s.excluded:
            continue
        tdf: pd.DataFrame = getattr(s, "_trials")
        tdf = tdf.copy()
        tdf["time_spent"] = pd.to_numeric(tdf["time_spent"], errors="coerce").fillna(0.0)
        tdf["visitation_count"] = pd.to_numeric(tdf["visitation_count"], errors="coerce").fillna(0)
        active = tdf[tdf["trial_ID"] % 2 == 0]
        melodies_map, labels_map = _load_samples(s.samples_path)

        for _, r in active.iterrows():
            tid = int(r["trial_ID"])
            roi = str(r["ROIs"])
            atype = _arm_type(r["frequency"])
            env = r["environment_association"] if atype == "grammar" else None
            env = env if env in ("EE", "SC") else None
            mel = melodies_map.get((tid, roi), []) if atype == "grammar" else []
            if atype == "grammar" and not mel:
                missing_melodies += 1
            if atype == "grammar":
                lab = labels_map.get((tid, roi))
                if lab is not None and (
                    lab[0] != r["grammar"] or lab[1] != r["tier"]
                    or lab[2] != r["environment_association"]
                ):
                    label_mismatches += 1
            rows.append({
                "mouse": s.mouse,
                "group": s.group,
                "day": s.day,
                "timestamp": s.timestamp,
                "trial_id": tid,
                "block": tid // 2,                 # 2->1, 4->2, 6->3, 8->4
                "roi": roi,
                "arm_type": atype,
                "environment": env,
                "grammar": r["grammar"] if r["grammar"] in ("A", "B") else None,
                "tier": r["tier"] if r["tier"] in ("dominant", "secondary", "rare") else None,
                "time_spent_s": float(r["time_spent"]),
                "visits": int(r["visitation_count"]),
                "melodies": mel,
            })

    df = pd.DataFrame(rows)

    report = {
        "results_root": root,
        "n_sessions_found": len(sessions),
        "excluded_sessions": [
            {"mouse": s.mouse, "day": s.day, "ts": s.timestamp, "reason": s.excluded}
            for s in sessions if s.excluded
        ],
        "duplicates_dropped": dup_dropped,
        "missing_melody_grammar_arms": missing_melodies,
        "trials_vs_samples_label_mismatches": label_mismatches,
        "integrity": integ,
    }
    if len(df):
        report["distinct_mice_by_day"] = (
            df.groupby("day")["mouse"].nunique().to_dict()
        )
        d1 = df[df["day"] == _DAY_PRIMARY].drop_duplicates("mouse")
        report["day1_distinct_mice"] = int(d1["mouse"].nunique())
        report["day1_group_balance"] = d1.groupby("group").size().to_dict()
    return LoadResult(arm_blocks=df, sessions=sessions, report=report)


# ---------------------------------------------------------------------------
# Day scope + PI
# ---------------------------------------------------------------------------

def load_day(df: pd.DataFrame, which: str = "primary") -> pd.DataFrame:
    """Slice the arm-block dataframe to one day. Never pools day 1 and day 2."""
    day = {"primary": _DAY_PRIMARY, "secondary": _DAY_SECONDARY}.get(which)
    if day is None:
        raise ValueError(f"which must be 'primary' or 'secondary', got {which!r}")
    return df[df["day"] == day].copy()


def session_pi(df: pd.DataFrame) -> pd.DataFrame:
    """EE-SC PI per (mouse, day), matching summary_analysis._compute_metrics.

    PI = (EE_time - SC_time) / (EE_time + SC_time) over grammar arms only.
    """
    g = df[df["arm_type"] == "grammar"]
    out = []
    for (mouse, day, group), sub in g.groupby(["mouse", "day", "group"]):
        ee = sub.loc[sub["environment"] == "EE", "time_spent_s"].sum()
        sc = sub.loc[sub["environment"] == "SC", "time_spent_s"].sum()
        tot = ee + sc
        out.append({
            "mouse": mouse, "day": day, "group": group,
            "EE_time_s": float(ee), "SC_time_s": float(sc),
            "PI": (ee - sc) / tot if tot > 0 else np.nan,
        })
    return pd.DataFrame(out)


def block_pi(df: pd.DataFrame) -> pd.DataFrame:
    """EE-SC PI per (mouse, day, block) for the within-session time-course."""
    g = df[df["arm_type"] == "grammar"]
    out = []
    for (mouse, day, group, block), sub in g.groupby(["mouse", "day", "group", "block"]):
        ee = sub.loc[sub["environment"] == "EE", "time_spent_s"].sum()
        sc = sub.loc[sub["environment"] == "SC", "time_spent_s"].sum()
        tot = ee + sc
        out.append({
            "mouse": mouse, "day": day, "group": group, "block": int(block),
            "PI": (ee - sc) / tot if tot > 0 else np.nan,
        })
    return pd.DataFrame(out)


def cell_pi(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(mouse, environment, tier) dwell time and share — the 6-cell grammar pattern."""
    g = df[df["arm_type"] == "grammar"].dropna(subset=["environment", "tier"])
    return (g.groupby(["mouse", "group", "environment", "tier"])["time_spent_s"]
            .sum().reset_index())
