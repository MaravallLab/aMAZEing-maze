"""
Analysis pipeline configuration.

Edit MOUSE_ID and BASE_PATH for your machine, then run any of the
01_*, 02_*, 03_* scripts.
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── edit these ──────────────────────────────────────────────────────
MOUSE_ID = "6357"
BASE_PATH = os.path.join(
    os.path.expanduser("~"), "Box", "Awake Project", "Maze data", "simplermaze"
)
# ────────────────────────────────────────────────────────────────────

MOUSE_DIR = os.path.join(BASE_PATH, f"mouse {MOUSE_ID}")

# analysis parameters
FPS = 30
BODYPART = "mid"
LIKELIHOOD_THRESH = 0.8
PX_PER_CM = 7.5


@dataclass
class SessionInfo:
    """Everything needed to analyse one session."""
    session_id: str          # e.g. "3.6"
    session_dir: str         # full path to session folder
    trial_csv: str           # path to trial info CSV
    roi_csv: Optional[str] = None
    dlc_csv: Optional[str] = None


def discover_sessions(mouse_dir: str = MOUSE_DIR,
                      mouse_id: str = MOUSE_ID) -> List[SessionInfo]:
    """Auto-discover all sessions for a mouse and return SessionInfo objects."""
    sessions = []

    for entry in sorted(os.listdir(mouse_dir)):
        full = os.path.join(mouse_dir, entry)
        if not os.path.isdir(full):
            continue

        # extract session id from folder name
        sid = _extract_session_id(entry)
        if sid is None:
            continue

        # find trial CSV (prefer clean_, then new_, then mouse*_trial_info)
        trial_csv = _find_trial_csv(full, mouse_id, sid)
        if trial_csv is None:
            continue

        # find ROI CSV
        roi_csv = _find_roi_csv(full)

        # find DLC CSV
        dlc_csv = _find_dlc_csv(mouse_dir, mouse_id, full, sid)

        sessions.append(SessionInfo(
            session_id=sid,
            session_dir=full,
            trial_csv=trial_csv,
            roi_csv=roi_csv,
            dlc_csv=dlc_csv,
        ))

    return sessions


def _extract_session_id(folder_name: str) -> Optional[str]:
    """Extract session number from folder name."""
    # "2024-08-29_10_23_026357session3.7" -> "3.7"
    m = re.search(r"session\s*(\d+\.\d+)", folder_name, re.IGNORECASE)
    if m:
        return m.group(1)
    # "habituation" -> "hab"
    if "habituation" in folder_name.lower():
        return "hab"
    return None


def _find_trial_csv(session_dir: str, mouse_id: str, sid: str) -> Optional[str]:
    """Find the best trial CSV in a session directory."""
    candidates = [
        # new_ format (sessions 3.6-3.8, has frame columns)
        os.path.join(session_dir, f"new_session{sid}_trials.csv"),
        # clean_ format (all sessions)
        os.path.join(session_dir, f"clean_mouse{mouse_id}_session{sid}_trial_info.csv"),
        # raw format
        os.path.join(session_dir, f"mouse{mouse_id}_session{sid}_trial_info.csv"),
    ]
    # habituation has different naming
    if sid == "hab":
        candidates = [
            os.path.join(session_dir, f"mouse{mouse_id}_session1.1_trial_info.csv"),
        ]
        # also try any *trial_info.csv
        candidates += glob.glob(os.path.join(session_dir, "*trial_info*.csv"))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _find_roi_csv(session_dir: str) -> Optional[str]:
    """Find ROI CSV in session directory."""
    for name in ["rois1.csv"]:
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            return p
    # fallback: any *rois*.csv
    matches = glob.glob(os.path.join(session_dir, "*rois*.csv"))
    return matches[0] if matches else None


def _find_dlc_csv(mouse_dir: str, mouse_id: str,
                  session_dir: str, sid: str) -> Optional[str]:
    """Find the DLC tracking CSV for a session."""
    dlc_base = os.path.join(mouse_dir, "deeplabcut")
    if not os.path.isdir(dlc_base):
        return None

    # search recursively for a CSV matching this session
    pattern = os.path.join(dlc_base, "**", f"*s{sid}DLC*.csv")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


# ── convenience ─────────────────────────────────────────────────────

def get_sessions_with_dlc(mouse_dir: str = MOUSE_DIR,
                          mouse_id: str = MOUSE_ID) -> List[SessionInfo]:
    """Return only sessions that have DLC tracking data."""
    return [s for s in discover_sessions(mouse_dir, mouse_id) if s.dlc_csv]


if __name__ == "__main__":
    print(f"Mouse {MOUSE_ID} sessions:")
    for s in discover_sessions():
        dlc_tag = "  [DLC]" if s.dlc_csv else ""
        roi_tag = "  [ROI]" if s.roi_csv else ""
        print(f"  {s.session_id:>5s}{dlc_tag}{roi_tag}  {os.path.basename(s.trial_csv)}")
