"""Start the command-line tools as child processes, from source or frozen.

The graphical interface never runs an experiment in its own process: it
launches the same entry points a terminal user would (``amaze-auditory``,
``amaze-summary``, ...) and shows their output. This module builds the
command line for that in the two situations the app runs in:

* from a source checkout / ``pip install``: ``python -m <module> ...``
* frozen by PyInstaller: the app executable itself, re-invoked as
  ``amazeing-app.exe --entry <name> ...`` (see :func:`run_entry`).

The research scripts in ``analysis/auditory/`` are not modules of the
package, so they get their own pair: :func:`command_for_script` runs
``python -u <script> ...`` from source and ``amazeing-app.exe --script <path>
...`` when frozen, and :func:`run_script` is what the executable does with it.
"""

from __future__ import annotations

import os
import runpy
import sys
from typing import Dict, List, Sequence

ENTRY_MODULES: Dict[str, str] = {
    "auditory": "amazeing.auditory.main",
    "analyse-session": "amazeing.auditory.run_analysis",
    "summary": "amazeing.auditory.run_summary_analysis",
    "summary-csv": "amazeing.auditory.run_summary_csv",
    "grammar": "amazeing.auditory.grammar_stimuli.run",
    "tactile": "amazeing.simplermaze.simplerCode",
    "tactile-segments": "amazeing.simplermaze.post_process_session",
    "draw-rois": "amazeing.app.draw_rois",
    "camera-check": "amazeing.auditory.camera_check",
}


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def command_for(entry: str, args: Sequence[str] = ()) -> List[str]:
    """Return the argv to run ``entry`` with ``args`` as a child process."""
    if entry not in ENTRY_MODULES:
        raise KeyError(f"Unknown entry {entry!r}; valid: {sorted(ENTRY_MODULES)}")
    if is_frozen():
        return [sys.executable, "--entry", entry, *args]
    return [sys.executable, "-u", "-m", ENTRY_MODULES[entry], *args]


def command_for_script(path: str, args: Sequence[str] = ()) -> List[str]:
    """Return the argv to run the research script at ``path`` with ``args``.

    Run it with the script's own folder as the working directory
    (:func:`script_cwd`): the scripts import their sibling modules.
    """
    path = os.path.abspath(path)
    if is_frozen():
        return [sys.executable, "--script", path, *args]
    return [sys.executable, "-u", path, *args]


def script_cwd(path: str) -> str:
    return os.path.dirname(os.path.abspath(path))


def child_environment() -> Dict[str, str]:
    """Environment for child processes: unbuffered UTF-8 output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_entry(entry: str, args: Sequence[str]) -> int:
    """Run ``entry`` in this process (used by the frozen executable)."""
    if entry not in ENTRY_MODULES:
        print(f"Unknown entry {entry!r}; valid: {sorted(ENTRY_MODULES)}", file=sys.stderr)
        return 2
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    sys.argv = [entry, *args]
    runpy.run_module(ENTRY_MODULES[entry], run_name="__main__")
    return 0


def run_script(path: str, args: Sequence[str]) -> int:
    """Run the research script at ``path`` in this process, as ``python path args`` would."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"No such script: {path}", file=sys.stderr)
        return 2
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    sys.argv = [path, *args]
    runpy.run_path(path, run_name="__main__")
    return 0
