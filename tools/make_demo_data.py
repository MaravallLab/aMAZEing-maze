#!/usr/bin/env python3
"""Build the demo dataset used by the analysis tutorial.

The thesis data are available on request, so the tutorial ships with a small
real subset instead: the habituation day and both test days of Experiment 2
for four mice, two from each counterbalancing group (enriched cage grammar A:
13535 and 13536; grammar B: 13539 and 13540). None of them is among the
sessions the thesis analyses exclude.

Only the tables a session writes are copied (the trials table, the visit log,
the maze entry log, the animal metadata and the grammar log). Videos, saved
sound arrays and figures are left out, so the zip stays small and the figures
are made fresh when the tutorial is followed.

    python tools/make_demo_data.py D:/AuditoryExperiments/grammar

writes docs/assets/demo-sessions.zip, which unzips to
demo-sessions/grammar/<day>/<session>/.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

MICE = ["13535", "13536", "13539", "13540"]
DAYS = ["habituation", "day_1", "day_2"]
KEEP = ("trials_", "grammar_samples_", "_detailed_visits.csv", "_maze_entries.csv", "_metadata.csv")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets" / "demo-sessions.zip"


def sessions_of(grammar_dir: Path, day: str, mouse: str):
    found = sorted((grammar_dir / day).glob(f"time_*mouse{mouse}"))
    if len(found) != 1:
        raise SystemExit(f"expected one {day} session for mouse {mouse}, found {len(found)}")
    return found[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grammar_dir", type=Path,
                    help="the Experiment 2 folder holding habituation/, day_1/ and day_2/")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "demo-sessions" / "grammar"
        n_files = 0
        for day in DAYS:
            for mouse in MICE:
                src = sessions_of(args.grammar_dir, day, mouse)
                dst = root / day / src.name
                dst.mkdir(parents=True)
                for f in sorted(src.iterdir()):
                    if f.suffix == ".csv" and any(k in f.name for k in KEEP):
                        shutil.copy2(f, dst / f.name)
                        n_files += 1
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(tmp).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp).as_posix())
    print(f"{n_files} files from {len(MICE)} mice and {len(DAYS)} days written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
