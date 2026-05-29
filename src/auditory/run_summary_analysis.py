"""Generate cross-session summary figures for grammar experiments.

Run after the last mouse of the day (or at any point to update the figures).

Usage
-----
# Summary for one day — all mice tested that day:
    python run_summary_analysis.py --day path/to/grammar/day_1

# Cross-day summary — all mice across all days:
    python run_summary_analysis.py --all path/to/grammar/

Figures are saved into the folder you pass (day folder or grammar folder).
"""

import argparse
import os
import sys
import traceback
from modules.summary_analysis import SummaryAnalyzer


def main():
    p = argparse.ArgumentParser(
        description="Generate summary figures across sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--day", metavar="DAY_FOLDER",
        help="Path to a single day folder, e.g. .../grammar/day_1",
    )
    group.add_argument(
        "--all", metavar="ROOT_FOLDER", dest="all_days",
        help="Path to the grammar folder containing all days, e.g. .../grammar/",
    )
    args = p.parse_args()

    root = os.path.normpath(args.day or args.all_days)

    if not os.path.isdir(root):
        print(f"Error: not a directory: {root}")
        sys.exit(1)

    print(f"\nRunning summary analysis on: {root}")
    try:
        analyzer = SummaryAnalyzer(root)
        out = analyzer.generate_report()
        print(f"\nSummary figures saved in: {out}")
    except Exception as e:
        print(f"Summary analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
