"""Export per-mouse summary CSVs for grammar experiments.

Writes one ``summary_<day>.csv`` inside each day folder (one row per mouse
tested that day) and a comprehensive ``summary_all_sessions.csv`` at the root
(one row per mouse per day). Use these CSVs as the starting point for data
analysis (stats, plotting, etc.).

Usage
-----
# CSVs for one day — all mice tested that day:
    python run_summary_csv.py --day path/to/grammar/day_1

# Cross-day CSVs — all mice across all days, plus a combined CSV:
    python run_summary_csv.py --all path/to/grammar/
"""

import argparse
import os
import sys
import traceback
from modules.summary_analysis import SummaryAnalyzer


def main():
    p = argparse.ArgumentParser(
        description="Export per-mouse summary CSVs across sessions.",
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

    print(f"\nExporting summary CSVs from: {root}")
    try:
        analyzer = SummaryAnalyzer(root)
        out = analyzer.export_csvs()
        print(f"\nSummary CSVs saved in: {out}")
    except Exception as e:
        print(f"CSV export failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
