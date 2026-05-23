"""Run the post-session analysis on an existing session folder.

Usage:
    python run_analysis.py <session_folder>
    python run_analysis.py <session_folder1> <session_folder2> ...

Figures are saved inside each session folder.
"""

import sys
import os
from modules.analysis import SessionAnalyzer


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <session_folder> [<session_folder2> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            print(f"Skipping — not a directory: {path}")
            continue
        print(f"\nAnalysing: {path}")
        try:
            SessionAnalyzer(path).generate_report()
        except Exception as e:
            import traceback
            print(f"Failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
