"""
check_visit_outliers.py
========================

Diagnostic + cleanup helper for the auditory maze analysis.

Scans every session, applies the DV-first / sanity-cap loader from
preference_analysis_config.py, and reports the longest remaining
visit-duration values per (mouse, day, trial, ROI).

Usage
-----
    # default 10 s individual-visit clip
    python check_visit_outliers.py

    # tighter clip (e.g. 5 s)
    VISIT_CLIP_MS=5000 python check_visit_outliers.py

    # disable clip entirely (raw DV / capped trials.csv only)
    VISIT_CLIP_MS=0 python check_visit_outliers.py

    # report the top 30 outliers instead of the default 15
    python check_visit_outliers.py --top 30

    # show outliers above a custom threshold (in ms)
    python check_visit_outliers.py --threshold 5000

After confirming the cleanup looks reasonable, re-run:
    python run_batch_preference.py "PATH/TO/8_arms_w_voc"
    python 02_within_trial_preference.py
to regenerate the figures using the same VISIT_CLIP_MS setting.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from preference_analysis_config import (
    discover_sessions, load_session_visits,
    INDIVIDUAL_VISIT_CLIP_MS,
    SOUND_TRIAL_IDS, SILENT_TRIAL_IDS,
    DAY_SHORT, EXPERIMENT_DAYS,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--top", type=int, default=15,
                    help="Show top N outliers (default 15)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Show all rows with avg visit duration > THRESHOLD ms "
                         "(overrides --top)")
    ap.add_argument("--export", type=str, default=None,
                    help="Optional CSV path to dump every flagged row")
    args = ap.parse_args()

    print(f"Per-visit clip in effect: {INDIVIDUAL_VISIT_CLIP_MS} ms "
          f"(set VISIT_CLIP_MS env var to change)")

    sessions = discover_sessions()
    print(f"Discovered {len(sessions)} sessions\n")

    rows = []
    source_counts = {}
    for sess in sessions:
        df, source = load_session_visits(sess)
        if df is None:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        sub = df[df["ROIs"].astype(str).str.startswith("ROI")
                 & (df["visitation_count"] > 0)].copy()
        if len(sub) == 0:
            continue
        sub["avg_dur_ms"] = sub["time_spent"] / sub["visitation_count"]
        for _, r in sub.iterrows():
            rows.append({
                "mouse_id": sess.mouse_id,
                "day": sess.day,
                "day_short": DAY_SHORT.get(sess.day, sess.day),
                "trial_ID": int(r["trial_ID"]),
                "ROI": r["ROIs"],
                "visits": int(r["visitation_count"]),
                "total_ms": float(r["time_spent"]),
                "avg_dur_ms": float(r["avg_dur_ms"]),
                "source": source,
            })

    out = pd.DataFrame(rows)
    print(f"Source breakdown across {len(sessions)} sessions: {source_counts}")
    print(f"Total (mouse, day, trial, ROI) rows with visits: {len(out)}")
    print()

    # Distribution summary
    arr = out["avg_dur_ms"].values
    print("Distribution of average visit duration (ms):")
    for q, label in [(50, "median"), (90, "p90"), (95, "p95"),
                     (99, "p99"), (100, "max")]:
        print(f"  {label:>6s}: {np.percentile(arr, q):>10.0f}")
    print()

    # Filter
    if args.threshold is not None:
        flagged = out[out["avg_dur_ms"] > args.threshold].copy()
        print(f"Rows with avg_dur_ms > {args.threshold:.0f}: {len(flagged)}")
    else:
        flagged = out.nlargest(args.top, "avg_dur_ms")
        print(f"Top {args.top} longest average visits:")

    flagged = flagged.sort_values("avg_dur_ms", ascending=False)
    print()
    print(flagged[["mouse_id", "day_short", "trial_ID", "ROI",
                   "visits", "total_ms", "avg_dur_ms", "source"]]
          .to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    print()

    # Export
    if args.export:
        out_path = args.export
    else:
        # Default to BATCH_ANALYSIS/visit_outliers.csv if writable
        from preference_analysis_config import OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "visit_outliers.csv")
    try:
        flagged.to_csv(out_path, index=False)
        print(f"Wrote flagged rows to: {out_path}")
    except OSError as e:
        print(f"Could not write {out_path}: {e}")

    print()
    print("=" * 60)
    print("Next steps")
    print("=" * 60)
    print("1. Inspect the flagged rows above.")
    print("2. If they look reasonable, regenerate figures:")
    print("     python run_batch_preference.py \"PATH/TO/8_arms_w_voc\"")
    print("     python 02_within_trial_preference.py")
    print("3. To use a tighter per-visit cap, set VISIT_CLIP_MS first:")
    print("     VISIT_CLIP_MS=5000 python run_batch_preference.py ...")
    print("   (Windows cmd:  set VISIT_CLIP_MS=5000  then run the script)")
    print("   (Windows PowerShell:  $env:VISIT_CLIP_MS=5000  then run)")


if __name__ == "__main__":
    main()
