import pandas as pd
from pathlib import Path
from segfuncs import BASE_DIR, get_video_csv_pairs

def main():
    # 1. Get the map of {csv_path: video_path}
    pairs = get_video_csv_pairs(BASE_DIR)
    
    if not pairs:
        print("No sessions found. Check BASE_DIR in common.py")
        return

    print(f"Checking {len(pairs)} sessions for manifests...")
    merged_count = 0

    for csv_path, video_path in pairs.items():
        tp = Path(csv_path)
        vp = Path(video_path)
        
        # 2. Construct the expected path to the manifest file
        # Matches logic: video_dir / "segments" / {video_name}_segments_manifest.csv
        manifest_path = vp.parent / "segments" / f"{vp.stem}_segments_manifest.csv"
        
        if not manifest_path.exists():
            # If no manifest, skip (maybe segmentation hasn't run yet)
            continue
            
        print(f"Merging: {vp.name}")
        
        # 3. Load DataFrames
        try:
            df_trials = pd.read_csv(tp)
            df_manifest = pd.read_csv(manifest_path)
        except Exception as e:
            print(f"  [Error] Could not read files: {e}")
            continue

        if df_manifest.empty:
            print("  [Skip] Manifest is empty.")
            continue

        # 4. Prepare columns in Trial CSV
        if "start_frame" not in df_trials.columns:
            df_trials["start_frame"] = pd.NA
        if "end_frame" not in df_trials.columns:
            df_trials["end_frame"] = pd.NA

        # 5. Merge Data
        # We assume row N in manifest corresponds to row N in trial_info
        # (The manifest usually has a 'trial_index' column we can use to be safe)
        
        updates = 0
        for _, row in df_manifest.iterrows():
            # Get the target index (default to row index if column missing)
            idx = int(row["trial_index"]) if "trial_index" in row else row.name
            
            # Boundary check
            if idx < len(df_trials):
                df_trials.at[idx, "start_frame"] = row["start_frame"]
                df_trials.at[idx, "end_frame"] = row["end_frame"]
                updates += 1

        # 6. Save the updated Trial CSV
        try:
            df_trials.to_csv(tp, index=False)
            print(f"  -> Updated {updates} rows in {tp.name}")
            merged_count += 1
        except Exception as e:
            print(f"  [Error] Failed to save CSV: {e}")

    print(f"\nDone! Merged frame data into {merged_count} CSV files.")

if __name__ == "__main__":
    main()