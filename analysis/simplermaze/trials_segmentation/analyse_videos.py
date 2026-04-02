import pandas as pd
import numpy as np
import cv2 as cv
from pathlib import Path
import plotly.express as px
import plotly.io as pio
from segfuncs import BASE_DIR, get_video_csv_pairs

# Use browser to render plots
pio.renderers.default = "browser"

def get_video_duration(path):
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened(): return 0.0
    fps = cap.get(cv.CAP_PROP_FPS)
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0: return frames / fps
    return 0.0

def generate_gaps(video_path, csv_path):
    """
    Reads the updated trial_info CSV to find coverage, then calculates gaps.
    """
    vp = Path(video_path)
    tp = Path(csv_path)
    
    # Check if segmentation updated the CSV
    df = pd.read_csv(tp)
    if "video_segment_path" not in df.columns: return None

    # Get valid segments from CSV (assuming logic maps rows to time roughly)
    # Note: Ideally, detect_bouts should save a manifest with timestamps. 
    # Since we only have the modified CSV and the files, we have to infer or 
    # rely on the previously generated `_uncovered_gaps.csv` if we saved it in Step 2.
    # To be robust, let's look for the side-car gaps file.
    
    # Note: In the original notebook, the gap calculation happened explicitly.
    # Let's recreate that logic here based on existing segments if gaps file missing.
    
    # Attempt to load specific gaps file if created previously, or calculate now.
    gaps_file = vp.parent / "segments" / f"{vp.stem}_uncovered_gaps.csv"
    
    total_dur = get_video_duration(vp)
    
    if gaps_file.exists():
        gaps_df = pd.read_csv(gaps_file)
        return gaps_df, total_dur
    
    return None, total_dur

def main():
    pairs = get_video_csv_pairs(BASE_DIR)
    plot_data = []
    EPOCH = pd.Timestamp("1970-01-01")

    print("Analyzing coverage...")

    for csv_p, vid_p in pairs.items():
        vp = Path(vid_p)
        # Look for the gaps CSV generated (Note: You might need to add the gap saving logic 
        # specifically to script 02 if you want it persistent, or recalculate here.
        # For simplicity, assuming script 02 generates detection segments).
        
        # Let's simply look for the segment files to reconstruct coverage
        segments_dir = vp.parent / "segments"
        if not segments_dir.exists(): continue
        
        # This is a simplification for visualization if exact timestamps weren't saved to CSV
        # Ideally, Script 02 should save a `manifest.csv` with start/end times.
        # Assuming we only have the gaps file as per original notebook flow:
        gaps_csv = segments_dir / f"{vp.stem}_uncovered_gaps.csv"
        
        if not gaps_csv.exists():
            print(f"No gap data for {vp.name}, skipping viz.")
            continue
            
        gaps_df = pd.read_csv(gaps_csv)
        total_dur = get_video_duration(vp)
        
        # Reconstruct "Covered" intervals from Gaps
        # Gaps are (start, end). Coverage is everything else.
        timeline = 0.0
        gaps = sorted(zip(gaps_df['start_s'], gaps_df['end_s']))
        
        # Add Covered
        for g_start, g_end in gaps:
            if g_start > timeline:
                plot_data.append({
                    "Video": vp.name, "Type": "Trial", 
                    "Start": EPOCH + pd.to_timedelta(timeline, unit='s'),
                    "End": EPOCH + pd.to_timedelta(g_start, unit='s')
                })
            # Add Gap
            plot_data.append({
                "Video": vp.name, "Type": "Gap", 
                "Start": EPOCH + pd.to_timedelta(g_start, unit='s'),
                "End": EPOCH + pd.to_timedelta(g_end, unit='s')
            })
            timeline = max(timeline, g_end)
            
        # Final coverage if video goes longer than last gap
        if timeline < total_dur:
             plot_data.append({
                "Video": vp.name, "Type": "Trial", 
                "Start": EPOCH + pd.to_timedelta(timeline, unit='s'),
                "End": EPOCH + pd.to_timedelta(total_dur, unit='s')
            })

    if not plot_data:
        print("No data to plot.")
        return

    df_plot = pd.DataFrame(plot_data)
    
    # Plotly Gantt
    fig = px.timeline(
        df_plot, 
        x_start="Start", x_end="End", y="Video", color="Type",
        color_discrete_map={"Trial": "#2ca02c", "Gap": "#d62728"},
        title="Segmentation Overview: Trials (Green) vs Gaps (Red)"
    )
    
    fig.update_yaxes(autorange="reversed") # Top to bottom
    fig.update_xaxes(tickformat="%H:%M:%S")
    
    out_html = "segmentation_summary.html"
    fig.write_html(out_html)
    print(f"Plot saved to {out_html}")
    fig.show()

if __name__ == "__main__":
    main()