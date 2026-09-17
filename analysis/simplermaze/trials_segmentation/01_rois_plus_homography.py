import cv2 as cv
import pandas as pd
from pathlib import Path
from segfuncs import (
    BASE_DIR, REF_ROIS_FILE, ROI_NAMES, PYR_SCALE,
    get_video_csv_pairs, grab_first_frame, load_rois, 
    save_rois_csv, overlay_rois, auto_rois_from_reference
)

def draw_rois_interactive(video_path: str, names: list, scale=2.0):
    frame = grab_first_frame(video_path)
    if frame is None: raise RuntimeError(f"Cannot open {video_path}")
    
    disp = cv.resize(frame, None, fx=scale, fy=scale)
    win = "Draw ROIs - ENTER to confirm, ESC to skip"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, disp.shape[1], disp.shape[0])
    
    selections = []
    for name in names:
        # Show existing selections on screen
        show_img = disp.copy()
        for _, sx, sy, sw, sh in selections:
            # Rescale back up for display
            cv.rectangle(show_img, (int(sx*scale), int(sy*scale)), 
                         (int((sx+sw)*scale), int((sy+sh)*scale)), (255,0,0), 2)
            
        cv.putText(show_img, f"Draw {name} (Enter)", (20, 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        
        r = cv.selectROI(win, show_img, fromCenter=False, showCrosshair=True)
        if r == (0,0,0,0): continue
        
        # Scale back down to original resolution
        x, y, w, h = int(r[0]/scale), int(r[1]/scale), int(r[2]/scale), int(r[3]/scale)
        selections.append((name, x, y, w, h))
        
    cv.destroyAllWindows()
    return selections

def main():
    csv_video = get_video_csv_pairs(BASE_DIR)
    if not csv_video:
        print("No videos found. Check BASE_DIR in common.py")
        return

    ref_csv_path = Path(BASE_DIR) / REF_ROIS_FILE
    
    # 1. Establish Reference ROIs
    first_video = next(iter(csv_video.values()))
    
    if not ref_csv_path.exists():
        print(f"Reference ROIs missing. Drawing on: {Path(first_video).name}")
        ref_rows = draw_rois_interactive(first_video, ROI_NAMES)
        save_rois_csv(ref_csv_path, ref_rows)
        print(f"Saved reference -> {ref_csv_path}")
    else:
        print(f"Loaded reference -> {ref_csv_path}")
        ref_rows = load_rois(str(ref_csv_path))

    ref_frame = grab_first_frame(first_video)

    # 2. Adapt to other videos
    for _, video in csv_video.items():
        vp = Path(video)
        out_csv = vp.with_suffix("").as_posix() + "_rois.csv"
        
        if Path(out_csv).exists():
            print(f"ROIs exist for {vp.name}, skipping.")
            continue

        if video == first_video:
            # Copy ref to specific file if not exists
            save_rois_csv(Path(out_csv), ref_rows)
            continue

        print(f"Processing: {vp.name}")
        cur_frame = grab_first_frame(str(vp))
        if cur_frame is None: continue

        # Auto-Adapt
        adapted = auto_rois_from_reference(cur_frame, ref_frame, ref_rows, PYR_SCALE)
        
        # Visualize
        viz = overlay_rois(cur_frame, adapted if adapted else ref_rows, 
                           color=(0,255,0) if adapted else (0,255,255))
        
        win = f"{vp.name} - Y=Accept, R=Redraw, Q=Skip"
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.imshow(win, viz)
        key = cv.waitKey(0) & 0xFF
        cv.destroyAllWindows()

        final_rows = None
        if key in (ord('y'), ord('Y')) and adapted:
            final_rows = adapted
        elif key in (ord('r'), ord('R')):
            final_rows = draw_rois_interactive(str(vp), ROI_NAMES)
        
        if final_rows:
            save_rois_csv(Path(out_csv), final_rows)
            print(f"Saved -> {Path(out_csv).name}")
        else:
            print("Skipped.")

if __name__ == "__main__":
    main()