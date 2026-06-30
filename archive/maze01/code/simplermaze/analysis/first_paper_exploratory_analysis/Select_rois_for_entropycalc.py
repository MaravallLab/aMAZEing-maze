# select rois from random frame of video

import cv2
import pandas as pd
import random
import os


VIDEO_PATH = r"C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_trial_003.mp4"
OUTPUT_CSV = 'C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/rois_for_entropy.csv'
NUM_ROIS = 16


# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
current_rois = []  # Stores (x, y, w, h)
temp_img = None    # Copy of image for display updates

def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, temp_img, current_rois

    # Access the original frame passed as param
    base_img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Refresh image to clear the "dragging" rectangle
            temp_img = base_img.copy()
            # Draw existing ROIs
            draw_existing_rois(temp_img)
            # Draw the current rectangle being dragged
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("ROI Selector", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Calculate width and height
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)
        w = x_max - x_min
        h = y_max - y_min

        # Only accept if it has some size
        if w > 5 and h > 5:
            current_rois.append((x_min, y_min, w, h))
            
            # Update the base display image with the new final rectangle
            temp_img = base_img.copy()
            draw_existing_rois(temp_img)
            cv2.imshow("ROI Selector", temp_img)
            
            print(f"Recorded ROI {len(current_rois)}: {current_rois[-1]}")

def draw_existing_rois(img):
    """Helper to draw all saved ROIs and their labels on the image."""
    for idx, (rx, ry, rw, rh) in enumerate(current_rois):
        # Draw Box (Red)
        cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
        # Draw Label (e.g., ROI 1)
        label = f"ROI {idx + 1}"
        cv2.putText(img, label, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    global temp_img

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return

    # 1. Load Video and Random Frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        random_frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
        ret, frame = cap.read()
    else:
        ret = False
        
    cap.release()

    if not ret:
        print("Error: Could not read video frame.")
        return

    # 2. Setup Window
    window_name = "ROI Selector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allows resizing
    cv2.resizeWindow(window_name, 1200, 800)        # Set a large default size
    
    temp_img = frame.copy()
    cv2.setMouseCallback(window_name, draw_roi, frame)

    print(f"--- Loaded Frame {random_frame_idx} ---")
    print(f"Please draw {NUM_ROIS} ROIs.")
    print("  - Drag mouse to draw.")
    print("  - Press 'u' to undo last box.")
    print("  - Press 'ESC' or 'ENTER' when finished.")

    while True:
        cv2.imshow(window_name, temp_img)
        k = cv2.waitKey(1) & 0xFF

        # Exit on ENTER or ESC
        if k == 13 or k == 27: 
            break
        
        # Undo on 'u'
        if k == ord('u'):
            if current_rois:
                print(f"Undoing ROI {len(current_rois)}")
                current_rois.pop()
                # Redraw
                temp_img = frame.copy()
                draw_existing_rois(temp_img)
                cv2.imshow(window_name, temp_img)

        # Stop if we reached the limit (optional auto-close)
        if len(current_rois) >= NUM_ROIS:
            cv2.putText(temp_img, "DONE! Press ENTER to save.", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow(window_name, temp_img)

    cv2.destroyAllWindows()

    # 3. Save Data
    if current_rois:
        data = []
        for i, (x, y, w, h) in enumerate(current_rois):
            data.append({
                'ROI_Name': f'Zone_{i+1}',
                'x_min': x,
                'x_max': x + w,
                'y_min': y,
                'y_max': y + h
            })
        
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Saved {len(df)} ROIs to '{OUTPUT_CSV}'.")
    else:
        print("No ROIs selected.")

if __name__ == "__main__":
    main()