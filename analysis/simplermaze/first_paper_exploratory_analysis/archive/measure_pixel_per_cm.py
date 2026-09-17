"""
This script uses OpenCV to open the video, grabs a random frame, and lets you click two points to measure the distance.

1. Edit the VIDEO_PATH variable in the USER CONFIGURATION section below.
2. Run the script.
3. Click once on one end of the maze wall, once again for the opposite wall.
4. A green line will appear connecting the dots.
5. In the terminal, insert the real world distance in cm and press enter.

It will print the number of px/cm.
"""

import cv2
import numpy as np
import random
import math

# insert path for video to get how many px there are in a cm in your setup

VIDEO_PATH = r"C:/Users/shahd/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_trial_003.mp4"



# Global variables to store points
ref_points = []
image = None
clone = None

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def mouse_callback(event, x, y, flags, param):
    global ref_points, image

    # If the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref_points) < 2:
            ref_points.append((x, y))
            
            # Draw a circle where the user clicked
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Frame", image)

            # If we have 2 points, draw the line and calculate distance
            if len(ref_points) == 2:
                # Draw the line
                cv2.line(image, ref_points[0], ref_points[1], (0, 255, 0), 2)
                
                # Calculate pixel distance
                dist_px = calculate_distance(ref_points[0], ref_points[1])
                
                # Display on image
                midpoint = ((ref_points[0][0] + ref_points[1][0]) // 2, 
                            (ref_points[0][1] + ref_points[1][1]) // 2)
                cv2.putText(image, f"{dist_px:.2f} px", (midpoint[0] + 10, midpoint[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Frame", image)
                
                print(f"\n[INFO] Line drawn.")
                print(f"  > Point 1: {ref_points[0]}")
                print(f"  > Point 2: {ref_points[1]}")
                print(f"  > Distance: {dist_px:.2f} pixels")
                
                # Ask user for real-world input in the terminal
                try:
                    real_dist = float(input("  > Enter the real-world distance in CM (or 0 to skip): "))
                    if real_dist > 0:
                        ratio = dist_px / real_dist
                        print(f"  > RESULT: {ratio:.4f} px/cm")
                        print(f"    (Use value {ratio:.4f} in your plotting script)")
                    else:
                        print("  > skipped calculation.")
                except ValueError:
                    print("  > Invalid input. Calculation skipped.")
                
                print("\n[Controls] Press 'r' to reset points, 'n' for new frame, 'q' to quit.")

def get_random_frame(cap):
    """Retrieves a random frame from the video capture."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not determine frame count.")
        return None
        
    random_index = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {random_index}.")
        return None
    return frame

def main():
    global image, clone, ref_points

    # Open Video using the Hardcoded Path
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        print("Please check the VIDEO_PATH ")
        return

    # Create a named window and attach the mouse callback
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    print("Fetching random frame")
    image = get_random_frame(cap)
    if image is None: return
    clone = image.copy()

    print("\n INSTRUCTIONS ")
    print("1. Click two points on the image to measure distance.")
    print("2. Look at the terminal to input the real-world CM distance.")
    print("3. Press 'r' to reset the points on the current frame.")
    print("4. Press 'n' to load a DIFFERENT random frame.")
    print("5. Press 'q' to quit.")

    while True:
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # Reset points ('r')
        if key == ord("r"):
            image = clone.copy()
            ref_points = []
            print("[INFO] Reset. Click two points.")

        # New Random Frame ('n')
        elif key == ord("n"):
            print("[INFO] Loading new random frame")
            new_frame = get_random_frame(cap)
            if new_frame is not None:
                image = new_frame
                clone = image.copy()
                ref_points = []

        # Quit ('q')
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()