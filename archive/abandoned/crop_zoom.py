import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

'''script to crop and zoom on a specific quadrant'''

selected_quadrant= None

def select_area(event, x, y, flags, param):
    global refPt, cropping, image_display

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        cv2.rectangle(image_display, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image_display)

def main():
    global refPt, cropping, image_display

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be accessed.")
        return

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_area)

    refPt = []
    cropping = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            image_display = frame.copy()

            if len(refPt) == 2:
                roi = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("ROI", roi)

            cv2.imshow("image", image_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    #print(selected_quadrant)
    if len(refPt) == 2:
        print("Selected quadrant coordinates: ", refPt)

if __name__ == "__main__":
    main()
