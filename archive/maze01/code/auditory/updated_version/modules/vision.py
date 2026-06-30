import cv2 as cv
import numpy as np
import pandas as pd
import time
import os
from typing import List, Dict, Tuple, Optional


## if define_rois doesn't work, take it from lines 242 in supfun_sequences.py

def define_rois(video_input: int, output_csv_path: str, roiNames: List[str] = ["entrance1", "entrance2", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6", "ROI7", "ROI8"] ):
    # open a frame from the video to manually select the rois, and then save the rois coordinates (x,y) in a csv in the output_csv_path
    cap = cv.VideoCapture(video_input)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera to define the Rois")
    
    # just get the camera running for the first 10 frames so that it gets auto-exposure instead of just grabbing the first frame
    for _ in range(10):
        ret, frame = cap.read()

    if not ret: #the frame is empty/broken
        raise RuntimeError(f"Cannot read frame")
    
    #setup the canvas to draw the rois and create dictionary 
    display_frame = frame.copy()
    rois= {}

    #create resizable window otherwise it's too small and selecting the rois is a challenge
    window_name = "Define ROIs"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720) # set an arbitrary defaiult size

    #prompt the user 
    print("\n--- ROI SELECTION MODE ---")
    print("1. Click and drag to draw a box.")
    print("2. Press SPACE or ENTER to confirm.")
    print("3. Press 'c' to cancel/retry the current box.")

    for name in roiNames:
        print(f"Please select location of {name}")

        # cv.selectROI returns a tuple (x, y, w, h)
        rect = cv.selectROI("Define ROIs", display_frame, fromCenter=False, showCrosshair=True)
        
        # Draw the confirmed box permanently on the display frame 
        x, y, w, h = rect
        cv.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(display_frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        rois[name] = [x, y, w, h]

    cv.destroyAllWindows()
    cap.release()

    # Save to CSV
    df = pd.DataFrame(rois, index=["xstart", "ystart", "xlen", "ylen"])
    df.to_csv(output_csv_path)

        

    
# here we are going to handle the ROI detection part and the debouncing 
# (aka a threshold of an arbitrary number of frames to basically confirm that there is activity or inactivity in the roi)
#the current camera is 30fps, so if exit_frames = 5 it means that the roi needs to be empty for 0.16 seconds to be considered empty
class ROIMonitor: 
    def __init__(self, 
                 roi_csv_path: str,
                 video_input: int,  
                 roiNames: List[str] = ["entrance1", "entrance2", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6", "ROI7", "ROI8"],
                 enter_frames: int = 1, 
                 exit_frames : int = 5 ):
        

        self.roiNames = roiNames
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames

        #check if the rois.csv exists, and if not, create it 

        if not os.path.exists(roi_csv_path):
            print(f"rois.csv not found at {roi_csv_path}")
            print("Drawing rois instead")
            # the path where the rois are being output is the same path selected as input for the class
            define_rois(video_input, roi_csv_path, roiNames)

        #now that the file exists, let's try to open it, in case it's damaged or corrupted

        try:
            self.rois_df = pd.read_csv(roi_csv_path, index_col= 0)
        except Exception as e: 
            raise RuntimeError(f"ROI file exists but it's corrupted. Delete {roi_csv_path} and restart. Error{e}")
        

        #initialise State Tracking variables
        self.thresholds: Dict[str, float]= {}
        self.is_occupied: Dict[str, bool] = {name: False for name in self.roiNames}
        self.present_streak: Dict[str, int] = {name : 0 for name in self.roiNames}
        self.absent_streak: Dict[str, int] = {name:0 for name in self.roiNames}


    def calibrate(self, binary_frame: np.ndarray): 
        # take a clean frame to establish the baseling pixel count
        #the frame needs to be without mouse
        print("Calibrating ROI thresholds...")
        for name in self.roi_names:
            if name not in self.rois_df.columns:
                print(f" Warning: ROI '{name}' in config.py but not in CSV. Skipping but double check.")
                continue


                
            cutout = self._crop_roi(binary_frame, name)
            self.thresholds[name] = np.sum(cutout)

    def update(self, binary_frame: np.ndarray) -> List[str]:
        
        #Process a new frame. Returns list of newly entered ROIs.
        
        just_entered_rois = []

        for name in self.roi_names:
            # Skip if something is wrong with CSV vs Config sync
            if name not in self.thresholds: continue

            cutout = self._crop_roi(binary_frame, name)
            current_sum = np.sum(cutout)
            
            # 1. If current sum is < 50% of baseline, something (mouse) is blocking the background
            raw_present = current_sum < (self.thresholds[name] * 0.5)

            # update Debouncing conters
            if raw_present:
                self.present_streak[name] += 1
                self.absent_streak[name] = 0
            else:
                self.absent_streak[name] += 1
                self.present_streak[name] = 0

            # check for State Change
            was_occupied = self.is_occupied[name]

            if not was_occupied and self.present_streak[name] >= self.enter_frames:
                self.is_occupied[name] = True
                just_entered_rois.append(name)

            elif was_occupied and self.absent_streak[name] >= self.exit_frames:
                self.is_occupied[name] = False
            
        return just_entered_rois

    def _crop_roi(self, frame: np.ndarray, name: str) -> np.ndarray:
        #helper function to slice the numpy array. Essentially just focuses on the rois rather than the whole image

        #look up coordinates in the rois.csv
        coords = self.rois_df[name]

        #slice the image array
        x, y, w, h = coords["xstart"], coords["ystart"], coords["xlen"], coords["ylen"]
        return frame[y:y+h, x:x+w]

    def draw_feedback(self, frame: np.ndarray):
        # The ROIs are blue if not visited, and red if visited. Basically a visual tracking 

        #skip roi if not in data
        for name in self.roi_names:
            if name not in self.rois_df.columns: continue
            
            #get the coordinates
            coords = self.rois_df[name]
            x, y, w, h = coords["xstart"], coords["ystart"], coords["xlen"], coords["ylen"]
            #pick colour based on state in BGR format
            colour = (0, 0, 255) if self.is_occupied[name] else (255, 0, 0)

            #draw box and name            
            cv.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv.putText(frame, name, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)





