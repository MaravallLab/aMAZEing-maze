# here we handle all things related to the outputs of the scripts
#This means we need a class that handles: 
# creation of a new folder that will go inside the parent folder that was defined in config.py as base_output_path (line 52)
    # create a subdirectory with the session type (so we organise straight away)
    # extrapolate date/time of the experiment
    # prompt user for mouse info (This will be handled later in main.py - should be the equivalent of collect_metadata)
    #create sub-subdirectory with mouse ID and time info inside the session directory
# create df and csv with mouse metadata
# get the stimulus string
# create visitation log with roi and stimulus info
#

import os
import csv
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class DataManager:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.session_directory = ""
        self.mouseID = ""
        self.timestamp = ""

    def setup_session(self, cfg) -> Tuple[str, str]:
        # automatically ask user input and create folder structure : base_output_path / experiment_session / time_YYYY...mouseID /
        # return session_directory_path, full mouse ID

        self.timestamp = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

        #get user input for mouseID
        user_input = input("insert mouse ID (number only):\n").strip()
        self.mouseID = f"mouse{user_input}"

        #create directory structure by creating the experiment session folder
        experiment_dir_name = cfg.experiment_mode

        #since we have different days of intervals experiments, if the experiment session involves intervals, we name them after the session and day of the session
        if cfg.experiment_mode == "complex_intervals":
            experiment_dir_name = f"{cfg.experiment_mode}_{cfg.complex_interval_day}"

        exp_sess_dir = os.join.path(self.base_output_path, experiment_dir_name)

        if not os.path.exist(exp_sess_dir):
            os.makedirs(exp_sess_dir)
            print(f"couldn't find directory, created session directory: {exp_sess_dir}")
        else:
            print(f"using existing directory {exp_sess_dir}")

        # now we create the mouse specific subdirectory inside the session directory
        mouse_dir_name = f"time_{self.timestamp}{self.mouseID}"
        self.session_directory = os.os.path.join (exp_sess_dir, mouse_dir_name)

        if not os.path.exist(self.session_directory):
            os.makedirs(self.session_directory)
            print(f"created directory {self.session_directory}")

        return self.session_directory, self.mouseID
    

    def save_metadata(self):
        # prompt the user for the mouse information to then save into a metadata csv
        print("Mouse info (press enter to skip)")
        ear_mark = input("ear mark identifiers?\n").strip()
        birth_date= input("insert mouse birth date:\n").strip()
        gender= input("insert mouse gender (m/f/whatever the mouse identifies with):\n").strip().lower()

        data = {
            "animal ID": self.mouseID,
            "session_date": self.timestamp,
            "ear_mark": ear_mark,
            "birth_date": birth_date,
            "gender":gender
        }

        filename = f"{self.mouseID}_{self.timestamp}_metadata.csv"

        path = os.path.join(self.session_directory, filename)

        pd.DataFrame([data]).to_csv(path, index = False)
        print(f"Metadata csv saved to {filename}")

    def initialise_visit_log(self, cfg):
        # create the csv file that is going to contain the visitation log (sequence of visitations)
        filename = f"{self.mouseID}_{cfg.experiment_mode}_detailed_visits.csv"
        full_path = os.path.join(self.session_directory, filename)
        
        headers = ["trial_ID", "ROI_visited", "stimulus","sound_on_time" , "sound_off_time","time_spent_seconds"]
        
        with open(full_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
        return full_path
    
    @staticmethod

    def get_stimulus_string(trials_df:pd.DataFrame, trial_id:int, roi:str) -> str:
        
        #Extract relevant stimulus info for a specific Trial/ROI combination and combine them into a single descriptive string.

        # Filter the dataframe for the specific trial and ROI
        condition = (trials_df['trial_ID'] == trial_id) & (trials_df['ROIs'] == roi)
        row = trials_df.loc[condition]

        if row.empty:
            return "Unknown_Stimulus"

        # List of potential columns to look for (based on your different experiment types)
        cols_to_check = [
            'frequency',
            'sound_type',
            'interval_type',
            'interval_ratio',
            'interval_name',
            'pattern',
            'temporal_modulation'
        ]

        details = []

        for col in cols_to_check:
            if col in row.columns:
                val = row[col].values[0]

                # --- handle arrays/lists vs scalars safely ---

                # Case 1: numpy array or Python list
                if isinstance(val, (np.ndarray, list)):
                    arr = np.array(val)

                    # skip if completely empty or all NaN
                    if arr.size == 0 or np.all(pd.isna(arr)):
                        continue

                    # make a compact string representation
                    flat = arr.flatten()
                    if flat.size > 5:
                        val_str = "[" + ", ".join(map(str, flat[:5])) + ", ...]"
                    else:
                        val_str = "[" + ", ".join(map(str, flat)) + "]"

                else:
                    # Case 2: scalar-like value
                    # only treat as missing if pandas thinks so
                    if pd.isna(val):
                        continue
                    val_str = str(val)

                details.append(f"{col}:{val_str}")

        return " | ".join(details) if details else "No_Stimulus_Info"

    
    @staticmethod
    def log_individual_visit(csv_path:str, trial_id:int , roi:str, stimulus_str:str, sound_onset:float, sound_offset:float, duration:float):
        #Appends to the log csv a single visit event
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, roi, stimulus_str, sound_onset, sound_offset, duration])








