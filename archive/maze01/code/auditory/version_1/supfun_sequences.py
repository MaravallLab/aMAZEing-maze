import pandas as pd
import cv2 as cv
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import simpledialog, Scale, HORIZONTAL, Label, ttk, filedialog, messagebox, Entry, font
import csv
import random
import wave
import sounddevice as sd
from sympy import *
from sympy.plotting import plot_parametric
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import soundfile as sf
from scipy.signal import resample, resample_poly, spectrogram, butter, sosfilt



def get_rois_list(rois_number):
    rois_list = []
    for i in range(rois_number):
        rois_list.append(f"ROI{i+1}")
    return rois_list

def collect_metadata(animal_ID):
    ear_mark = input("Ear mark identifiers? (y/n): \n").lower()
    birth_date = input("Insert animal birth date (dd/mm/yyyy): \n")
    gender = input("Insert animal assumed gender (m/f): \n").lower()


    data = {
    "animal ID": animal_ID,
    "ear mark identifier": ear_mark,
    "animal birth date": birth_date,
    "animal gender": gender,
    }

    return data

def save_metadata_to_csv(data, new_dir_path, file_name):
    df = pd.DataFrame([data])  # Convert dict to dataframe
    csv_path = os.path.join(new_dir_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")

def initialise_visit_log(directory, base_name):
    # create the csv file that is going to contain the visitation log (sequence of visitations)
    filename = f"{base_name}_detailed_visits.csv"
    full_path = os.path.join(directory, filename)
    
    headers = ["trial_ID", "ROI_visited", "stimulus","sound_on_time" , "sound_off_time","time_spent_seconds"]
    
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    return full_path

def get_stimulus_string(trials_df, trial_id, roi):
    """
    Extract relevant stimulus info for a specific Trial/ROI combination
    and combine them into a single descriptive string.
    """

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

def log_individual_visit(csv_path, trial_id, roi, stimulus_str, sound_onset, sound_offset, duration):
    #Appends to the log csv a single visit event
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([trial_id, roi, stimulus_str, sound_onset, sound_offset, duration])

def setup_directories(base_path, date_time, animal_ID):
    new_directory = f"{date_time}{animal_ID}"
    new_dir_path = os.path.join(base_path, new_directory)
    ensure_directory_exists(new_dir_path)
    return new_dir_path

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def get_user_inputs():
    animal_ID = input("Insert animal ID: \n").upper()

    return animal_ID, 

def get_current_time_formatted():
    return time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

def get_metadata():
    anID = input("enter animal identification:")
    date = input("enter date:")
    metadata = [anID, date]
    #figure out what else users want to store
    return metadata

def write_text(text="string",window="noWindow"):
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    
    cv.putText(window,text, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    return


def start_camera(videoInput=0):
    cap = cv.VideoCapture(videoInput)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap

def record_video(cap, recordFile, frame_width, frame_height, fps):
    cc = cv.VideoWriter_fourcc(*'mp4v')
    videoFileObject = cv.VideoWriter(recordFile, cc, fps, (frame_width, frame_height))
    return videoFileObject
    #if not videoFile.isOpened():
    #    print("Error: Failed to create VideoWriter object.")
    #    cap.release()
    #    exit()

    #while True:
    #    ret, frame = cap.read()
    #    if ret:
    #        videoFile.write(frame)
    #        cv.imshow("Frame", frame)
    #        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:
    #            break
    #    else:
    #        print("Error: Failed to read frame from camera.")
    #        break

    #videoFile.release()


def grab_n_convert_frame(cameraHandle):
    #capture a frame
    ret, frame = cameraHandle.read()
    #print(ret)
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray=frame
    return gray,ret 

def csv_to_dict(fileName="rois.csv"):
    output = dict()
    temp =  pd.read_csv(fileName,index_col=0)
    temp = temp.transpose()
    output = temp.to_dict()
    return output


def define_rois(videoInput=0,
                roiNames=["entrance1", "entrance2", "ROI1", "ROI2", "ROI3", "ROI4"],
                outputName="rois.csv"):
    
    cap = cv.VideoCapture(videoInput)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture one frame to know image size
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        cap.release()
        exit()

    # Convert frame for display
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # --- NEW: make the ROI window resizable ---
    cv.namedWindow('frame', cv.WINDOW_NORMAL)

    # Optionally: set an initial size (e.g. 1.2x original, or clamp to something)
    h, w = gray.shape[:2]
    scale = 1.2  # change this if you want it bigger/smaller by default
    cv.resizeWindow('frame', int(w * scale), int(h * scale))
    # ------------------------------------------

    # Dictionary to store ROIs
    rois = {}

    for entry in roiNames:
        print(f"Please select location of {entry}")

        # Display the current frame with already selected ROIs
        for name, roi in rois.items():
            x, y, rw, rh = roi
            cv.rectangle(gray, (x, y), (x + rw, y + rh), (255, 0, 0), 2)
            cv.putText(gray, name, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        
        # Show the frame and select ROI
        cv.imshow('frame', gray)
        rois[entry] = cv.selectROI('frame', gray,
                                   fromCenter=False, showCrosshair=True)

        # Draw the selected ROI on the frame
        x, y, rw, rh = rois[entry]
        cv.rectangle(gray, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        cv.putText(gray, entry, (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)


    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(rois)
    df.index = ["xstart", "ystart", "xlen", "ylen"]
    
    # Save the DataFrame to a CSV file
    df.to_csv(outputName, index=["xstart", "ystart", "xlen", "ylen"])
    
    # Release the capture and destroy windows
    cap.release()
    cv.destroyAllWindows()

    return df

#def draw_on_video(frame=gray,roi=(0,0,0,0)):
    #cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 0), -1)
def read_rois_from_csv(csvfile):
    with open(csvfile, newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        roi_data = {header: [] for header in headers}
        
        for row in reader:
            for header, value in zip(headers, row):
                roi_data[header].append(int(value))
                
    return roi_data
    

def grab_cut(frame,xstart,ystart,xlen,ylen):
    
    cut = frame[ystart:ystart+ylen,
                xstart:xstart+xlen]
    return cut


def choose_csv():
    root=tk.Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    # Print the file name to the console
    #print (filename)
    root.destroy()
    return filename

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Extract parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read frames and convert to numpy array
        frames = wav_file.readframes(n_frames)
        sound_data = np.frombuffer(frames, dtype=np.int16)
        
        # If stereo, reshape to two columns
        if n_channels == 2:
            sound_data = sound_data.reshape(-1, 2)
        
        return sound_data, framerate

def generate_voc_array(path_to_voc, sample_rate):
    """
    using resample_poly to downsample from fs_original → sample_rate.
    Returns a list [resampled_audio] if fs_original ≥ sample_rate, otherwise warns.
    """
    voc_sound_data = []
    voc, fs_original = sf.read(path_to_voc)  # voc: (n_samples,) or (n_samples, channels)

    if fs_original == sample_rate:
        voc_sound_data.append(voc)

    elif fs_original > sample_rate:
        from math import gcd
        up   = sample_rate
        down = fs_original
        g = gcd(up, down)
        up //= g
        down //= g

        voc_resampled = resample_poly(voc, up, down, axis=0)
        voc_sound_data.append(voc_resampled)

    else:
        raise ValueError(f"Recording fs={fs_original} Hz is below target {sample_rate} Hz.")

    return voc_sound_data[0]


def get_interval(interval_name):

    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
             "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
    intervals_values = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 64/45, 3/2, 8/5, 5/3, 16/9, 15/8, 2]
    intervals_values_strings= ["1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2", "8/5", "5/3", "16/9", "15/8", "2/1"]

    intervals = dict(zip(intervals_names, intervals_values))
    intervals_strings = dict(zip(intervals_names, intervals_values_strings))
    
    return intervals[interval_name], intervals_strings[interval_name]

def get_chord(chord):
    chords_dict= {
        "major": ["unison", "maj_3", "perf_5"],
        "minor": ["unison", "min_3", "perf_5"],
        "maj_6th": ["unison", "maj_3", "perf_5", "maj_6"],
        "min_6th": ["unison", "min_3", "perf_5", "maj_6"],
        "sus4": ["unison", "perf_4", "perf_5"],
        "dim": ["unison", "min_3", "tritone"],
        "aug": ["unison", "maj_3", "min_6"],
        "dominant_7th": ["unison", "maj_3", "perf_5", "min_7"],
        "min_7th": ["unison", "min_3", "perf_5", "min_7"],
        "maj_7th": ["unison", "maj_3", "perf_5", "maj_7"]
    }
    
    return chords_dict[chord]

def get_intervals_from_chord(root_frequency, chord):
    sequence_of_intervals = get_chord(chord)

    interval_values = []
    interval_strings = []
    frequencies = []
    for i in sequence_of_intervals: 
        interval_value, interval_string = get_interval(i)
        interval_values.append(interval_value)
        interval_strings.append(interval_string)
        frequencies.append(int(interval_value * root_frequency))

    return sequence_of_intervals, interval_strings, frequencies

def check_if_lower(pattern):
    lower = ["random", "RANDOM", "Random", "ran", "rdm", "Ran", "RDM", "silence", "SILENCE", "sil", "Sil", "Silence"]
    mixed = ["AoAo", "AOAO", "aoao", "A0A0", "a0a0"]
    
    if pattern in mixed: 
        pattern = "AoAo"
    else:
        if pattern in lower:
            pattern = pattern.lower()
        else:
            pattern = pattern.upper()
    return pattern

def ask_pattern(rois_number):
    pattern_list= []
    for i in range(1, rois_number+1):        
        pattern = input(f"insert pattern #{i}\n(e.g. AoAo, ABAB, ABCABC, ABCDABCD, random, silence, BABA, CBACBA)\n")
        pattern = check_if_lower(pattern)
        pattern_list.append(pattern)
    return (pattern_list)

def ask_pattern_chords(number_consonants, number_dissonants):
    consonant_patterns = []
    dissonant_patterns= []
    
    if number_consonants == number_dissonants: 
        for i in range(1, number_consonants+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)
            
    elif number_consonants >  number_dissonants : 
        difference= number_consonants -  number_dissonants
        same_n_rois = number_consonants - (difference)

        for i in range(1, same_n_rois+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)

        for i in range(1, difference+1):
            pattern = (input(f"insert the additional consonant pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)

    else: 
        difference=  number_dissonants - number_consonants
        same_n_rois =  number_dissonants - (difference)

        for i in range(1, same_n_rois+1):
            pattern = (input(f"insert pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            consonant_patterns.append(pattern)
            dissonant_patterns.append(pattern)

        for i in range(1, difference+1):
            pattern = (input(f"insert the additional dissonant pattern #{i}\n(ABAB, ABCABC, ABCDABCD, random, BABA, CBACBA):\n"))
            pattern = check_if_lower(pattern)
            dissonant_patterns.append(pattern)
            
    return consonant_patterns, dissonant_patterns

def ask_music_info_sequences(rois_number):

    intervals_vs_custom= input("would you like to add custom values or to generate sequences based on intervals? (custom/intervals)")

    if intervals_vs_custom == "custom":
        pattern_list = []
        
        #for each ROI, ask the user for the desired pattern
        ask_for_pattern_input = (input("would you like to make new patterns?\n (select y if you want to insert the patterns, n if you want to hard code the patterns)")).lower()
        
        if ask_for_pattern_input == "y":
            pattern_list= ask_pattern(rois_number)
        else:
            pattern_list = ['AAAAA','AoAo','ABAB','ABCABC', 'BABA', 'ABBA', "silence", "vocalisation"]   
        
        #to determine the individual events, exclude random and silence
        patterns_nonpatterns = ["silence", "vocalisation"]
        
        #get the sorted individual events in the sequences and map them to frequencies
        events = []
        freqs = []
        for i in sorted(pattern_list):
            if i not in patterns_nonpatterns: 
                for j in i:
                    if j not in events:
                        events.append(j)
                        #ask the user the frequency for the event
                        if j !='o':
                            freq = int(input(f"Insert frequency for sound {j}:\n"))
                            freqs.append(freq)
                        else:
                            freqs.append(0)

        
        #map frequency to event in a dictionary
        sound_dict = dict(zip(events,freqs))
        
        sequence_from_pattern= []
        repetitions = 50
        #generate the sequences of events from patterns 
        for index, item in enumerate(pattern_list):
            #print (index+1, item)
            if item not in patterns_nonpatterns: 
                #print (index, item)
                sequence_from_pattern.append(list(item*repetitions))
            elif item == "random": 
                sequence_from_pattern.append([random.choice(events) for _ in range(200)])
            elif item == "vocalisation":
                sequence_from_pattern.append("vocalisation")
            else:
                sequence_from_pattern.append(['o' for _ in range(200) ])
            
    #print(sequence_from_pattern)
        # handle vocalisations as a sentinel, otherwise map each event char
        sequence_of_frequencies = []
        for sublist in sequence_from_pattern:
            if sublist == "vocalisation":
                sequence_of_frequencies.append("vocalisation")
            else:
                sequence_of_frequencies.append([sound_dict[char] for char in sublist])

    elif intervals_vs_custom == "intervals":
        pass

    #
    #
    #
    #
    #
    #
    #
    #
    #
    ###continue with the intervals patterns one #### 

    return sequence_of_frequencies, pattern_list
    

def ask_music_info_simple_sounds(rois_number):
    frequency = []

    for i in range(1, rois_number+1): 
        freqs = int(input(f"Insert frequency for sound #{i}\n"))
        frequency.append(freqs)

    return frequency

def ask_info_intervals(rois_number):
    
    # create a dictionary with interval names and values
        
    #define intervals to get consonant and dissonant intervals
    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
                       "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]
    
    consonant_intervals = [intervals_names[i] for i in (3,4,5,7,8,9,12)]
    dissonant_intervals = [intervals_names[i] for i in (1,2,6,10,11)]
    
    

    usable_rois = rois_number - 2
    #one is going to be unison, the other one is going to be silence
    rois_per_nance= int(usable_rois/2)
    
    number_consonants = rois_per_nance
    number_dissonants = rois_per_nance
    
    if rois_number %2 !=0:
        
        choice_odd_roi = (input('''you have an odd number of ROIs.\nWould you like to have an extra consonant or dissonant ROI?\n''')).lower()
        if choice_odd_roi == "consonant":
            number_consonants +=1
            
        elif choice_odd_roi == "dissonant":
            number_dissonants +=1

            
    tonal_centre = int(input("insert the frequency that will be the tonal centre:\n"))
    tonal_centre_interval, tonal_centre_string = get_interval("unison")
    
    #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
    frequencies =[[tonal_centre, int(tonal_centre*tonal_centre_interval)]]
    interval_numerical_list = [tonal_centre_string]
    interval_string_names = ["unison"]
    
    
    
    for i in range(number_consonants):
        consonant_choice= input(f"insert the consonant interval of choice #{i+1} {consonant_intervals}:\n")
        consonant_choice= consonant_choice.lower()
        interval, interval_as_string = get_interval(consonant_choice)
        frequencies.append([tonal_centre, int(tonal_centre*interval)])
        interval_numerical_list.append(interval_as_string)
        interval_string_names.append(consonant_choice)
    
    for i in range(number_dissonants):
        dissonant_choice= input(f"insert the dissonant interval of choice #{i+1} {dissonant_intervals}:\n")
        dissonant_choice= dissonant_choice.lower()
        interval, interval_as_string = get_interval(dissonant_choice)
        frequencies.append([tonal_centre, int(tonal_centre*interval)])
        interval_numerical_list.append(interval_as_string)
        interval_string_names.append(dissonant_choice)
        
        #ask user if there will be a vocalisation.
    vocalisation = (input("want to select a .wav file containing a sound vocalisation?(y/n)\nif n, one of the ROIs will be silent \n")).lower()
    
    if vocalisation == "y":
        frequencies.append("vocalisation")
        interval_numerical_list.append([9]) # arbitrary number so that I can identify this later on
        interval_string_names.append("vocalisation")

    else:
        frequencies.append([0,0])
        interval_numerical_list.append(["0"])
        interval_string_names.append("no_interval")
        
        
    return frequencies, interval_numerical_list, interval_string_names

#hard code the freq and intervals list not to manually be prompted every time
def info_intervals_hc(rois_number, tonal_centre, intervals_list):
    
    #usable_rois exclude the unison and silent arm
    usable_rois = rois_number - 2

    tonal_centre_interval, tonal_centre_string = get_interval("unison")
    #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
    frequencies =[[tonal_centre, int(tonal_centre*tonal_centre_interval)]]
    interval_numerical_list = [tonal_centre_string]
    interval_string_names = ["unison"]


    if len(intervals_list) == usable_rois:

        for i in range(usable_rois):

    #check if we want a vocalisation as one of the inputs, if not, continues as normal
            if i != "vocalisation":
                interval, interval_as_string = get_interval(intervals_list[i])
                frequencies.append([tonal_centre, int(tonal_centre*interval)])
                interval_numerical_list.append(interval_as_string)
                interval_string_names.append(intervals_list[i])

    # if there is a vocalisation, it retrieves the path from the main script and checks if the sample rate is correct
            else: 
                frequencies.append("vocalisation")
                interval_numerical_list.append([9]) # arbitrary number so that I can identify this later on
                interval_string_names.append("vocalisation")


    # appends the silent frequency
        frequencies.append([0,0])
        interval_numerical_list.append(["0"])
        interval_string_names.append("no_interval")
    else:
        print("please check that the number of intervals is rois_number - 2")
     
    return frequencies, interval_numerical_list, interval_string_names



def info_complex_intervals_hc (rois_number, controls, tonal_centre, smooth_freq, rough_freq, consonant_intervals, dissonant_intervals, path_to_voc):
    

    all_intervals = consonant_intervals + dissonant_intervals
    
    #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
    frequencies =[]
    interval_numerical_list = []
    interval_string_names = []
    sound_type = []
    sounds_arrays = []

    for i in controls:
        interval_numerical_list.append(0)
        interval_string_names.append(i)          
        sound_type.append(i)

        if i == "silent":
            frequencies.append(0)
            # 10 s of silence at 192 kHz
            z = np.zeros(int(192000 * 10))
            sounds_arrays.append([z, z])
        else: 
            frequencies.append(i)
            voc = generate_voc_array(path_to_voc, 192000)
            silence = np.zeros_like(voc)
            sounds_arrays.append([voc, silence])

    if smooth_freq:
        tonal_centre_interval, tonal_centre_string = get_interval("unison") 
        frequencies.append([tonal_centre, int(tonal_centre*tonal_centre_interval)])
        interval_numerical_list.append(tonal_centre_string)
        interval_string_names.append("unison")
        sound_data_1= generate_sound_data(tonal_centre)
        sound_data_2= generate_sound_data(int(tonal_centre*tonal_centre_interval))

        sound_type.append("smooth")
        sounds_arrays.append([sound_data_1, sound_data_2])

    if rough_freq:

        tonal_centre_interval, tonal_centre_string = get_interval("unison")

        t_1, sound_data_1 = generate_sound_data(tonal_centre, give_t = True)
        t_2, sound_data_2 = generate_sound_data(int(tonal_centre*tonal_centre_interval), give_t = True)
        
        frequencies.append([tonal_centre, int(tonal_centre*tonal_centre_interval)])
        interval_numerical_list.append(tonal_centre_string)
        interval_string_names.append("unison")
        sound_type.append("rough")
        modulated_wave_1 = apply_constant_sinusoidal_envelope(t_1, sound_data_1)
        modulated_wave_2 = apply_constant_sinusoidal_envelope(t_2, sound_data_2)
        sounds_arrays.append([modulated_wave_1, modulated_wave_2])

    for i in all_intervals: 
        interval, interval_string = get_interval(i)
        freq_1 = tonal_centre
        freq_2 = tonal_centre*interval

        frequencies.append([freq_1, freq_2])
        interval_numerical_list.append(interval_string)
        interval_string_names.append(i)

        sound_1= generate_sound_data(tonal_centre)
        sound_2 = generate_sound_data(freq_2)
        
        sounds_arrays.append([sound_1, sound_2])

        if i in consonant_intervals: 
            sound_type.append("consonant")
        else:
            sound_type.append("dissonant")
        


    return frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays

def vocalisations_info_hc(rois_number, stimuli, file_names):

    #dict_files = dict(zip(stimuli, file_names))

    #here what we do is we check if the number of stimuli corresponds to the number of ROIs and adjusting the stimuli accordingly
    if rois_number == len(stimuli):
        pass
    # if there are more rois than stimuli, select random stimuli to repeat
    elif rois_number > len(stimuli): 

        difference = rois_number - len(stimuli)
        for i in range(difference):
            stimuli.append(random.choice(stimuli))
    else:
        stimuli = stimuli[:rois_number]


    #the frequencies list will contain lists containing the 2 frequencies that make up the interval. 
    frequencies =[]
    interval_numerical_list = []
    interval_string_names = []
    sound_type = []
    sounds_arrays = []

    for i in stimuli:
        interval_numerical_list.append(0)
        interval_string_names.append("none")          
        sound_type.append("control")

        if i == "silent":
            frequencies.append(0)
            sounds_arrays.append([0,0])
        else: 
            frequencies.append(i)
            voc_sound_array = generate_voc_array(i, 192000)
            sounds_arrays.append([voc_sound_array, 0])

    
    return frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays
    
def plotting_lissajous(interval):
    t = symbols('t')
    
    intervals_names = ["unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone", 
             "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave"]

    intervals_values = ["1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2", "8/5", "5/3", "16/9", "15/8", "2/1"]


    intervals = dict(zip(intervals_names, intervals_values))
    
    if len(intervals[interval])==5: 
        a= int(intervals[interval][0:2])
        b= int(intervals[interval][3:5])
    elif len(intervals[interval])==3:
        a= int(intervals[interval][0])
        b= int(intervals[interval][2])
        
    
    
    #s = 0 because the interval (a,b) plays at the same time
    s = 0 #pi/2


    title = f'a = {a}, b = {b}'
    x = sin(a*t + s)
    y = sin(b*t)
    p = plot_parametric(x, y, (t, 0, 2*pi), title=title)
    
    p.save(f"{interval}_plot.png")    

def compute_gain(frequency_hz):

    data = pd.read_csv("C:/Users/labuser/Documents/GitHub/mice-maze/code/auditory/speaker calibration scripts/frequency_response_speaker.csv")
    # interpolation function from the frequency response data extrapolated from the graph on the website
    frequency = data['Frequency_kHz'].values * 1000  # Convert kHz to Hz
    attenuation = data['Attenuation_dB'].values
    interp_func = interp1d(frequency, attenuation, kind='cubic', fill_value="extrapolate")
    #get the interpolated attenuation
    attenuation_db = interp_func(frequency_hz)
    #from there calculate the gain 
    gain = 10 ** (-attenuation_db / 20)
    return gain


    

 

def generate_sound_data(frequency, waveform="sine", duration=10, fs=192000, volume=1.0, ramp_duration = 0.01, compensate=True, give_t= False):
    """Generate sound data for a given frequency and waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint = False)
    gain = compute_gain(frequency) if compensate else 1.0
    adjusted_volume = volume * gain
    sound = np.zeros_like(t)    


    if waveform == 'sine':
        sound = np.sin(frequency * t * 2 * np.pi)  * adjusted_volume

    elif waveform == 'square':
        sound = np.sign(np.sin(frequency * t * 2 * np.pi)) * adjusted_volume

    elif waveform == 'sawtooth':
        sound = (2 * (t * frequency % 1) - 1) * adjusted_volume

    elif waveform == 'triangle':
        sound = (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1) * adjusted_volume

    elif waveform == 'pulse wave':
        sound = np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0)  * adjusted_volume  # 0.5 in this case is the duty cycle
    
    elif waveform == 'white noise':
        samples = int(fs * duration)
        sound = np.random.uniform(low=-1.0, high=1.0, size=samples) * adjusted_volume

    
    # apply a ramp up (fade in) in seconds so that the speaker doesn't click upon stimulus presentation
    ramp_samples = int(fs * ramp_duration)

    envelope = np.ones_like(sound)
    if ramp_samples > 0:
        # Linear ramp up
        envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
        # Linear ramp down
        # envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)

    sound *= envelope

    if give_t:
        return t, sound
    else:
        return sound


def apply_constant_sinusoidal_envelope(t, sound, mod_freq=50, depth=0.5):
    """
    Apply sinusoidal amplitude modulation to a waveform.
    
    Parameters:
        t (np.ndarray): Time array (same as from generate_sound_data)
        sound (np.ndarray): Sound waveform
        mod_freq (float): Frequency of amplitude modulation in Hz (e.g., 50 Hz)
        depth (float): Modulation depth (0 to 1)
        
    Returns:
        modulated_sound (np.ndarray): Sound with applied amplitude modulation
    """
    envelope = 1 + depth * np.sin(2 * np.pi * mod_freq * t)
    modulated_sound = sound * envelope
    return modulated_sound

def apply_segmented_am_modulation(
    t,
    sound,
    mod_freqs,
    segment_duration = 0.2,
    am_depth=0.5,
    fs=192000,
    normalize=True
):
    """
    Apply sequential sinusoidal amplitude modulation segments to an existing waveform.
    
    Parameters:
        t (np.ndarray): Time vector
        sound (np.ndarray): Input waveform
        mod_freqs (list): List of modulation frequencies (Hz)
        segment_duration (float): Duration of each modulation segment (s)
        am_depth (float): Depth of amplitude modulation (0–1)
        fs (int): Sampling rate
        normalize (bool): Whether to normalize to [-1, 1] after modulation
        
    Returns:
        modulated_sound (np.ndarray): The resulting waveform
    """
    n_samples = len(t)
    segment_samples = int(segment_duration * fs)
    num_segments = int(np.ceil(n_samples / segment_samples))
    
    # Repeat modulation frequencies if needed
    mod_freqs_full = (mod_freqs * ((num_segments // len(mod_freqs)) + 1))[:num_segments]
    
    # Copy the original sound to modulate
    modulated_sound = np.copy(sound)
    
    # Apply modulation segment by segment
    for i in range(num_segments):
        start = i * segment_samples
        end = min((i + 1) * segment_samples, n_samples)
        f_mod = mod_freqs_full[i]
        envelope = 1 + am_depth * np.sin(2 * np.pi * f_mod * t[start:end])
        modulated_sound[start:end] *= envelope

    # Normalize to [-1, 1] if needed
    if normalize and np.max(np.abs(modulated_sound)) > 1.0:
        modulated_sound /= np.max(np.abs(modulated_sound))

    return modulated_sound


def info_temporal_modulation_hc(rois_number, 
                                controls, 
                                smooth_freqs, 
                                constant_rough_freqs, 
                                complex_rough_freqs, 
                                constant_rough_modulation= 50, 
                                complex_rough_mod = [30, 50, 70], 
                                path_to_voc = None):
    # if you don't want controls, leave blank, but ensure that len(frequencies) is the same as rois_number. Obviously. 
    freqs = controls + smooth_freqs + constant_rough_freqs + complex_rough_freqs

    frequencies = []
    temporal_modulation = []
    sound_type = []
    sounds_arrays = []

    #here is a double check in case you're a bit distracted. Again, no judgement. 
    if len(freqs) != rois_number:
        print("Bestie, double check the number of stimuli and make sure they match the number of rois")

    else: 
        for i in controls:
            if i == "silent":
                sound_data = generate_sound_data(0)
                frequencies.append("silent_arm")
                temporal_modulation.append("no_stimulus")
                sound_type.append("control")
                sounds_arrays.append(sound_data)
            else: 
                frequencies.append("vocalisation")
                temporal_modulation.append("vocalisation")
                sound_type.append("control")
                voc_sound_array = generate_voc_array(path_to_voc, 192000)
                sounds_arrays.append(voc_sound_array)

        for i in smooth_freqs:
            sound_data = generate_sound_data(i)
            frequencies.append(i)
            temporal_modulation.append("none")
            sound_type.append("smooth")
            sounds_arrays.append(sound_data)

        for i in constant_rough_freqs:
            t, sound_data = generate_sound_data(i, give_t = True)
            frequencies.append(i)
            temporal_modulation.append(constant_rough_modulation)
            sound_type.append("rough")
            modulated_wave = apply_constant_sinusoidal_envelope(t, sound_data, mod_freq=constant_rough_modulation)
            sounds_arrays.append(modulated_wave)
            
        for i in complex_rough_freqs:
            t, sound_data = generate_sound_data(i, give_t = True)
            frequencies.append(i)
            temporal_modulation.append(complex_rough_mod)
            sound_type.append("rough_complex")
            modulated_wave = apply_segmented_am_modulation(t, sound_data, mod_freqs = complex_rough_mod)
            sounds_arrays.append(modulated_wave)


    return frequencies, temporal_modulation, sound_type, sounds_arrays

def get_trial_tuple(frequency, volume, waveform):
    return tuple(zip(frequency, volume, waveform))

def shuffle_data(frequency, volume, waveform):
    combined = list(zip(frequency, volume, waveform))
    random.shuffle(combined)
    return zip(*combined)

def create_simple_trials(rois, frequency,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):

    
    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialise lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    # volume_final = []
    # waveform_final = []
    wave_arrays = []
    repetition_numbers = []

    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    

    # Create trials
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                # volume_final.append(0)
                # waveform_final.append('none')
                temp_array = np.zeros(sample_rate*10)
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_tuple = tuple(frequency)  #get_trial_tuple(frequency, volume, waveform)
                else:  # Other trials should be unique and shuffled 
                    trial_list = list(frequency)
                    random.shuffle(trial_list)
                    trial_tuple = tuple(trial_list)
                    


                if trial_tuple not in previous_trials:
                    previous_trials.add(trial_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            # volume_final.append(volume[j])
                            # waveform_final.append(waveform[j])
                            wave_arrays.append(generate_sound_data(frequency[j]))
                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(trial_tuple[j])
                            # volume_final.append(shuffled_volume[j])
                            # waveform_final.append(shuffled_waveform[j])
                            wave_arrays.append(generate_sound_data(trial_tuple[j]))
                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "frequency": frequency_final,
        # "volume": volume_final,
        # "waveform": waveform_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["time_in_maze_ms"] = [0] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    # df["trial_time_in_maze_ms"] = [0] * len(df)
    

    return df,wave_arrays

def create_trials_for_sequences(rois, frequency, patterns, volume=100, waveform="sine", path_to_voc = None,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):
    
    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialise lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    wave_arrays = []
    repetition_numbers = []
    patterns_final= []

    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    # Create trials
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                patterns_final.append(0)
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_list = list(zip(frequency, patterns))

                else:  # Other trials should be unique and shuffled
                    trial_list = list(zip(frequency, patterns))
                    random.shuffle(trial_list)


                # Convert trial_list to a tuple of tuples ##add wavefrom as a tuple
                trial_tuple_as_tuple = tuple((tuple(freq), tuple(pat)) for freq, pat in trial_list)
                #print(type(trial_tuple_as_tuple))

                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            patterns_final.append(patterns[j])

                            #to generate the sounds data in a way that they play continuously
                            concatenated_array = []
                            if frequency[j] =="vocalisation":
                                sound = generate_voc_array(path_to_voc, 192000)
                                
                                wave_arrays.append(np.asarray(sound))

                            else:
                                for k in range(len(frequency[j])):
                                    sounds = generate_sound_data(frequency[j][k], duration = 0.04)  # Generate sound data
                                    concatenated_array.append(sounds)
                                concatenated_sounds= np.concatenate(concatenated_array)
                                wave_arrays.append(concatenated_sounds)


                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            freq, pat = trial_list[j]
                            frequency_final.append(freq)
                            patterns_final.append(pat)
                            concatenated_array = []
                            
                            if freq =="vocalisation":
                                sound = generate_voc_array(path_to_voc, 192000)
                                wave_arrays.append(np.asarray(sound))

                            else:

                                for k in range(len(freq)):
                                    sounds = generate_sound_data(freq[k], duration=0.04)  # Generate sound data
                                    #print(trial_list[j][k])
                                    concatenated_array.append(sounds)
                                concatenated_sounds= np.concatenate(concatenated_array)
                                wave_arrays.append(concatenated_sounds)



                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "pattern": patterns_final,
        "frequency": frequency_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["time_in_maze_ms"] = [0] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    # df["trial_time_in_maze_ms"] = [0] * len(df)

    return df, wave_arrays

def create_trials_for_intervals(rois, frequency, intervals, intervals_names,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate=192000):

    # Create a list of ROIs repeated total_repetitions times
    rois_repeated = rois * total_repetitions

    # Initialise lists to store the shuffled frequency, volume, and waveform
    frequency_final = []
    intervals_final = []
    intervals_names_final = []
    wave_arrays = []
    repetition_numbers = []


    # Total number of sound data sets
    num_sounds = len(frequency)
    previous_trials = set()

    #generate the sounds and create an array containing 4 lists, each containing 2 arrays containing each sound
    dual_array_sounds = []
    for i in frequency: 

        frequencies_sound_data = []
        #run the loop for the first 3 items that definitely contain a frequency
        if i[1] != 0:
            for j in i:
                sound = generate_sound_data(j)
                frequencies_sound_data.append([sound])

    #     #now the fourth item always has i[1] == 0
        if i[1] == 0 :
            if i[0] == 0:
                #if the first item of the 4th array is 0, they are both 0s and we can proceed 
                for j in i:
                        sound = generate_sound_data(j)
                        frequencies_sound_data.append([sound])

            #if in the 4th item, the first element of the array is not 0, we need to convert the .wav into a sound data array
            else:
                sound_data, framerate = read_wav_file(i[0])
                sound1 = generate_sound_data(sound_data, framerate)
                frequencies_sound_data.append([sound1])
                sound2 = generate_sound_data(i[1])
                frequencies_sound_data.append([sound2])

        dual_array_sounds.append(frequencies_sound_data)

    # Create trials
    #first trial is going to be silent 
    #i is the trial_ID
    for i in range(total_repetitions):
        if i % 2 == 0:  # Alternate zero data repetitions
            for j in range(len(rois)):
                repetition_numbers.append(i + 1)  # Add repetition number
                frequency_final.append(0)
                intervals_final.append(0)
                intervals_names_final.append(0)
                wave_arrays.append(np.zeros(sample_rate*10))  # Assuming 10 seconds of silence
        else:
            while True:
                if i == 1:  # Trial 2 should use the initial list
                    trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))

                    #print(f"trial 1: {trial_list}")


                else:  # Other trials should be unique and shuffled
                    trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))
                    random.shuffle(trial_list)
                    #print(f"trial {i}: {trial_list}")



                # Convert trial_list to a tuple of tuples ##add wavefrom as a tuple
                trial_tuple_as_tuple = tuple(item[2] for item in trial_list)
                #print(trial_tuple_as_tuple)

                #trying to convert the whole list of freq, int, wave requires too much, try to see if just affing the freq works

                #try to use nested lists instead
                #print(type(trial_list))
                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)
                    if i == 1:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            frequency_final.append(frequency[j])
                            intervals_final.append(intervals[j])
                            intervals_names_final.append(intervals_names[j])
                            wave_arrays.append(tuple(dual_array_sounds[j]))


                    else:
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)  # Add repetition number
                            freq, inter, inter_name, wave = trial_list[j]
                            frequency_final.append(freq)
                            intervals_final.append(inter)
                            intervals_names_final.append(inter_name)
                            wave_arrays.append(tuple(wave))



                    break

    # Create the DataFrame with the repetition numbers, repeated ROIs, and final data
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "interval": intervals_names_final,
        "interval_ratio": intervals_final,
        "frequency": frequency_final,
        "wave_arrays": wave_arrays
    })

    # Add other necessary columns filled with NaNs or default values
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["time_in_maze_ms"] = [0] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    # df["trial_time_in_maze_ms"] = [0] * len(df)

    return df, wave_arrays


def make_another_tuple(frequency, temporal_modulation, sound_type, sounds_array):
    return tuple(zip(frequency, temporal_modulation, sound_type, sounds_array))

def shuffle_another_data(frequency, temporal_modulation, sound_type, sounds_array):
    combined = list(zip(frequency, temporal_modulation, sound_type, sounds_array))
    random.shuffle(combined)
    # unpack each component
    freq, mod, typ, snd = zip(*combined)
    return freq, mod, typ, snd

def _make_hashable(x):
    """Convert lists to tuples, leave other types unchanged."""
    if isinstance(x, list):
        return tuple(x)
    return x


def create_temporally_modulated_trials(rois, frequency, temporal_modulation, sound_type, sounds_arrays,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):

    """
    Generate trials for temporally modulated sounds, ensuring unique shuffles.
    """

    # Repeat the list of ROIs for each repetition
    rois_repeated = rois * total_repetitions

    # Lists to collect final trial data
    frequency_final = []
    temporal_modulations_final = []
    sound_type_final = []
    wave_arrays = []
    repetition_numbers = []

    previous_trials = set()



    for i in range(total_repetitions):
        if i % 2 == 0:
            # === Silent‐trial sections ===
            for _ in rois:
                repetition_numbers.append(i + 1)
                frequency_final.append(0)
                temporal_modulations_final.append("none")
                sound_type_final.append("silent_trial")
                wave_arrays.append(np.zeros(sample_rate * 10))
        else:
            # === Non‐silent trials: either first ordering (i == 1) or shuffled (i > 1) ===
            while True:
                if i == 1:
                    # First non‐silent repetition: use the ORIGINAL order
                    trial_triples = []
                    for idx in range(len(rois)):
                        freq = frequency[idx]
                        # Convert any inner list (e.g. [30,50,70]) into a tuple for hashing
                        mod = _make_hashable(temporal_modulation[idx])
                        typ = sound_type[idx]
                        trial_triples.append((freq, mod, typ))
                    trial_tuple_as_tuple = tuple(trial_triples)

                    # Also prepare a “parallel” list of quadruples to pull sound arrays later
                    trial_list = list(zip(frequency, temporal_modulation, sound_type, sounds_arrays))

                else:
                    # Subsequent repetitions: fully shuffle (freq, mod, type, wave) together
                    combined = list(zip(frequency, temporal_modulation, sound_type, sounds_arrays))
                    random.shuffle(combined)

                    # Build a hashable tuple using only (freq, mod, type)
                    trial_triples = []
                    for (freq, mod, typ, snd) in combined:
                        trial_triples.append((freq, _make_hashable(mod), typ))
                    trial_tuple_as_tuple = tuple(trial_triples)

                    # Keep the full quadruples around to assign replayed order
                    trial_list = combined

                # Once we have a trial_tuple_as_tuple, check uniqueness
                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)

                    if i == 1:
                        # Assign original data (no shuffle)
                        for idx in range(len(rois)):
                            repetition_numbers.append(i + 1)
                            frequency_final.append(frequency[idx])
                            temporal_modulations_final.append(temporal_modulation[idx])
                            sound_type_final.append(sound_type[idx])
                            wave_arrays.append(sounds_arrays[idx])
                    else:
                        # Assign shuffled data from trial_list
                        for (freq_shuf, mod_shuf, typ_shuf, sounds_shuf) in trial_list:
                            repetition_numbers.append(i + 1)
                            frequency_final.append(freq_shuf)
                            temporal_modulations_final.append(mod_shuf)
                            sound_type_final.append(typ_shuf)
                            wave_arrays.append(sounds_shuf)
                    break

    # Build the DataFrame
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "frequency": frequency_final,
        "sound_type": sound_type_final,
        "temporal_modulation": temporal_modulations_final,
        "wave_arrays": wave_arrays
    })

    # Add extra columns for tracking mouse behaviour
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["time_in_maze_ms"] = [0] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    # df["trial_time_in_maze_ms"] = [0] * len(df)

    return df, wave_arrays



def create_complex_intervals_trials(rois, frequency,interval_numerical_list, interval_string_names, sound_type, sounds_arrays,
                  total_repetitions = 9,
                  zero_repetitions = 5,
                  sample_rate = 192000):

    """
    Generate trials for temporally modulated sounds, ensuring unique shuffles.
    """

    # Repeat the list of ROIs for each repetition
    rois_repeated = rois * total_repetitions

    # Lists to collect final trial data
    frequency_final = []
    interval_numerical_list_final = []
    interval_string_names_final= []
    sound_type_final = []
    wave_arrays = []
    repetition_numbers = []

    previous_trials = set()



    for i in range(total_repetitions):
        if i % 2 == 0:
            # === Silent‐trial sections ===
            for _ in rois:
                repetition_numbers.append(i + 1)
                frequency_final.append(0)
                interval_numerical_list_final.append(0)
                interval_string_names_final.append(0)
                sound_type_final.append("silent_trial")
                # two silent intervals of full length
                zeros = np.zeros(sample_rate * 10)
                wave_arrays.append((0,0))
        else:
            # === Non‐silent trials: either first ordering (i == 1) or shuffled (i > 1) ===
            while True:
                if i == 1:
                    # First non‐silent repetition: use the ORIGINAL order
                    trial_triples = []
                    for idx in range(len(rois)):
                        # _make_hashable Converts any inner list (e.g. [30,50,70]) into a tuple for hashing
                        freq = _make_hashable(frequency[idx])
                        int_num = interval_numerical_list[idx]
                        int_name = interval_string_names[idx]
                        typ = sound_type[idx]
                        trial_triples.append((freq, int_num, int_name, typ))
                    trial_tuple_as_tuple = tuple(trial_triples)

                    # Also prepare a “parallel” list of quadruples to pull sound arrays later
                    trial_list = list(zip(frequency, interval_numerical_list, interval_string_names, sound_type, sounds_arrays))

                else:
                    # Subsequent repetitions: fully shuffle (freq, mod, type, wave) together
                    combined = list(zip(frequency, interval_numerical_list, interval_string_names, sound_type, sounds_arrays))
                    random.shuffle(combined)

                    # Build a hashable tuple using only (freq, mod, type)
                    trial_triples = []
                    for (freq, int_num, int_name, typ, snd) in combined:
                        trial_triples.append((_make_hashable(freq),int_num, int_name, typ))
                    trial_tuple_as_tuple = tuple(trial_triples)

                    # Keep the full quadruples around to assign replayed order
                    trial_list = combined

                # Once we have a trial_tuple_as_tuple, check uniqueness
                if trial_tuple_as_tuple not in previous_trials:
                    previous_trials.add(trial_tuple_as_tuple)

                    if i == 1:
                        # Assign original data (no shuffle)
                        for idx in range(len(rois)):
                            repetition_numbers.append(i + 1)
                            frequency_final.append(frequency[idx])
                            interval_numerical_list_final.append(interval_numerical_list[idx])
                            interval_string_names_final.append(interval_string_names[idx])
                            
                            sound_type_final.append(sound_type[idx])
                            wave_arrays.append(tuple(sounds_arrays[idx]))
                    else:
                        # Assign shuffled data from trial_list
                        for (freq_shuf, int_num_shuf, int_name_shuf, typ_shuf, sounds_shuf) in trial_list:
                            repetition_numbers.append(i + 1)
                            frequency_final.append(freq_shuf)
                            interval_numerical_list_final.append(int_num_shuf)
                            interval_string_names_final.append(int_name_shuf)
                            sound_type_final.append(typ_shuf)
                            wave_arrays.append(tuple(sounds_shuf))
                    break

    # Build the DataFrame
    df = pd.DataFrame({
        "trial_ID": repetition_numbers,
        "ROIs": rois_repeated,
        "frequency": frequency_final,
        "interval_type": sound_type_final,
        "interval_ratio": interval_numerical_list_final,
        "interval_name": interval_string_names_final,
        "wave_arrays": wave_arrays
    })

    # Add extra columns for tracking mouse behaviour
    df["time_spent"] = [None] * len(df)
    df["visitation_count"] = [None] * len(df)
    df["time_in_maze_ms"] = [0] * len(df)
    df["trial_start_time"] = [None] * len(df)
    df["end_trial_time"] = [None] * len(df)
    # df["trial_time_in_maze_ms"] = [0] * len(df)

    return df, wave_arrays





def play_sound(sound_data, fs=192000):
    """Play sound using sounddevice library."""
        # down-mix multi-channel to mono
    print("play_sound got:", type(sound_data), 
    "  content type:", 
    type(sound_data[0]) if hasattr(sound_data, "__getitem__") else None)
     # ensure we have an ndarray
    sound_data = np.asarray(sound_data)

    # now .ndim is safe
    if sound_data.ndim > 1:
        # down-mix to mono
        sound_data = np.mean(sound_data, axis=1)

    sd.play(sound_data, fs)
    

def play_interval(sound_data1, sound_data2, fs=192000):
    """Play two sounds simultaneously—but treat any int as “all silence.”"""
    import numpy as np

    # If both are integers → nothing to play
    if isinstance(sound_data1, (int, float)) and isinstance(sound_data2, (int, float)):
        return

    # If one side is integer, replace with zeros array of the other side’s length
    if isinstance(sound_data1, (int, float)):
        sound_data1 = np.zeros_like(sound_data2)
    if isinstance(sound_data2, (int, float)):
        sound_data2 = np.zeros_like(sound_data1)

    # Now both are array‐like: trim to same length
    min_length = min(len(sound_data1), len(sound_data2))
    sound_data1 = sound_data1[:min_length]
    sound_data2 = sound_data2[:min_length]

    combined_sound_data = sound_data1 + sound_data2
    combined_sound_data_normalised = np.int16(
        (combined_sound_data / np.max(np.abs(combined_sound_data))) * 32767
    )
    sd.play(combined_sound_data_normalised, fs)



def save_sound(sound_data, frequency, waveform):
    parent_folder_selected = filedialog.askdirectory()

    if parent_folder_selected:
        local_time = time.localtime()
        date_time = time.strftime('%d-%m-%Y_%H', local_time)  # adding hour in case of trial/error

        # Create a new subdirectory within the selected folder that takes the day
        subfolder_path = os.path.join(parent_folder_selected, date_time)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Use the frequency as the name for the sound
        file_name = f"{frequency}.wav"

        # Save in the folder
        full_file_path = os.path.join(subfolder_path, file_name)
        save_sound_to_file(sound_data, full_file_path)

def save_sound_to_file(sound_data, file_path):
    '''Modify this if need different audio formats'''
    # Normalize sound_data to 16-bit PCM format for WAV file
    sound_data_normalized = np.int16((sound_data / np.max(np.abs(sound_data))) * 32767)
    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(44100)  # Sampling rate
        wf.writeframes(sound_data_normalized.tobytes())


def select_sound_folder(): 
    root = tk.Tk()
    sound_folder = filedialog.askdirectory(title="Select Folder Containing Sound Files")
    if sound_folder:
        root.mainloop()  # This will not do anything visible since the window is withdrawn
    else:
        root.destroy()
    
    return sound_folder



def time_in_millis():
    millis=round(time.time() * 1000)
    return millis

def write_data(file_name="tests.csv",mode="a",data=["test","test","test"]):
    data_fh = open(file_name,mode)
    data_writer = csv.writer(data_fh, delimiter=',')
    data_writer.writerow(data)
    data_fh.close()
