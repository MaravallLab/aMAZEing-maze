import numpy as np
import cv2 as cv
import pandas as pd
import supfun_sequences as sf
import time
#from serial import Serial
#import serial
import os
import csv
import sounddevice as sd
import soundfile as sf1
from tkinter import filedialog as fd
from tkinter import *
import serial

#set the default sample rate and channel for the sound card
samplerate = 192000
sd.default.samplerate = samplerate
#python -m sounddevice
sd.default.device = 3

#vol needs to be around 50dB


# Variables

# set to True for shorter time intervals to test that the sounds work fine
testing= True
#testing with longer silence in the middle of the sequence
longer_silence= False

# if this is the sort of situation where we require an arduino, either for ttls or reward delivery, set to true
microcontroller = False
arduino = None

# if the home cage is not connected, we are going to set this to true so that we will skip having the mouse enter the maze to start a new trial
no_homecage = False

# in case the homecage is connected but we still want the trials to start and end according to the "clock", set this to True
forced_timing= True

pause_between_frames = False

#set number of ROIs 
rois_number = 8

#set to true to create/modify ROIs .csv file
drawRois = False

#set to true to make individual sine sounds
make_simple_smooth_sounds = False
#set true to make individual complex sounds. This contains 2 intervals, 1 vocalisation and one pure tone
make_Simple_intervals= False

# set to true if you want to perform experiments testing the effects of temporal envelope modulation on sound preference
# w1_d1c
make_temporal_envelope_modulation = True

#w1_d2-d4
# set to true to perform experiments where the ROIS can be controls / frequencies of different AM/ intervals 
make_complex_intervals =  False
w1day2= False
w1day3= False
w1day4= False
another_day = False
#w2_d1
# set to true to make sequences of tones
make_sequences = False

#w2_d2
just_vocalisations = False

#If we are recording a video, this needs to be true and videoInput needs to be set to 0 (or 1, depending on the camera)
recordVideo = True
videoInput = 0
#"C:/Users/labadmin/Desktop/auditory_maze_experiments/maze_recordings/maze_recordings_sequences/time_2024-09-25_11_47_12mouse6705/mouse6705_time_2024-09-25_11_47_12.mp4"
#"C:/Users/aleja/OneDrive/Desktop/maze_experiments/maze_recordings/2024-08-15_16_12_305872/5872_2024-08-15_16_12_30.mp4"
#"C:/Users/aleja/Downloads/maze_test.mp4"


# setup arduino 
if microcontroller:
    try:
        # check if port is "com3" or "com5" and also in the arduino IDE, check that the baud rate is correct
        arduino = serial.Serial('COM4', 115200, timeout=0.1)
        time.sleep(2) # Wait for reboot
        print("Arduino Connected (State Machine Mode)")
    except Exception as e:
        print(f"Arduino Failed: {e}")
        arduino = None
    
# State Machine Variables
ttl_is_active = False
ttl_scheduled_off_time = 0




# Setup -- ask users for mouse info
date_time = sf.get_current_time_formatted()
date_time= f"time_{date_time}"
animal_ID, = sf.get_user_inputs()
animal_ID= f"mouse{animal_ID}"

#create new directory
base_path = os.path.join(os.path.expanduser('~'), 'Desktop/auditory_maze_experiments/maze_recordings', 'maze_recordings_sequences')
sf.ensure_directory_exists(base_path)
new_dir_path = sf.setup_directories(base_path, date_time, animal_ID)
#save recording in new directory
rec_name = f"{animal_ID}_{date_time}.mp4"
recordFile = os.path.join(new_dir_path, rec_name)
metadata = sf.collect_metadata(animal_ID)
sf.save_metadata_to_csv(metadata, new_dir_path, f"{animal_ID}_{date_time}.csv")

#define rois
entrance_rois= ["entrance1", "entrance2"]
rois_list = sf.get_rois_list(rois_number)

roiNames= entrance_rois + rois_list

#set base_name for trials list and sound array to be saved as 
base_name = f"trials_{date_time}"


#check that only one is set to true
if no_homecage and forced_timing:
    raise ValueError("Choose only one: no_homecage OR forced_timing.")


#make just sounds
if make_simple_smooth_sounds and not (make_sequences or make_Simple_intervals or make_temporal_envelope_modulation or make_complex_intervals):
    # frequency= sf.ask_music_info_simple_sounds(rois_number)
    frequency = [10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065]
    # for i in range(len(frequency)):
    #     sound_data = sf.generate_sound_data(frequency[i])
    trials, sound_array = sf.create_simple_trials(rois_list, frequency)

# Make sequences of smooth sounds
elif make_sequences and not (make_simple_smooth_sounds or make_Simple_intervals or make_temporal_envelope_modulation or make_complex_intervals):

    experimental_session = "sequences"
    frequency, patterns= sf.ask_music_info_sequences(rois_number)
    path_to_vocal = "C:/Users/labuser/Downloads/vocalisationzzzzzz/trimmed_vocalisations/run3_day2-2_male_w_female_oestrus.wav"
    trials, sound_array = sf.create_trials_for_sequences(rois_list, frequency, patterns, path_to_voc= path_to_vocal)
    #make all arrays the same size because otherwise it won't save sound arrays
    min_length = min(len(arr) for arr in sound_array)
    # Trim each array to the minimum length
    trimmed_sound_arrays = [arr[:min_length] for arr in sound_array]


#make complex sounds
elif make_Simple_intervals and not (make_sequences or make_simple_smooth_sounds or make_temporal_envelope_modulation or make_complex_intervals):
    #essentially this gives me the option to be prompted what I want as intervals vs hard coding them
    manual = False
    if manual:
        frequency, intervals, intervals_names = sf.ask_info_intervals(rois_number)
    else: 
        tonal_centre = 10000 #Hz
        intervals_list = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]
        frequency, intervals, intervals_names = sf.info_intervals_hc(rois_number, tonal_centre, intervals_list)

    trials, sound_array = sf.create_trials_for_intervals(rois_list, frequency, intervals, intervals_names)


elif make_temporal_envelope_modulation and not (make_sequences or make_simple_smooth_sounds or make_Simple_intervals or make_complex_intervals):

    experimental_session = "temporal_envelope_modulation"
    # okay, so, this adds another layer of control. You user can choose which frequencies will be smooth, which with constant AM, and which with variable AM 
    # you do you, depends on what level of control freak you are. No judgement, I understand. There there
    smooth_freqs = [10000, 20000]
    constant_rough_freqs = [10000, 20000]
    #constant temporal modulation
    ctm= 50 #Hz
    complex_rough_freqs = [10000, 20000]
    #complex temporal modulation
    #check in supfun apply_segmented_am_modulation if you want to change parameters
    complex_tm = [30, 50, 70] # Hz
    
    controls = ["vocalisation", "silent"]

    
    #insert your path to vocalisation
    path_to_vocalisation = "C:/Users/labuser/Downloads/vocalisationzzzzzz/trimmed_vocalisations/run3_day2-2_male_w_female_oestrus.wav"

   # look this next function could be easier but you bear with me, here is where we sort everything out

    frequencies, temporal_modulation, sound_type, sounds_arrays = sf.info_temporal_modulation_hc(rois_number,
                                                                                                controls, 
                                                                                                smooth_freqs, 
                                                                                                constant_rough_freqs, 
                                                                                                complex_rough_freqs, 
                                                                                                constant_rough_modulation= ctm, 
                                                                                                complex_rough_mod = complex_tm, 
                                                                                                path_to_voc = path_to_vocalisation)

    trials, sound_array = sf.create_temporally_modulated_trials(rois_list, frequencies, temporal_modulation, sound_type, sounds_arrays)


elif make_complex_intervals and not (make_sequences or make_simple_smooth_sounds or make_Simple_intervals or make_temporal_envelope_modulation):

    
    

    if w1day2 and not (w1day3 or w1day4):
        experimental_session = "intervals_pt1"
        tonal_centre = 15000
        smooth_freq= True
        rough_freq = True
            
        consonant_intervals = ["perf_5", "perf_4"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
        dissonant_intervals = ["tritone",  "min_7"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
        controls = ["vocalisation", "silent"] #"vocalisation", "silent"
    
    elif w1day3 and not (w1day2 or w1day4):
        experimental_session = "intervals_pt2"
        tonal_centre = 15000
        smooth_freq= True
        rough_freq = True
            
        consonant_intervals = ["maj_6", "min_3"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
        dissonant_intervals = ["maj_7",  "min_2"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
        controls = ["vocalisation", "silent"] #"vocalisation", "silent"

    elif w1day4 and not (w1day3 or w1day2):
        experimental_session = "pure_intervals"

        tonal_centre = 15000
        smooth_freq= False
        rough_freq = False
            
        consonant_intervals = ["maj_3", "perf_4", "perf_5", "min_6"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
        dissonant_intervals = ["min_7",  "maj_2", "tritone", "maj_7"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
        controls = [] #"vocalisation", "silent"

    elif another_day: 
        experimental_session = "intervals"

        tonal_centre = 15000
        smooth_freq= False
        rough_freq = False
            
        consonant_intervals = ["maj_3", "perf_4", "perf_5"] #"min_3", "maj_3", "perf_4", "perf_5", "min_6", "maj_6", "octave"
        dissonant_intervals = ["min_7",  "maj_2", "tritone"] # "min_2", "maj_2",  "tritone", "min_7", "maj_7"
        controls = ["vocalisation", "silent"] #"vocalisation", "silent"

    
    #insert your path to vocalisation
    path_to_vocalisation = "C:/Users/labuser/Downloads/vocalisationzzzzzz/trimmed_vocalisations/run3_day2-2_male_w_female_oestrus.wav"

    frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays = sf.info_complex_intervals_hc (rois_number, 
                                                                                                                           controls, 
                                                                                                                           tonal_centre, 
                                                                                                                           smooth_freq, 
                                                                                                                           rough_freq, 
                                                                                                                           consonant_intervals, 
                                                                                                                           dissonant_intervals, 
                                                                                                                           path_to_voc= path_to_vocalisation)
    
    trials, sound_array = sf.create_complex_intervals_trials(rois_list, frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays)

elif just_vocalisations and not (make_complex_intervals or make_sequences or make_simple_smooth_sounds or make_Simple_intervals or make_temporal_envelope_modulation):
    #define if there is a silent arm as control or not
    silent_arm = True

    experimental_session = "vocalisations"

    path_to_vocalisations_folder="C:/Users/labuser/Downloads/vocalisationzzzzzz/trimmed_vocalisations/"
    file_names = os.listdir(path_to_vocalisations_folder)

    stimuli=[]
    
    for file in file_names: 
        stimuli.append(path_to_vocalisations_folder+file)

    
    if silent_arm: 
        stimuli.append("silent")
     
    frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays = sf.vocalisations_info_hc(rois_number, stimuli, file_names)
    trials, sound_array = sf.create_complex_intervals_trials(rois_list, frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays)

else: 
    print("go back and make sure you only choose one experiment type")

# initialise the visit log
visit_log_name = f"{animal_ID}_{experimental_session}"
visit_log_path = sf.initialise_visit_log(new_dir_path, visit_log_name)

#save the data as npy to load the sounds quicker (more quickly?) and the trials df to csv                                                                                        
np.save(os.path.join(new_dir_path, f"{base_name}.npy"), np.array(sound_array, dtype=object))
trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv")) 

# base_name = sf.choose_csv()
# base_name = base_name[:base_name.find(".")]
# trials = pd.read_csv(base_name + ".csv")
# sound_array = np.load(base_name + ".npy", allow_pickle=True)
unique_trials = trials['trial_ID'].unique()


# Draw ROIs 
if drawRois:

    sf.define_rois(videoInput=videoInput,
                roiNames=roiNames,
                outputName=base_path + "/" + "rois1.csv")
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)
else:
    rois = pd.read_csv(base_path + "/" + "rois1.csv", index_col=0)

# Loading the ROI info and initialising the variables per roi
# Loading the ROI info and initialising the variables per roi
thresholds = {item: 0 for item in rois}
hasVisited = {item: False for item in rois}

# debounced "clean" occupancy state used everywhere else
mousePresent = {item: False for item in rois}

# raw per-frame occupancy before debouncing
rawMousePresent = {item: False for item in rois}
present_streak   = {item: 0 for item in rois}  # consecutive frames with mouse detected
absent_streak    = {item: 0 for item in rois}  # consecutive frames without mouse detected

# how many consecutive frames to confirm entry/exit
ENTER_FRAMES = 1  # set to 1 if you want absolutely immediate triggering
EXIT_FRAMES  = 15  # require 8 "empty" frames to count as leaving, almost 0.3 seconds

visitation_count   = {item: 0 for item in rois}
visit_start_times  = {item: None for item in rois}  # None = no ongoing visit
 # dictionary to track when a visit started


cap = sf.start_camera(videoInput=videoInput)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))


if recordVideo:
    videoFileObject = sf.record_video(cap, recordFile, frame_width, frame_height, fps)

valid, gray = cap.read()
ret, gray = cv.threshold(gray, 160,255, cv.THRESH_BINARY)


# back_sub = cv.createBackgroundSubtractorMOG2()
# fg_mask = back_sub.apply(gray)
# # cv.imshow('Foreground Mask', fg_mask)
# # cv.waitKey(0)
# #ret,gray_inv = cv.threshold(grayOriginal,180,255,cv.THRESH_BINARY_INV)

# try:
#     contours,hierarchy = cv.findContours(fg_mask, cv.RETR_TREE,
#                                 cv.CHAIN_APPROX_NONE)
#     if contours:
#         cv.drawContours(fg_mask, contours[0], -1, 255, 4)
#         cv.drawContours(gray, contours[0], 0, (0,255,255), 2)
#         print("contours found")
#     else:
#         print("No contours found")
# except:
#     print("no")


areas = {}
for _ in range(10):
    valid, gray = cap.read()
    if not valid:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    for item in rois:
        areas[item] = sf.grab_cut(gray,
                                xstart=rois[item]["xstart"],
                                ystart=rois[item]["ystart"],
                                xlen=rois[item]["xlen"],
                                ylen=rois[item]["ylen"])
        thresholds[item] = np.sum(areas[item])
        print(f"ROI: {item}, Threshold: {thresholds[item]}")

sessionStartTime = time.time()
cv.namedWindow('binary maze plus ROIs', cv.WINDOW_NORMAL)
cv.namedWindow('original image', cv.WINDOW_NORMAL)
absolute_time_start = sf.time_in_millis()



if testing:
    trial_lengths = [0.1, 1, 0.2, 2, 0.2, 2, 0.2, 2, 0.2]

    #inserted a longer silence trial in the middle, to check if the mice prefer the maze in silence
elif not testing and longer_silence:
    trial_lengths = [15, 15, 2, 15, 15, 15, 2, 15, 2]
elif not testing and microcontroller:
    trial_lengths = [15, 10, 2, 10, 2, 10, 2, 10, 2]
elif not testing and not microcontroller and not longer_silence:
    trial_lengths = [15, 15, 2, 15, 2, 15, 2, 15, 2]

total_session_time = (np.sum(trial_lengths))*60



for trial in unique_trials:
    time_trial = trial_lengths[trial - 1]
    print("trial_duration: ", time_trial)
    if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
        break
    reset_play = True

# initialising the variables for each trial
    for item in rois:
        hasVisited[item] = False
        visitation_count[item]= 0
        visit_start_times[item] = None
        present_streak[item] = 0
        absent_streak[item]  = 0
    
    trialOngoing = True
    trial_time_accum_ms = 0 #this will be the total time-in-maze for this trial
    last_entry_ts = None #timestamp when the mouse last entered
   
    enteredMaze = False
    hasLeft1 = False
    hasLeft2 = False
    e2Aftere1 = False
    e1Aftere2 = False
    ent1History = [False, False]
    ent2History = [False, False]
    started_trial_timer = False
    ongoing_trial_duration = 0
    time_old_frame = 0
    start_new_time = True


    if no_homecage:
        start_time = time.time()
        end_time = start_time + time_trial * 60
        enteredMaze = True

        # count the whole trial as "time in maze"
        last_entry_ts = start_time

        # Preload sounds for each ROI for this trial
        sounds = {}
        for roi in rois_list:
            sound = sound_array[trials.loc[
                (trials['trial_ID'] == trial) & (trials['ROIs'] == roi)
            ].index[0]]
            sounds[roi] = sound

        # Record trial start/end times in the DataFrame
        trials.loc[trials["trial_ID"] == trial, "trial_start_time"] = start_time
        trials.loc[trials["trial_ID"] == trial, "end_trial_time"] = end_time

        # We already started the trial; don't let the entrance logic restart it
        start_new_time = False
        print(f"Auto-starting trial {trial} for {time_trial} minutes (no_homecage mode).")

    elif forced_timing and not no_homecage:
        start_time = time.time()
        end_time = start_time + time_trial * 60
        sounds = {}

        for roi in rois_list:
            sound = sound_array[trials.loc[
                (trials['trial_ID'] == trial) & (trials['ROIs'] == roi)
            ].index[0]]
            sounds[roi] = sound

        trials.loc[trials["trial_ID"] == trial, "trial_start_time"] = start_time
        trials.loc[trials["trial_ID"] == trial, "end_trial_time"] = end_time

        start_new_time = False  # prevents entrance logic from restarting the timer
        print(f"Auto-starting trial {trial} for {time_trial} minutes (forced_timing mode).")


    while trialOngoing:

        session_elapsed_s = time.time() - sessionStartTime

        if pause_between_frames:
            time.sleep(0.01)


        if ttl_is_active and time.time() >= ttl_scheduled_off_time:
            if arduino:
                arduino.write(b'L') # Turn OFF
            ttl_is_active = False


        #read the video 
        valid, grayOriginal = cap.read()
        ret, gray = cv.threshold(grayOriginal, 160,255, cv.THRESH_BINARY)
        time_frame = sf.time_in_millis() - absolute_time_start # constantly updating - represents how many ms into the trial the frame(?) is

        if not valid:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if recordVideo:
            videoFileObject.write(grayOriginal)

        # show the rois on the video
        for item in rois:
            # draw the rectangle per ROI
            cv.rectangle(gray, (rois[item]["xstart"],
                                rois[item]["ystart"]),
                         (rois[item]["xstart"] + rois[item]["xlen"],
                          rois[item]["ystart"] + rois[item]["ylen"]),
                         color=(120, 0, 0), thickness=2)
            
            # Add the name of the ROI above the rectangle
            cv.putText(gray, item,
               (rois[item]["xstart"], rois[item]["ystart"] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

            

        cv.imshow('binary maze plus ROIs', gray)
        cv.imshow('original image', grayOriginal)
        #cv.imshow('frame_diff', fg_mask) 

        if cv.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break

        for item in rois:
            areas[item] = sf.grab_cut(gray,
                                      xstart=rois[item]["xstart"],
                                      ystart=rois[item]["ystart"],
                                      xlen=rois[item]["xlen"],
                                      ylen=rois[item]["ylen"])

            rawMousePresent[item] = np.sum(areas[item]) < thresholds[item] * 0.5

            # --- temporal debouncing: turn noisy raw signal into stable mousePresent
            if rawMousePresent[item]:
                present_streak[item] += 1
                absent_streak[item] = 0
            else:
                absent_streak[item] += 1
                present_streak[item] = 0

            previous_state = mousePresent[item]

            # switch ON only after enough consecutive "inside" frames
            if not previous_state and present_streak[item] >= ENTER_FRAMES:
                mousePresent[item] = True

            # switch OFF only after enough consecutive "outside" frames
            elif previous_state and absent_streak[item] >= EXIT_FRAMES:
                mousePresent[item] = False

            # otherwise keep mousePresent[item] as it was

            # rois_list = ['ROI1', 'ROI2', 'ROI3', 'ROI4', "ROI5", "ROI6", "ROI7", "ROI8"]

            # if no_homecage and item not in rois_list:
            # # We have already updated mousePresent[item], so
            # # entrance1/entrance2 can still be used elsewhere,
            # # but we skip visit counting and logging for them.
            #     continue

            
# --- CHECK FOR ENTRY (Start of Visit) ---
            if mousePresent[item]:
                
                # If this is a fresh entry (timer is None and we start the time)
                if visit_start_times[item] is None:
                    visit_start_times[item] = time.time() # Start the stopwatch
                    
                    # get the total visitation counts 
                    condition = (trials['trial_ID'] == trial) & (trials['ROIs'] == item)
                    if not hasVisited[item]: 
                        visitation_count[item] += 1
                        print(f'visitation count for {item} : {visitation_count[item]}')
                        hasVisited[item] = True
                        trials.loc[condition, 'visitation_count'] = visitation_count[item]

                # get the cumulative time spent
                duration_frame = time_frame - time_old_frame
                condition = (trials['trial_ID'] == trial) & (trials['ROIs'] == item)
                if pd.isna(trials.loc[condition, 'time_spent']).all():
                    trials.loc[condition, 'time_spent'] = duration_frame
                else:
                    trials.loc[condition, 'time_spent'] += duration_frame       

            # when the mouse leaves
            else:
                # If the mouse was previously in the ROI (timer is already running running)
                if visit_start_times[item] is not None:
                    end_visit_time = time.time()
                    duration_seconds = end_visit_time - visit_start_times[item]
                    
                    # Get the stimulus info string
                    stim_info = sf.get_stimulus_string(trials, trial, item)
                    
                    # Save to a row in the new CSV
                    sf.log_individual_visit(visit_log_path, trial, item, stim_info, visit_start_times[item], end_visit_time ,duration_seconds)
                    
                    print(f"Logged visit: {item} | Time: {duration_seconds:.2f}s | {stim_info}")
                    
                    # Reset the timer
                    visit_start_times[item] = None

                hasVisited[item] = False         


                    

        time_old_frame = time_frame

        if not no_homecage:
            ent1History.insert(0, mousePresent["entrance1"])
            ent1History.pop(-1)
            ent2History.insert(0, mousePresent["entrance2"])
            ent2History.pop(-1)

            if not ent1History[0] and ent1History[1]:
                hasLeft1 = True
                if hasLeft2:
                    e1Aftere2 = True
                    e2Aftere1 = False

            if not ent2History[0] and ent2History[1]:
                hasLeft2 = True
                if hasLeft1:
                    e1Aftere2 = False
                    e2Aftere1 = True

            # Mouse goes entrance2 after entrance1: start the trial
            if e2Aftere1 and not enteredMaze:
                print(f"mouse entered the maze for trial {trial}")
                
                if (not forced_timing) and start_new_time:
                    start_time = time.time()
                    start_new_time = False
                    print(f"trial {trial} for {time_trial} minutes.")
                    # Create a dictionary to store sounds dynamically for each ROI
                    sounds = {}
                    for roi in rois_list:
                        sound = sound_array[trials.loc[
                            (trials['trial_ID'] == trial) & (trials['ROIs'] == roi)
                        ].index[0]]
                        sounds[roi] = sound

                end_time = start_time + time_trial * 60
                remaining_total_time = end_time - time.time()
                remaining_minutes = int(remaining_total_time / 60)
                remaining_seconds = int(remaining_total_time % 60)
                print(f"Remaining Time: {remaining_minutes:02}:{remaining_seconds:02}")

                trials.loc[trials["trial_ID"] == trial, "trial_start_time"] = start_time
                trials.loc[trials["trial_ID"] == trial, "end_trial_time"] = end_time
                enteredMaze = True
                if last_entry_ts is None:
                    last_entry_ts = time.time()

            # Mouse goes entrance1 after entrance2: left the maze
            if e1Aftere2 and enteredMaze:
                print("mouse has left the maze")
                if time.time() >= end_time:
                    print(time.time() >= end_time)
                    trialOngoing = False

                # accumulate time in maze    
                if last_entry_ts is not None:
                    trial_time_accum_ms += int((time.time() - last_entry_ts) * 1000)
                    last_entry_ts = None
                enteredMaze = False



        # no_homecage mode or forced timing mode: auto end

        if no_homecage and enteredMaze and (time.time() >= end_time):
            print(f"Ending trial {trial} (no_homecage mode, timer elapsed)")
            # Do NOT touch last_entry_ts here – the final block
            # after the while-loop will add the remaining time.
            trialOngoing = False

        elif forced_timing and (time.time() >= end_time):
            print(f"Ending trial {trial} (forced_timing mode, timer elapsed)")
            trialOngoing = False


        # sound playback logic    

        if enteredMaze:
            #if time.time() >= end_time:
                #trialOngoing = False

            for item in mousePresent:
                # Dynamically check if all ROIs are False
                if all(not mousePresent[roi] for roi in rois_list):
                    reset_play = True
                    sd.stop()

                    # force stop the TTL
                    if ttl_is_active:
                        if arduino:
                            arduino.write(b'L')
                            ttl_is_active = False

                # Play sound when mouse is inside an ROI
                if item in rois_list:
                    if mousePresent[item]:
                        condition = (trials["trial_ID"] == trial) & (trials["ROIs"] == item)
                        frequency_value = trials.loc[condition, "frequency"].values[0]
                        # normalize to a 1D numpy array for easy testing
                        freq_arr = np.atleast_1d(frequency_value)

                        # if *all* entries are zero, skip playing
                        if np.all(freq_arr == 0):
                            continue

                        # otherwise play exactly once per trial
                        if reset_play:
                            reset_play = False

                            # find your preloaded sound for this ROI
                            sound_for_this_roi = sounds[item] 

                            duration_sec = 0



                            if isinstance(sound_for_this_roi, (list, tuple)):
                                # Use length of the first array in the tuple
                                duration_sec = len(sound_for_this_roi[0]) / samplerate
                            else:
                                duration_sec = len(sound_for_this_roi) / samplerate
                                    
                            # Trigger Arduino ON
                            if arduino:
                                arduino.write(b'H')
                                
                                # Set State Variables
                                ttl_is_active = True
                                ttl_scheduled_off_time = time.time() + duration_sec

                            # now dispatch based on what kind of sound it is
                            if make_Simple_intervals:
                                sf.play_interval(sound_for_this_roi[0][0],
                                                sound_for_this_roi[1][0])

                            elif make_complex_intervals or just_vocalisations:
                                sf.play_interval(sound_for_this_roi[0],
                                                sound_for_this_roi[1])
                                
                                        
                                # interval_pair = sound_for_this_roi

                                # # unwrap nested lists if present
                                # d1, d2 = interval_pair
                                # if isinstance(d1, (list, tuple)):  d1 = d1[0]
                                # if isinstance(d2, (list, tuple)):  d2 = d2[0]

                                # # debug: confirm shapes
                                # print(f"[DEBUG] ROI {item} → shapes:", getattr(d1, 'shape', None), getattr(d2, 'shape', None))

                                # sf.play_interval(d1, d2)

                            else:
                                sf.play_sound(sound_for_this_roi)
                                
                            print(f"Playing sound for {item}")


        time_old_frame = time_frame

    if last_entry_ts is not None:
        trial_time_accum_ms += int((time.time() - last_entry_ts) * 1000)
        last_entry_ts = None

    trials.loc[trials["trial_ID"] == trial, "time_in_maze_ms"] = trial_time_accum_ms

    # Save the updated trials DataFrame to CSV after each trial
    print(f"Saving trials data for trial {trial} to CSV")
    remaining_session_time = total_session_time - session_elapsed_s

    remaining_s_minutes = int(remaining_session_time/60)
    remaining_s_s = int(remaining_session_time%60)

    print (f"remaining time in the session: {remaining_s_minutes:02}:{remaining_s_s:02} minutes")
    trials.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"), index=False)

cap.release()
cv.destroyAllWindows()
if recordVideo:
    videoFileObject.release()

print("Experiment completed and data saved.")
