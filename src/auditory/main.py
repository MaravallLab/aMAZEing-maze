# figure out the logic of the session now that everything else is basically a function 

import cv2 as cv
import numpy as np
import pandas as pd
import time
import os
import sounddevice as sd

# our modules from the subdirectory `modules`
from config import ExperimentConfig
from modules.hardware import ArduinoController, Camera
from modules.vision import ROIMonitor
from modules.audio import Audio
from modules.experiments import ExperimentFactory
from modules.data_manager import DataManager
#from modules.analysis import SessionAnalyzer

def main():
    
    # 1. SETUP & INITIALIZATION
    
    cfg = ExperimentConfig()
    
    # Initialize Data Manager
    data_mgr = DataManager(cfg.base_output_path)
    
    # Interactive Setup:
    # This asks for mouse ID and creates the folder structure:
    # e.g., /data/complex_intervals_w1day2/time_2023..._mouse1/
    new_dir_path, animal_ID = data_mgr.setup_session(cfg)
    
    # Optional: Collect and save metadata (Gender, DOB, etc.)
    data_mgr.save_metadata()
    
    # Initialize the detailed CSV log for individual visits
    visit_log_path = data_mgr.init_visit_log(cfg.experiment_mode)
    
    print(f"Session Ready: {new_dir_path}")

    
    # 2. HARDWARE SETUP
    
    print("\n 🔌 Hardware Setup ")
    
    # Initialize Audio
    # We pass the defaults here so they are set globally
    audio = Audio(
        samplerate=cfg.samplerate, 
        device_id=cfg.audio_device_id,
        default_duration=10.0,
        default_ramp=0.02
    )
    
    # Initialize Arduino (if enabled in config)
    arduino = ArduinoController(
        port=cfg.arduino_port, 
        baud_rate=cfg.arduino_baud, 
        active=cfg.use_microcontroller
    )
    
    # Initialize Camera
    camera = Camera(device_id=cfg.video_input)
    
    # Setup Video Recording (optional)
    video_writer = None
    if cfg.record_video:
        rec_name = f"{animal_ID}_{data_mgr.timestamp}.mp4"
        rec_path = os.path.join(new_dir_path, rec_name)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(rec_path, fourcc, camera.fps, (camera.width, camera.height))
        print(f"📹 Recording started: {rec_name}")

    
    # 3. GENERATE TRIALS
    
    print("\n 🧪 Generating Trials ")
    
    # We pass 'audio' so the factory can generate the correct waveforms
    trials_df, sound_array = ExperimentFactory.generate_trials(cfg, audio)
    
    # Save the trial structure immediately (Safety first!)
    base_name = f"trials_{data_mgr.timestamp}"
    # Save the dataframe (readable plan)
    trials_df.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"), index=False)
    # Save the raw audio arrays (in case we need to debug sounds later)
    np.save(os.path.join(new_dir_path, f"{base_name}.npy"), np.array(sound_array, dtype=object))
    
    unique_trials = trials_df['trial_ID'].unique()
    trial_lengths = cfg.get_trial_lengths()

    
    # 4. VISION CALIBRATION
    
    # Construct the full list of ROIs (Entrances + numbered arms)
    generic_rois = [str(i+1) for i in range(cfg.rois_number)]
    full_rois_list = cfg.entrance_rois + generic_rois
    
    # Initialize Tracker
    tracker = ROIMonitor(
        roi_csv_path="rois1.csv", # Will prompt user if file missing
        roi_names_list=full_rois_list,
        video_input=cfg.video_input
    )
    
    print("\n Calibrating background (Please step away)...")
    time.sleep(1) # Allow camera to settle
    
    valid, frame = camera.get_frame()
    if valid:
        # Handle Grayscale vs Color
        if frame.ndim == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame 
            
        ret, binary = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)
        tracker.calibrate(binary)
        print("Calibration Complete.")
    else:
        raise RuntimeError("Failed to capture frame for calibration.")

    
    # 5. MAIN EXPERIMENT LOOP
    
    # Create window
    cv.namedWindow("Experiment View", cv.WINDOW_NORMAL)
    cv.resizeWindow("Experiment View", 1024, 768)

    for trial_idx in unique_trials:
        # Calculate timing
        # trial_idx is 1-based, list index is 0-based
        duration_mins = trial_lengths[trial_idx - 1]
        trial_end_time = time.time() + (duration_mins * 60)
        
        print(f"\n STARTING TRIAL {trial_idx} (Duration: {duration_mins} min)")
        print(f"   Ends at: {time.ctime(trial_end_time)}")
        
        # Reset trial variables
        visit_start_times = {roi: None for roi in full_rois_list}
        ttl_scheduled_off = 0
        ttl_active = False
        
        trial_running = True
        while trial_running:
            #  A. Capture 
            valid, raw_frame = camera.get_frame()
            if not valid:
                print("Camera stream ended unexpectedly.")
                break
                
            if video_writer:
                video_writer.write(raw_frame)

            #  B. Image Processing 
            if raw_frame.ndim == 3:
                gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
                display_frame = raw_frame.copy()
            else:
                gray = raw_frame
                # Convert to BGR for display so drawn boxes are colored
                display_frame = cv.cvtColor(raw_frame, cv.COLOR_GRAY2BGR)

            ret, binary = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)
            
            #  C. Tracking 
            # 'entered_rois' is a list of ROIs entered *this frame*
            entered_rois = tracker.update(binary)
            
            #  D. Handle Entries 
            for roi in entered_rois:
                visit_start_times[roi] = time.time()
                
                # Check dataframe for this Trial/ROI combo
                mask = (trials_df['trial_ID'] == trial_idx) & (trials_df['ROIs'] == roi)
                if not mask.any(): continue
                
                # Update visitation count
                current_count = trials_df.loc[mask, 'visitation_count'].values[0]
                new_count = 1 if pd.isna(current_count) else current_count + 1
                trials_df.loc[mask, 'visitation_count'] = new_count
                
                # Get the sound for this ROI
                sound_index = trials_df.loc[mask].index[0]
                sound_clip = sound_array[sound_index]
                
                # Check if we should play (ignore silence/control)
                should_play = True
                if isinstance(sound_clip, (int, float)) and sound_clip == 0:
                    should_play = False
                elif isinstance(sound_clip, (list, np.ndarray, tuple)):
                     # If it's a tuple (interval), check if both are zero
                     if isinstance(sound_clip, tuple):
                         if all(isinstance(x, (int, float)) and x==0 for x in sound_clip):
                             should_play = False
                     # If it's an array, check if all zeros
                     elif isinstance(sound_clip, np.ndarray) and np.all(sound_clip == 0):
                        should_play = False

                if should_play:
                    print(f"    Playing sound for {roi}")
                    
                    # Handle Tuple (Intervals) vs Single Sound
                    if isinstance(sound_clip, tuple):
                         # Mix the two channels/sounds
                         mixed = audio.mix_sounds(sound_clip[0], sound_clip[1])
                         audio.play(mixed)
                         duration = len(mixed) / cfg.samplerate
                    else:
                         audio.play(sound_clip)
                         duration = len(sound_clip) / cfg.samplerate

                    # Trigger Arduino
                    arduino.trigger_on()
                    ttl_active = True
                    ttl_scheduled_off = time.time() + duration

            #  E. Handle Exits & Logging 
            for roi in full_rois_list:
                # If mouse WAS in ROI (start_time set) but is NOT there now:
                if not tracker.is_occupied[roi] and visit_start_times[roi] is not None:
                    start_t = visit_start_times[roi]
                    end_t = time.time()
                    visit_dur = end_t - start_t
                    
                    # Log the visit to CSV
                    stim_info = DataManager.get_stimulus_string(trials_df, trial_idx, roi)
                    data_mgr.log_visit(visit_log_path, trial_idx, roi, stim_info, start_t, end_t, visit_dur)
                    
                    print(f"   Visit Logged: {roi} ({visit_dur:.2f}s)")
                    
                    # Reset timer
                    visit_start_times[roi] = None
                    
                    # Update total time spent in main dataframe
                    mask = (trials_df['trial_ID'] == trial_idx) & (trials_df['ROIs'] == roi)
                    current_time = trials_df.loc[mask, 'time_spent'].values[0]
                    new_time = visit_dur if pd.isna(current_time) else current_time + visit_dur
                    trials_df.loc[mask, 'time_spent'] = new_time

            #  F. Hardware Logic (Stop Sound/TTL) 
            # Stop sound if mouse leaves ALL ROIs
            if not any(tracker.is_occupied.values()):
                audio.stop()
                if ttl_active:
                    arduino.trigger_off()
                    ttl_active = False

            # Stop TTL if sound finished but mouse is still inside
            if ttl_active and time.time() >= ttl_scheduled_off:
                arduino.trigger_off()
                ttl_active = False
            
            #  G. Render Feedback 
            tracker.draw_feedback(display_frame)
            
            remaining = int(trial_end_time - time.time())
            cv.putText(display_frame, f"Trial {trial_idx}: {remaining}s left", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow("Experiment View", display_frame)

            if cfg.show_binary_view:
                cv.imshow("Binary Debug View", binary)

            #  H. Loop Checks 
            if time.time() >= trial_end_time:
                print("Trial Time Ended")
                trial_running = False
                
            if cv.waitKey(1) & 0xFF in [ord('q'), 27]: # q or ESC
                print("User Quit")
                trial_running = False
                # Break outer loop too
                trial_idx = unique_trials[-1] + 1 
                break


        end_of_trial_t = time.time()
        for roi in full_rois_list:
            if visit_start_times[roi] is not None:
                start_t = visit_start_times[roi]
                visit_dur = end_of_trial_t - start_t
                # Log + accumulate as a normal exit
                stim_info = DataManager.get_stimulus_string(trials_df, trial_idx, roi)
                data_mgr.log_visit(visit_log_path, trial_idx, roi, stim_info,
                                start_t, end_of_trial_t, visit_dur)
                mask = (trials_df['trial_ID'] == trial_idx) & (trials_df['ROIs'] == roi)
                current_time = trials_df.loc[mask, 'time_spent'].values[0]
                new_time = visit_dur if pd.isna(current_time) else current_time + visit_dur
                trials_df.loc[mask, 'time_spent'] = new_time
                # Re-arm so the visit "continues" into the next trial
                visit_start_times[roi] = end_of_trial_t

        # Save data after every trial (Safety)
        trials_df.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"), index=False)
        
        # Stop everything between trials
        audio.stop()
        arduino.trigger_off()

    
    # 6. CLEANUP & ANALYSIS
    
    print("\n Experiment Finished ")
    camera.release()
    if video_writer:
        video_writer.release()
    arduino.close()
    cv.destroyAllWindows()

    # 
    # print("\n Running Post-Experiment Analysis ")
    # try:
    #     analyzer = SessionAnalyzer(new_dir_path)
    #     analyzer.generate_report()
    #     print(f"Graphs saved in: {new_dir_path}")
    # except Exception as e:
    #     print(f" Analysis failed (check logs): {e}")
    #     # Print full trace for debugging if needed
    #     # import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
