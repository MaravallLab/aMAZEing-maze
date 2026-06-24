# figure out the logic of the session now that everything else is basically a function

import argparse
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
from modules.experiments import ExperimentFactory, GrammarStimulus
from modules.data_manager import DataManager
from modules.analysis import SessionAnalyzer


def _parse_cli():
    """CLI flags. Anything passed here overrides the matching field in
    config.py; anything omitted keeps the config default."""
    p = argparse.ArgumentParser(
        description="Run a single maze session. Grammar experiment flags "
                    "below override the defaults in config.py."
    )
    p.add_argument("--grammar-mode",
                   choices=["training", "silent_baseline", "test"],
                   default=None,
                   help="Which day of the test protocol this is.")
    p.add_argument("--enriched-grammar", choices=["A", "B"], default=None,
                   help="Which grammar (A or B) this mouse heard in the EE cage "
                        "during training. Determines arm assignment on test day.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (reproducible melody draws). Omit for random.")
    p.add_argument("--draw-rois", action="store_true",
                   help="Force interactive re-drawing of ROIs even if rois1.csv exists.")
    p.add_argument("--day", default=None,
                   help="Day label used as a parent folder in the output path "
                        "(e.g. habituation, day_1, day_2).")
    p.add_argument("--no-record-video", action="store_true",
                   help="Disable video file saving while keeping camera tracking active. "
                        "Useful for habituation days to save disk space.")
    return p.parse_args()


def main():
    # ==========================================
    # 1. SETUP & INITIALIZATION
    # ==========================================
    args = _parse_cli()
    cfg = ExperimentConfig()

    # Apply CLI overrides (only if the flag was passed)
    if args.grammar_mode is not None:
        cfg.grammar_mode = args.grammar_mode
    if args.enriched_grammar is not None:
        cfg.enriched_grammar = args.enriched_grammar
    if args.seed is not None:
        cfg.grammar_seed = args.seed
    if args.draw_rois:
        cfg.draw_rois = True
    if args.day is not None:
        cfg.experiment_day = args.day
    if args.no_record_video:
        cfg.record_video = False

    print(f"Session config: experiment_mode={cfg.experiment_mode!r}  "
          f"grammar_mode={cfg.grammar_mode!r}  "
          f"enriched_grammar={cfg.enriched_grammar!r}  "
          f"draw_rois={cfg.draw_rois}")
    
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

    # Initialize the maze entry/exit log
    maze_log_path = data_mgr.init_maze_log(cfg.experiment_mode)

    print(f"📂 Session Ready: {new_dir_path}")

    # ==========================================
    # 2. HARDWARE SETUP
    # ==========================================
    print("\n--- 🔌 Hardware Setup ---")
    
    # Initialize Audio — pulls samplerate/device/defaults from cfg.
    audio = Audio(cfg, calibration_gain_path=cfg.calibration_gain_path)
    
    # Initialize Arduino (if enabled in config)
    arduino = ArduinoController(cfg=cfg, active=cfg.use_microcontroller)
    
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

    # ==========================================
    # 3. GENERATE TRIALS
    # ==========================================
    print("\n--- 🧪 Generating Trials ---")
    
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

    def _fmt(seconds: float) -> str:
        s = max(0, int(seconds))
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ==========================================
    # 4. VISION CALIBRATION
    # ==========================================
    # Construct the full list of ROIs (Entrances + numbered arms)
    generic_rois = [str(i+1) for i in range(cfg.rois_number)]
    full_rois_list = cfg.entrance_rois + generic_rois

    # If cfg.draw_rois is True, force re-drawing by removing the existing CSV
    # so ROIMonitor's "missing → interactive draw" path fires.
    roi_csv_path = "rois1.csv"
    if cfg.draw_rois and os.path.exists(roi_csv_path):
        print(f"📐 draw_rois=True → removing existing {roi_csv_path} so you can re-draw.")
        os.remove(roi_csv_path)

    # Initialize Tracker
    tracker = ROIMonitor(
        roi_csv_path=roi_csv_path,  # Will prompt user (cv.selectROI) if file missing
        roiNames=full_rois_list,
        video_input=cfg.video_input,
        detection_sensitivity=cfg.detection_sensitivity,
        debug_roi=cfg.debug_roi
    )

    print("\n Calibrating background — please keep the maze empty...")
    calib_frames = []
    for i in range(40):  # discard first 30 warmup frames, collect 10 raw frames
        valid, frame = camera.get_frame()
        if valid and i >= 30:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            calib_frames.append(gray)

    if len(calib_frames) < 5:
        raise RuntimeError("Not enough frames captured for calibration. Check camera.")

    tracker.calibrate(calib_frames)
    print("Calibration Complete.")

    # ==========================================
    # 5. MAIN EXPERIMENT LOOP
    # ==========================================
    # Create window
    cv.namedWindow("Experiment View", cv.WINDOW_NORMAL)
    cv.resizeWindow("Experiment View", 1024, 768)

    # Maze entry/exit tracking (persists across trials)
    last_entrance = None       # which entrance was triggered most recently
    maze_entry_time = None     # wall-clock time when mouse entered the maze
    total_maze_time = 0.0      # cumulative seconds inside the maze

    session_end_time = time.time() + sum(trial_lengths) * 60

    for trial_idx in unique_trials:
        # Calculate timing
        # trial_idx is 1-based, list index is 0-based
        duration_mins = trial_lengths[trial_idx - 1]

        # Skip blocks with 0 duration entirely (no frame captured, no data bleed)
        if duration_mins == 0:
            continue

        trial_end_time = time.time() + (duration_mins * 60)

        print(f"\n STARTING TRIAL {trial_idx} (Duration: {duration_mins} min)")
        print(f"   Ends at: {time.ctime(trial_end_time)}")
        
        # Reset trial variables
        visit_start_times = {roi: None for roi in full_rois_list}
        ttl_scheduled_off = 0
        ttl_active = False
        
        trial_running = True
        while trial_running:
            # --- A. Capture ---
            valid, raw_frame = camera.get_frame()
            if not valid:
                print("❌ Camera stream ended unexpectedly.")
                break
                
            if video_writer:
                video_writer.write(raw_frame)

            # --- B. Image Processing ---
            if raw_frame.ndim == 3:
                gray = cv.cvtColor(raw_frame, cv.COLOR_BGR2GRAY)
                display_frame = raw_frame.copy()
            else:
                gray = raw_frame
                # Convert to BGR for display so drawn boxes are colored
                display_frame = cv.cvtColor(raw_frame, cv.COLOR_GRAY2BGR)

            ret, binary = cv.threshold(gray, cfg.binary_threshold, 255, cv.THRESH_BINARY)

            # --- C. Tracking ---
            # 'entered_rois' is a list of ROIs entered *this frame*
            entered_rois = tracker.update(binary)
            
            # --- D. Handle Entries ---
            for roi in entered_rois:
                visit_start_times[roi] = time.time()

                # --- Entrance/exit logic ---
                if roi in ("entrance1", "entrance2"):
                    if roi == "entrance2" and last_entrance == "entrance1":
                        # Passed through entrance1 → entrance2: entered maze
                        maze_entry_time = time.time()
                        print("  [MAZE] Mouse ENTERED the maze")
                        DataManager.log_maze_event(maze_log_path, trial_idx, "entered", maze_entry_time, None)
                    elif roi == "entrance1" and last_entrance == "entrance2":
                        # Passed through entrance2 → entrance1: left maze
                        now = time.time()
                        duration = (now - maze_entry_time) if maze_entry_time is not None else 0.0
                        total_maze_time += duration
                        maze_entry_time = None
                        print(f"  [MAZE] Mouse LEFT the maze (time inside: {duration:.1f}s, total: {total_maze_time:.1f}s)")
                        DataManager.log_maze_event(maze_log_path, trial_idx, "exited", now, duration)
                    last_entrance = roi

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
                if isinstance(sound_clip, GrammarStimulus):
                    pass  # always play — rendered live below
                elif isinstance(sound_clip, (int, float)) and sound_clip == 0:
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
                    row = trials_df.loc[sound_index]
                    freq_val = row['frequency']
                    if freq_val == 'grammar':
                        stim_label = f"{row['tier']} {row['environment_association']} (Grammar {row['grammar']})"
                    elif freq_val == 'vocalisation':
                        stim_label = "vocalisation"
                    else:
                        stim_label = f"{freq_val} Hz"
                    print(f"   Playing: {roi} -> {stim_label}")

                    # Grammar arm: sample a fresh melody on every entry
                    if isinstance(sound_clip, GrammarStimulus):
                         wave = sound_clip.render(audio, roi=roi, trial_id=trial_idx)
                         audio.play(wave)
                         duration = len(wave) / cfg.samplerate
                    # Handle Tuple (Intervals) vs Single Sound
                    elif isinstance(sound_clip, tuple):
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

            # --- E. Handle Exits & Logging ---
            for roi in full_rois_list:
                # If mouse WAS in ROI (start_time set) but is NOT there now:
                if not tracker.is_occupied[roi] and visit_start_times[roi] is not None:
                    start_t = visit_start_times[roi]
                    end_t = time.time()
                    visit_dur = end_t - start_t
                    
                    # Log the visit to CSV
                    stim_info = DataManager.get_stimulus_string(trials_df, trial_idx, roi)
                    DataManager.log_individual_visit(visit_log_path, trial_idx, roi, stim_info, start_t, end_t, visit_dur)
                    
                    print(f"   📝 Visit Logged: {roi} ({visit_dur:.2f}s)")
                    
                    # Reset timer
                    visit_start_times[roi] = None
                    
                    # Update total time spent in main dataframe (entrance ROIs won't be in trials_df)
                    mask = (trials_df['trial_ID'] == trial_idx) & (trials_df['ROIs'] == roi)
                    if mask.any():
                        current_time = trials_df.loc[mask, 'time_spent'].values[0]
                        new_time = visit_dur if pd.isna(current_time) else current_time + visit_dur
                        trials_df.loc[mask, 'time_spent'] = new_time

            # --- F. Hardware Logic (Stop Sound/TTL) ---
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
            
            # --- G. Render Feedback ---
            tracker.draw_feedback(display_frame)
            
            now = time.time()
            cv.putText(display_frame, f"Trial {trial_idx}:   {_fmt(trial_end_time - now)} left", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_frame, f"Session: {_fmt(session_end_time - now)} left", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            cv.imshow("Experiment View", display_frame)

            if cfg.show_binary_view:
                cv.imshow("Binary Debug View", binary)

            # --- H. Loop Checks ---
            if time.time() >= trial_end_time:
                print("🛑 Trial Time Ended")
                trial_running = False
                
            if cv.waitKey(1) & 0xFF in [ord('q'), 27]: # q or ESC
                print("User Quit")
                trial_running = False
                # Break outer loop too
                trial_idx = unique_trials[-1] + 1 
                break

        # Save data after every trial (Safety)
        trials_df.to_csv(os.path.join(new_dir_path, f"{base_name}.csv"), index=False)
        
        # Stop everything between trials
        audio.stop()
        arduino.trigger_off()

    # ==========================================
    # 6. CLEANUP & ANALYSIS
    # ==========================================
    print("\n--- 🏁 Experiment Finished ---")

    # If mouse was still inside the maze when the session ended, close that bout
    if maze_entry_time is not None:
        duration = time.time() - maze_entry_time
        total_maze_time += duration
        DataManager.log_maze_event(maze_log_path, trial_idx, "session_end_still_inside", time.time(), duration)

    print(f"  Total time spent in maze: {total_maze_time:.1f}s ({total_maze_time/60:.2f} min)")

    # Dump grammar sampling history (only if grammar mode ran).
    # The same GrammarStimulus instance appears in sound_array at multiple
    # positions (shuffled across blocks), so dedupe by id().
    grammar_rows = []
    seen = set()
    for clip in sound_array:
        if isinstance(clip, GrammarStimulus) and id(clip) not in seen:
            seen.add(id(clip))
            grammar_rows.extend(clip.history)
    if grammar_rows:
        grammar_log = os.path.join(new_dir_path, f"grammar_samples_{data_mgr.timestamp}.csv")
        pd.DataFrame(grammar_rows).to_csv(grammar_log, index=False)
        print(f"📝 Grammar samples logged: {grammar_log}")

    camera.release()
    if video_writer:
        video_writer.release()
    arduino.close()
    cv.destroyAllWindows()

    # --- AUTO-ANALYSIS ---
    print("\n--- Running Post-Experiment Analysis ---")
    try:
        analyzer = SessionAnalyzer(new_dir_path)
        analyzer.generate_report()
        print(f"Graphs saved in: {new_dir_path}")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
