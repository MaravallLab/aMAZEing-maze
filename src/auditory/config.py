'''Here you get to define/modify the experimental variables.

All paths default to locations relative to the user's home directory.
Override them here or pass them when constructing the config.
'''

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Default base directory: ~/Desktop/auditory_maze_experiments/maze_recordings
_DEFAULT_BASE = os.path.join(os.path.expanduser("~"), "Desktop", "auditory_maze_experiments", "maze_recordings")

@dataclass
class ExperimentConfig:

    # Audio settings
    samplerate: int = 192000
    channel_id: int = 3

    # Sound settings — for specific changes (e.g. you want some sounds to be
    # longer or shorter than others) you can override these when calling the
    # generation functions in main.py
    default_sound_duration: float = 10.0
    default_waveform: str = "sine"
    default_volume: float = 1.0
    default_ramp_length_s: float = 0.02

    # Arduino settings
    arduino_port: str = "COM4"
    arduino_baud: int = 115200
    use_microcontroller: bool = False

    # Camera settings
    video_input: int = 0
    record_video: bool = True
    draw_rois: bool = False
    pause_between_frames: bool = False
    show_binary_view: bool = True

    # Testing / timing
    testing: bool = False
    longer_middle_silence: bool = False

    ### Experiment mode ###
    # Options: "simple_smooth", "simple_intervals", "temporal_envelope_modulation",
    #          "complex_intervals", "sequences", "vocalisation",
    #          "semantic_predictive_complexity"
    experiment_mode: str = "complex_intervals"

    # Only used if experiment_mode == "complex_intervals"
    # Options: "w1day2", "w1day3", "w1day4", "another_day"
    complex_interval_day: str = "w1day3"

    # Trial Settings
    rois_number: int = 8
    entrance_rois: List[str] = field(default_factory=lambda: ["entrance1", "entrance2"])

    # PATHS — override these for your machine
    base_output_path: str = _DEFAULT_BASE

    # Path to vocalisation control files
    path_to_vocalisation_folder: str = ""
    path_to_vocalisation_control: str = ""


    def get_trial_lengths(self) -> List[float]:

        if self.testing:
            return [0.1, 1, 0.2, 2, 0.2, 2, 0.2, 2, 0.2]
        elif self.longer_middle_silence:
            return [15, 15, 2, 15, 15, 15, 2, 15, 2]
        elif self.use_microcontroller:
            return [15, 10, 2, 10, 2, 10, 2, 10, 2]
        else:
            return [15, 15, 2, 15, 2, 15, 2, 15, 2]








