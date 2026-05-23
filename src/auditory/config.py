'''Here you get to define/modify the experimental variables.

All paths default to locations relative to the user's home directory.
Override them here or pass them when constructing the config.
'''

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Default base directory: ~/Desktop/auditory_maze_experiments/maze_recordings
_DEFAULT_BASE = os.path.join(os.path.expanduser("~"), "Desktop", "auditory_maze_experiments", "maze_recordings")

# Calibration CSV shipped with the repo (analysis/calibration/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CALIBRATION = os.path.join(_REPO_ROOT, "analysis", "calibration", "frequency_response_speaker.csv")

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
    binary_threshold: int = 160        # applied to live frames before detection (tune to your lighting)
    detection_sensitivity: float = 0.5  # mouse detected when binary sum drops below this fraction of raw baseline
    debug_roi: str = ""            # set to a ROI name (e.g. "1") or "all" to print live pixel sums

    # Testing / timing
    testing: bool = False
    longer_middle_silence: bool = False

    ### Experiment mode ###
    # Options: "simple_smooth", "simple_intervals", "temporal_envelope_modulation",
    #          "complex_intervals", "sequences", "vocalisation", "grammar",
    #          "semantic_predictive_complexity"
    experiment_mode: str = "grammar"

    # Only used if experiment_mode == "complex_intervals"
    # Options: "w1day2", "w1day3", "w1day4", "another_day"
    complex_interval_day: str = "w1day3"

    # Only used if experiment_mode == "grammar"
    # grammar_mode:
    #   "training"        — main.py refuses; you must use grammar_stimuli.run CLI
    #                       (default so a forgotten config raises a clear error)
    #   "silent_baseline" — Day 1 of the 3-test-day protocol: 1 hour in the
    #                       maze with NO audio on any arm. Logs ROI visits
    #                       only, to establish baseline preference.
    #   "test"            — Day 2 / Day 3: 1 hour with the full grammar test
    #                       (9-block shuffle of 8 stimuli).
    # enriched_grammar: which physical grammar (A or B) this mouse heard
    #   in the EE (enriched) cage during training. The other grammar is the
    #   one it heard in the SC cage. This drives arm assignment on test
    #   day: arms 1-3 play this grammar (EE-associated), arms 4-6 play
    #   the other (SC-associated). Unused in silent_baseline mode.
    grammar_mode: str = "training"
    enriched_grammar: str = "A"
    grammar_seed: Optional[int] = None      # RNG seed; None = nondeterministic

    # Per-day session length, in minutes. Override these if you want a
    # different schedule than the 1-hour default. The test list must have
    # exactly 9 entries (the 9-block silent/active cycle): even-indexed
    # entries are silent blocks, odd-indexed are active.
    grammar_silent_baseline_minutes: float = 45.0 #60.0
    grammar_test_block_minutes: List[float] = field(
        default_factory=lambda: [0, 15.0, 0, 15.0, 0, 15.0, 0, 15.0, 0]
    )

    # Trial Settings
    rois_number: int = 8
    entrance_rois: List[str] = field(default_factory=lambda: ["entrance1", "entrance2"])

    # PATHS — override these for your machine
    base_output_path: str = _DEFAULT_BASE
    calibration_gain_path: str = _DEFAULT_CALIBRATION

    # Path to vocalisation control files
    path_to_vocalisation_folder: str = r"C:\Users\labuser\Downloads\vocalisationzzzzzz\trimmed_vocalisations"
    path_to_vocalisation_control: str = r"c:\Users\labuser\Downloads\vocalisationzzzzzz\trimmed_vocalisations\run1_day2_male_w_female_oestrus.wav"


    def get_trial_lengths(self) -> List[float]:

        # Grammar experiment: durations are taken from the dedicated
        # grammar_* fields above so you can override them without touching
        # this function.
        if self.experiment_mode == "grammar":
            if self.grammar_mode == "silent_baseline":
                return [float(self.grammar_silent_baseline_minutes)]
            # test mode: must be a 9-element list (the 9-block cycle is
            # hard-coded in _make_grammar). Validate so a stray change
            # fails loudly.
            if len(self.grammar_test_block_minutes) != 9:
                raise ValueError(
                    f"grammar_test_block_minutes must have exactly 9 entries "
                    f"(got {len(self.grammar_test_block_minutes)}). The grammar "
                    f"test uses a fixed 9-block silent/active cycle."
                )
            return list(self.grammar_test_block_minutes)

        if self.testing:
            return [0.1, 1, 0.2, 2, 0.2, 2, 0.2, 2, 0.2]
        elif self.longer_middle_silence:
            return [15, 15, 2, 15, 15, 15, 2, 15, 2]
        elif self.use_microcontroller:
            return [15, 10, 2, 10, 2, 10, 2, 10, 2]
        else:
            return [15, 15, 2, 15, 2, 15, 2, 15, 2]








