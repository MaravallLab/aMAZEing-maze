'''Here you get to define/modify the experimental variables.

All paths default to locations relative to the user's home directory.
Override them here or pass them when constructing the config.
'''

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Default base directory: ~/Desktop/auditory_maze_experiments/maze_recordings
_DEFAULT_BASE = os.path.join(os.path.expanduser("~"), "Desktop", "auditory_maze_experiments", "maze_recordings")

# Speaker calibration CSV shipped as package data (amazeing/auditory/data/).
# Replace it with your own speaker's curve via ExperimentConfig.calibration_gain_path.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIBRATION = os.path.join(_PACKAGE_DIR, "data", "frequency_response_speaker.csv")

# Presets for experiment_mode == "complex_intervals". Each day fixes which
# consonant and dissonant intervals are presented, whether a smooth and a rough
# unison arm are included, and which control arms are present. Taken verbatim
# from the original experiments.py so existing protocols are unchanged.
COMPLEX_INTERVAL_DAYS: Dict[str, Dict[str, Any]] = {
    "w1day2": {"consonant": ["perf_5", "perf_4"], "dissonant": ["tritone", "min_7"],
               "smooth": True, "rough": True, "controls": ["vocalisation", "silent"]},
    "w1day3": {"consonant": ["maj_6", "min_3"], "dissonant": ["maj_7", "min_2"],
               "smooth": True, "rough": True, "controls": ["vocalisation", "silent"]},
    "w1day4": {"consonant": ["maj_3", "perf_4", "perf_5", "min_6"],
               "dissonant": ["min_7", "maj_2", "tritone", "maj_7"],
               "smooth": False, "rough": False, "controls": []},
    "another_day": {"consonant": ["maj_3", "perf_4", "perf_5"],
                    "dissonant": ["min_7", "maj_2", "tritone"],
                    "smooth": False, "rough": False,
                    "controls": ["vocalisation", "silent"]},
}

# Every interval the stimulus generator understands (just intonation).
INTERVAL_NAMES: List[str] = [
    "unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone",
    "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave",
]

@dataclass
class ExperimentConfig:

    # Audio settings
    samplerate: int = 192000
    channel_id: int = 3

    # Sound settings - for specific changes (e.g. you want some sounds to be
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
    detection_sensitivity: float = 0.6  # mouse detected when binary sum drops below this fraction of raw baseline
    debug_roi: str = ""            # set to a ROI name (e.g. "1") or "all" to print live pixel sums

    # Testing / timing
    testing: bool = False
    longer_middle_silence: bool = False

    ### Experiment mode ###
    # Options: "simple_smooth", "simple_intervals", "temporal_envelope_modulation",
    #          "complex_intervals", "sequences", "vocalisation", "grammar",
    #          "custom" (stimuli described in custom_stimuli below),
    #          "semantic_predictive_complexity"
    experiment_mode: str = "grammar"

    # Only used if experiment_mode == "complex_intervals"
    # Options: "w1day2", "w1day3", "w1day4", "another_day"
    complex_interval_day: str = "w1day3"

    # Only used if experiment_mode == "grammar"
    # grammar_mode:
    #   "training" - main.py refuses; you must use grammar_stimuli.run CLI
    #                       (default so a forgotten config raises a clear error)
    #   "silent_baseline" - Day 1 of the 3-test-day protocol: 1 hour in the
    #                       maze with NO audio on any arm. Logs ROI visits
    #                       only, to establish baseline preference.
    #   "test" - Day 2 / Day 3: 1 hour with the full grammar test
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

    # ---------------------------------------------------------------
    # Per-mode stimulus parameters
    # ---------------------------------------------------------------
    # These were literals inside experiments.py until v2; the defaults below
    # reproduce the original values exactly, so an untouched config generates
    # the same stimuli as before. Edit them here or in the interface.

    # experiment_mode == "simple_smooth": one pure tone per arm
    smooth_frequencies: List[float] = field(
        default_factory=lambda: [10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065])

    # experiment_mode == "simple_intervals": two-tone chords on one tonal centre
    simple_interval_tonal_centre: float = 10000.0
    simple_intervals_list: List[str] = field(
        default_factory=lambda: ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"])

    # experiment_mode == "temporal_envelope_modulation"
    tem_controls: List[str] = field(default_factory=lambda: ["vocalisation", "silent"])
    tem_smooth_freqs: List[float] = field(default_factory=lambda: [10000, 20000])
    tem_constant_rough_freqs: List[float] = field(default_factory=lambda: [10000, 20000])
    tem_complex_rough_freqs: List[float] = field(default_factory=lambda: [10000, 20000])
    tem_constant_mod_freq: float = 50.0            # Hz, constant AM rate
    tem_complex_mod_freqs: List[float] = field(default_factory=lambda: [30, 50, 70])
    tem_mod_depth: float = 0.5                     # AM depth, 0-1

    # experiment_mode == "complex_intervals": consonant vs dissonant contrasts.
    # complex_interval_day selects a preset (see COMPLEX_INTERVAL_DAYS); the
    # override fields below take precedence when non-empty, so the interface can
    # edit a day's stimuli without inventing a new day name.
    complex_interval_tonal_centre: float = 15000.0
    complex_consonant_intervals: List[str] = field(default_factory=list)
    complex_dissonant_intervals: List[str] = field(default_factory=list)
    complex_controls: Optional[List[str]] = None   # None = use the day preset
    complex_include_smooth: Optional[bool] = None
    complex_include_rough: Optional[bool] = None

    # experiment_mode == "sequences": tone patterns per arm.
    # sequence_tone_map maps each letter in the patterns to a frequency in Hz
    # ("o" is a silent slot). Leave both empty to get the original interactive
    # console prompts.
    sequence_patterns: List[str] = field(
        default_factory=lambda: ["AAAAA", "AoAo", "ABAB", "ABCABC", "BABA", "ABBA",
                                 "silence", "vocalisation"])
    sequence_tone_map: Dict[str, float] = field(default_factory=dict)
    sequence_repetitions: int = 50

    # experiment_mode == "vocalisation"
    vocalisation_include_silent_arm: bool = True

    # Only used if experiment_mode == "custom": one entry per numbered ROI
    # ("1", "2", ...) describing the stimulus played there. ROIs without an
    # entry are silent. Each entry is a mapping with a ``kind``:
    #   tone    : frequency (Hz); optional waveform, duration_s, volume,
    #             ramp_s. Speaker compensation is applied.
    #   am_tone : as tone, plus mod_freq (Hz) and depth (0-1) for a constant
    #             amplitude-modulation envelope.
    #   wav     : path to a .wav file (resampled to the session sample rate).
    #   silent  : plays nothing (explicit silent control arm).
    # Optional ``label`` names the stimulus in the logs and figures.
    custom_stimuli: List[Dict[str, Any]] = field(default_factory=list)
    # Block schedule in minutes, applied to every experiment mode. The session
    # always runs the 9-block cycle: even-numbered entries (0, 2, 4, 6, 8) are
    # silent blocks and odd-numbered entries (1, 3, 5, 7) are active blocks.
    # Set an entry to 0 to skip that block entirely.
    #   None  = use the mode's own default (the legacy 15/15/2 schedule, or
    #           grammar_test_block_minutes in grammar test mode).
    # Grammar silent-baseline days ignore this and use
    # grammar_silent_baseline_minutes, which is a single block.
    block_minutes: Optional[List[float]] = None

    # Trial Settings
    rois_number: int = 8
    entrance_rois: List[str] = field(default_factory=lambda: ["entrance1", "entrance2"])

    # Apply the speaker frequency-response compensation to the six grammar
    # tones. Gains are normalised so the least-attenuated tone keeps its
    # nominal amplitude and the others are boosted relative to it, then the
    # whole set is scaled so nothing exceeds 1.0 (no clipping). Set False to
    # reproduce sessions recorded before this option existed (uncompensated).
    grammar_apply_speaker_gain: bool = True

    # PATHS - override these for your machine
    base_output_path: str = _DEFAULT_BASE
    calibration_gain_path: str = _DEFAULT_CALIBRATION

    # Where the ROI rectangles are stored. ROIs belong to a rig, not to the
    # code, so the default lives next to the recordings rather than in the
    # source tree. Leave empty to use <base_output_path>/rois1.csv.
    roi_csv_path: str = ""

    # Top-level day label inserted between base_output_path and experiment_mode.
    # Set via --day on the CLI (e.g. habituation, day_1, day_2).
    # Leave empty to keep the old flat structure.
    experiment_day: str = ""

    # Vocalisation stimuli. The folder defaults to a "vocalisations" directory
    # next to the recordings folder (i.e. <parent of base_output_path>/vocalisations)
    # so your .wav files live with your data, not inside the installed package.
    # The control file is the single recording used on the "vocalisation" arm
    # of the mixed experiments; leave empty to make that arm silent (a warning
    # is printed at session start).
    path_to_vocalisation_folder: str = ""
    path_to_vocalisation_control: str = ""

    def __post_init__(self):
        if not self.roi_csv_path:
            self.roi_csv_path = os.path.join(self.base_output_path, "rois1.csv")
        if not self.path_to_vocalisation_folder:
            self.path_to_vocalisation_folder = os.path.join(
                os.path.dirname(os.path.normpath(self.base_output_path)), "vocalisations")


    def resolve_complex_interval_day(self) -> Dict[str, Any]:
        """Return the complex-intervals stimulus set, applying any overrides.

        Starts from the COMPLEX_INTERVAL_DAYS preset named by
        ``complex_interval_day`` and replaces any field the user has set
        explicitly. Raises ValueError for an unknown day name.
        """
        if self.complex_interval_day not in COMPLEX_INTERVAL_DAYS:
            raise ValueError(
                f"Unknown complex_interval_day: {self.complex_interval_day!r}. "
                f"Valid: {', '.join(COMPLEX_INTERVAL_DAYS)}")
        preset = dict(COMPLEX_INTERVAL_DAYS[self.complex_interval_day])
        if self.complex_consonant_intervals:
            preset["consonant"] = list(self.complex_consonant_intervals)
        if self.complex_dissonant_intervals:
            preset["dissonant"] = list(self.complex_dissonant_intervals)
        if self.complex_controls is not None:
            preset["controls"] = list(self.complex_controls)
        if self.complex_include_smooth is not None:
            preset["smooth"] = bool(self.complex_include_smooth)
        if self.complex_include_rough is not None:
            preset["rough"] = bool(self.complex_include_rough)
        return preset

    def get_trial_lengths(self) -> List[float]:

        # Grammar experiment: durations are taken from the dedicated
        # grammar_* fields above so you can override them without touching
        # this function.
        if self.experiment_mode == "grammar" and self.grammar_mode == "silent_baseline":
            # One continuous block; block_minutes does not apply.
            return [float(self.grammar_silent_baseline_minutes)]

        # A schedule set here wins for every mode.
        if self.block_minutes is not None:
            if len(self.block_minutes) != 9:
                raise ValueError(
                    f"block_minutes must have exactly 9 entries "
                    f"(got {len(self.block_minutes)}); even indices are silent "
                    f"blocks, odd indices are active blocks.")
            return [float(x) for x in self.block_minutes]

        if self.experiment_mode == "grammar":
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








