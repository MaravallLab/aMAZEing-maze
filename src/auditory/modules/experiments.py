# here we handle all things related to the trials generations

# basically anything that in version_1's supfun_sequences starts with create_ ..._trial or has info in their names

import numpy as np
import pandas as pd
import random
import os
from typing import Tuple, List, Any, Dict, Union, Optional
from modules.audio import Audio
from config import ExperimentConfig

#this will be the structure of the output of the trial generation. A dataframe containing all the trials information + the list of the sound sound_arrays
TrialData = Tuple[pd.DataFrame, List[Any]]

# ── helpers ──────────────────────────────────────────────────────────

def _make_hashable(x):
    """Convert lists to tuples so they can go in a set."""
    if isinstance(x, list):
        return tuple(x)
    return x


def _add_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append the standard behavioural-tracking columns that main.py expects."""
    n = len(df)
    df["time_spent"] = [None] * n
    df["visitation_count"] = [None] * n
    df["time_in_maze_ms"] = [0] * n
    df["trial_start_time"] = [None] * n
    df["end_trial_time"] = [None] * n
    return df


class ExperimentFactory:

    # this class will act like a "menu" that generates the correct trial structure based on the experiment mode in config.py
    # we generate the sound data that will go in the df with the functions we defined in audio.py

    # ── trial-structure constants ────────────────────────────────────
    TOTAL_REPETITIONS = 9   # how many blocks (silent + active interleaved)
    SAMPLE_RATE = 192000    # used only for silence arrays

    @staticmethod
    def generate_trials(cfg: ExperimentConfig, audio: Audio) -> TrialData:

        generic_rois = [f"ROI{str(i+1)}" for i in range(cfg.rois_number)]
        rois_list = cfg.entrance_rois + generic_rois

        experiment_type = cfg.experiment_mode

        print(f"generating trials for {experiment_type}")

        if experiment_type == "simple_smooth":
            return ExperimentFactory._make_simple_smooth(generic_rois, cfg, audio)
        elif experiment_type == "simple_intervals":
            return ExperimentFactory._make_simple_intervals(generic_rois, cfg, audio)
        elif experiment_type == "temporal_envelope_modulation":
            return ExperimentFactory._make_temporal_envelope_modulation(generic_rois, cfg, audio)
        elif experiment_type == "complex_intervals":
            return ExperimentFactory._make_complex_intervals(generic_rois, cfg, audio)
        elif experiment_type == "sequences":
            return ExperimentFactory._make_sequences(generic_rois, cfg, audio)
        elif experiment_type == "vocalisation":
            return ExperimentFactory._make_vocalisation(generic_rois, cfg, audio)
        elif experiment_type == "semantic_predictive_complexity":
            raise NotImplementedError("semantic_predictive_complexity is not yet implemented")
        else:
            raise ValueError(f"Unknown experiment mode: {experiment_type}")


    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT-SPECIFIC SETUP
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_simple_smooth(rois: List[str], cfg: ExperimentConfig, audio: Audio) -> TrialData:
        frequencies = [10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065]

        if len(frequencies) < len(rois):
            print("not enough frequencies for ROIs. Recycling. If you want to add/modify, go to experiments.py,  _make_simple_smooth().")
            frequencies = (frequencies * 2)[:len(rois)]

        #create trial structure
        return ExperimentFactory._create_simple_trials_logic(rois, frequencies, audio)

    @staticmethod
    def _make_simple_intervals(rois: List[str], cfg: ExperimentConfig, audio: Audio, manual: bool = False) -> TrialData:
        rois_number = cfg.rois_number

        if manual: # if manual =True, prompt the user for the intervals
            frequency, intervals, intervals_names = ExperimentFactory._ask_info_intervals(rois_number)
        else:
            tonal_centre = 10000 #Hz
            intervals_list = ["perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]
            frequency, intervals, intervals_names = ExperimentFactory._get_info_intervals_hard_coded(rois, tonal_centre, intervals_list)

        return ExperimentFactory._create_intervals_trials_logic(rois, frequency, intervals, intervals_names, audio)

    @staticmethod
    def _make_temporal_envelope_modulation(rois: List[str], cfg: ExperimentConfig, audio: Audio) -> TrialData:
        rois_number = cfg.rois_number

        # okay, so, this adds another layer of control. You user can choose which frequencies will be smooth, which with constant Amplitude Modulation, and which with variable AM

        smooth_freqs = [10000, 20000]
        constant_rough_freqs = [10000, 20000]
        #constant temporal modulation
        ctemporal_modulation = 50 #Hz
        complex_rough_freqs = [10000, 20000]

        #complex temporal modulation
        complex_temporal_modulation = [30, 50, 70] # Hz

        controls = ["vocalisation", "silent"]

        path_to_vocalisation = cfg.path_to_vocalisation_control

        frequencies, temporal_modulation, sound_type, sound_arrays = ExperimentFactory._get_info_tem_hard_coded(
            rois_number,
            controls,
            smooth_freqs,
            constant_rough_freqs,
            complex_rough_freqs,
            constant_rough_modulation=ctemporal_modulation,
            complex_rough_mod=complex_temporal_modulation,
            audio=audio,
            path_to_voc=path_to_vocalisation,
        )

        return ExperimentFactory._create_tem_trials_logic(rois, frequencies, temporal_modulation, sound_type, sound_arrays, audio)


    @staticmethod
    def _make_complex_intervals(rois: List[str], cfg: ExperimentConfig, audio: Audio) -> TrialData:
        experiment_day = cfg.complex_interval_day

        tonal_centre = 15000
        path_to_voc = cfg.path_to_vocalisation_control
        smooth_freq = False
        rough_freq = False
        controls = ["vocalisation", "silent"] #"vocalisation", "silent"

        #"w1day2", "w1day3", "w1day4", "another_day"
        if experiment_day == "w1day2":
            smooth_freq = True; rough_freq = True
            consonant_intervals = ["perf_5", "perf_4"]
            dissonant_intervals = ["tritone", "min_7"]

        elif experiment_day == "w1day3":
            smooth_freq = True; rough_freq = True
            consonant_intervals = ["maj_6", "min_3"]
            dissonant_intervals = ["maj_7", "min_2"]

        elif experiment_day == "w1day4":
            consonant_intervals = ["maj_3", "perf_4", "perf_5", "min_6"]
            dissonant_intervals = ["min_7", "maj_2", "tritone", "maj_7"]
            controls = []

        elif experiment_day == "another_day":
            consonant_intervals = ["maj_3", "perf_4", "perf_5"]
            dissonant_intervals = ["min_7", "maj_2", "tritone"]
        else:
            raise ValueError(f"Unknown complex_interval_day: {experiment_day}")

        frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays = ExperimentFactory._get_info_complex_intervals_hard_coded(
            len(rois),
            controls,
            tonal_centre,
            smooth_freq,
            rough_freq,
            consonant_intervals,
            dissonant_intervals,
            audio=audio,
            path_to_voc=path_to_voc,
        )

        return ExperimentFactory._create_complex_intervals_trials_logic(
            rois, frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays, audio
        )


    @staticmethod
    def _make_sequences(rois: List[str], cfg: ExperimentConfig, audio: Audio) -> TrialData:
        # Interactive Setup (Ported from ask_music_info_sequences)
        intervals_vs_custom = input("Would you like to add CUSTOM values or generate sequences based on INTERVALS? (c / i): ").lower().strip()

        sequence_of_frequencies = []
        pattern_list = []

        if intervals_vs_custom in ("c", "custom"):
            ask_input = input("Make new patterns? (y=insert manually, n=use defaults): ").lower().strip()

            if ask_input == 'y':
                # Manual Entry
                for i in range(len(rois)):
                    p = input(f"Insert pattern #{i+1} (e.g. AoAo, ABAB, random): ").strip()
                    # Standardize names
                    if p.lower() in ["random", "silence", "vocalisation"]:
                        p = p.lower()
                    elif p in ["AoAo", "aoao", "AOAO"]:
                        p = "AoAo"
                    else:
                        p = p.upper()
                    pattern_list.append(p)
            else:
                # Hardcoded defaults
                defaults = ['AAAAA', 'AoAo', 'ABAB', 'ABCABC', 'BABA', 'ABBA', "silence", "vocalisation"]
                # Adjust to ROI count
                if len(rois) > len(defaults):
                    defaults = (defaults * 2)
                pattern_list = defaults[:len(rois)]

            # Map characters to frequencies
            # Exclude special keywords
            patterns_nonpatterns = ["silence", "vocalisation", "random"]

            # get the sorted individual events in the sequences and map them to frequencies
            events = []
            freqs = []
            for i in sorted(pattern_list):
                if i not in patterns_nonpatterns:
                    for j in i:
                        if j not in events:
                            events.append(j)
                            # ask the user the frequency for the event
                            if j != 'o':
                                freq = int(input(f"Insert frequency for sound {j}:\n"))
                                freqs.append(freq)
                            else:
                                freqs.append(0)

            # map frequency to event in a dictionary
            sound_dict = dict(zip(events, freqs))

            # Generate Sequences (Lists of frequencies)
            repetitions = 50
            for item in pattern_list:
                if item == "random":
                    # Random selection from defined events
                    keys = list(sound_dict.keys())
                    if not keys:
                        keys = [10000]  # Fallback
                    seq = [random.choice(keys) for _ in range(200)]
                    sequence_of_frequencies.append([sound_dict.get(k, 0) for k in seq])

                elif item == "vocalisation":
                    sequence_of_frequencies.append("vocalisation")

                elif item == "silence":
                    sequence_of_frequencies.append([0] * 200)

                else:
                    # Standard pattern (e.g. ABAB)
                    full_str = (item * repetitions)[:200]  # Cap length
                    seq_freqs = []
                    for char in full_str:
                        seq_freqs.append(sound_dict.get(char, 0))
                    sequence_of_frequencies.append(seq_freqs)

        else:
            # intervals mode (not fully implemented in legacy)
            raise NotImplementedError("Intervals-based sequence generation is not implemented. Use custom mode instead.")

        path_to_voc = cfg.path_to_vocalisation_control

        return ExperimentFactory._create_sequence_trials_logic(rois, sequence_of_frequencies, pattern_list, audio, path_to_voc)


    @staticmethod
    def _make_vocalisation(rois: List[str], cfg: ExperimentConfig, audio: Audio) -> TrialData:
        """All-vocalisation experiment: each ROI plays a different vocalisation file."""
        path_to_vocalisations_folder = cfg.path_to_vocalisation_folder
        silent_arm = True

        if not os.path.isdir(path_to_vocalisations_folder):
            raise FileNotFoundError(f"Vocalisation folder not found: {path_to_vocalisations_folder}")

        file_names = os.listdir(path_to_vocalisations_folder)
        stimuli = [os.path.join(path_to_vocalisations_folder, f) for f in file_names]

        if silent_arm:
            stimuli.append("silent")

        # Adjust to ROI count
        if len(stimuli) < len(rois):
            difference = len(rois) - len(stimuli)
            for _ in range(difference):
                stimuli.append(random.choice(stimuli[:-1]))  # don't duplicate "silent"
        elif len(stimuli) > len(rois):
            stimuli = stimuli[:len(rois)]

        # Build per-ROI info (mirrors vocalisations_info_hc)
        frequencies = []
        interval_numerical_list = []
        interval_string_names = []
        sound_type = []
        sounds_arrays = []

        for stim in stimuli:
            interval_numerical_list.append(0)
            interval_string_names.append("none")
            sound_type.append("control")

            if stim == "silent":
                frequencies.append(0)
                z = np.zeros(int(audio.fs * audio.default_duration))
                sounds_arrays.append([z, z])
            else:
                frequencies.append(stim)
                voc = audio.load_wav(stim)
                silence = np.zeros_like(voc)
                sounds_arrays.append([voc, silence])

        return ExperimentFactory._create_complex_intervals_trials_logic(
            rois, frequencies, interval_numerical_list, interval_string_names,
            sound_type, sounds_arrays, audio
        )


    # ════════════════════════════════════════════════════════════════
    # TRIAL CREATION LOGIC
    # ════════════════════════════════════════════════════════════════
    #
    # These methods build the trials DataFrame + wave_arrays list.
    # The pattern is the same for every experiment type:
    #   - 9 blocks (total_repetitions), odd blocks are silent
    #   - even blocks shuffle the ROI↔stimulus mapping
    #   - first active block (i==1) keeps the original order
    #   - subsequent active blocks are unique random permutations
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_simple_trials_logic(
        rois: List[str],
        frequencies: List[float],
        audio: Audio,
        total_repetitions: int = 9,
    ) -> TrialData:
        """Create trials for simple smooth sounds (one frequency per ROI)."""

        rois_repeated = rois * total_repetitions
        frequency_final = []
        wave_arrays = []
        repetition_numbers = []
        previous_trials = set()

        for i in range(total_repetitions):
            if i % 2 == 0:
                # Silent block
                for _ in rois:
                    repetition_numbers.append(i + 1)
                    frequency_final.append(0)
                    wave_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                while True:
                    if i == 1:
                        trial_tuple = tuple(frequencies)
                    else:
                        trial_list = list(frequencies)
                        random.shuffle(trial_list)
                        trial_tuple = tuple(trial_list)

                    if trial_tuple not in previous_trials:
                        previous_trials.add(trial_tuple)
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)
                            freq = trial_tuple[j] if i != 1 else frequencies[j]
                            frequency_final.append(freq)
                            wave_arrays.append(audio.generate_sound_data(freq))
                        break

        df = pd.DataFrame({
            "trial_ID": repetition_numbers,
            "ROIs": rois_repeated,
            "frequency": frequency_final,
            "wave_arrays": wave_arrays,
        })
        df = _add_tracking_columns(df)
        return df, wave_arrays


    @staticmethod
    def _create_intervals_trials_logic(
        rois: List[str],
        frequency: List,         # list of [f1, f2] pairs or "vocalisation"
        intervals: List,         # interval ratio strings
        intervals_names: List,   # interval name strings
        audio: Audio,
        total_repetitions: int = 9,
    ) -> TrialData:
        """Create trials for simple interval experiments (two-tone chords per ROI)."""

        rois_repeated = rois * total_repetitions
        frequency_final = []
        intervals_final = []
        intervals_names_final = []
        wave_arrays = []
        repetition_numbers = []
        previous_trials = set()

        # Pre-generate dual sound arrays for each ROI
        dual_array_sounds = []
        for freq_pair in frequency:
            freq_sounds = []
            if isinstance(freq_pair, str) and freq_pair == "vocalisation":
                # handled at playback; store sentinel
                freq_sounds = ["vocalisation"]
            elif freq_pair[1] != 0:
                for f in freq_pair:
                    freq_sounds.append(audio.generate_sound_data(f))
            else:
                # Silent interval
                for f in freq_pair:
                    freq_sounds.append(audio.generate_sound_data(f))
            dual_array_sounds.append(freq_sounds)

        for i in range(total_repetitions):
            if i % 2 == 0:
                for _ in rois:
                    repetition_numbers.append(i + 1)
                    frequency_final.append(0)
                    intervals_final.append(0)
                    intervals_names_final.append(0)
                    wave_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                while True:
                    if i == 1:
                        trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))
                    else:
                        trial_list = list(zip(frequency, intervals, intervals_names, dual_array_sounds))
                        random.shuffle(trial_list)

                    trial_tuple_as_tuple = tuple(item[2] for item in trial_list)

                    if trial_tuple_as_tuple not in previous_trials:
                        previous_trials.add(trial_tuple_as_tuple)
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)
                            freq, inter, inter_name, wave = trial_list[j]
                            frequency_final.append(freq)
                            intervals_final.append(inter)
                            intervals_names_final.append(inter_name)
                            wave_arrays.append(tuple(wave))
                        break

        df = pd.DataFrame({
            "trial_ID": repetition_numbers,
            "ROIs": rois_repeated,
            "interval": intervals_names_final,
            "interval_ratio": intervals_final,
            "frequency": frequency_final,
            "wave_arrays": wave_arrays,
        })
        df = _add_tracking_columns(df)
        return df, wave_arrays


    @staticmethod
    def _create_tem_trials_logic(
        rois: List[str],
        frequency: List,
        temporal_modulation: List,
        sound_type: List,
        sounds_arrays: List,
        audio: Audio,
        total_repetitions: int = 9,
    ) -> TrialData:
        """Create trials for temporally envelope-modulated sounds."""

        rois_repeated = rois * total_repetitions
        frequency_final = []
        temporal_modulations_final = []
        sound_type_final = []
        wave_arrays = []
        repetition_numbers = []
        previous_trials = set()

        for i in range(total_repetitions):
            if i % 2 == 0:
                for _ in rois:
                    repetition_numbers.append(i + 1)
                    frequency_final.append(0)
                    temporal_modulations_final.append("none")
                    sound_type_final.append("silent_trial")
                    wave_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                while True:
                    if i == 1:
                        trial_triples = []
                        for idx in range(len(rois)):
                            freq = frequency[idx]
                            mod = _make_hashable(temporal_modulation[idx])
                            typ = sound_type[idx]
                            trial_triples.append((freq, mod, typ))
                        trial_tuple_as_tuple = tuple(trial_triples)
                        trial_list = list(zip(frequency, temporal_modulation, sound_type, sounds_arrays))
                    else:
                        combined = list(zip(frequency, temporal_modulation, sound_type, sounds_arrays))
                        random.shuffle(combined)
                        trial_triples = []
                        for (freq, mod, typ, snd) in combined:
                            trial_triples.append((freq, _make_hashable(mod), typ))
                        trial_tuple_as_tuple = tuple(trial_triples)
                        trial_list = combined

                    if trial_tuple_as_tuple not in previous_trials:
                        previous_trials.add(trial_tuple_as_tuple)

                        if i == 1:
                            for idx in range(len(rois)):
                                repetition_numbers.append(i + 1)
                                frequency_final.append(frequency[idx])
                                temporal_modulations_final.append(temporal_modulation[idx])
                                sound_type_final.append(sound_type[idx])
                                wave_arrays.append(sounds_arrays[idx])
                        else:
                            for (freq_shuf, mod_shuf, typ_shuf, sounds_shuf) in trial_list:
                                repetition_numbers.append(i + 1)
                                frequency_final.append(freq_shuf)
                                temporal_modulations_final.append(mod_shuf)
                                sound_type_final.append(typ_shuf)
                                wave_arrays.append(sounds_shuf)
                        break

        df = pd.DataFrame({
            "trial_ID": repetition_numbers,
            "ROIs": rois_repeated,
            "frequency": frequency_final,
            "sound_type": sound_type_final,
            "temporal_modulation": temporal_modulations_final,
            "wave_arrays": wave_arrays,
        })
        df = _add_tracking_columns(df)
        return df, wave_arrays


    @staticmethod
    def _create_complex_intervals_trials_logic(
        rois: List[str],
        frequency: List,
        interval_numerical_list: List,
        interval_string_names: List,
        sound_type: List,
        sounds_arrays: List,
        audio: Audio,
        total_repetitions: int = 9,
    ) -> TrialData:
        """Create trials for complex intervals / vocalisation experiments."""

        rois_repeated = rois * total_repetitions
        frequency_final = []
        interval_numerical_list_final = []
        interval_string_names_final = []
        sound_type_final = []
        wave_arrays = []
        repetition_numbers = []
        previous_trials = set()

        for i in range(total_repetitions):
            if i % 2 == 0:
                for _ in rois:
                    repetition_numbers.append(i + 1)
                    frequency_final.append(0)
                    interval_numerical_list_final.append(0)
                    interval_string_names_final.append(0)
                    sound_type_final.append("silent_trial")
                    wave_arrays.append((0, 0))
            else:
                while True:
                    if i == 1:
                        trial_triples = []
                        for idx in range(len(rois)):
                            freq = _make_hashable(frequency[idx])
                            int_num = interval_numerical_list[idx]
                            int_name = interval_string_names[idx]
                            typ = sound_type[idx]
                            trial_triples.append((freq, int_num, int_name, typ))
                        trial_tuple_as_tuple = tuple(trial_triples)
                        trial_list = list(zip(frequency, interval_numerical_list, interval_string_names, sound_type, sounds_arrays))
                    else:
                        combined = list(zip(frequency, interval_numerical_list, interval_string_names, sound_type, sounds_arrays))
                        random.shuffle(combined)
                        trial_triples = []
                        for (freq, int_num, int_name, typ, snd) in combined:
                            trial_triples.append((_make_hashable(freq), int_num, int_name, typ))
                        trial_tuple_as_tuple = tuple(trial_triples)
                        trial_list = combined

                    if trial_tuple_as_tuple not in previous_trials:
                        previous_trials.add(trial_tuple_as_tuple)

                        if i == 1:
                            for idx in range(len(rois)):
                                repetition_numbers.append(i + 1)
                                frequency_final.append(frequency[idx])
                                interval_numerical_list_final.append(interval_numerical_list[idx])
                                interval_string_names_final.append(interval_string_names[idx])
                                sound_type_final.append(sound_type[idx])
                                wave_arrays.append(tuple(sounds_arrays[idx]))
                        else:
                            for (freq_shuf, int_num_shuf, int_name_shuf, typ_shuf, sounds_shuf) in trial_list:
                                repetition_numbers.append(i + 1)
                                frequency_final.append(freq_shuf)
                                interval_numerical_list_final.append(int_num_shuf)
                                interval_string_names_final.append(int_name_shuf)
                                sound_type_final.append(typ_shuf)
                                wave_arrays.append(tuple(sounds_shuf))
                        break

        df = pd.DataFrame({
            "trial_ID": repetition_numbers,
            "ROIs": rois_repeated,
            "frequency": frequency_final,
            "interval_type": sound_type_final,
            "interval_ratio": interval_numerical_list_final,
            "interval_name": interval_string_names_final,
            "wave_arrays": wave_arrays,
        })
        df = _add_tracking_columns(df)
        return df, wave_arrays


    @staticmethod
    def _create_sequence_trials_logic(
        rois: List[str],
        frequency: List,       # list of frequency-sequences per ROI, or "vocalisation"
        patterns: List[str],   # pattern strings per ROI
        audio: Audio,
        path_to_voc: Optional[str] = None,
        total_repetitions: int = 9,
    ) -> TrialData:
        """Create trials for sequence experiments (tone patterns per ROI)."""

        rois_repeated = rois * total_repetitions
        frequency_final = []
        wave_arrays = []
        repetition_numbers = []
        patterns_final = []
        previous_trials = set()

        for i in range(total_repetitions):
            if i % 2 == 0:
                for _ in rois:
                    repetition_numbers.append(i + 1)
                    frequency_final.append(0)
                    patterns_final.append(0)
                    wave_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                while True:
                    if i == 1:
                        trial_list = list(zip(frequency, patterns))
                    else:
                        trial_list = list(zip(frequency, patterns))
                        random.shuffle(trial_list)

                    # Build hashable key from (frequency-tuple, pattern-tuple) pairs
                    trial_tuple_as_tuple = tuple(
                        (tuple(freq) if isinstance(freq, list) else freq, pat)
                        for freq, pat in trial_list
                    )

                    if trial_tuple_as_tuple not in previous_trials:
                        previous_trials.add(trial_tuple_as_tuple)
                        for j in range(len(rois)):
                            repetition_numbers.append(i + 1)
                            freq, pat = trial_list[j]
                            frequency_final.append(freq)
                            patterns_final.append(pat)

                            # Generate sound: concatenate short tones for each element
                            if freq == "vocalisation":
                                if path_to_voc:
                                    sound = audio.load_wav(path_to_voc)
                                else:
                                    sound = np.zeros(int(audio.fs * audio.default_duration))
                                wave_arrays.append(sound)
                            elif isinstance(freq, list):
                                concatenated = []
                                for f in freq:
                                    concatenated.append(audio.generate_sound_data(f, duration_s=0.04))
                                wave_arrays.append(np.concatenate(concatenated))
                            else:
                                wave_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
                        break

        df = pd.DataFrame({
            "trial_ID": repetition_numbers,
            "ROIs": rois_repeated,
            "pattern": patterns_final,
            "frequency": frequency_final,
            "wave_arrays": wave_arrays,
        })
        df = _add_tracking_columns(df)
        return df, wave_arrays


    # ════════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS — stimulus info getters
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_interval(interval_name: str) -> Tuple[float, str]:
        """Return (numerical_ratio, ratio_string) for a named musical interval."""

        intervals_names = [
            "unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone",
            "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave",
        ]
        intervals_values = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 64/45, 3/2, 8/5, 5/3, 16/9, 15/8, 2]
        intervals_values_strings = [
            "1/1", "16/15", "9/8", "6/5", "5/4", "4/3", "45/32", "3/2",
            "8/5", "5/3", "16/9", "15/8", "2/1",
        ]

        intervals = dict(zip(intervals_names, intervals_values))
        intervals_strings = dict(zip(intervals_names, intervals_values_strings))

        return intervals[interval_name], intervals_strings[interval_name]

    @staticmethod
    def _ask_info_intervals(rois_number: int):
        """Interactive prompt for interval selection."""

        intervals_names = [
            "unison", "min_2", "maj_2", "min_3", "maj_3", "perf_4", "tritone",
            "perf_5", "min_6", "maj_6", "min_7", "maj_7", "octave",
        ]

        consonant_intervals = [intervals_names[i] for i in (0, 3, 4, 5, 7, 8, 9, 12)]
        dissonant_intervals = [intervals_names[i] for i in (1, 2, 6, 10, 11)]

        print(f"You will now be prompted to select the stimuli for the {rois_number} ROIs")
        new_rois_number = rois_number

        frequencies = []
        interval_numerical_list = []
        interval_string_names = []

        # ask if vocalisation
        vocalisation = input("Do you want to include a vocalisation recording?(y/n)\n").strip().lower()
        if vocalisation == "y":
            frequencies.append("vocalisation")
            interval_numerical_list.append([9])
            interval_string_names.append("vocalisation")
            new_rois_number -= 1

        # ask if silence
        print(f"You have {new_rois_number} ROIs available")
        silence = input("do you want a Silent ROI?(y/n)\n").strip().lower()
        if silence == "y":
            frequencies.append([0, 0])
            interval_numerical_list.append([0])
            interval_string_names.append("no_interval")
            new_rois_number -= 1

        print(f"You have {new_rois_number} ROIs available")
        number_consonants = int(input("insert the number of consonant rois: \n").strip())
        new_rois_number = new_rois_number - number_consonants
        number_dissonants = new_rois_number

        print(f"your number of dissonant rois is: {number_dissonants}")

        tonal_centre = int(input("insert the frequency that will be the tonal centre:\n"))

        for i in range(number_consonants):
            consonant_choice = input(f"insert the consonant interval of choice #{i+1} {consonant_intervals}:\n")
            consonant_choice = consonant_choice.lower()
            interval, interval_as_string = ExperimentFactory._get_interval(consonant_choice)
            frequencies.append([tonal_centre, int(tonal_centre * interval)])
            interval_numerical_list.append(interval_as_string)
            interval_string_names.append(consonant_choice)

        for i in range(number_dissonants):
            dissonant_choice = input(f"insert the dissonant interval of choice #{i+1} {dissonant_intervals}:\n")
            dissonant_choice = dissonant_choice.lower()
            interval, interval_as_string = ExperimentFactory._get_interval(dissonant_choice)
            frequencies.append([tonal_centre, int(tonal_centre * interval)])
            interval_numerical_list.append(interval_as_string)
            interval_string_names.append(dissonant_choice)

        return frequencies, interval_numerical_list, interval_string_names


    @staticmethod
    def _get_info_intervals_hard_coded(rois, tonal_centre, intervals_list):
        """Hard-coded interval info (no user prompts)."""
        rois_number = len(rois) if isinstance(rois, list) else rois

        # usable_rois exclude the unison and silent arm
        usable_rois = rois_number - 2

        tonal_centre_interval, tonal_centre_string = ExperimentFactory._get_interval("unison")
        frequencies = [[tonal_centre, int(tonal_centre * tonal_centre_interval)]]
        interval_numerical_list = [tonal_centre_string]
        interval_string_names = ["unison"]

        if len(intervals_list) == usable_rois:
            for i in range(usable_rois):
                if intervals_list[i] != "vocalisation":
                    interval, interval_as_string = ExperimentFactory._get_interval(intervals_list[i])
                    frequencies.append([tonal_centre, int(tonal_centre * interval)])
                    interval_numerical_list.append(interval_as_string)
                    interval_string_names.append(intervals_list[i])
                else:
                    frequencies.append("vocalisation")
                    interval_numerical_list.append([9])
                    interval_string_names.append("vocalisation")

            # append the silent frequency
            frequencies.append([0, 0])
            interval_numerical_list.append(["0"])
            interval_string_names.append("no_interval")
        else:
            print("please check that the number of intervals is rois_number - 2")

        return frequencies, interval_numerical_list, interval_string_names


    @staticmethod
    def _get_info_tem_hard_coded(
        rois_number,
        controls,
        smooth_freqs,
        constant_rough_freqs,
        complex_rough_freqs,
        constant_rough_modulation=50,
        complex_rough_mod=None,
        audio: Optional[Audio] = None,
        path_to_voc: Optional[str] = None,
    ):
        """Build stimulus lists for temporal-envelope-modulation experiments."""
        if complex_rough_mod is None:
            complex_rough_mod = [30, 50, 70]

        freqs = controls + smooth_freqs + constant_rough_freqs + complex_rough_freqs
        frequencies = []
        temporal_modulation = []
        sound_type = []
        sound_arrays = []

        if len(freqs) != rois_number:
            print("Bestie, double check the number of stimuli and make sure they match the number of rois")

        for item in controls:
            if item == "silent":
                frequencies.append("silent_arm")
                temporal_modulation.append("no_stimulus")
                sound_type.append("control")
                sound_arrays.append(np.zeros(int(audio.fs * audio.default_duration)))
            else:
                frequencies.append("vocalisation")
                temporal_modulation.append("vocalisation")
                sound_type.append("control")
                sound_arrays.append(audio.load_wav(path_to_voc))

        for f in smooth_freqs:
            frequencies.append(f)
            temporal_modulation.append("none")
            sound_type.append("smooth")
            sound_arrays.append(audio.generate_sound_data(f))

        for f in constant_rough_freqs:
            frequencies.append(f)
            temporal_modulation.append(constant_rough_modulation)
            sound_type.append("rough")
            sound_arrays.append(audio.generate_simple_tem_sound_data(f, modulated_frequency=constant_rough_modulation))

        for f in complex_rough_freqs:
            frequencies.append(f)
            temporal_modulation.append(complex_rough_mod)
            sound_type.append("rough_complex")
            sound_arrays.append(audio.generate_complex_tem_sound_data(f, modulated_frequencies_list=complex_rough_mod))

        return frequencies, temporal_modulation, sound_type, sound_arrays


    @staticmethod
    def _get_info_complex_intervals_hard_coded(
        rois_number,
        controls,
        tonal_centre,
        smooth_freq,
        rough_freq,
        consonant_intervals,
        dissonant_intervals,
        audio: Optional[Audio] = None,
        path_to_voc: Optional[str] = None,
    ):
        """Build stimulus lists for complex-intervals experiments."""

        all_intervals = consonant_intervals + dissonant_intervals

        frequencies = []
        interval_numerical_list = []
        interval_string_names = []
        sound_type = []
        sounds_arrays = []

        for ctrl in controls:
            interval_numerical_list.append(0)
            interval_string_names.append(ctrl)
            sound_type.append(ctrl)

            if ctrl == "silent":
                frequencies.append(0)
                z = np.zeros(int(audio.fs * audio.default_duration))
                sounds_arrays.append([z, z])
            else:
                frequencies.append(ctrl)
                voc = audio.load_wav(path_to_voc)
                silence = np.zeros_like(voc)
                sounds_arrays.append([voc, silence])

        if smooth_freq:
            tonal_centre_interval, tonal_centre_string = ExperimentFactory._get_interval("unison")
            frequencies.append([tonal_centre, int(tonal_centre * tonal_centre_interval)])
            interval_numerical_list.append(tonal_centre_string)
            interval_string_names.append("unison")
            s = audio.generate_sound_data(tonal_centre)
            sounds_arrays.append([s, s])
            sound_type.append("smooth")

        if rough_freq:
            tonal_centre_interval, tonal_centre_string = ExperimentFactory._get_interval("unison")
            frequencies.append([tonal_centre, int(tonal_centre * tonal_centre_interval)])
            interval_numerical_list.append(tonal_centre_string)
            interval_string_names.append("unison")
            sound_type.append("rough")
            modulated_wave = audio.generate_simple_tem_sound_data(tonal_centre)
            sounds_arrays.append([modulated_wave, modulated_wave])

        for interval_name in all_intervals:
            interval, interval_string = ExperimentFactory._get_interval(interval_name)
            freq_1 = tonal_centre
            freq_2 = tonal_centre * interval

            frequencies.append([freq_1, freq_2])
            interval_numerical_list.append(interval_string)
            interval_string_names.append(interval_name)

            sound_1 = audio.generate_sound_data(tonal_centre)
            sound_2 = audio.generate_sound_data(freq_2)
            sounds_arrays.append([sound_1, sound_2])

            if interval_name in consonant_intervals:
                sound_type.append("consonant")
            else:
                sound_type.append("dissonant")

        return frequencies, interval_numerical_list, interval_string_names, sound_type, sounds_arrays
