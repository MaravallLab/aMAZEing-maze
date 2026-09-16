"""The per-mode stimulus parameters are now config fields.

The first class pins the DEFAULTS to the values that were hard-coded in
experiments.py before v2, so an untouched config reproduces the stimuli used
for the original experiments. The rest check that editing a field actually
changes the stimuli.
"""

import numpy as np
import pytest


class TestDefaultsMatchOriginalHardCodedValues:
    """Guard against a default drifting away from the published protocol."""

    def test_simple_smooth_frequencies(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 8
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert list(df[df["trial_ID"] == 2]["frequency"]) == [
            10000, 12000, 14000, 16000, 18735, 20957, 22543, 24065]

    def test_simple_intervals_defaults(self, experiment_config):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.simple_interval_tonal_centre == 10000.0
        assert cfg.simple_intervals_list == [
            "perf_5", "perf_4", "maj_6", "tritone", "min_2", "maj_7"]

    def test_tem_defaults(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.tem_smooth_freqs == [10000, 20000]
        assert cfg.tem_constant_rough_freqs == [10000, 20000]
        assert cfg.tem_complex_rough_freqs == [10000, 20000]
        assert cfg.tem_constant_mod_freq == 50.0
        assert cfg.tem_complex_mod_freqs == [30, 50, 70]
        assert cfg.tem_mod_depth == 0.5
        assert cfg.tem_controls == ["vocalisation", "silent"]

    @pytest.mark.parametrize("day,consonant,dissonant,smooth,rough,controls", [
        ("w1day2", ["perf_5", "perf_4"], ["tritone", "min_7"], True, True,
         ["vocalisation", "silent"]),
        ("w1day3", ["maj_6", "min_3"], ["maj_7", "min_2"], True, True,
         ["vocalisation", "silent"]),
        ("w1day4", ["maj_3", "perf_4", "perf_5", "min_6"],
         ["min_7", "maj_2", "tritone", "maj_7"], False, False, []),
        ("another_day", ["maj_3", "perf_4", "perf_5"],
         ["min_7", "maj_2", "tritone"], False, False, ["vocalisation", "silent"]),
    ])
    def test_complex_interval_day_presets(self, day, consonant, dissonant,
                                          smooth, rough, controls):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(complex_interval_day=day)
        d = cfg.resolve_complex_interval_day()
        assert d["consonant"] == consonant
        assert d["dissonant"] == dissonant
        assert d["smooth"] is smooth and d["rough"] is rough
        assert d["controls"] == controls
        assert cfg.complex_interval_tonal_centre == 15000.0

    def test_sequence_defaults(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.sequence_patterns == ["AAAAA", "AoAo", "ABAB", "ABCABC",
                                         "BABA", "ABBA", "silence", "vocalisation"]
        assert cfg.sequence_repetitions == 50
        assert cfg.sequence_tone_map == {}      # empty = ask on the console


class TestEditingParametersChangesStimuli:

    def test_custom_smooth_frequencies(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 3
        experiment_config.smooth_frequencies = [5000, 6000, 7000]
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert list(df[df["trial_ID"] == 2]["frequency"]) == [5000, 6000, 7000]

    def test_complex_interval_overrides_beat_the_preset(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(complex_interval_day="w1day2",
                               complex_consonant_intervals=["octave"],
                               complex_controls=[],
                               complex_include_rough=False)
        d = cfg.resolve_complex_interval_day()
        assert d["consonant"] == ["octave"]
        assert d["dissonant"] == ["tritone", "min_7"]   # untouched preset value
        assert d["controls"] == []
        assert d["rough"] is False
        assert d["smooth"] is True

    def test_unknown_day_raises(self):
        from amazeing.auditory.config import ExperimentConfig
        with pytest.raises(ValueError, match="Unknown complex_interval_day"):
            ExperimentConfig(complex_interval_day="nope").resolve_complex_interval_day()

    def test_tem_depth_reaches_the_waveform(self, mock_audio):
        """The depth argument was previously ignored by the complex AM generator."""
        shallow = mock_audio.generate_complex_tem_sound_data(
            10000, duration_s=0.2, ramp_duration_s=0.0, depth=0.1)
        deep = mock_audio.generate_complex_tem_sound_data(
            10000, duration_s=0.2, ramp_duration_s=0.0, depth=0.9)
        # deeper modulation = larger spread between envelope peaks and troughs
        assert np.std(np.abs(deep)) > np.std(np.abs(shallow))

    def test_sequences_from_config_need_no_prompts(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "sequences"
        experiment_config.rois_number = 3
        experiment_config.sequence_patterns = ["ABAB", "silence", "AoAo"]
        experiment_config.sequence_tone_map = {"A": 8000, "B": 12000}
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        active = df[df["trial_ID"] == 2]
        assert list(active["pattern"]) == ["ABAB", "silence", "AoAo"]
        first = active[active["ROIs"] == "1"]["frequency"].iloc[0]
        assert first[:4] == [8000, 12000, 8000, 12000]

    def test_sequences_missing_tone_raises(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "sequences"
        experiment_config.rois_number = 1
        experiment_config.sequence_patterns = ["ABC"]
        experiment_config.sequence_tone_map = {"A": 8000}
        with pytest.raises(ValueError, match="no frequency for"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    def test_vocalisation_silent_arm_flag(self):
        from amazeing.auditory.config import ExperimentConfig
        assert ExperimentConfig().vocalisation_include_silent_arm is True


class TestShuffleTerminates:
    """Few arms means few possible orderings; the search must not spin forever."""

    @pytest.mark.parametrize("mode,extra", [
        ("simple_smooth", {"smooth_frequencies": [8000, 9000]}),
        ("custom", {"custom_stimuli": [{"roi": "1", "kind": "tone", "frequency": 8000},
                                       {"roi": "2", "kind": "tone", "frequency": 9000}]}),
    ])
    def test_two_arms_still_finishes(self, mock_audio, experiment_config, mode, extra):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = mode
        experiment_config.rois_number = 2
        for k, v in extra.items():
            setattr(experiment_config, k, v)
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert len(df) == 9 * 2
        # four active blocks, each with both stimuli present
        for tid in (2, 4, 6, 8):
            assert len(df[df["trial_ID"] == tid]) == 2

    def test_single_arm_finishes(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = 1
        experiment_config.smooth_frequencies = [8000]
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert len(df) == 9

    @pytest.mark.parametrize("n_freqs,n_arms", [(1, 8), (3, 8), (2, 5), (8, 8)])
    def test_short_frequency_list_is_recycled_to_fill_every_arm(
            self, mock_audio, experiment_config, n_freqs, n_arms):
        """Doubling the list once was not enough below half the arm count."""
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.rois_number = n_arms
        experiment_config.smooth_frequencies = [10000 + 100 * k for k in range(n_freqs)]
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert len(df) == 9 * n_arms
        assert len(df[df["trial_ID"] == 2]) == n_arms

    def test_empty_frequency_list_is_reported(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_smooth"
        experiment_config.smooth_frequencies = []
        with pytest.raises(ValueError, match="at least one frequency"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)
