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


class TestStimulusCountMatchesArmCount:
    """A mismatch used to print a warning and build a malformed trial table."""

    def test_intervals_mismatch_is_refused(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "simple_intervals"
        experiment_config.rois_number = 4            # needs 2 intervals, has 6
        with pytest.raises(ValueError, match="8 stimuli.*4 arms"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    def test_tem_mismatch_is_refused(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "temporal_envelope_modulation"
        experiment_config.rois_number = 4            # the default set fills 8
        with pytest.raises(ValueError, match="8 stimuli.*4 arms"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    def test_complex_intervals_mismatch_is_refused(self, mock_audio, experiment_config):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = "complex_intervals"
        experiment_config.complex_interval_day = "w1day2"
        experiment_config.rois_number = 5
        with pytest.raises(ValueError, match="arms"):
            ExperimentFactory.generate_trials(experiment_config, mock_audio)

    @pytest.mark.parametrize("mode,arms", [
        ("simple_intervals", 8),
        ("temporal_envelope_modulation", 8),
        ("complex_intervals", 8),
    ])
    def test_the_published_configurations_pass(self, mock_audio, experiment_config, mode, arms):
        from amazeing.auditory.experiments import ExperimentFactory
        experiment_config.experiment_mode = mode
        experiment_config.rois_number = arms
        df, _ = ExperimentFactory.generate_trials(experiment_config, mock_audio)
        assert len(df) == 9 * arms

    def test_the_plan_explains_the_breakdown(self):
        from amazeing.auditory.config import ExperimentConfig
        total, breakdown = ExperimentConfig(
            experiment_mode="temporal_envelope_modulation").stimulus_arm_plan()
        assert total == 8
        assert "control" in breakdown and "complex AM" in breakdown

        total, breakdown = ExperimentConfig(
            experiment_mode="simple_intervals").stimulus_arm_plan()
        assert total == 8
        assert "unison" in breakdown and "silent" in breakdown

    def test_adaptive_modes_have_no_fixed_count(self):
        from amazeing.auditory.config import ExperimentConfig
        for mode in ("simple_smooth", "sequences", "vocalisation", "custom"):
            assert ExperimentConfig(experiment_mode=mode).stimulus_arm_plan() is None
            ExperimentConfig(experiment_mode=mode).check_stimulus_count()   # no raise

    def test_grammar_test_day_needs_eight_arms(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="grammar", grammar_mode="test", rois_number=6)
        with pytest.raises(ValueError, match="8 stimuli.*6 arms"):
            cfg.check_stimulus_count()
        ExperimentConfig(experiment_mode="grammar", grammar_mode="test",
                         rois_number=8).check_stimulus_count()


class TestMissingAudioFiles:
    """A missing recording plays as silence, so it must be reported."""

    def test_vocalisation_control_not_set(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="temporal_envelope_modulation")
        problems = cfg.missing_audio_files()
        assert problems and "control vocalisation" in problems[0]
        with pytest.raises(ValueError, match="control vocalisation"):
            cfg.check_audio_files()

    def test_vocalisation_control_set_but_absent(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="complex_intervals",
                               path_to_vocalisation_control=str(tmp_path / "nope.wav"))
        assert "does not exist" in cfg.missing_audio_files()[0]

    def test_a_real_file_satisfies_it(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        wav = tmp_path / "call.wav"
        wav.write_bytes(b"RIFF")
        cfg = ExperimentConfig(experiment_mode="temporal_envelope_modulation",
                               path_to_vocalisation_control=str(wav))
        assert cfg.missing_audio_files() == []
        cfg.check_audio_files()

    def test_no_control_arm_means_no_file_needed(self):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="temporal_envelope_modulation",
                               tem_controls=["silent"])
        assert cfg.missing_audio_files() == []

    def test_vocalisation_mode_needs_a_folder_with_wavs(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="vocalisation",
                               path_to_vocalisation_folder=str(tmp_path))
        assert "no .wav files" in cfg.missing_audio_files()[0]
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        assert ExperimentConfig(experiment_mode="vocalisation",
                                path_to_vocalisation_folder=str(tmp_path)
                                ).missing_audio_files() == []

    def test_custom_wav_arm_is_checked(self, tmp_path):
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="custom", rois_number=2,
                               custom_stimuli=[{"roi": "1", "kind": "wav",
                                                "path": str(tmp_path / "x.wav")}])
        assert "arm 1" in cfg.missing_audio_files()[0]

    def test_preview_explains_the_silence(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="temporal_envelope_modulation",
                               rois_number=8)
        voc = [p for p in preview_stimuli(cfg, duration_s=0.02)
               if p.label == "vocalisation"]
        assert voc, "no vocalisation arm in the preview"
        assert voc[0].is_silent
        assert "vocalisation" in voc[0].note.lower()

    def test_a_deliberately_silent_arm_has_no_note(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        cfg = ExperimentConfig(experiment_mode="custom", rois_number=2,
                               custom_stimuli=[{"roi": "1", "kind": "tone",
                                                "frequency": 9000}])
        silent = preview_stimuli(cfg, duration_s=0.02)[1]
        assert silent.is_silent and silent.note == ""

    def test_label_does_not_read_interval_vocalisation(self):
        from amazeing.app.stimulus_preview import preview_stimuli
        from amazeing.auditory.config import ExperimentConfig
        labels = [p.label for p in preview_stimuli(
            ExperimentConfig(experiment_mode="complex_intervals", rois_number=8),
            duration_s=0.02)]
        assert "vocalisation" in labels
        assert not any(l.startswith("interval vocalisation") for l in labels)
