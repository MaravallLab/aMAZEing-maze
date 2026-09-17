"""Auditory maze paradigm.

Modules
-------
config            ExperimentConfig dataclass (all session parameters)
main              session loop (``amaze-auditory`` entry point)
audio             sound synthesis, playback, speaker compensation
vision            ROI tracking with temporal debouncing
hardware          camera and Arduino TTL controller
experiments       trial-structure factory for every experiment mode
data_manager      session folders, visit / maze-entry logs
analysis          per-session figures (``amaze-analyse-session``)
summary_analysis  cross-session figures and CSVs (``amaze-summary``)
grammar_stimuli   Markov-grammar melody generation (``amaze-grammar``)
"""
