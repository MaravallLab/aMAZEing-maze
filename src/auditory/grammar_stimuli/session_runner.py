"""Run a training or test session of grammar stimuli.

The runner is intentionally decoupled from the maze tracking loop in
``code/auditory/updated_version/main.py``. That file drives stimulus onset
from ROI entries; here we focus only on producing the melodies, scheduling
them on a fixed 4.4 s cycle, and writing a per-melody log.

Use cases:
- Training days: :meth:`run_training_session` emits one grammar (trained
  grammar for the animal's group+condition), dominant tier only, for
  ``session_duration_s`` seconds.
- Test day: :meth:`run_test_session` iterates over the 8 arms in
  :data:`config.TEST_ARM_PLAN`, producing a separate log and waveform
  summary per arm.

If ``cfg.dry_run`` is True, no audio is played: the runner still generates
and logs symbols so it can be used for offline simulation.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from . import config as cfg
from .sequence_sampler import MarkovSampler, SampleMeta
from .tone_generator import generate_melody, generate_silence_gap


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_LOG_FIELDS = [
    "melody_index",
    "arm_id",
    "grammar",
    "tier",
    "start_symbol",
    "symbols",
    "per_step_bits",
    "mean_bits",
    "onset_s",
    "offset_s",
]


def _init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        writer.writeheader()


def _append_row(path: str, row: Dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        writer.writerow(row)


def _row_from_meta(
    meta: SampleMeta,
    *,
    melody_index: int,
    arm_id: str,
    onset_s: float,
    offset_s: float,
) -> Dict:
    return {
        "melody_index": melody_index,
        "arm_id": arm_id,
        "grammar": meta.grammar_name,
        "tier": meta.tier,
        "start_symbol": meta.start_symbol,
        "symbols": "".join(meta.symbols),
        "per_step_bits": ";".join(f"{b:.4f}" for b in meta.per_step_bits),
        "mean_bits": round(meta.mean_bits, 6),
        "onset_s": round(onset_s, 4),
        "offset_s": round(offset_s, 4),
    }


# ---------------------------------------------------------------------------
# Sound device wrapper
# ---------------------------------------------------------------------------

class _Player:
    """Thin wrapper around sounddevice that no-ops in dry-run mode."""

    def __init__(self, session_cfg: cfg.GrammarSessionConfig) -> None:
        self.cfg = session_cfg
        self._sd = None
        if not session_cfg.dry_run:
            import sounddevice as sd  # local import so dry-run needs no device
            sd.default.samplerate = session_cfg.sample_rate
            sd.default.device = session_cfg.device_id
            self._sd = sd

    def play_blocking(self, waveform: np.ndarray) -> None:
        if self._sd is None:
            # Dry run: simulate elapsed time without actually sleeping the
            # full amount (tests and offline simulation want speed).
            return
        self._sd.play(waveform, self.cfg.sample_rate)
        self._sd.wait()

    def stop(self) -> None:
        if self._sd is not None:
            self._sd.stop()


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    mode: str                                # 'training' or 'test'
    group: int
    condition: str
    trained_grammar: str
    novel_grammar: str
    arm_logs: Dict[str, str]                 # arm_id -> csv path
    n_melodies: Dict[str, int]               # arm_id -> melody count
    duration_s: Dict[str, float]             # arm_id -> wall-clock-ish duration


class SessionRunner:
    """Orchestrate stimulus generation + logging for one session."""

    def __init__(self, session_cfg: cfg.GrammarSessionConfig) -> None:
        self.cfg = session_cfg
        self._rng = np.random.default_rng(session_cfg.seed)
        self._player = _Player(session_cfg)
        os.makedirs(session_cfg.output_dir, exist_ok=True)

    # -- internal ---------------------------------------------------------
    def _run_arm(
        self,
        *,
        arm_id: str,
        grammar_name: str,
        tier: str,
        log_path: str,
        duration_s: float,
    ) -> int:
        """Stream melodies for one arm until ``duration_s`` elapses.

        Returns the number of melodies played.
        """
        sampler = MarkovSampler(grammar_name=grammar_name, tier=tier, rng=self._rng)

        gap = generate_silence_gap(sample_rate=self.cfg.sample_rate)
        melody_s = cfg.MELODY_DURATION_MS * 1e-3
        cycle_s = cfg.MELODY_CYCLE_MS * 1e-3

        # We use a simulated clock: each melody advances the clock by
        # exactly cycle_s. In dry-run this means no real sleeping; in live
        # mode the player blocks for the melody, and we sleep for the
        # inter-melody gap.
        sim_t = 0.0
        melody_index = 0

        while sim_t + cycle_s <= duration_s + 1e-6:
            meta = sampler.sample_melody(length=self.cfg.melody_length)
            onset = sim_t
            offset = onset + melody_s

            wave = generate_melody(
                meta.symbols,
                sample_rate=self.cfg.sample_rate,
                amplitude=self.cfg.amplitude,
            )
            # Append inter-melody gap so the full cycle is one buffer.
            full_buf = np.concatenate([wave, gap])

            self._player.play_blocking(full_buf)

            _append_row(
                log_path,
                _row_from_meta(
                    meta,
                    melody_index=melody_index,
                    arm_id=arm_id,
                    onset_s=onset,
                    offset_s=offset,
                ),
            )

            sim_t += cycle_s
            melody_index += 1

        return melody_index

    def _silent_arm(self, arm_id: str, log_path: str, duration_s: float) -> int:
        """Silent arm: log one zero-duration row marking the arm existed."""
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
            writer.writerow({
                "melody_index": 0,
                "arm_id": arm_id,
                "grammar": "",
                "tier": "",
                "start_symbol": "",
                "symbols": "",
                "per_step_bits": "",
                "mean_bits": 0.0,
                "onset_s": 0.0,
                "offset_s": round(duration_s, 4),
            })
        return 0

    # -- training ---------------------------------------------------------
    def run_training_session(self) -> SessionSummary:
        """Run a full training session (single grammar, dominant tier)."""
        grammar = self.cfg.resolve_training_grammar()
        novel = "B" if grammar == "A" else "A"

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(
            self.cfg.output_dir,
            f"training_g{self.cfg.group}_{self.cfg.condition}_{grammar}_{stamp}",
        )
        log_path = base + ".csv"
        meta_path = base + ".json"
        _init_csv(log_path)

        n = self._run_arm(
            arm_id="training",
            grammar_name=grammar,
            tier=cfg.TRAINING_COMPLEXITY,
            log_path=log_path,
            duration_s=self.cfg.session_duration_s,
        )

        summary = SessionSummary(
            mode="training",
            group=self.cfg.group,
            condition=self.cfg.condition,
            trained_grammar=grammar,
            novel_grammar=novel,
            arm_logs={"training": log_path},
            n_melodies={"training": n},
            duration_s={"training": self.cfg.session_duration_s},
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(summary), fh, indent=2)
        return summary

    # -- test -------------------------------------------------------------
    def run_test_session(
        self,
        per_arm_duration_s: Optional[float] = None,
    ) -> SessionSummary:
        """Run the 8-arm test plan.

        ``per_arm_duration_s`` defaults to ``session_duration_s / 8`` so a
        4-hour session distributes 30 minutes per arm. This is a generation
        convenience only: in the live maze, arm exposure is driven by the
        animal's behaviour, not by a fixed schedule.
        """
        if per_arm_duration_s is None:
            per_arm_duration_s = self.cfg.session_duration_s / len(cfg.TEST_ARM_PLAN)

        trained = self.cfg.resolve_training_grammar()
        novel = "B" if trained == "A" else "A"

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(
            self.cfg.output_dir,
            f"test_g{self.cfg.group}_{self.cfg.condition}_{stamp}",
        )
        os.makedirs(base_dir, exist_ok=True)

        arm_logs: Dict[str, str] = {}
        n_melodies: Dict[str, int] = {}
        durations: Dict[str, float] = {}

        for arm in cfg.TEST_ARM_PLAN:
            arm_id = arm["arm_id"]
            log_path = os.path.join(base_dir, f"{arm_id}.csv")
            _init_csv(log_path)

            if arm["kind"] == "grammar":
                role = arm["grammar_role"]
                grammar_name = trained if role == "trained" else novel
                n = self._run_arm(
                    arm_id=arm_id,
                    grammar_name=grammar_name,
                    tier=arm["tier"],
                    log_path=log_path,
                    duration_s=per_arm_duration_s,
                )
            elif arm["kind"] == "vocalisation":
                # We don't synthesise vocalisations here; the live session
                # controller loads a WAV. Log a placeholder row.
                _append_row(log_path, {
                    "melody_index": 0,
                    "arm_id": arm_id,
                    "grammar": "",
                    "tier": "vocalisation",
                    "start_symbol": "",
                    "symbols": "",
                    "per_step_bits": "",
                    "mean_bits": 0.0,
                    "onset_s": 0.0,
                    "offset_s": round(per_arm_duration_s, 4),
                })
                n = 0
            elif arm["kind"] == "silent":
                n = self._silent_arm(arm_id, log_path, per_arm_duration_s)
            else:
                raise RuntimeError(f"Unknown arm kind {arm['kind']!r}")

            arm_logs[arm_id] = log_path
            n_melodies[arm_id] = n
            durations[arm_id] = per_arm_duration_s

        summary = SessionSummary(
            mode="test",
            group=self.cfg.group,
            condition=self.cfg.condition,
            trained_grammar=trained,
            novel_grammar=novel,
            arm_logs=arm_logs,
            n_melodies=n_melodies,
            duration_s=durations,
        )
        with open(os.path.join(base_dir, "session_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(summary), fh, indent=2)
        return summary
