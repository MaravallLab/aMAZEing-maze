"""CLI entry point for grammar stimulus sessions.

Examples
--------
Dry-run a 4-hour training session for counterbalance group 1 / EE housing::

    python -m grammar_stimuli.run --mode training --group 1 --condition EE --dry-run

Run a test session, 30 minutes per arm::

    python -m grammar_stimuli.run --mode test --group 2 --condition SC \\
        --per-arm-seconds 1800 --output-dir ./sessions/anim042

Run a short sanity check (10 s) on real audio::

    python -m grammar_stimuli.run --mode training --duration-seconds 10
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import config as cfg
from .session_runner import SessionRunner


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="grammar_stimuli",
        description="Generate and play Markov-grammar melodies for maze sessions.",
    )
    p.add_argument("--mode", choices=["training", "test"], default="training")
    p.add_argument("--group", type=int, choices=[1, 2], default=1,
                   help="Counterbalancing group")
    p.add_argument("--condition", choices=["EE", "SC"], default="EE",
                   help="Housing condition: Enriched Environment or Standard Conditions")
    p.add_argument("--training-grammar", choices=["A", "B"], default=None,
                   help="Override trained grammar (otherwise inferred from group/condition)")
    p.add_argument("--duration-seconds", type=float, default=None,
                   help="Session length in seconds (default: 4 hours)")
    p.add_argument("--per-arm-seconds", type=float, default=None,
                   help="Test-only: seconds per arm. Default = duration/8")
    p.add_argument("--sample-rate", type=int, default=cfg.SAMPLE_RATE)
    p.add_argument("--amplitude", type=float, default=cfg.AMPLITUDE)
    p.add_argument("--device-id", type=int, default=3, help="sounddevice output device")
    p.add_argument("--output-dir", default="./sessions")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Do not play audio; still generate and log symbols")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    session_cfg = cfg.GrammarSessionConfig(
        sample_rate=args.sample_rate,
        amplitude=args.amplitude,
        device_id=args.device_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
        group=args.group,
        condition=args.condition,
        mode=args.mode,
        training_grammar=args.training_grammar,
        session_duration_s=(
            args.duration_seconds if args.duration_seconds is not None
            else cfg.SESSION_DURATION_S
        ),
    )

    runner = SessionRunner(session_cfg)
    if args.mode == "training":
        summary = runner.run_training_session()
    else:
        summary = runner.run_test_session(per_arm_duration_s=args.per_arm_seconds)

    print(f"Done. Mode={summary.mode} trained={summary.trained_grammar} "
          f"novel={summary.novel_grammar}")
    for arm_id, path in summary.arm_logs.items():
        print(f"  {arm_id}: {summary.n_melodies[arm_id]} melodies -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
