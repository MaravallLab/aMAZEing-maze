"""Load and save a session configuration as a YAML file.

The YAML file is the contract between the command line, the graphical
interface and the analysis tools: every field of ``ExperimentConfig`` can be
set from it, and ``amaze-auditory --config session.yaml`` runs a session from
it. Nothing in the session loop changes; this module only builds the same
``ExperimentConfig`` object that ``config.py`` defines.

File format
-----------
A flat mapping of ``ExperimentConfig`` field names to values, plus a
``schema_version`` key so future readers can migrate old files::

    schema_version: 1
    experiment_mode: custom
    rois_number: 4
    base_output_path: C:/data/maze_recordings
    custom_stimuli:
      - roi: "1"
        kind: tone
        frequency: 10000
      - roi: "2"
        kind: wav
        path: C:/data/vocalisations/call.wav
      - roi: "3"
        kind: silent

Unknown keys are an error rather than being silently ignored, so a typo in
a field name cannot quietly leave a default in place.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict

import yaml

from amazeing.auditory.config import ExperimentConfig

SCHEMA_VERSION = 1

_FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)}


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Plain dict of every field, with ``schema_version`` first."""
    data: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    data.update(dataclasses.asdict(cfg))
    return data


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from a mapping, validating the keys."""
    if not isinstance(data, dict):
        raise ValueError("Session config must be a mapping of field names to values")
    data = dict(data)
    version = data.pop("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported session config schema_version {version!r} "
            f"(this version of amazeing reads {SCHEMA_VERSION})"
        )
    unknown = sorted(set(data) - _FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown field(s) in session config: {', '.join(unknown)}. "
            f"Valid fields: {', '.join(sorted(_FIELDS))}"
        )
    return ExperimentConfig(**data)


def load_config(path: str) -> ExperimentConfig:
    """Read a YAML session config and return the ``ExperimentConfig``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        return config_from_dict(data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: {e}") from e


def save_config(cfg: ExperimentConfig, path: str) -> str:
    """Write ``cfg`` as YAML (creating parent folders) and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False, allow_unicode=True)
    return path
