"""Shared default arguments for the POD study scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STUDY_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "StudyConfigs"
POD_STUDY_CONFIG = STUDY_CONFIGS_DIR / "config_landmarks_grid_3.json"


def load_script_defaults(defaults_file: str | Path) -> dict[str, Any]:
    """Load default arguments from a JSON file, given as a name inside StudyConfigs or as a path."""
    with (STUDY_CONFIGS_DIR / defaults_file).open(encoding="utf-8") as file:
        return json.load(file)
