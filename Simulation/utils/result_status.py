"""Track successful FEA cases and expose only usable result archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATUS_FILENAME = "simulation_status.json"


def write_simulation_status(
    experiment_dir: str | Path,
    cases: list[dict[str, Any]],
) -> Path:
    """Persist the current case status for an experiment."""
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)
    status_file = experiment_path / STATUS_FILENAME
    successful_count = sum(case["status"] == "success" for case in cases)
    failed_count = sum(case["status"] == "failed" for case in cases)
    payload = {
        "cases": cases,
        "successful_count": successful_count,
        "failed_count": failed_count,
    }
    with status_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return status_file


def successful_npz_files(experiment_dir: str | Path) -> list[Path]:
    """Return only NPZ files recorded as successful in the status manifest."""
    experiment_path = Path(experiment_dir)
    status_file = experiment_path / STATUS_FILENAME
    archive_root = experiment_path
    if not status_file.is_file():
        legacy_status_file = experiment_path / "FEA" / STATUS_FILENAME
        if legacy_status_file.is_file():
            status_file = legacy_status_file
            archive_root = legacy_status_file.parent
    if not status_file.is_file():
        raise FileNotFoundError(
            f"Simulation status file not found: {status_file}. "
            "Run the parameter sweep before starting a study."
        )

    with status_file.open(encoding="utf-8") as file:
        payload = json.load(file)

    successful_files: list[Path] = []
    for case in payload.get("cases", []):
        if case.get("status") != "success":
            continue
        relative_path = case.get("npz_file")
        if not relative_path:
            continue
        npz_file = (archive_root / relative_path).resolve()
        if not npz_file.is_file():
            raise FileNotFoundError(
                f"Successful simulation result is missing: {npz_file}"
            )
        successful_files.append(npz_file)

    if not successful_files:
        raise FileNotFoundError(
            f"No successful simulation NPZ files found in {experiment_path}."
        )
    return sorted(successful_files)
