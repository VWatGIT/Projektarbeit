"""Run study scripts from a batch CSV on one experiment, one after another.

The CSV has the columns 'script' and 'config' (a file name inside StudyConfigs).
Landmark studies receive the file as --config (their study config). The POD and
interpolation studies receive it as --settings (their default arguments). Each row runs as
its own process.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parent
STUDY_CONFIGS_DIR = SIMULATION_DIR / "StudyConfigs"
DEFAULT_BATCH_FILE = STUDY_CONFIGS_DIR / "batch_studies.csv"
SETTINGS_SCRIPTS = {"study_interpolation_resolution", "study_pod_modes", "study_pod_surfaces"}


def available_scripts() -> set[str]:
    return {path.stem for path in SIMULATION_DIR.glob("study_*.py")}


def read_batch_file(batch_file: Path) -> list[tuple[str, str]]:
    with batch_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"script", "config"} <= set(reader.fieldnames):
            raise ValueError(f"{batch_file} must contain 'script' and 'config' columns.")
        rows = [
            (row["script"].strip(), row["config"].strip())
            for row in reader
            if row["script"].strip() or row["config"].strip()
        ]
    scripts = available_scripts()
    for script, config in rows:
        if script not in scripts:
            raise ValueError(f"Unknown script '{script}' in {batch_file}.")
        if not config:
            raise ValueError(f"Empty config for script '{script}' in {batch_file}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH_FILE)
    args = parser.parse_args()

    rows = read_batch_file(args.batch)
    if not rows:
        raise SystemExit(f"No rows in {args.batch}.")

    failures = []
    for script, config in rows:
        print(f"\n=== {script} with {config} on {args.experiment} ===")
        flag = "--settings" if script in SETTINGS_SCRIPTS else "--config"
        command = [
            sys.executable,
            str(SIMULATION_DIR / f"{script}.py"),
            flag,
            str(STUDY_CONFIGS_DIR / config),
            "--experiment",
            args.experiment,
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            failures.append(f"{script} {config}")
            print(f"{script} {config} failed with exit code {result.returncode}.")
    if failures:
        raise SystemExit(f"Failed runs: {'; '.join(failures)}")


if __name__ == "__main__":
    main()
