"""Run a configured Gappy POD surface study on existing FEM results."""

import argparse
from pathlib import Path

from utils.study_pipeline import evaluate_study, load_study_config


if __name__ == "__main__":
    simulation_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run a configured Gappy POD study on existing FEM results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=simulation_dir / "StudyConfigs" / "grid_on_sauger_3.json",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = simulation_dir / "Results" / args.experiment
    config = load_study_config(
        args.config,
        results_dir=results_dir,
        output_dir=args.output_dir,
    )
    results = evaluate_study(config)
    print(f"Evaluated {len(results)} held-out cases.")
