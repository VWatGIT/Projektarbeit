"""Run a configured Gappy POD surface study on existing FEM results."""

from pathlib import Path

from utils.study_pipeline import evaluate_study, load_study_config


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[0] / "StudyConfigs" / "grid_on_sauger_3.json" 
    results = evaluate_study(load_study_config(config_path))
    print(f"Evaluated {len(results)} held-out cases.")
