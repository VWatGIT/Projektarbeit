"""Run seeded sauger-centered landmark offset trials on existing FEA results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError

try:
    from .utils.create_parameters import detect_sauger_count
    from .utils.sauger_landmark_study import (
        generate_landmarks,
        invalid_landmarks,
        load_sauger_centers,
        load_sauger_landmark_study_config,
        trial_seed,
    )
    from .utils.gappy_pod import reconstruction_errors
    from .utils.study_pipeline import Landmark, evaluate_landmark_scenarios
    from .utils.surface_resampling import (
        ResampledSnapshotSet,
        load_resampled_snapshots,
    )
except ImportError:
    from utils.create_parameters import detect_sauger_count
    from utils.sauger_landmark_study import (
        GeneratedLandmark,
        generate_landmarks,
        invalid_landmarks,
        load_sauger_centers,
        load_sauger_landmark_study_config,
        trial_seed,
    )
    from utils.gappy_pod import reconstruction_errors
    from utils.study_pipeline import Landmark, evaluate_landmark_scenarios
    from utils.surface_resampling import (
        ResampledSnapshotSet,
        load_resampled_snapshots,
    )


METRICS = (
    "relative_l2_error",
    "absolute_l2_error",
    "maximum_absolute_error",
    "deformed_coordinate_relative_l2_error",
)


def _parameter_sauger_count(parameter_file: Path) -> int:
    if not parameter_file.is_file():
        raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
    with parameter_file.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        columns = next(reader, [])
    return detect_sauger_count(columns)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if metric in row and math.isfinite(float(row[metric]))
    ]
    return fmean(values) if values else None


def aggregate_trials(
    trial_rows: list[dict[str, Any]],
    radii: tuple[float, ...],
    case_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    case_rows = case_rows or []
    summaries = []
    for radius in radii:
        successful = [
            row
            for row in trial_rows
            if row["status"] == "success" and float(row["radius"]) == radius
        ]
        failed_count = sum(
            row["status"] == "failed" and float(row["radius"]) == radius
            for row in trial_rows
        )
        summary: dict[str, Any] = {
            "radius": radius,
            "successful_trials": len(successful),
            "failed_trials": failed_count,
        }
        radius_cases = [
            row for row in case_rows if float(row["radius"]) == radius
        ]
        for metric in METRICS:
            values = [
                float(row[f"mean_{metric}"])
                for row in successful
                if row.get(f"mean_{metric}") is not None
                and math.isfinite(float(row[f"mean_{metric}"]))
            ]
            summary[f"mean_{metric}"] = fmean(values) if values else None
            summary[f"std_{metric}"] = pstdev(values) if values else None

            metrics_by_case: dict[str, list[float]] = {}
            for case in radius_cases:
                value = case.get(metric)
                if value is not None and math.isfinite(float(value)):
                    metrics_by_case.setdefault(str(case["case_name"]), []).append(
                        float(value)
                    )
            case_means = [fmean(values) for values in metrics_by_case.values()]
            summary[f"min_case_mean_{metric}"] = (
                min(case_means) if case_means else None
            )
            summary[f"max_case_mean_{metric}"] = (
                max(case_means) if case_means else None
            )
        summaries.append(summary)
    return summaries


def write_true_surface_fem_metrics(
    output_file: Path,
    snapshots: ResampledSnapshotSet,
) -> dict[str, Any]:
    """Write mean FEM-versus-undeformed surface metrics across FEA cases."""
    coordinates = np.zeros((snapshots.grid.point_count, 3), dtype=np.float64)
    axis_a, axis_b = snapshots.grid.axis_indices
    coordinates[:, axis_a] = snapshots.grid.coordinates_2d[:, 0]
    coordinates[:, axis_b] = snapshots.grid.coordinates_2d[:, 1]
    coordinates[:, snapshots.grid.fixed_axis] = snapshots.grid.fixed_value
    true_surface = coordinates[snapshots.valid_mask]

    case_metrics = []
    for displacement in snapshots.displacements:
        fem_displacement = displacement[snapshots.valid_mask]
        displacement_metrics = reconstruction_errors(
            fem_displacement.reshape(-1),
            np.zeros(fem_displacement.size, dtype=np.float64),
        )
        fem_surface = true_surface + fem_displacement
        displacement_metrics["deformed_coordinate_relative_l2_error"] = float(
            np.linalg.norm(fem_surface - true_surface)
            / max(np.linalg.norm(fem_surface), 1e-12)
        )
        case_metrics.append(displacement_metrics)

    report = {
        "comparison": "true_surface_vs_fem_surface",
        "true_surface": "undeformed_resampled_coordinates",
        "fem_surface": "undeformed_coordinates_plus_fem_displacement",
        "averaging": "arithmetic_mean_of_per_case_metrics",
        "case_count": len(case_metrics),
        "metrics": {
            metric: fmean(float(case[metric]) for case in case_metrics)
            for metric in METRICS
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def plot_radius_summary(output_file: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    titles = (
        "Relative displacement L2 error",
        "Absolute displacement L2 error",
        "Maximum absolute displacement error",
        "Relative deformed-coordinate L2 error",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        rows = [row for row in summaries if row[f"mean_{metric}"] is not None]
        x_values = [float(row["radius"]) for row in rows]
        y_values = [float(row[f"mean_{metric}"]) for row in rows]
        y_errors = [float(row[f"std_{metric}"] or 0.0) for row in rows]
        line = axis.plot(
            x_values,
            y_values,
            marker="o",
            label="Overall mean",
        )[0]
        case_min = [row[f"min_case_mean_{metric}"] for row in rows]
        case_max = [row[f"max_case_mean_{metric}"] for row in rows]
        if all(value is not None for value in case_min + case_max):
            axis.fill_between(
                x_values,
                [float(value) for value in case_min],
                [float(value) for value in case_max],
                color=line.get_color(),
                alpha=0.18,
                label="Min-max across held-out cases",
            )
        axis.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="none",
            ecolor="black",
            capsize=3,
            label="+/- 1 SD across trial means",
        )
        axis.set_title(title)
        axis.set_ylabel(
            "Mean relative error"
            if "relative" in metric
            else "Mean absolute error (model units)"
        )
        axis.grid(True, alpha=0.3)
        if "relative" in metric:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Landmark radius from nominal sauger position (model units)")
    figure.suptitle("Leave-one-out error by sauger-centered landmark radius")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def run_study(
    config_path: str | Path,
    radii_override: tuple[float, ...] | None = None,
    trials_per_radius_override: int | None = None,
    output_dir_override: Path | None = None,
) -> Path:
    config = load_sauger_landmark_study_config(config_path)
    if radii_override is not None:
        if not radii_override or any(
            not math.isfinite(radius) or radius <= 0 for radius in radii_override
        ):
            raise ValueError("Override radii must be finite and positive.")
        if len(set(radii_override)) != len(radii_override):
            raise ValueError("Override radii must not contain duplicates.")
        config = replace(config, radii=radii_override)
    if trials_per_radius_override is not None:
        if trials_per_radius_override < 1:
            raise ValueError("trials_per_radius override must be positive.")
        config = replace(config, trials_per_radius=trials_per_radius_override)
    if output_dir_override is not None:
        config = replace(config, output_dir=output_dir_override.resolve())

    centers = load_sauger_centers(config.initial_positions_file)
    parameter_file = config.results_dir.parent / "parameters.csv"
    expected_sauger_count = _parameter_sauger_count(parameter_file)
    if len(centers) != expected_sauger_count:
        raise ValueError(
            f"Initial position file has {len(centers)} saugers, but "
            f"parameters.csv has {expected_sauger_count}."
        )

    snapshots = load_resampled_snapshots(
        config.results_dir,
        nx=config.nx,
        nz=config.nz,
    )
    valid_coordinates = snapshots.grid.coordinates_2d[snapshots.valid_mask]
    if valid_coordinates.shape[0] < 3:
        raise ValueError("At least three valid surface points are required.")
    try:
        valid_hull = Delaunay(valid_coordinates)
    except QhullError as error:
        raise ValueError("Could not construct the valid surface boundary.") from error

    scenarios: list[tuple[str, tuple[Landmark, ...]]] = []
    scenario_metadata: dict[str, dict[str, Any]] = {}
    trial_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    for radius_index, radius in enumerate(config.radii):
        for trial_index in range(config.trials_per_radius):
            scenario_id = f"radius_{radius_index + 1:02d}_trial_{trial_index + 1:03d}"
            seed = trial_seed(
                config.random_seed,
                radius_index,
                trial_index,
                config.trials_per_radius,
            )
            generated = generate_landmarks(
                centers,
                config.landmarks_per_sauger,
                radius,
                random.Random(seed),
            )
            invalid = invalid_landmarks(snapshots, generated, hull=valid_hull)
            invalid_keys = {
                (point.sauger_index, point.landmark_index) for point in invalid
            }
            status = "failed" if invalid else "success"
            error = (
                "; ".join(
                    f"sauger {point.sauger_index} landmark {point.landmark_index} "
                    f"({point.point.x:.6g}, {point.point.z:.6g}) outside valid surface"
                    for point in invalid
                )
                or None
            )
            trial_rows.append(
                {
                    "scenario_id": scenario_id,
                    "radius": radius,
                    "trial": trial_index + 1,
                    "seed": seed,
                    "status": status,
                    "error": error,
                    "case_count": 0,
                }
            )
            for point in generated:
                placement_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "radius": radius,
                        "trial": trial_index + 1,
                        "seed": seed,
                        "sauger_index": point.sauger_index,
                        "landmark_index": point.landmark_index,
                        "nastran_x": point.point.x,
                        "nastran_z": point.point.z,
                        "status": status,
                    }
                )
            if invalid:
                continue

            landmarks = tuple(point.point for point in generated)
            scenarios.append((scenario_id, landmarks))
            scenario_metadata[scenario_id] = {
                "radius": radius,
                "trial": trial_index + 1,
                "seed": seed,
            }

    case_metrics = evaluate_landmark_scenarios(
        snapshots=snapshots,
        scenarios=tuple(scenarios),
        rank=config.rank,
        energy_fraction=config.energy_fraction,
    ) if scenarios else []
    metrics_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in case_metrics:
        metrics_by_scenario.setdefault(row["scenario_id"], []).append(row)

    case_rows = []
    for scenario_id, rows in metrics_by_scenario.items():
        metadata = scenario_metadata[scenario_id]
        matching_trial = next(
            trial for trial in trial_rows if trial["scenario_id"] == scenario_id
        )
        matching_trial["case_count"] = len(rows)
        trial_means = {metric: _mean_metric(rows, metric) for metric in METRICS}
        matching_trial.update(
            {f"mean_{metric}": value for metric, value in trial_means.items()}
        )
        for row in rows:
            case_rows.append(
                {
                    **metadata,
                    "scenario_id": scenario_id,
                    **{key: value for key, value in row.items() if key != "scenario_id"},
                }
            )

    summaries = aggregate_trials(trial_rows, config.radii, case_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_true_surface_fem_metrics(
        config.output_dir / "true_surface_vs_fem_surface_metrics.json",
        snapshots,
    )
    _write_csv(
        config.output_dir / "landmark_placements.csv",
        [
            "scenario_id",
            "radius",
            "trial",
            "seed",
            "sauger_index",
            "landmark_index",
            "nastran_x",
            "nastran_z",
            "status",
        ],
        placement_rows,
    )
    case_metric_fields = [
        "radius",
        "trial",
        "seed",
        "scenario_id",
        "case_name",
        "pod_rank",
        *METRICS,
        "maximum_relative_error",
        "rank",
        "condition_number",
        "observation_residual",
        "least_squares_residual",
    ]
    _write_csv(
        config.output_dir / "case_metrics.csv",
        case_metric_fields,
        case_rows,
    )
    trial_metric_fields = [
        "scenario_id",
        "radius",
        "trial",
        "seed",
        "status",
        "error",
        "case_count",
        *(f"mean_{metric}" for metric in METRICS),
    ]
    _write_csv(
        config.output_dir / "trial_metrics.csv",
        trial_metric_fields,
        trial_rows,
    )
    summary_fields = ["radius", "successful_trials", "failed_trials"]
    summary_fields.extend(
        field
        for metric in METRICS
        for field in (
            f"mean_{metric}",
            f"std_{metric}",
            f"min_case_mean_{metric}",
            f"max_case_mean_{metric}",
        )
    )
    _write_csv(
        config.output_dir / "summary.csv",
        summary_fields,
        summaries,
    )
    manifest = {
        "name": config.name,
        "results_dir": str(config.results_dir),
        "initial_positions_file": str(config.initial_positions_file),
        "output_dir": str(config.output_dir),
        "grid": {"nx": config.nx, "nz": config.nz},
        "energy_fraction": config.energy_fraction,
        "rank": config.rank,
        "landmarks_per_sauger": config.landmarks_per_sauger,
        "radii": list(config.radii),
        "trials_per_radius": config.trials_per_radius,
        "random_seed": config.random_seed,
        "coordinate_units": "same_as_source_model",
        "angle_sampling": "independent_uniform_0_to_2pi",
        "radius_sampling": "fixed_per_trial",
        "boundary_policy": "fail_trial",
        "case_names": list(snapshots.case_names),
        "sauger_count": len(centers),
        "successful_trials": sum(row["status"] == "success" for row in trial_rows),
        "failed_trials": sum(row["status"] == "failed" for row in trial_rows),
        "metric_files": [
            "case_metrics.csv",
            "trial_metrics.csv",
            "summary.csv",
            "landmark_placements.csv",
        ],
    }
    with (config.output_dir / "study_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    plot_radius_summary(config.output_dir / "error_by_radius.png", summaries)
    print(
        f"Completed {len(scenarios)} of {len(trial_rows)} placements across "
        f"{len(snapshots.case_names)} held-out cases."
    )
    print(f"Results: {config.output_dir}")
    return config.output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    simulation_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate sauger-centered random landmark offsets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=simulation_dir / "StudyConfigs" / "sauger_landmark_offsets.json",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=None,
        help="Override the configured radius list, useful for a smoke run.",
    )
    parser.add_argument(
        "--trials-per-radius",
        type=int,
        default=None,
        help="Override the configured trial count, useful for a smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    run_study(
        args.config,
        radii_override=None if args.radii is None else tuple(args.radii),
        trials_per_radius_override=args.trials_per_radius,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()