"""Evaluate how the interpolation grid resolution affects Gappy POD reconstruction.

A fixed square landmark grid (default 3x3) observes each held-out surface. For
every square interpolation grid nx = nz from min to max resolution, the surfaces
are resampled and each case is reconstructed from a POD basis fitted on all
remaining cases (leave-one-out).

Without --experiment, compare the three three-sauger experiments using both
surface resampling methods. Use --experiment and --compare-resampling to compare
the two methods for one experiment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .compare_surface_generation import DISPLACEMENT_EXPERIMENTS
    from .study_landmarks_grid import METRICS, build_grid_landmarks
    from .study_pod_surfaces import evaluate_training_sizes, summarize, write_csv
    from .utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from .utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        StudyConfig,
        load_study_config,
    )
    from .utils.surface_resampling import load_resampled_snapshots
    from .utils.sauger_landmark_study import load_sauger_centers
except ImportError:
    from compare_surface_generation import DISPLACEMENT_EXPERIMENTS
    from study_landmarks_grid import METRICS, build_grid_landmarks
    from study_pod_surfaces import evaluate_training_sizes, summarize, write_csv
    from utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        StudyConfig,
        load_study_config,
    )
    from utils.surface_resampling import load_resampled_snapshots
    from utils.sauger_landmark_study import load_sauger_centers


def plot_resolution(
    output_file: Path,
    summaries: list[dict[str, Any]],
    landmark_count: int,
) -> None:
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
    resolutions = [row["interpolation_resolution"] for row in summaries]
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        mean = np.array([row[f"mean_{metric}"] for row in summaries], dtype=float)
        std = np.array([row[f"std_{metric}"] for row in summaries], dtype=float)
        axis.plot(resolutions, mean, marker="o", linewidth=1.8, label="Mean")
        axis.fill_between(
            resolutions,
            np.maximum(mean - std, 0),
            mean + std,
            alpha=0.25,
            label="±1 std",
        )
        axis.set_title(title)
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=7)
        if "relative" in metric:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for axis in axes[-1]:
        axis.set_xlabel("Interpolation grid resolution (nx = nz)")
    figure.suptitle(
        f"Leave-one-out error by interpolation grid resolution ({landmark_count} landmarks)"
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="config_interpolation_resolution.json",
        help="Default arguments file in StudyConfigs, or a path to one.",
    )
    parser.add_argument(
        "--experiment",
        help="Evaluate one experiment instead of comparing the three-sauger experiments.",
    )
    parser.add_argument("--grid-size", type=int)
    parser.add_argument(
        "--compare-resampling",
        action="store_true",
        help="Compare linear and element resampling for the selected experiment.",
    )
    parser.add_argument(
        "--surface-resampling",
        choices=("linear", "elements"),
        default="linear",
        help="Resampling method for single-experiment mode; comparison mode uses both.",
    )
    parser.add_argument("--min-resolution", type=int)
    parser.add_argument("--max-resolution", type=int)
    parser.add_argument("--resolution-step", type=int)
    parser.add_argument(
        "--energy-fraction",
        type=float,
        help="Cumulative POD energy in (0, 1] that determines the number of modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to Study/<name> for one experiment, or Comparison/interpolation_resolution.",
    )
    pre_args, _ = parser.parse_known_args(arguments)
    parser.set_defaults(**load_script_defaults(pre_args.settings))
    args = parser.parse_args(arguments)
    raw_arguments = arguments if arguments is not None else sys.argv[1:]
    explicit_method = any(
        value == "--surface-resampling" or value.startswith("--surface-resampling=")
        for value in raw_arguments
    )
    if args.compare_resampling and args.experiment is None:
        parser.error("--compare-resampling requires --experiment")
    if args.compare_resampling and explicit_method:
        parser.error("--surface-resampling cannot be combined with --compare-resampling")
    if args.experiment is None and explicit_method:
        parser.error("--surface-resampling requires --experiment; comparison mode uses both methods")
    return args


def _validate_settings(args: argparse.Namespace) -> None:
    if args.min_resolution < 2:
        raise ValueError("min-resolution must be at least 2.")
    if args.max_resolution < args.min_resolution:
        raise ValueError("max-resolution must not be below min-resolution.")
    if args.resolution_step < 1:
        raise ValueError("resolution-step must be positive.")
    if not 0 < args.energy_fraction <= 1:
        raise ValueError("energy-fraction must be in (0, 1].")


def _evaluate_resolution(
    config: StudyConfig,
    resolution: int,
    method: str,
    grid_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...], int]:
    snapshots = load_resampled_snapshots(
        config.results_dir,
        nx=resolution,
        nz=resolution,
        method=method,
    )
    case_count = len(snapshots.case_names)
    if case_count < 2:
        raise ValueError(f"{config.results_dir}: at least two cases are required.")
    landmarks = build_grid_landmarks(config, grid_size)
    operator, _ = _landmark_operator(snapshots, landmarks)
    valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]
    print(f"[{method} {resolution}x{resolution}] {case_count} cases, {len(landmarks)} landmarks")
    rows = evaluate_training_sizes(
        states=snapshots.state_matrix(),
        operator=operator,
        valid_coordinates=valid_coordinates,
        case_names=snapshots.case_names,
        training_sizes=[case_count - 1],
        trials=1,
        seed=0,
        rank=config.rank,
        energy_fraction=config.energy_fraction,
    )
    for row in rows:
        row["interpolation_resolution"] = resolution
    summary = summarize(rows)[0]
    summary["interpolation_resolution"] = resolution
    return rows, summary, snapshots.case_names, int(snapshots.valid_mask.sum())


def run_study(args: argparse.Namespace) -> None:
    _validate_settings(args)
    simulation_dir = Path(__file__).resolve().parent
    results_dir = simulation_dir / "Results" / args.experiment
    base_config = load_study_config(
        POD_STUDY_CONFIG,
        results_dir=results_dir,
        output_dir=args.output_dir,
    )
    base_config = replace(
        base_config,
        energy_fraction=args.energy_fraction,
        rank=None,
    )
    if args.output_dir is None:
        base_config = replace(
            base_config,
            output_dir=base_config.results_dir / "Study" / args.name,
        )

    resolutions = list(
        range(args.min_resolution, args.max_resolution + 1, args.resolution_step)
    )
    print(
        f"Interpolation resolutions {resolutions}, {args.grid_size}x{args.grid_size} "
        f"landmark grid, energy fraction {args.energy_fraction}."
    )

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for resolution in resolutions:
        config = replace(base_config, nx=resolution, nz=resolution)
        resolution_rows, summary, _, _ = _evaluate_resolution(
            config, resolution, args.surface_resampling, args.grid_size,
        )
        rows.extend(resolution_rows)
        summaries.append(summary)

    output_dir = base_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "interpolation_cases.csv", rows)
    write_csv(output_dir / "interpolation_summary.csv", summaries)
    plot_resolution(output_dir / "interpolation_error.png", summaries, args.grid_size**2)
    print(f"Results written to {output_dir}")


def plot_three_sauger_comparison(
    output_file: Path,
    summaries: list[dict[str, Any]],
    method: str,
    y_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for displacement, experiment in DISPLACEMENT_EXPERIMENTS.items():
        records = sorted(
            (row for row in summaries if row["surface_resampling"] == method
             and row["experiment"] == experiment),
            key=lambda row: row["interpolation_resolution"],
        )
        if not records:
            raise ValueError(f"No {method} comparison results for {experiment}.")
        axis.plot(
            [row["interpolation_resolution"] for row in records],
            [row["mean_maximum_absolute_error"] for row in records],
            marker="o",
            label=f"{displacement} mm",
        )
    axis.set_xlabel("Interpolation grid resolution (nx = nz)")
    axis.set_ylabel("Mean maximum absolute displacement error (model units)")
    axis.set_ylim(0, y_max)
    axis.set_title(f"Three-sauger reconstruction: {method} resampling")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Displacement range")
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def _compare_resampling(
    config: StudyConfig,
    resolutions: range,
    grid_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiment = config.results_dir.name
    expected_cases: set[str] | None = None
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for resolution in resolutions:
        for method in ("linear", "elements"):
            case_rows, _, case_names, valid_points = _evaluate_resolution(
                config, resolution, method, grid_size,
            )
            current_cases = set(case_names)
            if len(current_cases) != len(case_names):
                raise ValueError(f"{experiment}: duplicate case names.")
            if expected_cases is None:
                expected_cases = current_cases
            elif current_cases != expected_cases:
                raise ValueError(f"{experiment}: case names differ between resolutions or methods.")
            if len(case_rows) != len(case_names) or {
                row["case_name"] for row in case_rows
            } != current_cases:
                raise ValueError(f"{experiment}: evaluation did not produce one row per case.")
            errors = [float(row["maximum_absolute_error"]) for row in case_rows]
            if not all(np.isfinite(error) and error >= 0 for error in errors):
                raise ValueError(f"{experiment}: maximum absolute errors must be finite and non-negative.")
            for row in case_rows:
                row.update({
                    "experiment": experiment,
                    "surface_resampling": method,
                    "valid_grid_points": valid_points,
                })
            rows.extend(case_rows)
            summaries.append({
                "experiment": experiment,
                "surface_resampling": method,
                "interpolation_resolution": resolution,
                "valid_grid_points": valid_points,
                "evaluation_count": len(errors),
                "mean_maximum_absolute_error": float(np.mean(errors)),
                "std_maximum_absolute_error": float(np.std(errors)),
            })
    return rows, summaries


def plot_resampling_comparison(
    output_file: Path,
    summaries: list[dict[str, Any]],
    experiment: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for method, line_style in (("linear", "-"), ("elements", "--")):
        records = sorted(
            (row for row in summaries if row["surface_resampling"] == method),
            key=lambda row: row["interpolation_resolution"],
        )
        if not records:
            raise ValueError(f"No {method} comparison results for {experiment}.")
        axis.plot(
            [row["interpolation_resolution"] for row in records],
            [row["mean_maximum_absolute_error"] for row in records],
            marker="o",
            linestyle=line_style,
            label=method,
        )
    axis.set_xlabel("Interpolation grid resolution (nx = nz)")
    axis.set_ylabel("Mean maximum absolute displacement error (model units)")
    axis.set_ylim(bottom=0)
    axis.set_title(f"Resampling comparison: {experiment}")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Surface resampling")
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def run_resampling_comparison(args: argparse.Namespace) -> None:
    _validate_settings(args)
    simulation_dir = Path(__file__).resolve().parent
    results_dir = simulation_dir / "Results" / args.experiment
    config = load_study_config(
        POD_STUDY_CONFIG,
        results_dir=results_dir,
        output_dir=args.output_dir,
    )
    config = replace(config, energy_fraction=args.energy_fraction, rank=None)
    if args.output_dir is None:
        config = replace(config, output_dir=config.results_dir / "Study" / args.name)
    resolutions = range(args.min_resolution, args.max_resolution + 1, args.resolution_step)
    rows, summaries = _compare_resampling(config, resolutions, args.grid_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.output_dir / "interpolation_resampling_cases.csv", rows)
    write_csv(config.output_dir / "interpolation_resampling_summary.csv", summaries)
    plot_resampling_comparison(
        config.output_dir / "interpolation_resampling_comparison.png",
        summaries,
        args.experiment,
    )
    print(f"Results written to {config.output_dir}")


def run_three_sauger_comparison(args: argparse.Namespace) -> None:
    _validate_settings(args)
    simulation_dir = Path(__file__).resolve().parent
    resolutions = range(args.min_resolution, args.max_resolution + 1, args.resolution_step)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for displacement, experiment in DISPLACEMENT_EXPERIMENTS.items():
        results_dir = simulation_dir / "Results" / experiment
        centers = load_sauger_centers(results_dir / "initial_sauger_positions.json")
        if len(centers) != 3:
            raise ValueError(f"{experiment}: expected 3 saugers, found {len(centers)}.")
        config = load_study_config(POD_STUDY_CONFIG, results_dir=results_dir)
        config = replace(config, energy_fraction=args.energy_fraction, rank=None)
        experiment_rows, experiment_summaries = _compare_resampling(
            config, resolutions, args.grid_size,
        )
        for record in experiment_rows + experiment_summaries:
            record["displacement_range"] = displacement
        rows.extend(experiment_rows)
        summaries.extend(experiment_summaries)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else simulation_dir / "Comparison" / "interpolation_resolution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "three_sauger_interpolation_cases.csv", rows)
    write_csv(output_dir / "three_sauger_interpolation_summary.csv", summaries)
    highest_error = max(row["mean_maximum_absolute_error"] for row in summaries)
    y_max = highest_error * 1.05 if highest_error > 0 else 1.0
    for method in ("linear", "elements"):
        plot_three_sauger_comparison(
            output_dir / f"three_sauger_interpolation_{method}.png",
            summaries,
            method,
            y_max,
        )
    print(f"Results written to {output_dir}")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.experiment is None:
        run_three_sauger_comparison(args)
    elif args.compare_resampling:
        run_resampling_comparison(args)
    else:
        run_study(args)


if __name__ == "__main__":
    main()
