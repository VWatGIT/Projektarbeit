"""Evaluate how the interpolation grid resolution affects Gappy POD reconstruction.

A fixed square landmark grid (default 3x3) observes each held-out surface. For
every square interpolation grid nx = nz from min to max resolution, the surfaces
are resampled and each case is reconstructed from a POD basis fitted on all
remaining cases (leave-one-out).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .study_landmarks_grid import METRICS, build_grid_landmarks
    from .study_pod_surfaces import evaluate_training_sizes, summarize, write_csv
    from .utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from .utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        load_study_config,
    )
    from .utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from study_landmarks_grid import METRICS, build_grid_landmarks
    from study_pod_surfaces import evaluate_training_sizes, summarize, write_csv
    from utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        load_study_config,
    )
    from utils.surface_resampling import load_resampled_snapshots


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
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--grid-size", type=int)
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
        help="Defaults to Study/<name> from the settings file.",
    )
    pre_args, _ = parser.parse_known_args(arguments)
    parser.set_defaults(**load_script_defaults(pre_args.settings))
    return parser.parse_args(arguments)


def run_study(args: argparse.Namespace) -> None:
    if args.min_resolution < 2:
        raise ValueError("min-resolution must be at least 2.")
    if args.max_resolution < args.min_resolution:
        raise ValueError("max-resolution must not be below min-resolution.")
    if args.resolution_step < 1:
        raise ValueError("resolution-step must be positive.")
    if not 0 < args.energy_fraction <= 1:
        raise ValueError("energy-fraction must be in (0, 1].")

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
        snapshots = load_resampled_snapshots(
            config.results_dir,
            nx=resolution,
            nz=resolution,
        )
        case_count = len(snapshots.case_names)
        landmarks = build_grid_landmarks(config, args.grid_size)
        operator, _ = _landmark_operator(snapshots, landmarks)
        valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]
        print(f"[{resolution}x{resolution}] {case_count} cases, {len(landmarks)} landmarks")

        resolution_rows = evaluate_training_sizes(
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
        for row in resolution_rows:
            row["interpolation_resolution"] = resolution
        rows.extend(resolution_rows)

        summary = summarize(resolution_rows)[0]
        summary["interpolation_resolution"] = resolution
        summaries.append(summary)

    output_dir = base_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "interpolation_cases.csv", rows)
    write_csv(output_dir / "interpolation_summary.csv", summaries)
    plot_resolution(output_dir / "interpolation_error.png", summaries, len(landmarks))
    print(f"Results written to {output_dir}")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    run_study(args)


if __name__ == "__main__":
    main()
