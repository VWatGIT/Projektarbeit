"""Compare leave-one-out reconstruction errors across landmark grid sizes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .utils.create_study_configs import grid_values
    from .utils.study_pipeline import (
        Landmark,
        StudyConfig,
        evaluate_study,
        load_study_config,
    )
    from .utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from utils.create_study_configs import grid_values
    from utils.study_pipeline import (
        Landmark,
        StudyConfig,
        evaluate_study,
        load_study_config,
    )
    from utils.surface_resampling import load_resampled_snapshots


METRICS = (
    "relative_l2_error",
    "absolute_l2_error",
    "maximum_absolute_error",
    "deformed_coordinate_relative_l2_error",
)


def build_grid_landmarks(
    base_config: StudyConfig,
    grid_size: int,
) -> tuple[Landmark, ...]:
    """Create a square, evenly spaced landmark grid over the configured bounds."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive.")

    x_values = [landmark.x for landmark in base_config.landmarks]
    z_values = [landmark.z for landmark in base_config.landmarks]
    x_grid = grid_values(min(x_values), max(x_values), grid_size)
    z_grid = grid_values(min(z_values), max(z_values), grid_size)
    return tuple(Landmark(x=x, z=z) for x in x_grid for z in z_grid)


def summarize_metrics(
    grid_size: int,
    case_metrics: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    """Average each finite per-case error, retaining its valid case count."""
    summary: dict[str, int | float | None] = {
        "grid_size": grid_size,
        "landmark_count": grid_size * grid_size,
        "case_count": len(case_metrics),
    }
    for metric in METRICS:
        finite_values = [
            float(case[metric])
            for case in case_metrics
            if metric in case and math.isfinite(float(case[metric]))
        ]
        summary[f"mean_{metric}"] = (
            sum(finite_values) / len(finite_values) if finite_values else None
        )
        summary[f"finite_case_count_{metric}"] = len(finite_values)
        summary[f"min_case_{metric}"] = min(finite_values) if finite_values else None
        summary[f"max_case_{metric}"] = max(finite_values) if finite_values else None
    return summary


def write_summary_csv(
    output_file: Path,
    summaries: list[dict[str, int | float | None]],
) -> None:
    fieldnames = ["grid_size", "landmark_count", "case_count"]
    for metric in METRICS:
        fieldnames.extend(
            (
                f"mean_{metric}",
                f"finite_case_count_{metric}",
                f"min_case_{metric}",
                f"max_case_{metric}",
            )
        )

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def plot_summaries(
    output_file: Path,
    summaries: list[dict[str, int | float | None]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    titles = (
        "Relative displacement L2 error",
        "Absolute displacement L2 error",
        "Maximum absolute displacement error",
        "Relative deformed-coordinate L2 error",
    )

    for axis, metric, title in zip(axes.flat, METRICS, titles):
        plotted_rows = [
            row for row in summaries if row[f"mean_{metric}"] is not None
        ]
        x_values = [int(row["landmark_count"]) for row in plotted_rows]
        y_values = [float(row[f"mean_{metric}"]) for row in plotted_rows]
        line = axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            label="Mean across held-out cases",
        )[0]
        case_min = [row[f"min_case_{metric}"] for row in plotted_rows]
        case_max = [row[f"max_case_{metric}"] for row in plotted_rows]
        if all(value is not None for value in case_min + case_max):
            axis.fill_between(
                x_values,
                [float(value) for value in case_min],
                [float(value) for value in case_max],
                color=line.get_color(),
                alpha=0.18,
                label="Min-max across held-out cases",
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
        axis.set_xlabel("Number of landmarks")
    axes[-1, 0].set_xticks([int(row["landmark_count"]) for row in summaries])
    figure.suptitle("Leave-one-out reconstruction error by landmark count")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    simulation_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate leave-one-out reconstruction errors for square landmark grids."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=simulation_dir / "StudyConfigs" / "grid_on_sauger_3.json",
        help="Study config providing the domain, interpolation grid, and POD settings.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the FEA results directory from the study config.",
    )
    parser.add_argument("--min-grid", type=int, default=2)
    parser.add_argument("--max-grid", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the directory for per-grid studies, CSV, and plot.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.min_grid < 1 or args.max_grid < args.min_grid:
        raise ValueError("Grid range must satisfy 1 <= min-grid <= max-grid.")

    base_config = load_study_config(args.config)
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else base_config.results_dir
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir.parent / "Study" / "grid_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_resampled_snapshots(
        results_dir,
        nx=base_config.nx,
        nz=base_config.nz,
    )
    summaries = []
    for grid_size in range(args.min_grid, args.max_grid + 1):
        grid_config = StudyConfig(
            name=f"grid_{grid_size}x{grid_size}",
            results_dir=results_dir,
            output_dir=output_dir / f"grid_{grid_size}x{grid_size}",
            nx=base_config.nx,
            nz=base_config.nz,
            landmarks=build_grid_landmarks(base_config, grid_size),
            energy_fraction=base_config.energy_fraction,
            rank=base_config.rank,
        )
        print(
            f"Evaluating {grid_size}x{grid_size} grid "
            f"({grid_size * grid_size} landmarks)..."
        )
        case_metrics = evaluate_study(
            grid_config,
            snapshots=snapshots,
            persist=False,
        )
        summaries.append(summarize_metrics(grid_size, case_metrics))
        print(f"  averaged {len(case_metrics)} held-out cases")

    summary_file = output_dir / "grid_comparison.csv"
    plot_file = output_dir / "grid_comparison.png"
    write_summary_csv(summary_file, summaries)
    plot_summaries(plot_file, summaries)
    print(f"Wrote {summary_file}")
    print(f"Wrote {plot_file}")


if __name__ == "__main__":
    main()