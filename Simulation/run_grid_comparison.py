"""Compare leave-one-out reconstruction errors across landmark grid sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .utils.create_study_configs import grid_values
    from .utils.study_pipeline import (
        Landmark,
        StudyConfig,
        evaluate_study,
        load_study_config,
    )
    from .utils.coordinate_transforms import suction_position_in_nastran_coordinates
    from .utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from utils.create_study_configs import grid_values
    from utils.study_pipeline import (
        Landmark,
        StudyConfig,
        evaluate_study,
        load_study_config,
    )
    from utils.coordinate_transforms import suction_position_in_nastran_coordinates
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


def select_best_worst_cases(
    case_metrics: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the lowest and highest finite maximum absolute error cases."""
    finite_cases = [
        case
        for case in case_metrics
        if math.isfinite(float(case["maximum_absolute_error"]))
    ]
    if not finite_cases:
        raise ValueError("No cases have a finite maximum absolute error.")
    best = min(
        finite_cases,
        key=lambda case: (float(case["maximum_absolute_error"]), case["case_name"]),
    )
    worst = max(
        finite_cases,
        key=lambda case: (float(case["maximum_absolute_error"]), case["case_name"]),
    )
    return best, worst


def load_case_parameters(parameter_file: Path) -> dict[str, dict[str, str]]:
    """Load the exact generated parameter row for each simulation case."""
    with parameter_file.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return {
            f"simulation_{int(row['simulation_id']):03d}": dict(row)
            for row in rows
        }


def case_record(
    *,
    grid_size: int,
    case_metrics: dict[str, Any],
    landmarks: tuple[Landmark, ...],
    simulation_parameters: dict[str, str],
    nx: int,
    nz: int,
    energy_fraction: float | None,
    rank: int | None,
) -> dict[str, Any]:
    sauger_positions = sauger_plot_coordinates(simulation_parameters)
    return {
        "case_name": case_metrics["case_name"],
        "grid_size": grid_size,
        "landmark_count": len(landmarks),
        "interpolation_grid": {"nx": nx, "nz": nz},
        "pod": {"rank": rank, "energy_fraction": energy_fraction},
        "selection_metric": "maximum_absolute_error",
        "maximum_absolute_error": float(case_metrics["maximum_absolute_error"]),
        "metrics": {
            metric: float(case_metrics[metric])
            for metric in METRICS
        },
        "landmarks_nastran_xz": [
            {"landmark_index": index, "x": point.x, "z": point.z}
            for index, point in enumerate(landmarks, start=1)
        ],
        "sauger_positions_nastran_xyz": sauger_positions,
        "simulation_parameters": simulation_parameters,
    }


def sauger_plot_coordinates(
    simulation_parameters: dict[str, str],
) -> list[dict[str, float | int | None]]:
    """Return actual sauger x/y/z positions in the Nastran plotting frame."""
    positions = []
    sauger_index = 1
    while f"s{sauger_index}_x" in simulation_parameters:
        inventor_x = float(simulation_parameters[f"s{sauger_index}_x"])
        inventor_z = float(simulation_parameters[f"s{sauger_index}_z"])
        nastran_x, nastran_z = suction_position_in_nastran_coordinates(
            inventor_x,
            inventor_z,
        )
        displacement_y = simulation_parameters.get(
            f"sauger_{sauger_index}_displacement"
        )
        positions.append(
            {
                "sauger_index": sauger_index,
                "x": nastran_x,
                "y": None if displacement_y is None else float(displacement_y),
                "z": nastran_z,
            }
        )
        sauger_index += 1
    return positions


def format_sauger_coordinates(simulation_parameters: dict[str, str]) -> str:
    """Format sauger x/z coordinates and y positions in the plot's frame."""
    positions = sauger_plot_coordinates(simulation_parameters)
    if not positions:
        return "Sauger coordinates unavailable"
    return "Saugers Nastran (x,z), y position:\n" + "\n".join(
        f"S{position['sauger_index']}=({position['x']:.1f}, {position['z']:.1f}), "
        f"s{position['sauger_index']}_y="
        f"{position['y']:.3f}"
        if position["y"] is not None
        else f"S{position['sauger_index']}=({position['x']:.1f}, {position['z']:.1f}), "
        f"s{position['sauger_index']}_y unavailable"
        for position in positions
    )


def plot_surface_comparison(
    output_file: Path,
    cases: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]],
    axis_indices: tuple[int, int],
) -> None:
    """Render FEM, reconstruction, and absolute-error fields for selected cases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(len(cases), 3, figsize=(15, 4.5 * len(cases)))
    axes = np.atleast_2d(axes)
    panel_fields = (
        ("reference_displacement", "FEM displacement magnitude"),
        ("reconstructed_displacement", "Reconstructed displacement magnitude"),
    )

    for row_index, (case_label, metrics, surface) in enumerate(cases):
        valid_mask = np.asarray(surface["valid_mask"], dtype=bool)
        coordinates = surface["coordinates"][valid_mask]
        x_values = coordinates[:, axis_indices[0]]
        z_values = coordinates[:, axis_indices[1]]
        triangulation = mtri.Triangulation(x_values, z_values)
        reference = surface["reference_displacement"][valid_mask]
        reconstructed = surface["reconstructed_displacement"][valid_mask]
        values_to_plot = [
            (np.linalg.norm(reference, axis=1), panel_fields[0][1]),
            (np.linalg.norm(reconstructed, axis=1), panel_fields[1][1]),
            (
                np.max(np.abs(reference - reconstructed), axis=1),
                "Maximum component absolute error",
            ),
        ]
        for axis, (values, title) in zip(axes[row_index], values_to_plot):
            contour = axis.tricontourf(
                triangulation,
                values,
                levels=20,
                cmap="viridis",
            )
            axis.set_title(title)
            axis.set_xlabel(f"Coordinate axis {axis_indices[0]}")
            axis.set_ylabel(f"Coordinate axis {axis_indices[1]}")
            axis.set_aspect("equal", adjustable="box")
            landmarks = metrics["landmarks_nastran_xz"]
            axis.scatter(
                [point["x"] for point in landmarks],
                [point["z"] for point in landmarks],
                s=8,
                c="white",
                edgecolors="black",
                linewidths=0.35,
                zorder=5,
            )
            saugers = metrics["sauger_positions_nastran_xyz"]
            axis.scatter(
                [point["x"] for point in saugers],
                [point["z"] for point in saugers],
                s=10,
                c="#e53935",
                edgecolors="white",
                linewidths=0.35,
                zorder=6,
            )
            figure.colorbar(contour, ax=axis, label="Displacement (model units)")
        figure.text(
            0.005,
            1.0 - (row_index + 0.5) / len(cases),
            f"{case_label}: grid {metrics['grid_size']}x{metrics['grid_size']}\n"
            f"{metrics['case_name']}\n"
            f"max abs error={metrics['maximum_absolute_error']:.6g}\n"
            f"{format_sauger_coordinates(metrics['simulation_parameters'])}",
            ha="left",
            va="center",
            fontsize=8,
        )

    figure.suptitle("Selected leave-one-out reconstruction surfaces")
    figure.legend(
        handles=(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor="black", markersize=4, label="Landmarks",
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#e53935",
                markeredgecolor="white", markersize=4, label="Saugers",
            ),
        ),
        loc="lower center",
        ncol=2,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.23, 0.06, 1, 0.97))
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def write_summary_csv(
    output_file: Path,
    summaries: list[dict[str, int | float | None]],
) -> None:
    fieldnames = [
        "grid_size",
        "landmark_count",
        "case_count",
        "best_case_name",
        "best_maximum_absolute_error",
        "worst_case_name",
        "worst_maximum_absolute_error",
    ]
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
            axis.errorbar(
                x_values,
                y_values,
                yerr=[
                    [mean - float(minimum) for mean, minimum in zip(y_values, case_min)],
                    [float(maximum) - mean for mean, maximum in zip(y_values, case_max)],
                ],
                fmt="none",
                ecolor=line.get_color(),
                elinewidth=1.1,
                capsize=3,
                alpha=0.65,
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
    figure.savefig(output_file, dpi=300)
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
    simulation_parameters = load_case_parameters(
        results_dir.parent / "parameters.csv"
    )
    summaries = []
    per_grid_cases = []
    global_candidates: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
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
        best_case, worst_case = select_best_worst_cases(case_metrics)
        summaries[-1].update(
            {
                "best_case_name": best_case["case_name"],
                "best_maximum_absolute_error": best_case["maximum_absolute_error"],
                "worst_case_name": worst_case["case_name"],
                "worst_maximum_absolute_error": worst_case["maximum_absolute_error"],
            }
        )
        selected_case_names = {best_case["case_name"], worst_case["case_name"]}
        captured_surfaces: dict[str, dict[str, np.ndarray]] = {}
        selected_metrics = evaluate_study(
            grid_config,
            snapshots=snapshots,
            persist=False,
            case_names=selected_case_names,
            capture_surfaces=captured_surfaces,
        )
        selected_by_name = {row["case_name"]: row for row in selected_metrics}
        grid_records = {}
        grid_plot_cases = []
        for label, selected in (("best", best_case), ("worst", worst_case)):
            case_name = selected["case_name"]
            case_record_data = case_record(
                grid_size=grid_size,
                case_metrics=selected,
                landmarks=grid_config.landmarks,
                simulation_parameters=simulation_parameters.get(case_name, {}),
                nx=base_config.nx,
                nz=base_config.nz,
                energy_fraction=base_config.energy_fraction,
                rank=base_config.rank,
            )
            grid_records[label] = case_record_data
            captured = captured_surfaces[case_name]
            grid_plot_cases.append((label, case_record_data, captured))
            global_candidates.append((case_record_data, captured))
        per_grid_cases.append(
            {
                "grid_size": grid_size,
                "landmark_count": len(grid_config.landmarks),
                **grid_records,
            }
        )
        plot_surface_comparison(
            output_dir / f"grid_{grid_size}x{grid_size}_best_worst.png",
            grid_plot_cases,
            snapshots.grid.axis_indices,
        )
        print(f"  averaged {len(case_metrics)} held-out cases")

    summary_file = output_dir / "grid_comparison.csv"
    plot_file = output_dir / "grid_comparison.png"
    write_summary_csv(summary_file, summaries)
    plot_summaries(plot_file, summaries)
    global_best = min(
        global_candidates,
        key=lambda candidate: candidate[0]["maximum_absolute_error"],
    )
    global_worst = max(
        global_candidates,
        key=lambda candidate: candidate[0]["maximum_absolute_error"],
    )
    global_records = {"best": global_best[0], "worst": global_worst[0]}
    with (output_dir / "global_best_worst_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selection_metric": "maximum_absolute_error",
                "results_dir": str(results_dir),
                "per_grid": per_grid_cases,
                "global": global_records,
            },
            file,
            indent=2,
        )
        file.write("\n")

    best_surface = global_best[1]
    worst_surface = global_worst[1]
    np.savez_compressed(
        output_dir / "global_best_worst_surfaces.npz",
        coordinates=best_surface["coordinates"],
        valid_mask=best_surface["valid_mask"],
        best_reference_displacement=best_surface["reference_displacement"],
        best_reconstructed_displacement=best_surface["reconstructed_displacement"],
        worst_reference_displacement=worst_surface["reference_displacement"],
        worst_reconstructed_displacement=worst_surface["reconstructed_displacement"],
    )
    plot_surface_comparison(
        output_dir / "global_best_worst_surfaces.png",
        [
            ("Global best", global_best[0], best_surface),
            ("Global worst", global_worst[0], worst_surface),
        ],
        snapshots.grid.axis_indices,
    )
    print(f"Wrote {summary_file}")
    print(f"Wrote {plot_file}")
    print(f"Wrote best/worst surface comparisons to {output_dir}")


if __name__ == "__main__":
    main()