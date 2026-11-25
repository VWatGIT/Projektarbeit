"""Compare grid-study reconstruction errors across surface-generation sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

import numpy as np

try:
    from .utils.gappy_pod import reconstruction_errors
    from .utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from utils.gappy_pod import reconstruction_errors
    from utils.surface_resampling import load_resampled_snapshots


SIMULATION_DIR = Path(__file__).resolve().parent
METRICS = {
    "relative": ("relative_l2_error", "max_case_relative_l2_error"),
    "absolute_max": ("maximum_absolute_error", "max_case_maximum_absolute_error"),
    "absolute_mean": ("maximum_absolute_error", "mean_maximum_absolute_error"),
    "absolute": ("absolute_l2_error", "max_case_absolute_l2_error"),
}
DEFAULT_METRICS = ("relative", "absolute_max", "absolute_mean")
CLAMP_EXPERIMENTS = {
    1: {1: "cl1_sf25_d1_r0", 3: "cl3_sf25_d1_r0", 9: "cl9_sf25_d1_r0"},
    5: {1: "cl1_sf25_d5_r0", 3: "cl3_sf25_d5_r0", 9: "cl9_sf25_d5_r0"},
}
DISPLACEMENT_EXPERIMENTS = {
    1: "cl3_sf25_d1_r0",
    5: "cl3_sf25_d5_r0",
    10: "cl3_sf25_d10_r0",
}
CL9_DISPLACEMENT_EXPERIMENTS = {
    1: "cl9_sf25_d1_r0",
    5: "cl9_sf25_d5_r0",
}
GRID_EXPERIMENTS = {3: DISPLACEMENT_EXPERIMENTS, 9: CL9_DISPLACEMENT_EXPERIMENTS}


@dataclass(frozen=True)
class Baseline:
    case_count: int
    values: dict[str, float]


def load_baseline(results_dir: Path, experiment: str, study: str) -> Baseline:
    """Evaluate the flat-surface estimate on the grid used by the study."""
    metadata_file = results_dir / experiment / "Study" / study / "global_best_worst_metadata.json"
    with metadata_file.open(encoding="utf-8") as file:
        metadata = json.load(file)
    try:
        grids = [
            (record[selection]["interpolation_grid"]["nx"],
             record[selection]["interpolation_grid"]["nz"])
            for record in metadata["per_grid"]
            for selection in ("best", "worst")
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{metadata_file}: missing interpolation grid metadata") from exc
    if not grids or any(grid != grids[0] for grid in grids):
        raise ValueError(f"{metadata_file}: inconsistent or missing interpolation grids")
    nx, nz = grids[0]
    if not isinstance(nx, int) or isinstance(nx, bool) or not isinstance(nz, int) or isinstance(nz, bool):
        raise ValueError(f"{metadata_file}: invalid interpolation grid dimensions")
    method = metadata.get("surface_resampling", "linear")
    if method not in ("linear", "elements"):
        raise ValueError(f"{metadata_file}: invalid surface resampling method {method!r}")
    if method == "linear":
        snapshots = load_resampled_snapshots(results_dir / experiment, nx=nx, nz=nz)
    else:
        snapshots = load_resampled_snapshots(results_dir / experiment, nx=nx, nz=nz, method=method)
    states = snapshots.state_matrix()
    if states.shape[0] == 0 or states.shape[1] == 0 or not np.isfinite(states).all():
        raise ValueError(f"{experiment}: no finite valid FEM displacement states")
    case_errors = [
        reconstruction_errors(state, np.zeros_like(state))
        for state in states.T
    ]
    values = {
        "relative": max(case["relative_l2_error"] for case in case_errors),
        "absolute_max": max(case["maximum_absolute_error"] for case in case_errors),
        "absolute_mean": fmean(case["maximum_absolute_error"] for case in case_errors),
        "absolute": max(case["absolute_l2_error"] for case in case_errors),
    }
    return Baseline(case_count=len(case_errors), values=values)


def load_grid_errors(
    results_dir: Path, experiment: str, study: str, metric: str = "relative",
    expected_cases: int | None = None,
) -> dict[int, float]:
    """Read a finite held-out-case error summary for each grid."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    metric_name, value_column = METRICS[metric]
    finite_column = f"finite_case_count_{metric_name}"
    path = results_dir / experiment / "Study" / study / "grid_comparison.csv"
    required = {
        "grid_size",
        "landmark_count",
        "case_count",
        finite_column,
        value_column,
    }
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        errors: dict[int, float] = {}
        for line, row in enumerate(reader, start=2):
            try:
                grid_size = int(row["grid_size"])
                count = int(row["landmark_count"])
                case_count = int(row["case_count"])
                finite_count = int(row[finite_column])
                error = float(row[value_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid grid or error data") from exc
            if grid_size < 1 or count != grid_size**2 or grid_size in errors:
                raise ValueError(f"{path}:{line}: invalid or duplicate square grid")
            if case_count < 1 or finite_count != case_count:
                raise ValueError(
                    f"{path}:{line}: {metric_name} errors are not finite for all cases"
                )
            if expected_cases is not None and case_count != expected_cases:
                raise ValueError(
                    f"{path}:{line}: study has {case_count} cases but current FEM data "
                    f"has {expected_cases}; rerun the grid study"
                )
            if not math.isfinite(error) or error < 0:
                raise ValueError(f"{path}:{line}: invalid {value_column}")
            errors[grid_size] = error
    if not errors:
        raise ValueError(f"{path}: no grid results")
    return errors


def validate_clamps(errors: dict[int, dict[int, float]]) -> None:
    missing = [clamps for clamps, values in errors.items() if 3 not in values]
    if missing:
        raise ValueError(f"Missing 3x3 grid results for clamp counts: {missing}")


def validate_grids(errors: dict[int, dict[int, float]]) -> None:
    grid_sets = [set(values) for values in errors.values()]
    if not grid_sets or any(grids != grid_sets[0] for grids in grid_sets[1:]):
        raise ValueError("Displacement studies must contain the same grid sizes")


def format_error_axis(axis, metric: str) -> None:
    from matplotlib.ticker import PercentFormatter

    if metric == "relative":
        axis.set_ylabel("Maximum relative L2 displacement error")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    elif metric == "absolute":
        axis.set_ylabel("Maximum absolute L2 displacement error (model units)")
    elif metric == "absolute_max":
        axis.set_ylabel("Maximum case-wise max. absolute error (model units)")
    elif metric == "absolute_mean":
        axis.set_ylabel("Mean case-wise max. absolute error (model units)")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_clamps(
    output_file: Path,
    errors: dict[int, dict[int, dict[int, float]]],
    metric: str = "relative",
    *,
    baselines: dict[int, dict[int, float]],
) -> None:
    import matplotlib.pyplot as plt

    if not errors:
        raise ValueError("No displacement series supplied for clamp comparison")
    for values in errors.values():
        validate_clamps(values)
    counts = sorted(next(iter(errors.values())))
    if any(sorted(values) != counts for values in errors.values()):
        raise ValueError("Clamp displacement series must contain the same clamp counts")
    fig, axis = plt.subplots(figsize=(7, 5))
    for displacement, values in sorted(errors.items()):
        line = axis.plot(
            counts, [values[count][3] for count in counts],
            marker="o", label=f"{displacement} mm",
        )[0]
        axis.plot(
            counts, [baselines[displacement][count] for count in counts],
            linestyle="--", color=line.get_color(), marker="x",
            label=f"{displacement} mm flat baseline",
        )
    axis.set_xticks(counts)
    axis.set_xlabel("Number of clamps")
    format_error_axis(axis, metric)
    axis.set_title("Clamp count comparison (3x3 grid)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Displacement")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_grid_sizes(
    output_file: Path, errors: dict[int, dict[int, float]],
    clamps: int = 3, metric: str = "relative",
    *, baselines: dict[int, float],
) -> None:
    import matplotlib.pyplot as plt

    validate_grids(errors)
    grid_sizes = sorted(next(iter(errors.values())))
    counts = [size**2 for size in grid_sizes]
    fig, axis = plt.subplots(figsize=(8, 5))
    for displacement, values in sorted(errors.items()):
        line = axis.plot(
            counts,
            [values[size] for size in grid_sizes],
            marker="o",
            label=f"{displacement} mm",
        )[0]
        axis.axhline(
            baselines[displacement], color=line.get_color(), linestyle="--",
            label=f"{displacement} mm flat baseline",
        )
    axis.set_xticks(counts)
    axis.set_xlabel("Number of landmarks (square grid)")
    format_error_axis(axis, metric)
    axis.set_title(f"Grid-size comparison ({clamps} clamps)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Displacement")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SIMULATION_DIR / "Results")
    parser.add_argument("--output-dir", type=Path, default=SIMULATION_DIR / "Comparison")
    parser.add_argument("--study", default="landmarks_grid", help="Grid study directory name")
    parser.add_argument(
        "--plot", choices=("all", "clamps", "grids"), default="all",
        help="Choose a comparison to generate (default: all).",
    )
    parser.add_argument(
        "--metric", choices=("all", *METRICS), default="all",
        help="Select the error summary; all excludes the optional absolute L2 metric.",
    )
    parser.add_argument(
        "--grid-clamps", choices=("all", "3", "9"), default="all",
        help="Choose clamp counts for grid-size comparisons (default: both).",
    )
    args = parser.parse_args(argv)

    # Load every required input before producing any output.
    names = (
        {name for experiments in CLAMP_EXPERIMENTS.values() for name in experiments.values()}
        if args.plot in ("all", "clamps") else set()
    )
    selected_grids = {
        count: experiments for count, experiments in GRID_EXPERIMENTS.items()
        if args.plot in ("all", "grids") and args.grid_clamps in ("all", str(count))
    }
    if args.plot in ("all", "grids"):
        for experiments in selected_grids.values():
            names.update(experiments.values())
    metrics = DEFAULT_METRICS if args.metric == "all" else (args.metric,)
    baselines = {
        name: load_baseline(args.results_dir, name, args.study)
        for name in sorted(names)
    }
    loaded = {
        metric: {
            name: load_grid_errors(
                args.results_dir, name, args.study, metric, baselines[name].case_count
            )
            for name in sorted(names)
        }
        for metric in metrics
    }

    for metric in metrics:
        if args.plot in ("all", "clamps"):
            for displacement, experiments in CLAMP_EXPERIMENTS.items():
                try:
                    validate_clamps({key: loaded[metric][name] for key, name in experiments.items()})
                except ValueError as exc:
                    raise ValueError(f"{displacement} mm: {exc}") from exc
        for experiments in selected_grids.values():
            validate_grids({key: loaded[metric][name] for key, name in experiments.items()})

    import matplotlib

    matplotlib.use("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        suffix = {
            "relative": "",
            "absolute_max": "_maximum_absolute",
            "absolute_mean": "_mean_maximum_absolute",
            "absolute": "_absolute_l2",
        }[metric]
        if args.plot in ("all", "clamps"):
            path = args.output_dir / f"clamps_d1_d5_grid3{suffix}.png"
            clamps = {
                displacement: {key: loaded[metric][name] for key, name in experiments.items()}
                for displacement, experiments in CLAMP_EXPERIMENTS.items()
            }
            clamp_baselines = {
                displacement: {
                    count: baselines[name].values[metric]
                    for count, name in experiments.items()
                }
                for displacement, experiments in CLAMP_EXPERIMENTS.items()
            }
            plot_clamps(path, clamps, metric, baselines=clamp_baselines)
            print(f"Wrote {path}")
        for count, experiments in selected_grids.items():
            path = args.output_dir / f"cl{count}_displacement_by_grid{suffix}.png"
            grids = {key: loaded[metric][name] for key, name in experiments.items()}
            grid_baselines = {key: baselines[name].values[metric] for key, name in experiments.items()}
            plot_grid_sizes(path, grids, count, metric, baselines=grid_baselines)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
