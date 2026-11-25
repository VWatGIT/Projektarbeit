"""Evaluate Gappy POD reconstruction over the cumulative energy fraction.

A fixed square landmark grid (default 3x3) observes each held-out surface. For
every energy fraction on a logarithmic scale (default 0.1 up to 1), the POD
basis is fitted on all remaining cases in a leave-one-out evaluation. The
number of modes follows from the energy fraction.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .study_landmarks_grid import METRICS, build_grid_landmarks
    from .study_pod_surfaces import format_duration, write_csv
    from .utils.gappy_pod import PODBasis, reconstruction_errors
    from .utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from .utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        load_study_config,
    )
    from .utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from study_landmarks_grid import METRICS, build_grid_landmarks
    from study_pod_surfaces import format_duration, write_csv
    from utils.gappy_pod import PODBasis, reconstruction_errors
    from utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        load_study_config,
    )
    from utils.surface_resampling import load_resampled_snapshots


def evaluate_energy_fractions(
    states: np.ndarray,
    operator: Any,
    valid_coordinates: np.ndarray,
    case_names: Sequence[str],
    energy_fractions: Sequence[float],
) -> list[dict[str, Any]]:
    """Return one metric row per (energy fraction, held-out case).

    The SVD of each training set depends only on the held-out case, so it is
    computed once and reused for every energy fraction.
    """
    case_count = states.shape[1]
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    start = time.monotonic()
    for held_out in range(case_count):
        training = np.delete(np.arange(case_count), held_out)
        modes, singular_values, _ = np.linalg.svd(
            states[:, training],
            full_matrices=False,
        )
        reference = states[:, held_out]
        reference_deformed = valid_coordinates + reference.reshape(-1, 3)
        observations = operator.observe(reference)
        for fraction_index, energy_fraction in enumerate(energy_fractions):
            basis = PODBasis.from_svd(
                modes,
                singular_values,
                rank=None,
                energy_fraction=float(energy_fraction),
            )
            reconstructed, diagnostics = basis.reconstruct(observations, operator)
            errors = reconstruction_errors(reference, reconstructed)
            rows_by_key[(fraction_index, held_out)] = {
                "energy_fraction": float(energy_fraction),
                "case_name": case_names[held_out],
                "pod_rank": int(basis.modes.shape[1]),
                **errors,
                "condition_number": diagnostics["condition_number"],
                "deformed_coordinate_relative_l2_error": float(
                    errors["absolute_l2_error"]
                    / max(np.linalg.norm(reference_deformed), 1e-12)
                ),
            }
        elapsed = time.monotonic() - start
        eta = elapsed / (held_out + 1) * (case_count - held_out - 1)
        print(
            f"[{held_out + 1}/{case_count}] held-out {case_names[held_out]} "
            f"| elapsed {format_duration(elapsed)} | ETA ~{format_duration(eta)}"
        )
    return [
        rows_by_key[(fraction_index, held_out)]
        for fraction_index in range(len(energy_fractions))
        for held_out in range(case_count)
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for energy_fraction in sorted({row["energy_fraction"] for row in rows}):
        group = [row for row in rows if row["energy_fraction"] == energy_fraction]
        summary: dict[str, Any] = {
            "energy_fraction": energy_fraction,
            "evaluation_count": len(group),
            "mean_pod_rank": float(np.mean([row["pod_rank"] for row in group])),
            "min_pod_rank": int(min(row["pod_rank"] for row in group)),
            "max_pod_rank": int(max(row["pod_rank"] for row in group)),
        }
        for metric in METRICS:
            values = np.array(
                [row[metric] for row in group if math.isfinite(row[metric])]
            )
            summary[f"mean_{metric}"] = float(values.mean()) if values.size else None
            summary[f"std_{metric}"] = float(values.std()) if values.size else None
            summary[f"min_{metric}"] = float(values.min()) if values.size else None
            summary[f"max_{metric}"] = float(values.max()) if values.size else None
        summaries.append(summary)
    return summaries


def _set_energy_log_axis(axis) -> None:
    """Log x-axis with evenly spaced, labelled ticks that adapt to the data range."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.xaxis.set_minor_locator(NullLocator())


def plot_mode_count(output_file: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies = [row["energy_fraction"] for row in summaries]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        energies,
        [row["mean_pod_rank"] for row in summaries],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Mean",
    )
    axis.fill_between(
        energies,
        [row["min_pod_rank"] for row in summaries],
        [row["max_pod_rank"] for row in summaries],
        alpha=0.25,
        label="Min-max",
    )
    _set_energy_log_axis(axis)
    axis.set_ylabel("Number of POD modes")
    axis.set_title("POD modes required by energy fraction")
    axis.grid(True, alpha=0.3, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def plot_error_metrics(
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
    energies = [row["energy_fraction"] for row in summaries]
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        mean = np.array([row[f"mean_{metric}"] for row in summaries], dtype=float)
        std = np.array([row[f"std_{metric}"] for row in summaries], dtype=float)
        axis.plot(energies, mean, marker="o", markersize=3, linewidth=1.8, label="Mean")
        axis.fill_between(
            energies,
            np.maximum(mean - std, 0),
            mean + std,
            alpha=0.25,
            label="±1 std",
        )
        axis.set_title(title)
        _set_energy_log_axis(axis)
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=7)
        if "relative" in metric:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for axis in axes[-1]:
        axis.set_xlabel("Cumulative energy fraction")
    figure.suptitle(
        f"Leave-one-out error by energy fraction ({landmark_count} landmarks)"
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="config_pod_modes.json",
        help="Default arguments file in StudyConfigs, or a path to one.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--surface-resampling", choices=("linear", "elements"), default="linear")
    parser.add_argument("--min-energy", type=float)
    parser.add_argument(
        "--points",
        type=int,
        help="Number of logarithmically spaced energy fractions from --min-energy to 1.",
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
    simulation_dir = Path(__file__).resolve().parent
    results_dir = simulation_dir / "Results" / args.experiment
    config = load_study_config(
        POD_STUDY_CONFIG,
        results_dir=results_dir,
        output_dir=args.output_dir,
    )
    snapshots = load_resampled_snapshots(
        config.results_dir,
        nx=config.nx,
        nz=config.nz,
        method=args.surface_resampling,
    )
    config = replace(config, rank=None)
    if args.output_dir is None:
        config = replace(config, output_dir=config.results_dir / "Study" / args.name)
    landmarks = build_grid_landmarks(config, args.grid_size)
    operator, _ = _landmark_operator(snapshots, landmarks)
    valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]

    energy_fractions = np.logspace(np.log10(args.min_energy), 0, num=args.points)
    print(
        f"{len(snapshots.case_names)} cases, {len(landmarks)} landmarks, "
        f"{args.points} energy fractions from {args.min_energy:g} to 1."
    )
    rows = evaluate_energy_fractions(
        states=snapshots.state_matrix(),
        operator=operator,
        valid_coordinates=valid_coordinates,
        case_names=snapshots.case_names,
        energy_fractions=energy_fractions,
    )
    summaries = summarize(rows)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "energy_fraction_cases.csv", rows)
    write_csv(output_dir / "energy_fraction_summary.csv", summaries)
    plot_mode_count(output_dir / "energy_fraction_modes.png", summaries)
    plot_error_metrics(
        output_dir / "energy_fraction_errors.png",
        summaries,
        len(landmarks),
    )
    print(f"Results written to {output_dir}")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if not 0 < args.min_energy < 1:
        raise ValueError("min-energy must be in (0, 1).")
    if args.points < 2:
        raise ValueError("points must be at least 2.")
    run_study(args)


if __name__ == "__main__":
    main()
