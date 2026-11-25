"""Evaluate Gappy POD reconstruction error over the number of training surfaces.

A fixed square landmark grid (default 3x3) observes each held-out surface. For
for every training size N (1 up to 2 x the landmark count by default), the POD
basis is fitted on N surfaces drawn at random from all remaining cases,
repeated for several trials.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .study_landmarks_grid import METRICS, build_grid_landmarks
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
    from utils.gappy_pod import PODBasis, reconstruction_errors
    from utils.script_defaults import POD_STUDY_CONFIG, load_script_defaults
    from utils.study_pipeline import (
        _grid_coordinates_3d,
        _landmark_operator,
        load_study_config,
    )
    from utils.surface_resampling import load_resampled_snapshots


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


class ProgressReporter:
    """Single-line console progress with elapsed time and a rough ETA."""

    def __init__(self, total: int, min_interval: float = 0.5) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.min_interval = min_interval
        self.start = time.monotonic()
        self.last_print = 0.0

    def update(self, current_size: int) -> None:
        self.done += 1
        now = time.monotonic()
        if self.done < self.total and now - self.last_print < self.min_interval:
            return
        self.last_print = now
        elapsed = now - self.start
        eta = elapsed / self.done * (self.total - self.done)
        print(
            f"\r[{self.done}/{self.total}] {self.done / self.total:6.1%} | "
            f"training size {current_size} | elapsed {format_duration(elapsed)} | "
            f"ETA ~{format_duration(eta)}   ",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        print(f"\nFinished in {format_duration(time.monotonic() - self.start)}.")


def evaluate_training_sizes(
    states: np.ndarray,
    operator: Any,
    valid_coordinates: np.ndarray,
    case_names: Sequence[str],
    training_sizes: Sequence[int],
    trials: int,
    seed: int,
    rank: int | None,
    energy_fraction: float | None,
) -> list[dict[str, Any]]:
    """Return one metric row per (training size, held-out case, trial)."""
    case_count = states.shape[1]
    rows: list[dict[str, Any]] = []
    total = sum(
        case_count * (1 if size == case_count - 1 else trials)
        for size in training_sizes
    )
    progress = ProgressReporter(total)
    for size in training_sizes:
        for held_out in range(case_count):
            pool = np.delete(np.arange(case_count), held_out)
            # Only one distinct subset exists when all remaining cases are used.
            trial_count = 1 if size == pool.size else trials
            reference = states[:, held_out]
            reference_deformed = valid_coordinates + reference.reshape(-1, 3)
            observed = operator.observe(reference)
            for trial in range(trial_count):
                rng = np.random.default_rng([seed, size, held_out, trial])
                training = np.sort(rng.choice(pool, size=size, replace=False))
                basis = PODBasis.fit(
                    states[:, training],
                    rank=None if rank is None else min(rank, size),
                    energy_fraction=energy_fraction,
                )
                reconstructed, diagnostics = basis.reconstruct(observed, operator)
                errors = reconstruction_errors(reference, reconstructed)
                rows.append(
                    {
                        "training_size": size,
                        "case_name": case_names[held_out],
                        "trial": trial,
                        "pod_rank": int(basis.modes.shape[1]),
                        **errors,
                        "condition_number": diagnostics["condition_number"],
                        "deformed_coordinate_relative_l2_error": float(
                            errors["absolute_l2_error"]
                            / max(np.linalg.norm(reference_deformed), 1e-12)
                        ),
                    }
                )
                progress.update(size)
    progress.finish()
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for size in sorted({row["training_size"] for row in rows}):
        size_rows = [row for row in rows if row["training_size"] == size]
        summary: dict[str, Any] = {
            "training_size": size,
            "evaluation_count": len(size_rows),
            "mean_pod_rank": float(np.mean([row["pod_rank"] for row in size_rows])),
            "min_pod_rank": int(min(row["pod_rank"] for row in size_rows)),
            "max_pod_rank": int(max(row["pod_rank"] for row in size_rows)),
        }
        for metric in METRICS:
            values = np.array(
                [row[metric] for row in size_rows if math.isfinite(row[metric])]
            )
            summary[f"mean_{metric}"] = float(values.mean()) if values.size else None
            summary[f"std_{metric}"] = float(values.std()) if values.size else None
            summary[f"min_{metric}"] = float(values.min()) if values.size else None
            summary[f"max_{metric}"] = float(values.max()) if values.size else None
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summaries(
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
    sizes = [row["training_size"] for row in summaries]
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        mean = np.array([row[f"mean_{metric}"] for row in summaries], dtype=float)
        std = np.array([row[f"std_{metric}"] for row in summaries], dtype=float)
        axis.plot(sizes, mean, marker="o", linewidth=1.8, label="Mean")
        axis.fill_between(
            sizes,
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
        axis.set_xlabel("Number of training surfaces")
    figure.suptitle(
        f"Leave-one-out error by training set size ({landmark_count} landmarks)"
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def plot_mode_count(
    output_file: Path,
    summaries: list[dict[str, Any]],
    energy_fraction: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [row["training_size"] for row in summaries]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        sizes,
        [row["mean_pod_rank"] for row in summaries],
        marker="o",
        linewidth=1.8,
        label="Mean",
    )
    axis.fill_between(
        sizes,
        [row["min_pod_rank"] for row in summaries],
        [row["max_pod_rank"] for row in summaries],
        alpha=0.25,
        label="Min-max",
    )
    axis.set_xlabel("Number of training surfaces")
    axis.set_ylabel("Number of POD modes")
    title = "Modes required to reach the energy level"
    if energy_fraction is not None:
        title += f" (energy fraction {energy_fraction:g})"
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="config_pod_surfaces.json",
        help="Default arguments file in StudyConfigs, or a path to one.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--min-training", type=int)
    parser.add_argument(
        "--max-training",
        type=int,
        help="Defaults to unlimited, i.e. the number of cases minus one.",
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--surface-resampling", choices=("linear", "elements"), default="linear")
    parser.add_argument(
        "--energy-fraction",
        type=float,
        help=(
            "Cumulative POD energy in (0, 1] that determines the number of modes. "
            "Overrides the config's energy_fraction or fixed rank."
        ),
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
    case_count = len(snapshots.case_names)
    if args.energy_fraction is not None:
        if not 0 < args.energy_fraction <= 1:
            raise ValueError("energy-fraction must be in (0, 1].")
        config = replace(config, energy_fraction=args.energy_fraction, rank=None)
    if args.output_dir is None:
        config = replace(config, output_dir=config.results_dir / "Study" / args.name)
    landmarks = build_grid_landmarks(config, args.grid_size)
    max_training = args.max_training or case_count - 1
    if not 1 <= args.min_training <= max_training <= case_count - 1:
        raise ValueError(
            f"Training sizes must satisfy 1 <= min <= max <= {case_count - 1}."
        )
    if args.trials < 1:
        raise ValueError("trials must be positive.")

    operator, _ = _landmark_operator(snapshots, landmarks)
    valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]
    print(
        f"{case_count} cases, {len(landmarks)} landmarks, training sizes "
        f"{args.min_training}..{max_training}, {args.trials} trials each, "
        f"energy fraction {config.energy_fraction}, rank {config.rank}."
    )
    rows = evaluate_training_sizes(
        states=snapshots.state_matrix(),
        operator=operator,
        valid_coordinates=valid_coordinates,
        case_names=snapshots.case_names,
        training_sizes=range(args.min_training, max_training + 1),
        trials=args.trials,
        seed=args.seed,
        rank=config.rank,
        energy_fraction=config.energy_fraction,
    )
    summaries = summarize(rows)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_size_cases.csv", rows)
    write_csv(output_dir / "training_size_summary.csv", summaries)
    plot_summaries(output_dir / "training_size_summary.png", summaries, len(landmarks))
    plot_mode_count(
        output_dir / "training_size_modes.png",
        summaries,
        config.energy_fraction,
    )
    print(f"Results written to {output_dir}")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    run_study(args)


if __name__ == "__main__":
    main()
