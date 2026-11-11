"""Evaluate leave-one-out reconstruction across random landmark counts."""

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

try:
	from .utils.coordinate_transforms import suction_position_in_nastran_coordinates
	from .utils.random_landmark_study import (
		AreaWeightedSurfaceSampler,
		RandomLandmarkStudyConfig,
		STUDY_METRICS,
		load_random_landmark_study_config,
	)
	from .utils.study_pipeline import (
		Landmark,
		evaluate_landmark_scenarios,
	)
	from .utils.surface_resampling import (
		ResampledSnapshotSet,
		load_resampled_snapshots,
	)
except ImportError:
	from utils.coordinate_transforms import suction_position_in_nastran_coordinates
	from utils.random_landmark_study import (
		AreaWeightedSurfaceSampler,
		RandomLandmarkStudyConfig,
		STUDY_METRICS,
		load_random_landmark_study_config,
	)
	from utils.study_pipeline import Landmark, evaluate_landmark_scenarios
	from utils.surface_resampling import (
		ResampledSnapshotSet,
		load_resampled_snapshots,
	)


METRICS = STUDY_METRICS


def _load_case_parameters(parameter_file: Path) -> dict[str, dict[str, str]]:
	with parameter_file.open(newline="", encoding="utf-8") as file:
		rows = csv.DictReader(file)
		return {
			f"simulation_{int(row['simulation_id']):03d}": dict(row)
			for row in rows
		}


def _sauger_positions_nastran(
	simulation_parameters: dict[str, str],
) -> list[dict[str, float | int | None]]:
	positions = []
	sauger_index = 1
	while f"s{sauger_index}_x" in simulation_parameters:
		inventor_x = float(simulation_parameters[f"s{sauger_index}_x"])
		inventor_z = float(simulation_parameters[f"s{sauger_index}_z"])
		nastran_x, nastran_z = suction_position_in_nastran_coordinates(
			inventor_x,
			inventor_z,
		)
		y_value = simulation_parameters.get(f"sauger_{sauger_index}_displacement")
		positions.append(
			{
				"sauger_index": sauger_index,
				"x": nastran_x,
				"y": None if y_value is None else float(y_value),
				"z": nastran_z,
			}
		)
		sauger_index += 1
	return positions


def derive_trial_seed(base_seed: int, amount: int, trial: int) -> int:
	"""Return a deterministic seed for a landmark-count/trial pair."""
	return base_seed + amount * 1_000_003 + trial


def select_best_worst_cases(
	case_rows: list[dict[str, Any]],
	metric: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Select finite minimum and maximum records for a per-case metric."""
	if metric not in METRICS:
		choices = ", ".join(METRICS)
		raise ValueError(f"metric must be one of {choices}; got {metric!r}.")
	finite_rows = [
		row
		for row in case_rows
		if metric in row and math.isfinite(float(row[metric]))
	]
	if not finite_rows:
		raise ValueError(f"No cases have a finite {metric}.")
	best = min(
		finite_rows,
		key=lambda row: (
			float(row[metric]),
			str(row["case_name"]),
			str(row["scenario_id"]),
		),
	)
	worst = max(
		finite_rows,
		key=lambda row: (
			float(row[metric]),
			str(row["case_name"]),
			str(row["scenario_id"]),
		),
	)
	return best, worst


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		raise ValueError(f"Cannot write an empty CSV: {path}")
	with path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
	values = [
		float(row[metric])
		for row in rows
		if metric in row and math.isfinite(float(row[metric]))
	]
	return fmean(values) if values else None


def aggregate_amount_metrics(
	trial_rows: list[dict[str, Any]],
	case_rows: list[dict[str, Any]],
	amounts: tuple[int, ...],
) -> list[dict[str, Any]]:
	"""Summarize trial means and held-out-case spread for each amount."""
	summaries = []
	for amount in amounts:
		successful_trials = [
			row
			for row in trial_rows
			if row["status"] == "success" and int(row["amount"]) == amount
		]
		amount_cases = [row for row in case_rows if int(row["amount"]) == amount]
		summary: dict[str, Any] = {
			"amount": amount,
			"successful_trials": len(successful_trials),
			"failed_trials": sum(
				row["status"] == "failed"
				and int(row["amount"]) == amount
				for row in trial_rows
			),
			"case_count": len(amount_cases),
		}
		for metric in METRICS:
			trial_values = [
				float(row[f"mean_{metric}"])
				for row in successful_trials
				if row.get(f"mean_{metric}") is not None
				and math.isfinite(float(row[f"mean_{metric}"]))
			]
			case_means = []
			case_names = sorted({row["case_name"] for row in amount_cases})
			for case_name in case_names:
				case_mean = _mean_metric(
					[row for row in amount_cases if row["case_name"] == case_name],
					metric,
				)
				if case_mean is not None:
					case_means.append(case_mean)
			summary[f"mean_{metric}"] = (
				fmean(trial_values) if trial_values else None
			)
			summary[f"std_{metric}"] = pstdev(trial_values) if trial_values else None
			summary[f"min_case_mean_{metric}"] = (
				min(case_means) if case_means else None
			)
			summary[f"max_case_mean_{metric}"] = (
				max(case_means) if case_means else None
			)
		summaries.append(summary)
	return summaries


def _selected_case_record(
	row: dict[str, Any],
	*,
	config: RandomLandmarkStudyConfig,
	scenario_metadata: dict[str, dict[str, Any]],
	case_parameters: dict[str, dict[str, str]],
) -> dict[str, Any]:
	return {
		"case_name": row["case_name"],
		"scenario_id": row["scenario_id"],
		"amount": int(row["amount"]),
		"trial": int(row["trial"]),
		"seed": int(row["seed"]),
		"selection_metric": config.best_worst_selection_metric,
		"selection_metric_value": float(row[config.best_worst_selection_metric]),
		"maximum_absolute_error": float(row["maximum_absolute_error"]),
		"metrics": {metric: float(row[metric]) for metric in METRICS},
		"grid": {"nx": config.nx, "nz": config.nz},
		"landmark_count": int(row["amount"]),
		"landmarks_nastran_xz": scenario_metadata[row["scenario_id"]]["landmarks"],
		"sauger_positions_nastran_xyz": _sauger_positions_nastran(
			case_parameters.get(row["case_name"], {})
		),
	}


def _plot_surface_comparison(
	output_file: Path,
	cases: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]],
	axis_indices: tuple[int, int],
) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import matplotlib.tri as mtri
	from matplotlib.lines import Line2D

	figure, axes = plt.subplots(len(cases), 3, figsize=(15, 4.5 * len(cases)))
	axes = np.atleast_2d(axes)
	for row_index, (case_label, record, surface) in enumerate(cases):
		valid_mask = np.asarray(surface["valid_mask"], dtype=bool)
		coordinates = surface["coordinates"][valid_mask]
		triangulation = mtri.Triangulation(
			coordinates[:, axis_indices[0]],
			coordinates[:, axis_indices[1]],
		)
		reference = surface["reference_displacement"][valid_mask]
		reconstructed = surface["reconstructed_displacement"][valid_mask]
		fields = (
			(np.linalg.norm(reference, axis=1), "FEM displacement magnitude"),
			(
				np.linalg.norm(reconstructed, axis=1),
				"Reconstructed displacement magnitude",
			),
			(
				np.max(np.abs(reference - reconstructed), axis=1),
				"Maximum component absolute error",
			),
		)
		landmarks = record["landmarks_nastran_xz"]
		saugers = record["sauger_positions_nastran_xyz"]
		for axis, (values, title) in zip(axes[row_index], fields):
			contour = axis.tricontourf(
				triangulation,
				values,
				levels=20,
				cmap="viridis",
			)
			axis.scatter(
				[point["x"] for point in landmarks],
				[point["z"] for point in landmarks],
				s=8,
				c="white",
				edgecolors="black",
				linewidths=0.35,
				zorder=5,
			)
			axis.scatter(
				[point["x"] for point in saugers],
				[point["z"] for point in saugers],
				s=18,
				c="#e53935",
				edgecolors="white",
				linewidths=0.5,
				zorder=6,
			)
			axis.set_title(title)
			axis.set_xlabel(f"Coordinate axis {axis_indices[0]}")
			axis.set_ylabel(f"Coordinate axis {axis_indices[1]}")
			axis.set_aspect("equal", adjustable="box")
			figure.colorbar(contour, ax=axis, label="Displacement (model units)")
		figure.text(
			0.005,
			1.0 - (row_index + 0.5) / len(cases),
			f"{case_label}: {record['amount']} landmarks\n"
			f"{record['case_name']}, trial {record['trial']}\n"
			f"max abs error={record['maximum_absolute_error']:.6g}\n"
			f"selection {record['selection_metric']}="
			f"{record['selection_metric_value']:.6g}",
			ha="left",
			va="center",
			fontsize=8,
		)
	figure.suptitle("Selected random-landmark reconstructions")
	figure.legend(
		handles=(
			Line2D(
				[0], [0], marker="o", color="none", markerfacecolor="white",
				markeredgecolor="black", markersize=4, label="Landmarks",
			),
			Line2D(
				[0], [0], marker="o", color="none", markerfacecolor="#e53935",
				markeredgecolor="white", markersize=5, label="Saugers",
			),
		),
		loc="lower center",
		ncol=1,
		fontsize=8,
	)
	figure.tight_layout(rect=(0.23, 0.06, 1, 0.97))
	figure.savefig(output_file, dpi=300)
	plt.close(figure)


def _plot_amount_summary(output_file: Path, summaries: list[dict[str, Any]]) -> None:
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
		x_values = [int(row["amount"]) for row in rows]
		y_values = [float(row[f"mean_{metric}"]) for row in rows]
		errors = [float(row[f"std_{metric}"]) for row in rows]
		axis.errorbar(
			x_values,
			y_values,
			yerr=errors,
			marker="o",
			linewidth=1.8,
			capsize=3,
		)
		axis.set_title(title)
		axis.set_ylabel("Error" if metric in ("relative_l2_error", "deformed_coordinate_relative_l2_error") else "Error (model units)")
		axis.grid(True, alpha=0.25)
		if metric in ("relative_l2_error", "deformed_coordinate_relative_l2_error"):
			axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
	for axis in axes[-1]:
		axis.set_xlabel("Number of random landmarks")
	figure.suptitle("Leave-one-out error by random landmark count")
	figure.tight_layout()
	figure.savefig(output_file, dpi=300)
	plt.close(figure)


def run_random_landmark_study(
	config: RandomLandmarkStudyConfig,
	*,
	amounts: tuple[int, ...] | None = None,
	trials_per_amount: int | None = None,
	snapshots: ResampledSnapshotSet | None = None,
) -> dict[str, Any]:
	"""Run seeded placements for each amount and evaluate all held-out cases."""
	selected_amounts = config.amounts if amounts is None else amounts
	if not selected_amounts or any(amount < 1 for amount in selected_amounts):
		raise ValueError("amounts must be a non-empty list of positive integers.")
	if len(set(selected_amounts)) != len(selected_amounts):
		raise ValueError("amounts must not contain duplicates.")
	selected_trials = (
		config.trials_per_amount
		if trials_per_amount is None
		else trials_per_amount
	)
	if selected_trials < 1:
		raise ValueError("trials_per_amount must be positive.")

	if snapshots is None:
		snapshots = load_resampled_snapshots(
			config.results_dir,
			nx=config.nx,
			nz=config.nz,
		)
	sampler = AreaWeightedSurfaceSampler.from_snapshots(snapshots)
	case_parameters = _load_case_parameters(config.results_dir / "parameters.csv")
	scenarios: list[tuple[str, tuple[Landmark, ...]]] = []
	scenario_metadata: dict[str, dict[str, Any]] = {}
	placement_rows = []
	trial_rows = []

	for amount in selected_amounts:
		for trial in range(1, selected_trials + 1):
			seed = derive_trial_seed(config.random_seed, amount, trial)
			landmarks = sampler.sample(amount, random.Random(seed))
			scenario_id = f"amount_{amount:02d}_trial_{trial:03d}"
			scenarios.append((scenario_id, landmarks))
			landmark_records = [
				{"landmark_index": index, "x": point.x, "z": point.z}
				for index, point in enumerate(landmarks, start=1)
			]
			scenario_metadata[scenario_id] = {
				"amount": amount,
				"trial": trial,
				"seed": seed,
				"landmarks": landmark_records,
			}
			placement_rows.extend(
				{
					"amount": amount,
					"trial": trial,
					"seed": seed,
					**landmark,
				}
				for landmark in landmark_records
			)
			trial_rows.append(
				{
					"scenario_id": scenario_id,
					"amount": amount,
					"trial": trial,
					"seed": seed,
					"status": "success",
					"case_count": 0,
				}
			)

	raw_case_rows = evaluate_landmark_scenarios(
		snapshots=snapshots,
		scenarios=tuple(scenarios),
		rank=config.rank,
		energy_fraction=config.energy_fraction,
	)
	scenario_rows: dict[str, list[dict[str, Any]]] = {
		scenario_id: [] for scenario_id, _ in scenarios
	}
	for row in raw_case_rows:
		scenario_rows[row["scenario_id"]].append(row)

	case_rows = []
	for scenario_id, rows in scenario_rows.items():
		metadata = scenario_metadata[scenario_id]
		trial_row = next(
			item for item in trial_rows if item["scenario_id"] == scenario_id
		)
		trial_row["case_count"] = len(rows)
		for metric in METRICS:
			trial_row[f"mean_{metric}"] = _mean_metric(rows, metric)
		case_rows.extend(
			{
				"amount": metadata["amount"],
				"trial": metadata["trial"],
				"seed": metadata["seed"],
				**row,
			}
			for row in rows
		)

	if not case_rows:
		raise ValueError("No held-out case metrics were produced.")
	summaries = aggregate_amount_metrics(trial_rows, case_rows, selected_amounts)

	per_amount_best_worst = []
	capture_requests: dict[str, set[str]] = {}
	for amount in selected_amounts:
		rows = [row for row in case_rows if int(row["amount"]) == amount]
		best_row, worst_row = select_best_worst_cases(
			rows,
			config.best_worst_selection_metric,
		)
		best_record = _selected_case_record(
			best_row,
			config=config,
			scenario_metadata=scenario_metadata,
			case_parameters=case_parameters,
		)
		worst_record = _selected_case_record(
			worst_row,
			config=config,
			scenario_metadata=scenario_metadata,
			case_parameters=case_parameters,
		)
		per_amount_best_worst.append(
			{"amount": amount, "best": best_record, "worst": worst_record}
		)
		for record in (best_record, worst_record):
			capture_requests.setdefault(record["scenario_id"], set()).add(
				record["case_name"]
			)

	global_best_row, global_worst_row = select_best_worst_cases(
		case_rows,
		config.best_worst_selection_metric,
	)
	global_best = _selected_case_record(
		global_best_row,
		config=config,
		scenario_metadata=scenario_metadata,
		case_parameters=case_parameters,
	)
	global_worst = _selected_case_record(
		global_worst_row,
		config=config,
		scenario_metadata=scenario_metadata,
		case_parameters=case_parameters,
	)
	for record in (global_best, global_worst):
		capture_requests.setdefault(record["scenario_id"], set()).add(
			record["case_name"]
		)

	capture_scenarios = tuple(
		scenario for scenario in scenarios if scenario[0] in capture_requests
	)
	captured_surfaces: dict[tuple[str, str], dict[str, np.ndarray]] = {}
	evaluate_landmark_scenarios(
		snapshots=snapshots,
		scenarios=capture_scenarios,
		rank=config.rank,
		energy_fraction=config.energy_fraction,
		case_names_by_scenario=capture_requests,
		capture_surfaces=captured_surfaces,
	)

	config.output_dir.mkdir(parents=True, exist_ok=True)
	_write_csv(config.output_dir / "landmark_placements.csv", placement_rows)
	_write_csv(config.output_dir / "case_metrics.csv", case_rows)
	_write_csv(config.output_dir / "trial_metrics.csv", trial_rows)
	_write_csv(config.output_dir / "summary.csv", summaries)
	_plot_amount_summary(config.output_dir / "error_by_amount.png", summaries)

	best_worst_images = []
	for selected in per_amount_best_worst:
		selected_cases = []
		for label in ("best", "worst"):
			record = selected[label]
			surface_key = (record["scenario_id"], record["case_name"])
			selected_cases.append(
				(label.title(), record, captured_surfaces[surface_key])
			)
		image_name = f"amount_{selected['amount']:02d}_best_worst.png"
		_plot_surface_comparison(
			config.output_dir / image_name,
			selected_cases,
			snapshots.grid.axis_indices,
		)
		best_worst_images.append(image_name)

	global_cases = []
	for label, record in (("Global best", global_best), ("Global worst", global_worst)):
		surface_key = (record["scenario_id"], record["case_name"])
		global_cases.append((label, record, captured_surfaces[surface_key]))
	global_image = "global_best_worst_surfaces.png"
	_plot_surface_comparison(
		config.output_dir / global_image,
		global_cases,
		snapshots.grid.axis_indices,
	)
	global_surface_file = "global_best_worst_surfaces.npz"
	best_surface = global_cases[0][2]
	worst_surface = global_cases[1][2]
	np.savez_compressed(
		config.output_dir / global_surface_file,
		coordinates=best_surface["coordinates"],
		valid_mask=best_surface["valid_mask"],
		best_reference_displacement=best_surface["reference_displacement"],
		best_reconstructed_displacement=best_surface["reconstructed_displacement"],
		worst_reference_displacement=worst_surface["reference_displacement"],
		worst_reconstructed_displacement=worst_surface["reconstructed_displacement"],
	)

	best_worst_metadata = {
		"selection_metric": config.best_worst_selection_metric,
		"results_dir": str(config.results_dir),
		"amounts": list(selected_amounts),
		"trials_per_amount": selected_trials,
		"per_amount": per_amount_best_worst,
		"global": {"best": global_best, "worst": global_worst},
		"images": best_worst_images + [global_image],
		"global_surfaces_file": global_surface_file,
	}
	with (config.output_dir / "best_worst_cases.json").open(
		"w",
		encoding="utf-8",
	) as file:
		json.dump(best_worst_metadata, file, indent=2)
		file.write("\n")

	manifest = {
		"name": config.name,
		"results_dir": str(config.results_dir),
		"output_dir": str(config.output_dir),
		"grid": {"nx": config.nx, "nz": config.nz},
		"energy_fraction": config.energy_fraction,
		"rank": config.rank,
		"amounts": list(selected_amounts),
		"trials_per_amount": selected_trials,
		"random_seed": config.random_seed,
		"sampling": "area_weighted_triangles_within_valid_grid_cells",
		"best_worst_selection_metric": config.best_worst_selection_metric,
		"coordinate_units": "same_as_source_model",
		"case_names": list(snapshots.case_names),
		"successful_trials": sum(row["status"] == "success" for row in trial_rows),
		"failed_trials": sum(row["status"] == "failed" for row in trial_rows),
		"selection_file": "best_worst_cases.json",
		"selection_images": best_worst_images + [global_image],
		"global_surfaces_file": global_surface_file,
		"metric_files": [
			"landmark_placements.csv",
			"case_metrics.csv",
			"trial_metrics.csv",
			"summary.csv",
		],
		"summary_plot": "error_by_amount.png",
	}
	with (config.output_dir / "study_manifest.json").open(
		"w",
		encoding="utf-8",
	) as file:
		json.dump(manifest, file, indent=2)
		file.write("\n")
	return manifest


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
	simulation_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(
		description="Evaluate leave-one-out reconstruction across random landmarks."
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=simulation_dir / "StudyConfigs" / "random_landmarks.json",
	)
	parser.add_argument(
		"--experiment",
		required=True,
		help="Experiment folder name under Simulation/Results.",
	)
	parser.add_argument("--amounts", type=int, nargs="+", default=None)
	parser.add_argument("--trials-per-amount", type=int, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
	args = parse_args(arguments)
	simulation_dir = Path(__file__).resolve().parent
	config = load_random_landmark_study_config(
		args.config,
		results_dir=simulation_dir / "Results" / args.experiment,
		output_dir=args.output_dir,
	)
	manifest = run_random_landmark_study(
		config,
		amounts=None if args.amounts is None else tuple(args.amounts),
		trials_per_amount=args.trials_per_amount,
	)
	print(
		f"Completed {manifest['successful_trials']} trials across "
		f"{len(manifest['amounts'])} landmark amounts."
	)
	print(f"Results written to {config.output_dir}")


if __name__ == "__main__":
	main()
