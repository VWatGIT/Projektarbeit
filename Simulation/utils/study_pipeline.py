"""Configurable surface-based Gappy POD study evaluation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .gappy_pod import ObservationOperator, PODBasis, reconstruction_errors
    from .surface_resampling import ResampledSnapshotSet, load_resampled_snapshots
except ImportError:
    from utils.gappy_pod import ObservationOperator, PODBasis, reconstruction_errors
    from utils.surface_resampling import ResampledSnapshotSet, load_resampled_snapshots


@dataclass(frozen=True)
class Landmark:
    """A physical x/z landmark used as a displacement observation."""

    x: float
    z: float


@dataclass(frozen=True)
class StudyConfig:
    """Resolved configuration for one surface reconstruction study."""

    name: str
    results_dir: Path
    output_dir: Path
    nx: int
    nz: int
    landmarks: tuple[Landmark, ...]
    energy_fraction: float | None = 0.999
    rank: int | None = None


def load_study_config(
    config_file: str | Path,
    *,
    results_dir: str | Path,
    output_dir: str | Path | None = None,
) -> StudyConfig:
    """Load study settings and resolve I/O directories supplied by the caller."""
    path = Path(config_file).resolve()
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)

    base_dir = path.parent
    landmarks = tuple(
        Landmark(
            x=float(item["x"]),
            z=float(item["z"]),
        )
        for item in raw["landmarks"]
    )
    if not landmarks:
        raise ValueError("At least one landmark is required.")

    rank = raw.get("rank")
    energy_fraction = raw.get("energy_fraction")
    if rank is None and energy_fraction is None:
        energy_fraction = 0.999
    if rank is not None and energy_fraction is not None:
        raise ValueError("Configure either rank or energy_fraction, not both.")

    results_path = Path(results_dir).resolve()
    if output_dir is None:
        output_path = results_path / "Study" / str(raw["name"])
    else:
        output_path = Path(output_dir).resolve()
    study_root = results_path / "Study"
    if output_path != study_root and study_root not in output_path.parents:
        raise ValueError(
            "output_dir must be the experiment's Study directory or one of its "
            f"descendants: {study_root}"
        )

    return StudyConfig(
        name=str(raw["name"]),
        results_dir=results_path,
        output_dir=output_path,
        nx=int(raw.get("interpolation_grid", {}).get("nx", 50)),
        nz=int(raw.get("interpolation_grid", {}).get("nz", 50)),
        landmarks=landmarks,
        energy_fraction=None if energy_fraction is None else float(energy_fraction),
        rank=None if rank is None else int(rank),
    )


def _landmark_operator(
    snapshot_set: ResampledSnapshotSet,
    landmarks: tuple[Landmark, ...],
) -> tuple[ObservationOperator, tuple[dict[str, Any], ...]]:
    """Interpolate each landmark from its four nearest valid grid points."""
    valid_indices = np.flatnonzero(snapshot_set.valid_mask)
    if valid_indices.size < 4:
        raise ValueError("At least four valid interpolation-grid points are required.")

    valid_coordinates = snapshot_set.grid.coordinates_2d[valid_indices]
    neighbor_indices: list[np.ndarray] = []
    neighbor_weights: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for landmark in landmarks:
        distances = np.linalg.norm(
            valid_coordinates - np.array([landmark.x, landmark.z]),
            axis=1,
        )
        nearest = np.argsort(distances, kind="stable")[:4]
        nearest_distances = distances[nearest]
        if nearest_distances[0] == 0:
            weights = np.zeros(4, dtype=np.float64)
            weights[0] = 1.0
        else:
            inverse_distances = 1.0 / nearest_distances**2
            weights = inverse_distances / inverse_distances.sum()

        full_indices = valid_indices[nearest]
        compact_indices = np.searchsorted(valid_indices, full_indices)
        neighbor_indices.append(compact_indices)
        neighbor_weights.append(weights)
        metadata.append(
            {
                "requested_x": landmark.x,
                "requested_z": landmark.z,
                "interpolation": "inverse_distance_weighted_4_nearest",
                "neighbors": [
                    {
                        "grid_index": int(index),
                        "grid_x": float(coordinate[0]),
                        "grid_z": float(coordinate[1]),
                        "distance": float(distance),
                        "weight": float(weight),
                    }
                    for index, coordinate, distance, weight in zip(
                        full_indices,
                        valid_coordinates[nearest],
                        nearest_distances,
                        weights,
                    )
                ],
            }
        )

    operator = ObservationOperator.from_weighted_point_indices(
        point_count=valid_indices.size,
        point_indices=np.asarray(neighbor_indices, dtype=np.int64),
        point_weights=np.asarray(neighbor_weights, dtype=np.float64),
    )
    return operator, tuple(metadata)


def _state_to_field(
    state: np.ndarray,
    valid_mask: np.ndarray,
    point_count: int,
) -> np.ndarray:
    field = np.full((point_count, 3), np.nan, dtype=np.float64)
    field[np.asarray(valid_mask, dtype=bool)] = np.asarray(
        state,
        dtype=np.float64,
    ).reshape(-1, 3)
    return field


def evaluate_study(
    config: StudyConfig,
    snapshots: ResampledSnapshotSet | None = None,
    persist: bool = True,
    case_names: set[str] | None = None,
    capture_surfaces: dict[str, dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate leave-one-out Gappy POD, optionally without writing artifacts."""
    if persist and case_names is not None:
        raise ValueError("A filtered case evaluation cannot persist a partial study.")
    if snapshots is None:
        snapshots = load_resampled_snapshots(
            config.results_dir,
            nx=config.nx,
            nz=config.nz,
        )
    available_case_names = set(snapshots.case_names)
    if case_names is not None:
        missing_case_names = case_names - available_case_names
        if missing_case_names:
            raise ValueError(
                "Requested cases are not present in the snapshot set: "
                f"{sorted(missing_case_names)}"
            )
    observation_operator, landmark_metadata = _landmark_operator(
        snapshots,
        config.landmarks,
    )
    states = snapshots.state_matrix()
    reconstructed_surfaces_dir = config.output_dir / "ReconstructedSurfaces"
    if persist:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        reconstructed_surfaces_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": config.name,
            "results_dir": str(config.results_dir),
            "output_dir": str(config.output_dir),
            "interpolation_grid": {"nx": config.nx, "nz": config.nz},
            "rank": config.rank,
            "energy_fraction": config.energy_fraction,
            "landmark_interpolation": "inverse_distance_weighted_4_nearest_grid_points",
            "landmarks": list(landmark_metadata),
            "case_names": list(snapshots.case_names),
            "valid_grid_points": int(snapshots.valid_mask.sum()),
        }
        with (config.output_dir / "study_manifest.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(manifest, file, indent=2)

    metrics: list[dict[str, Any]] = []
    valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]
    case_indices = np.arange(len(snapshots.case_names))

    for held_out_index, case_name in enumerate(snapshots.case_names):
        if case_names is not None and case_name not in case_names:
            continue
        basis = PODBasis.fit(
            states[:, np.delete(case_indices, held_out_index)],
            rank=config.rank,
            energy_fraction=config.energy_fraction,
        )
        observed = observation_operator.observe(states[:, held_out_index])
        reconstructed_state, diagnostics = basis.reconstruct(
            observed,
            operator=observation_operator,
        )
        reference_state = states[:, held_out_index]
        errors = reconstruction_errors(reference_state, reconstructed_state)
        reference_deformed = valid_coordinates + reference_state.reshape(-1, 3)
        metric_row: dict[str, Any] = {
            "case_name": case_name,
            "pod_rank": int(basis.modes.shape[1]),
            **errors,
            **diagnostics,
            "deformed_coordinate_relative_l2_error": float(
                errors["absolute_l2_error"]
                / max(np.linalg.norm(reference_deformed), 1e-12)
            ),
        }
        if capture_surfaces is not None:
            coordinates = _grid_coordinates_3d(snapshots)
            reference_field = _state_to_field(
                reference_state,
                snapshots.valid_mask,
                snapshots.grid.point_count,
            )
            reconstructed_field = _state_to_field(
                reconstructed_state,
                snapshots.valid_mask,
                snapshots.grid.point_count,
            )
            capture_surfaces[case_name] = {
                "coordinates": coordinates,
                "reference_displacement": reference_field,
                "reconstructed_displacement": reconstructed_field,
                "valid_mask": snapshots.valid_mask.copy(),
            }
        if persist:
            reference_field = _state_to_field(
                reference_state,
                snapshots.valid_mask,
                snapshots.grid.point_count,
            )
            reconstructed_field = _state_to_field(
                reconstructed_state,
                snapshots.valid_mask,
                snapshots.grid.point_count,
            )
            output_file = reconstructed_surfaces_dir / f"{case_name}_reconstruction.npz"
            coordinates = _grid_coordinates_3d(snapshots)
            np.savez_compressed(
                output_file,
                coordinates=coordinates,
                reference_displacement=reference_field,
                reconstructed_displacement=reconstructed_field,
                reference_deformed_coordinates=coordinates + reference_field,
                reconstructed_deformed_coordinates=coordinates + reconstructed_field,
                valid_mask=snapshots.valid_mask,
                landmark_neighbor_indices=observation_operator.point_indices,
                landmark_weights=observation_operator.point_weights,
            )
            metric_row["reconstruction_file"] = str(output_file)
        metrics.append(metric_row)

    if persist:
        serializable_metrics = _json_safe(metrics)
        with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(serializable_metrics, file, indent=2)
        with (config.output_dir / "metrics.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(serializable_metrics)
        with (config.output_dir / "landmarks.json").open("w", encoding="utf-8") as file:
            json.dump(list(landmark_metadata), file, indent=2)
    return metrics


def evaluate_landmark_scenarios(
    snapshots: ResampledSnapshotSet,
    scenarios: tuple[tuple[str, tuple[Landmark, ...]], ...],
    rank: int | None = None,
    energy_fraction: float | None = 0.999,
    case_names_by_scenario: dict[str, set[str]] | None = None,
    capture_surfaces: dict[tuple[str, str], dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate landmark sets, optionally filtering cases and capturing surfaces."""
    if not scenarios:
        raise ValueError("At least one landmark scenario is required.")
    scenario_ids = {scenario_id for scenario_id, _ in scenarios}
    if case_names_by_scenario is not None:
        unknown_scenario_ids = set(case_names_by_scenario) - scenario_ids
        if unknown_scenario_ids:
            raise ValueError(
                f"Unknown scenario IDs requested: {sorted(unknown_scenario_ids)}"
            )

    operators = {
        scenario_id: _landmark_operator(snapshots, landmarks)[0]
        for scenario_id, landmarks in scenarios
    }
    states = snapshots.state_matrix()
    case_indices = np.arange(len(snapshots.case_names))
    valid_coordinates = _grid_coordinates_3d(snapshots)[snapshots.valid_mask]
    full_coordinates = _grid_coordinates_3d(snapshots)
    metric_rows: list[dict[str, Any]] = []

    for held_out_index, case_name in enumerate(snapshots.case_names):
        selected_scenario_ids = [
            scenario_id
            for scenario_id, _ in scenarios
            if case_names_by_scenario is None
            or case_name in case_names_by_scenario.get(scenario_id, set())
        ]
        if not selected_scenario_ids:
            continue
        basis = PODBasis.fit(
            states[:, np.delete(case_indices, held_out_index)],
            rank=rank,
            energy_fraction=energy_fraction,
        )
        reference_state = states[:, held_out_index]
        reference_deformed = valid_coordinates + reference_state.reshape(-1, 3)
        for scenario_id in selected_scenario_ids:
            operator = operators[scenario_id]
            reconstructed_state, diagnostics = basis.reconstruct(
                operator.observe(reference_state),
                operator,
            )
            errors = reconstruction_errors(reference_state, reconstructed_state)
            metric_rows.append(
                {
                    "scenario_id": scenario_id,
                    "case_name": case_name,
                    "pod_rank": int(basis.modes.shape[1]),
                    **errors,
                    **diagnostics,
                    "deformed_coordinate_relative_l2_error": float(
                        errors["absolute_l2_error"]
                        / max(np.linalg.norm(reference_deformed), 1e-12)
                    ),
                }
            )
            if capture_surfaces is not None:
                capture_surfaces[(scenario_id, case_name)] = {
                    "coordinates": full_coordinates,
                    "reference_displacement": _state_to_field(
                        reference_state,
                        snapshots.valid_mask,
                        snapshots.grid.point_count,
                    ),
                    "reconstructed_displacement": _state_to_field(
                        reconstructed_state,
                        snapshots.valid_mask,
                        snapshots.grid.point_count,
                    ),
                    "valid_mask": snapshots.valid_mask.copy(),
                }
    return metric_rows


def _grid_coordinates_3d(snapshot_set: ResampledSnapshotSet) -> np.ndarray:
    coordinates = np.zeros((snapshot_set.grid.point_count, 3), dtype=np.float64)
    axis_a, axis_b = snapshot_set.grid.axis_indices
    coordinates[:, axis_a] = snapshot_set.grid.coordinates_2d[:, 0]
    coordinates[:, axis_b] = snapshot_set.grid.coordinates_2d[:, 1]
    coordinates[:, snapshot_set.grid.fixed_axis] = snapshot_set.grid.fixed_value
    return coordinates


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric values to JSON-compatible nulls."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
