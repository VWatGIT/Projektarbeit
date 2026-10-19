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
    landmark_tolerance: float = 1e-6


def load_study_config(config_file: str | Path) -> StudyConfig:
    """Load and validate a JSON study configuration.

    ``results_dir`` is the FEA result directory. The study output directory
    must be its sibling, for example ``Results/Experiment/FEA`` and
    ``Results/Experiment/MyStudy``.
    """
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

    output_dir = (base_dir / raw["output_dir"]).resolve()
    results_dir = (base_dir / raw["results_dir"]).resolve()
    if output_dir.parent != results_dir.parent:
        raise ValueError(
            "output_dir must be next to results_dir under the same experiment "
            f"directory: {results_dir.parent}"
        )

    return StudyConfig(
        name=str(raw["name"]),
        results_dir=results_dir,
        output_dir=output_dir,
        nx=int(raw.get("grid", {}).get("nx", 50)),
        nz=int(raw.get("grid", {}).get("nz", 50)),
        landmarks=landmarks,
        energy_fraction=None if energy_fraction is None else float(energy_fraction),
        rank=None if rank is None else int(rank),
        landmark_tolerance=float(raw.get("landmark_tolerance", 1e-6)),
    )


def _landmark_indices(
    snapshot_set: ResampledSnapshotSet,
    landmarks: tuple[Landmark, ...],
    tolerance: float,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Resolve physical x/z landmarks to common-grid point indices."""
    axis_a, axis_b = snapshot_set.grid.axis_indices
    grid_3d = np.zeros((snapshot_set.grid.point_count, 3), dtype=np.float64)
    grid_3d[:, axis_a] = snapshot_set.grid.coordinates_2d[:, 0]
    grid_3d[:, axis_b] = snapshot_set.grid.coordinates_2d[:, 1]
    grid_3d[:, snapshot_set.grid.fixed_axis] = snapshot_set.grid.fixed_value

    indices: list[int] = []
    metadata: list[dict[str, Any]] = []
    for landmark in landmarks:
        distances = np.linalg.norm(
            grid_3d[:, (axis_a, axis_b)] - np.array([landmark.x, landmark.z]),
            axis=1,
        )
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance > tolerance:
            raise ValueError(
                f"Landmark ({landmark.x:g}, {landmark.z:g}) is {distance:g} away from the "
                f"interpolation grid; tolerance is {tolerance:g}."
            )
        if not snapshot_set.valid_mask[index]:
            raise ValueError(
                f"Landmark ({landmark.x:g}, {landmark.z:g}) is outside the valid grid."
            )
        indices.append(index)
        metadata.append(
            {
                "requested_x": landmark.x,
                "requested_z": landmark.z,
                "grid_index": index,
                "grid_x": float(grid_3d[index, axis_a]),
                "grid_z": float(grid_3d[index, axis_b]),
                "distance": distance,
            }
        )

    if len(set(indices)) != len(indices):
        raise ValueError("Multiple landmarks resolve to the same grid point.")
    valid_indices = np.flatnonzero(snapshot_set.valid_mask)
    compact_indices = [
        int(np.searchsorted(valid_indices, index))
        for index in indices
    ]
    return np.asarray(compact_indices, dtype=np.int64), tuple(metadata)


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


def evaluate_study(config: StudyConfig) -> list[dict[str, Any]]:
    """Evaluate leave-one-out Gappy POD and persist all reconstructions."""
    snapshots = load_resampled_snapshots(
        config.results_dir,
        nx=config.nx,
        nz=config.nz,
    )
    landmark_indices, landmark_metadata = _landmark_indices(
        snapshots,
        config.landmarks,
        config.landmark_tolerance,
    )
    observed_state_indices = np.concatenate(
        [3 * landmark_indices[:, None] + component for component in range(3)],
        axis=1,
    ).reshape(-1)
    states = snapshots.state_matrix()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_surfaces_dir = config.output_dir / "ReconstructedSurfaces"
    reconstructed_surfaces_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": config.name,
        "results_dir": str(config.results_dir),
        "output_dir": str(config.output_dir),
        "grid": {"nx": config.nx, "nz": config.nz},
        "rank": config.rank,
        "energy_fraction": config.energy_fraction,
        "landmark_tolerance": config.landmark_tolerance,
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

    for held_out_index, case_name in enumerate(snapshots.case_names):
        training_indices = np.delete(np.arange(len(snapshots.case_names)), held_out_index)
        basis = PODBasis.fit(
            states[:, training_indices],
            rank=config.rank,
            energy_fraction=config.energy_fraction,
        )
        observed = states[observed_state_indices, held_out_index]
        reconstructed_state, diagnostics = basis.reconstruct(
            observed,
            operator=ObservationOperator(observed_state_indices),
        )
        reference_state = states[:, held_out_index]
        errors = reconstruction_errors(reference_state, reconstructed_state)
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
        reference_deformed = (
            _grid_coordinates_3d(snapshots) + reference_field
        )
        reconstructed_deformed = (
            _grid_coordinates_3d(snapshots) + reconstructed_field
        )
        output_file = reconstructed_surfaces_dir / f"{case_name}_reconstruction.npz"
        np.savez_compressed(
            output_file,
            coordinates=_grid_coordinates_3d(snapshots),
            reference_displacement=reference_field,
            reconstructed_displacement=reconstructed_field,
            reference_deformed_coordinates=reference_deformed,
            reconstructed_deformed_coordinates=reconstructed_deformed,
            valid_mask=snapshots.valid_mask,
            landmark_indices=landmark_indices,
        )
        metrics.append(
            {
                "case_name": case_name,
                "pod_rank": int(basis.modes.shape[1]),
                **errors,
                **diagnostics,
                "deformed_coordinate_relative_l2_error": float(
                    np.linalg.norm(
                        (reference_deformed - reconstructed_deformed)[
                            snapshots.valid_mask
                        ]
                    )
                    / max(
                        np.linalg.norm(reference_deformed[snapshots.valid_mask]),
                        1e-12,
                    )
                ),
                "reconstruction_file": str(output_file),
            }
        )

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
