"""Configuration and geometry helpers for sauger-centered landmark studies."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

try:
    from .study_pipeline import Landmark
    from .surface_resampling import ResampledSnapshotSet, validate_surface_resampling
except ImportError:
    from utils.study_pipeline import Landmark
    from utils.surface_resampling import ResampledSnapshotSet, validate_surface_resampling


STUDY_METRICS = (
    "relative_l2_error",
    "absolute_l2_error",
    "maximum_absolute_error",
    "deformed_coordinate_relative_l2_error",
)
DEFAULT_BEST_WORST_SELECTION_METRIC = "maximum_absolute_error"


@dataclass(frozen=True)
class SaugerCenter:
    """One fixed, nominal suction center in Nastran x/z coordinates."""

    sauger_index: int
    x: float
    z: float


@dataclass(frozen=True)
class GeneratedLandmark:
    """A trial landmark and the sauger it is associated with."""

    sauger_index: int
    landmark_index: int
    point: Landmark


@dataclass(frozen=True)
class SaugerLandmarkStudyConfig:
    """Settings for reproducible random landmark-offset experiments."""

    name: str
    results_dir: Path
    output_dir: Path
    initial_positions_file: Path
    nx: int
    nz: int
    landmarks_per_sauger: int
    radii: tuple[float, ...]
    trials_per_radius: int
    random_seed: int
    best_worst_selection_metric: str = DEFAULT_BEST_WORST_SELECTION_METRIC
    energy_fraction: float | None = 0.999
    rank: int | None = None
    surface_resampling: str = "linear"


def load_sauger_landmark_study_config(
    config_file: str | Path,
    *,
    results_dir: str | Path,
    output_dir: str | Path | None = None,
) -> SaugerLandmarkStudyConfig:
    path = Path(config_file).resolve()
    with path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    radii = tuple(float(radius) for radius in raw["radii"])
    if not radii or any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("radii must be a non-empty list of finite positive values.")
    if len(set(radii)) != len(radii):
        raise ValueError("radii must not contain duplicates.")

    grid = raw.get("interpolation_grid", {})
    nx = int(grid.get("nx", 51))
    nz = int(grid.get("nz", 51))
    if nx < 2 or nz < 2:
        raise ValueError("Interpolation grid dimensions nx and nz must be at least 2.")

    landmarks_per_sauger = int(raw["landmarks_per_sauger"])
    trials_per_radius = int(raw["trials_per_radius"])
    if landmarks_per_sauger < 1:
        raise ValueError("landmarks_per_sauger must be positive.")
    if trials_per_radius < 1:
        raise ValueError("trials_per_radius must be positive.")

    best_worst_selection_metric = raw.get(
        "best_worst_selection_metric",
        DEFAULT_BEST_WORST_SELECTION_METRIC,
    )
    if best_worst_selection_metric not in STUDY_METRICS:
        choices = ", ".join(STUDY_METRICS)
        raise ValueError(
            "best_worst_selection_metric must be one of "
            f"{choices}; got {best_worst_selection_metric!r}."
        )

    rank = raw.get("rank")
    energy_fraction = raw.get("energy_fraction")
    if rank is not None and energy_fraction is not None:
        raise ValueError("Configure either rank or energy_fraction, not both.")
    if rank is None and energy_fraction is None:
        energy_fraction = 0.999
    if rank is not None and int(rank) < 1:
        raise ValueError("rank must be positive.")
    if energy_fraction is not None and not 0 < float(energy_fraction) <= 1:
        raise ValueError("energy_fraction must be in (0, 1].")

    results_path = Path(results_dir).resolve()
    initial_positions_file = results_path / "initial_sauger_positions.json"
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

    return SaugerLandmarkStudyConfig(
        name=str(raw["name"]),
        results_dir=results_path,
        output_dir=output_path,
        initial_positions_file=initial_positions_file,
        nx=nx,
        nz=nz,
        landmarks_per_sauger=landmarks_per_sauger,
        radii=radii,
        trials_per_radius=trials_per_radius,
        random_seed=int(raw["random_seed"]),
        best_worst_selection_metric=best_worst_selection_metric,
        energy_fraction=(
            None if energy_fraction is None else float(energy_fraction)
        ),
        rank=None if rank is None else int(rank),
        surface_resampling=validate_surface_resampling(raw.get("surface_resampling", "linear")),
    )


def load_sauger_centers(position_file: str | Path) -> tuple[SaugerCenter, ...]:
    path = Path(position_file)
    with path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    positions = raw.get("saugers")
    if not isinstance(positions, list) or not positions:
        raise ValueError("Initial-position JSON must contain a non-empty 'saugers' list.")

    centers = tuple(
        SaugerCenter(
            sauger_index=int(item["sauger_index"]),
            x=float(item["nastran_x"]),
            z=float(item["nastran_z"]),
        )
        for item in positions
    )
    indices = tuple(center.sauger_index for center in centers)
    if indices != tuple(range(1, len(centers) + 1)):
        raise ValueError("Sauger indices must be contiguous and start at 1.")
    if any(not math.isfinite(center.x) or not math.isfinite(center.z) for center in centers):
        raise ValueError("Initial sauger coordinates must be finite.")
    return centers


def trial_seed(
    base_seed: int,
    radius_index: int,
    trial_index: int,
    trials_per_radius: int,
) -> int:
    """Return a stable unique seed for one radius/trial pair."""
    return base_seed + radius_index * trials_per_radius + trial_index


def generate_landmarks(
    sauger_centers: tuple[SaugerCenter, ...],
    landmarks_per_sauger: int,
    radius: float,
    random_generator: random.Random,
) -> tuple[GeneratedLandmark, ...]:
    """Place each landmark at a fixed radius and an independent random angle."""
    if landmarks_per_sauger < 1:
        raise ValueError("landmarks_per_sauger must be positive.")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")

    landmarks = []
    for center in sauger_centers:
        for landmark_index in range(1, landmarks_per_sauger + 1):
            angle = random_generator.uniform(0.0, math.tau)
            landmarks.append(
                GeneratedLandmark(
                    sauger_index=center.sauger_index,
                    landmark_index=landmark_index,
                    point=Landmark(
                        x=center.x + radius * math.cos(angle),
                        z=center.z + radius * math.sin(angle),
                    ),
                )
            )
    return tuple(landmarks)


def invalid_landmarks(
    snapshots: ResampledSnapshotSet,
    landmarks: tuple[GeneratedLandmark, ...],
    hull: Delaunay | None = None,
) -> tuple[GeneratedLandmark, ...]:
    """Return generated points outside the convex hull of valid resampled points."""
    valid_coordinates = snapshots.grid.coordinates_2d[snapshots.valid_mask]
    if valid_coordinates.shape[0] < 3:
        raise ValueError("At least three valid grid points are required for boundary checks.")
    if hull is None:
        try:
            hull = Delaunay(valid_coordinates)
        except QhullError as error:
            raise ValueError("Could not construct the valid surface boundary.") from error

    points = np.asarray(
        [(landmark.point.x, landmark.point.z) for landmark in landmarks],
        dtype=np.float64,
    )
    inside = hull.find_simplex(points) >= 0
    return tuple(
        landmark
        for landmark, is_inside in zip(landmarks, inside)
        if not bool(is_inside)
    )