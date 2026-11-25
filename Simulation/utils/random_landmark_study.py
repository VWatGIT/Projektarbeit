"""Configuration and area sampling for random surface landmark studies."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
class RandomLandmarkStudyConfig:
    """Validated settings and resolved paths for a random landmark sweep."""

    name: str
    results_dir: Path
    output_dir: Path
    nx: int
    nz: int
    amounts: tuple[int, ...]
    trials_per_amount: int
    random_seed: int
    best_worst_selection_metric: str
    energy_fraction: float | None = 0.999
    rank: int | None = None
    surface_resampling: str = "linear"


def load_random_landmark_study_config(
    config_file: str | Path,
    *,
    results_dir: str | Path,
    output_dir: str | Path | None = None,
) -> RandomLandmarkStudyConfig:
    """Load settings from JSON and resolve experiment-relative directories."""
    path = Path(config_file).resolve()
    with path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    amounts = tuple(int(amount) for amount in raw["amounts"])
    if not amounts or any(amount < 1 for amount in amounts):
        raise ValueError("amounts must be a non-empty list of positive integers.")
    if len(set(amounts)) != len(amounts):
        raise ValueError("amounts must not contain duplicates.")

    trials_per_amount = int(raw["trials_per_amount"])
    if trials_per_amount < 1:
        raise ValueError("trials_per_amount must be positive.")

    grid = raw.get("interpolation_grid", {})
    nx = int(grid.get("nx", 51))
    nz = int(grid.get("nz", 51))
    if nx < 2 or nz < 2:
        raise ValueError("Interpolation grid dimensions nx and nz must be at least 2.")

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

    selection_metric = raw.get(
        "best_worst_selection_metric",
        DEFAULT_BEST_WORST_SELECTION_METRIC,
    )
    if selection_metric not in STUDY_METRICS:
        choices = ", ".join(STUDY_METRICS)
        raise ValueError(
            "best_worst_selection_metric must be one of "
            f"{choices}; got {selection_metric!r}."
        )

    results_path = Path(results_dir).resolve()
    study_root = results_path / "Study"
    if output_dir is None:
        output_path = study_root / str(raw["name"])
    else:
        output_path = Path(output_dir).resolve()
    if output_path != study_root and study_root not in output_path.parents:
        raise ValueError(
            "output_dir must be the experiment's Study directory or one of its "
            f"descendants: {study_root}"
        )

    return RandomLandmarkStudyConfig(
        name=str(raw["name"]),
        results_dir=results_path,
        output_dir=output_path,
        nx=nx,
        nz=nz,
        amounts=amounts,
        trials_per_amount=trials_per_amount,
        random_seed=int(raw["random_seed"]),
        best_worst_selection_metric=selection_metric,
        energy_fraction=(
            None if energy_fraction is None else float(energy_fraction)
        ),
        rank=None if rank is None else int(rank),
        surface_resampling=validate_surface_resampling(raw.get("surface_resampling", "linear")),
    )


@dataclass(frozen=True)
class AreaWeightedSurfaceSampler:
    """Triangles and cumulative areas for sampling points on valid grid cells."""

    triangles: np.ndarray
    cumulative_areas: np.ndarray
    total_area: float

    @classmethod
    def from_snapshots(
        cls,
        snapshots: ResampledSnapshotSet,
    ) -> AreaWeightedSurfaceSampler:
        nx, nz = snapshots.grid.shape
        points = snapshots.grid.coordinates_2d.reshape(nx, nz, 2)
        valid = snapshots.valid_mask.reshape(nx, nz)
        triangles: list[np.ndarray] = []
        areas: list[float] = []

        def append_triangle(vertices: tuple[tuple[int, int], ...]) -> None:
            triangle = np.asarray([points[index] for index in vertices])
            edge_a = triangle[1] - triangle[0]
            edge_b = triangle[2] - triangle[0]
            area = abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])) / 2.0
            if math.isfinite(area) and area > 0:
                triangles.append(triangle)
                areas.append(area)

        for i in range(nx - 1):
            for j in range(nz - 1):
                corners = (
                    (i, j),
                    (i + 1, j),
                    (i + 1, j + 1),
                    (i, j + 1),
                )
                valid_corners = tuple(index for index in corners if valid[index])
                if len(valid_corners) == 4:
                    append_triangle((corners[0], corners[1], corners[2]))
                    append_triangle((corners[0], corners[2], corners[3]))
                elif len(valid_corners) == 3:
                    append_triangle(valid_corners)

        if not triangles:
            raise ValueError(
                "The valid resampled surface contains no sampleable grid-cell area."
            )
        triangle_array = np.asarray(triangles, dtype=np.float64)
        cumulative_areas = np.cumsum(np.asarray(areas, dtype=np.float64))
        return cls(
            triangles=triangle_array,
            cumulative_areas=cumulative_areas,
            total_area=float(cumulative_areas[-1]),
        )

    def sample(
        self,
        amount: int,
        random_generator: random.Random,
    ) -> tuple[Landmark, ...]:
        """Draw ``amount`` points uniformly by area from the valid triangles."""
        if amount < 1:
            raise ValueError("amount must be positive.")
        landmarks = []
        for _ in range(amount):
            area_position = random_generator.random() * self.total_area
            triangle_index = int(
                np.searchsorted(self.cumulative_areas, area_position, side="right")
            )
            triangle = self.triangles[triangle_index]
            root = math.sqrt(random_generator.random())
            split = random_generator.random()
            weights = (1.0 - root, root * (1.0 - split), root * split)
            point = weights[0] * triangle[0]
            point = point + weights[1] * triangle[1] + weights[2] * triangle[2]
            landmarks.append(Landmark(x=float(point[0]), z=float(point[1])))
        return tuple(landmarks)