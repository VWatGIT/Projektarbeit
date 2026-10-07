"""Resample planar FEM displacement fields onto a common 2D surface grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator


@dataclass(frozen=True)
class SurfaceGrid:
    """Common grid defined in the two varying coordinates of a planar surface."""

    axis_indices: tuple[int, int]
    fixed_axis: int
    fixed_value: float
    coordinates_2d: np.ndarray
    shape: tuple[int, int]

    @property
    def point_count(self) -> int:
        return self.coordinates_2d.shape[0]


@dataclass(frozen=True)
class ResampledSurface:
    """Displacement field sampled on a common surface grid."""

    coordinates_2d: np.ndarray
    displacement: np.ndarray
    valid_mask: np.ndarray
    source_case: str | None = None


@dataclass(frozen=True)
class ResampledSnapshotSet:
    """Multiple displacement fields represented on one common grid."""

    case_names: tuple[str, ...]
    grid: SurfaceGrid
    displacements: np.ndarray
    valid_mask: np.ndarray

    def state_matrix(self) -> np.ndarray:
        """Return valid grid displacements as POD columns."""
        valid_displacements = self.displacements[:, self.valid_mask, :]
        return valid_displacements.reshape(len(self.case_names), -1).T


def create_planar_grid(
    coordinates: np.ndarray,
    nx: int = 50,
    nz: int = 50,
    planarity_tolerance: float = 1e-8,
) -> SurfaceGrid:
    """Create a grid from the two varying axes of a planar point cloud."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape (n, 3), got {points.shape}.")
    if nx < 2 or nz < 2:
        raise ValueError("nx and nz must be at least 2.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Surface coordinates contain non-finite values.")

    ranges = np.ptp(points, axis=0)
    varying_axes = tuple(np.argsort(ranges)[-2:])
    fixed_axis = ({0, 1, 2} - set(varying_axes)).pop()
    scale = max(float(ranges[list(varying_axes)].max()), 1.0)
    if ranges[fixed_axis] > planarity_tolerance * scale:
        raise ValueError(
            "The input surface is not planar within the requested tolerance: "
            f"axis {fixed_axis} range is {ranges[fixed_axis]:.6g}."
        )

    axis_a, axis_b = varying_axes
    values_a = np.linspace(points[:, axis_a].min(), points[:, axis_a].max(), nx)
    values_b = np.linspace(points[:, axis_b].min(), points[:, axis_b].max(), nz)
    grid_a, grid_b = np.meshgrid(values_a, values_b, indexing="ij")
    coordinates_2d = np.column_stack((grid_a.ravel(), grid_b.ravel()))

    return SurfaceGrid(
        axis_indices=(axis_a, axis_b),
        fixed_axis=fixed_axis,
        fixed_value=float(points[:, fixed_axis].mean()),
        coordinates_2d=coordinates_2d,
        shape=(nx, nz),
    )


def resample_displacement(
    coordinates: np.ndarray,
    displacement: np.ndarray,
    grid: SurfaceGrid,
    source_case: str | None = None,
) -> ResampledSurface:
    """Linearly interpolate a planar FEM displacement field onto ``grid``."""
    source_coordinates = np.asarray(coordinates, dtype=np.float64)
    source_displacement = np.asarray(displacement, dtype=np.float64)
    if source_coordinates.shape != source_displacement.shape:
        raise ValueError(
            "Coordinates and displacement must have the same shape, got "
            f"{source_coordinates.shape} and {source_displacement.shape}."
        )
    if source_coordinates.ndim != 2 or source_coordinates.shape[1] != 3:
        raise ValueError("Coordinates and displacement must have shape (n, 3).")

    source_2d = source_coordinates[:, grid.axis_indices]
    interpolated_components = np.column_stack(
        [
            LinearNDInterpolator(source_2d, source_displacement[:, component])(
                grid.coordinates_2d
            )
            for component in range(3)
        ]
    )
    valid_mask = np.all(np.isfinite(interpolated_components), axis=1)

    return ResampledSurface(
        coordinates_2d=grid.coordinates_2d,
        displacement=interpolated_components,
        valid_mask=valid_mask,
        source_case=source_case,
    )


def resample_npz(
    npz_file: str | Path,
    grid: SurfaceGrid,
) -> ResampledSurface:
    """Load one existing result archive and resample its displacement field."""
    npz_path = Path(npz_file)
    with np.load(npz_path) as archive:
        return resample_displacement(
            coordinates=archive["original_coordinates"],
            displacement=archive["displacement"],
            grid=grid,
            source_case=npz_path.parent.name,
        )


def load_resampled_snapshots(
    results_dir: str | Path,
    nx: int = 50,
    nz: int = 50,
) -> ResampledSnapshotSet:
    """Resample every result archive onto a grid derived from the first case."""
    results_path = Path(results_dir)
    files = sorted(results_path.glob("simulation_*/simulation_*.npz"))
    if not files:
        raise FileNotFoundError(f"No simulation NPZ files found in {results_path}.")

    with np.load(files[0]) as archive:
        grid = create_planar_grid(
            archive["original_coordinates"],
            nx=nx,
            nz=nz,
        )

    case_names: list[str] = []
    displacements: list[np.ndarray] = []
    valid_mask: np.ndarray | None = None
    for file_path in files:
        resampled = resample_npz(file_path, grid)
        if valid_mask is None:
            valid_mask = resampled.valid_mask
        elif not np.array_equal(valid_mask, resampled.valid_mask):
            raise ValueError(
                f"{file_path}: valid grid region differs from the first case."
            )
        case_names.append(resampled.source_case or file_path.parent.name)
        displacements.append(resampled.displacement)

    assert valid_mask is not None
    return ResampledSnapshotSet(
        case_names=tuple(case_names),
        grid=grid,
        displacements=np.stack(displacements),
        valid_mask=valid_mask,
    )
