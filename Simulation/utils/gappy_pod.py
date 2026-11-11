"""Gappy POD tools for the node-based Nastran displacement snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .result_status import successful_npz_files
except ImportError:
    from utils.result_status import successful_npz_files


def _as_state(displacement: np.ndarray) -> np.ndarray:
    """Flatten node-major displacement data to [Ux, Uy, Uz, ...]."""
    array = np.asarray(displacement, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"Expected displacement with shape (n, 3), got {array.shape}."
        )
    return array.reshape(-1)


def _as_field(state: np.ndarray, node_count: int) -> np.ndarray:
    array = np.asarray(state, dtype=np.float64)
    expected_size = node_count * 3
    if array.shape != (expected_size,):
        raise ValueError(
            f"Expected state with shape ({expected_size},), got {array.shape}."
        )
    return array.reshape(node_count, 3)


@dataclass(frozen=True)
class SnapshotSet:
    """Validated FEM snapshots sharing one node space."""

    case_names: tuple[str, ...]
    node_ids: np.ndarray
    original_coordinates: np.ndarray
    displacements: np.ndarray

    @property
    def node_count(self) -> int:
        return self.node_ids.size

    @property
    def state_size(self) -> int:
        return self.node_count * 3

    def state_matrix(self) -> np.ndarray:
        """Return states as columns with shape (3*n_nodes, n_cases)."""
        return self.displacements.reshape(len(self.case_names), -1).T


def load_snapshots(results_dir: str | Path) -> SnapshotSet:
    """Load and validate all simulation_*.npz files in ``results_dir``."""
    results_path = Path(results_dir)
    files = successful_npz_files(results_path)

    reference_ids: np.ndarray | None = None
    reference_coordinates: np.ndarray | None = None
    case_names: list[str] = []
    displacements: list[np.ndarray] = []

    for file_path in files:
        with np.load(file_path) as archive:
            required = {
                "node_ids",
                "original_coordinates",
                "displacement",
                "displacement_magnitude",
                "deformed_coordinates",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{file_path} is missing arrays: {sorted(missing)}")

            node_ids = np.asarray(archive["node_ids"], dtype=np.int64)
            coordinates = np.asarray(archive["original_coordinates"], dtype=np.float64)
            displacement = np.asarray(archive["displacement"], dtype=np.float64)
            magnitude = np.asarray(archive["displacement_magnitude"], dtype=np.float64)
            deformed = np.asarray(archive["deformed_coordinates"], dtype=np.float64)

        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError(f"{file_path}: node_ids must be a non-empty vector.")
        if np.unique(node_ids).size != node_ids.size:
            raise ValueError(f"{file_path}: node_ids are not unique.")
        if not np.all(node_ids[:-1] < node_ids[1:]):
            raise ValueError(f"{file_path}: node_ids must be strictly ascending.")
        if coordinates.shape != (node_ids.size, 3):
            raise ValueError(f"{file_path}: invalid coordinate shape {coordinates.shape}.")
        if displacement.shape != coordinates.shape:
            raise ValueError(f"{file_path}: invalid displacement shape {displacement.shape}.")
        if magnitude.shape != (node_ids.size,):
            raise ValueError(f"{file_path}: invalid magnitude shape {magnitude.shape}.")
        if deformed.shape != coordinates.shape:
            raise ValueError(f"{file_path}: invalid deformed-coordinate shape {deformed.shape}.")
        if not all(np.all(np.isfinite(array)) for array in (coordinates, displacement, magnitude, deformed)):
            raise ValueError(f"{file_path}: arrays contain non-finite values.")
        if not np.allclose(magnitude, np.linalg.norm(displacement, axis=1)):
            raise ValueError(f"{file_path}: displacement magnitudes are inconsistent.")
        if not np.allclose(deformed, coordinates + displacement):
            raise ValueError(f"{file_path}: deformed coordinates are inconsistent.")

        if reference_ids is None:
            reference_ids = node_ids
            reference_coordinates = coordinates
        else:
            if not np.array_equal(node_ids, reference_ids):
                raise ValueError(f"{file_path}: node ordering differs from the first case.")
            if not np.allclose(coordinates, reference_coordinates):
                raise ValueError(f"{file_path}: reference coordinates differ from the first case.")

        case_names.append(file_path.parent.name)
        displacements.append(displacement)

    assert reference_ids is not None
    assert reference_coordinates is not None
    return SnapshotSet(
        case_names=tuple(case_names),
        node_ids=reference_ids,
        original_coordinates=reference_coordinates,
        displacements=np.stack(displacements),
    )


@dataclass(frozen=True)
class ObservationOperator:
    """Select or interpolate displacement components from a state."""

    state_indices: np.ndarray | None = None
    point_indices: np.ndarray | None = None
    point_weights: np.ndarray | None = None

    @property
    def observation_count(self) -> int:
        if self.state_indices is not None:
            return self.state_indices.size
        if self.point_indices is not None:
            return self.point_indices.shape[0] * 3
        raise ValueError("An observation operator requires point indices.")

    @classmethod
    def from_point_indices(
        cls,
        point_count: int,
        observed_point_indices: Iterable[int],
        components: Iterable[int] = (0, 1, 2),
    ) -> "ObservationOperator":
        point_indices = tuple(int(index) for index in observed_point_indices)
        if any(index < 0 or index >= point_count for index in point_indices):
            raise ValueError("Observed point indices are outside the state point range.")
        component_values = tuple(int(component) for component in components)
        if not component_values or any(component not in (0, 1, 2) for component in component_values):
            raise ValueError("components must contain only 0, 1, or 2.")
        if not point_indices:
            raise ValueError("At least one observed point is required.")

        state_indices = [
            3 * point_index + component
            for point_index in point_indices
            for component in component_values
        ]
        return cls(np.asarray(state_indices, dtype=np.int64))

    @classmethod
    def from_node_ids(
        cls,
        node_ids: np.ndarray,
        observed_node_ids: Iterable[int],
        components: Iterable[int] = (0, 1, 2),
    ) -> "ObservationOperator":
        node_ids = np.asarray(node_ids, dtype=np.int64)
        lookup = {int(node_id): index for index, node_id in enumerate(node_ids)}
        component_values = tuple(int(component) for component in components)
        if not component_values or any(component not in (0, 1, 2) for component in component_values):
            raise ValueError("components must contain only 0, 1, or 2.")

        state_indices: list[int] = []
        for node_id in observed_node_ids:
            if int(node_id) not in lookup:
                raise ValueError(f"Observed node ID {node_id} is not in the snapshot mesh.")
            node_index = lookup[int(node_id)]
            state_indices.extend(3 * node_index + component for component in component_values)

        if not state_indices:
            raise ValueError("At least one observed node is required.")
        return cls(np.asarray(state_indices, dtype=np.int64))

    @classmethod
    def from_weighted_point_indices(
        cls,
        point_count: int,
        point_indices: np.ndarray,
        point_weights: np.ndarray,
    ) -> "ObservationOperator":
        indices = np.asarray(point_indices, dtype=np.int64)
        weights = np.asarray(point_weights, dtype=np.float64)
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("point_indices must have shape (landmarks, 4).")
        if weights.shape != indices.shape:
            raise ValueError("point_weights must match point_indices shape.")
        if indices.shape[0] == 0:
            raise ValueError("At least one observed point is required.")
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise ValueError("Observed point indices are outside the state point range.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("point_weights must be finite and non-negative.")
        if not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("Each landmark's point weights must sum to 1.")
        return cls(point_indices=indices, point_weights=weights)

    def observe(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state)
        if self.state_indices is not None:
            return state_array[self.state_indices]
        if self.point_indices is None or self.point_weights is None:
            raise ValueError("An observation operator requires point indices.")
        if state_array.ndim not in (1, 2) or state_array.shape[0] % 3:
            raise ValueError("State must have shape (point_count * 3,) or (point_count * 3, n).")

        point_count = state_array.shape[0] // 3
        point_state = state_array.reshape((point_count, 3) + state_array.shape[1:])
        neighbor_values = point_state[self.point_indices]
        observations = np.einsum(
            "lk,lkc...->lc...",
            self.point_weights,
            neighbor_values,
        )
        return observations.reshape((self.observation_count,) + state_array.shape[1:])


NodeObservationOperator = ObservationOperator


@dataclass(frozen=True)
class PODBasis:
    """Zero-reference, orthonormal POD basis for displacement snapshots."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    energy: np.ndarray

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        rank: int | None = None,
        energy_fraction: float | None = 0.999,
    ) -> "PODBasis":
        snapshot_matrix = np.asarray(states, dtype=np.float64)
        if snapshot_matrix.ndim != 2:
            raise ValueError("states must have shape (state_size, snapshot_count).")
        modes, singular_values, _ = np.linalg.svd(
            snapshot_matrix,
            full_matrices=False,
        )
        return cls.from_svd(
            modes,
            singular_values,
            rank=rank,
            energy_fraction=energy_fraction,
        )

    @classmethod
    def from_svd(
        cls,
        modes: np.ndarray,
        singular_values: np.ndarray,
        rank: int | None = None,
        energy_fraction: float | None = 0.999,
    ) -> "PODBasis":
        """Build a basis from a precomputed SVD of the snapshot matrix."""
        if singular_values.size == 0:
            raise ValueError("At least two non-identical snapshots are required.")

        total_energy = float(np.sum(singular_values**2))
        if total_energy == 0:
            raise ValueError("Cannot fit a POD basis with zero displacement energy.")
        energy = np.cumsum(singular_values**2) / total_energy
        if rank is None:
            if energy_fraction is None or not 0 < energy_fraction <= 1:
                raise ValueError("energy_fraction must be in (0, 1].")
            rank = min(int(np.searchsorted(energy, energy_fraction) + 1), modes.shape[1])
        if not 1 <= rank <= modes.shape[1]:
            raise ValueError(f"rank must be between 1 and {modes.shape[1]}.")

        return cls(
            mean=np.zeros(modes.shape[0], dtype=np.float64),
            modes=modes[:, :rank],
            singular_values=singular_values,
            energy=energy,
        )

    def reconstruct(
        self,
        observations: np.ndarray,
        operator: ObservationOperator,
    ) -> tuple[np.ndarray, dict[str, float]]:
        observed = np.asarray(observations, dtype=np.float64)
        expected = operator.observation_count
        if observed.shape != (expected,):
            raise ValueError(f"Expected {expected} observations, got {observed.shape}.")

        sampled_modes = operator.observe(self.modes)
        right_hand_side = observed - operator.observe(self.mean)
        coefficients, residuals, matrix_rank, singular_values = np.linalg.lstsq(
            sampled_modes,
            right_hand_side,
            rcond=None,
        )
        state = self.mean + self.modes @ coefficients
        condition_number = float("inf")
        if singular_values.size and singular_values[-1] > 0:
            condition_number = float(singular_values[0] / singular_values[-1])
        diagnostics = {
            "rank": float(matrix_rank),
            "condition_number": condition_number,
            "observation_residual": float(np.linalg.norm(sampled_modes @ coefficients - right_hand_side)),
            "least_squares_residual": float(residuals.sum()) if residuals.size else 0.0,
        }
        return state, diagnostics


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_state = np.asarray(reference, dtype=np.float64)
    estimate_state = np.asarray(estimate, dtype=np.float64)
    denominator = np.linalg.norm(reference_state)
    if denominator == 0:
        return float(np.linalg.norm(estimate_state))
    return float(np.linalg.norm(reference_state - estimate_state) / denominator)


def reconstruction_errors(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    """Return absolute and relative errors over all state entries."""
    reference_state = np.asarray(reference, dtype=np.float64)
    estimate_state = np.asarray(estimate, dtype=np.float64)
    if reference_state.shape != estimate_state.shape:
        raise ValueError(
            f"Reference and estimate shapes differ: "
            f"{reference_state.shape} != {estimate_state.shape}."
        )

    absolute_errors = np.abs(reference_state - estimate_state)
    reference_absolute = np.abs(reference_state)
    relative_errors = np.zeros_like(absolute_errors)
    nonzero_reference = reference_absolute > 0
    relative_errors[nonzero_reference] = (
        absolute_errors[nonzero_reference] / reference_absolute[nonzero_reference]
    )
    relative_errors[~nonzero_reference & (absolute_errors > 0)] = np.inf

    return {
        "absolute_l2_error": float(np.linalg.norm(reference_state - estimate_state)),
        "relative_l2_error": relative_l2_error(reference_state, estimate_state),
        "maximum_absolute_error": float(np.max(absolute_errors)),
        "maximum_relative_error": float(np.max(relative_errors)),
    }


def leave_one_out_evaluation(
    snapshots: SnapshotSet,
    observed_node_ids: Iterable[int],
    components: Iterable[int] = (0, 1, 2),
    rank: int | None = None,
    energy_fraction: float | None = 0.999,
) -> list[dict[str, float | str]]:
    """Evaluate sparse reconstruction while keeping each case out of training."""
    states = snapshots.state_matrix()
    operator = NodeObservationOperator.from_node_ids(
        snapshots.node_ids,
        observed_node_ids,
        components,
    )
    results: list[dict[str, float | str]] = []

    for held_out_index, case_name in enumerate(snapshots.case_names):
        training_indices = np.delete(np.arange(len(snapshots.case_names)), held_out_index)
        basis = PODBasis.fit(
            states[:, training_indices],
            rank=rank,
            energy_fraction=energy_fraction,
        )
        reference_state = states[:, held_out_index]
        reconstructed_state, diagnostics = basis.reconstruct(
            operator.observe(reference_state),
            operator,
        )
        errors = reconstruction_errors(reference_state, reconstructed_state)
        results.append(
            {
                "case_name": case_name,
                "pod_rank": float(basis.modes.shape[1]),
                "relative_displacement_error": errors["relative_l2_error"],
                **errors,
                **diagnostics,
            }
        )
    return results


def deformed_coordinates(snapshot_set: SnapshotSet, state: np.ndarray) -> np.ndarray:
    """Return deformed coordinates from a reconstructed displacement state."""
    return snapshot_set.original_coordinates + _as_field(state, snapshot_set.node_count)