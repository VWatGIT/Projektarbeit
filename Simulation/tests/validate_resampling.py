"""Validate planar surface resampling on one existing FEM result."""

from pathlib import Path

import numpy as np

try:
    from ..utils.gappy_pod import ObservationOperator, PODBasis, reconstruction_errors
    from ..utils.surface_resampling import load_resampled_snapshots
except ImportError:
    from utils.gappy_pod import ObservationOperator, PODBasis, reconstruction_errors
    from Simulation.utils.surface_resampling import load_resampled_snapshots


def main() -> None:
    results_dir = Path(__file__).parents[1] / "Versuche" / "Results"
    snapshots = load_resampled_snapshots(results_dir, nx=50, nz=50)

    axis_a, axis_b = snapshots.grid.axis_indices
    print(f"Loaded cases: {len(snapshots.case_names)}")
    print(f"Source axes: {axis_a}, {axis_b}; fixed axis: {snapshots.grid.fixed_axis}")
    print(f"Grid shape: {snapshots.grid.shape}")
    print(
        f"Valid grid points: {snapshots.valid_mask.sum()} / "
        f"{snapshots.grid.point_count}"
    )
    print(f"POD state matrix shape: {snapshots.state_matrix().shape}")
    states = snapshots.state_matrix()
    valid_point_count = int(snapshots.valid_mask.sum())
    observed_point_indices = np.arange(0, valid_point_count, 100) # This here represents the number of landmarks TODO place landmarks on sauger_positions?
    operator = ObservationOperator.from_point_indices(
        point_count=valid_point_count,
        observed_point_indices=observed_point_indices,
    )

    print(f"Observed grid points per test case: {observed_point_indices.size}")
    print("case                 rank  rel_disp_error  rel_surface_error  condition")
    displacement_errors = []
    surface_errors = []
    reference_surface = np.zeros((snapshots.grid.point_count, 3), dtype=np.float64)
    reference_surface[:, snapshots.grid.axis_indices] = snapshots.grid.coordinates_2d
    reference_surface[:, snapshots.grid.fixed_axis] = snapshots.grid.fixed_value
    reference_surface = reference_surface[snapshots.valid_mask]

    for held_out_index, case_name in enumerate(snapshots.case_names):
        training_indices = np.delete(np.arange(len(snapshots.case_names)), held_out_index)
        basis = PODBasis.fit(states[:, training_indices], energy_fraction=0.999)
        reference_state = states[:, held_out_index]
        reconstructed_state, diagnostics = basis.reconstruct(
            operator.observe(reference_state),
            operator,
        )
        errors = reconstruction_errors(reference_state, reconstructed_state)
        reconstructed_displacement = reconstructed_state.reshape(-1, 3)
        reference_displacement = reference_state.reshape(-1, 3)
        surface_error = reconstruction_errors(
            reference_surface + reference_displacement,
            reference_surface + reconstructed_displacement,
        )
        displacement_errors.append(errors["relative_l2_error"])
        surface_errors.append(surface_error["relative_l2_error"])
        print(
            f"{case_name:<20} {basis.modes.shape[1]:>4}  "
            f"{errors['relative_l2_error']:.6e}  "
            f"{surface_error['relative_l2_error']:.6e}  "
            f"{diagnostics['condition_number']:.6e}"
        )

    print(f"Mean relative displacement error: {np.mean(displacement_errors):.6e}")
    print(f"Mean relative surface error: {np.mean(surface_errors):.6e}")
    print(
        "Maximum sampled displacement magnitude: "
        f"{np.linalg.norm(snapshots.displacements[:, snapshots.valid_mask, :], axis=2).max():.6e}"
    )


if __name__ == "__main__":
    main()