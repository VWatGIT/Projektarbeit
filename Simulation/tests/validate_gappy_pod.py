"""Run basic validation of the node-based Gappy POD implementation."""

from pathlib import Path

import numpy as np

try:
    from ..utils.gappy_pod import leave_one_out_evaluation, load_snapshots
except ImportError:
    from utils.gappy_pod import leave_one_out_evaluation, load_snapshots


def main() -> None:
    results_dir = Path(__file__).parents[1] / "Versuche" / "Results"
    snapshots = load_snapshots(results_dir)
    observed_node_ids = snapshots.node_ids[::10]
    results = leave_one_out_evaluation(
        snapshots,
        observed_node_ids,
        energy_fraction=0.999,
    )

    print(f"Loaded {len(snapshots.case_names)} cases with {snapshots.node_count} nodes.")
    print(f"Observed nodes: {observed_node_ids.size}")
    print(
        "case                 rank  abs_l2_error  rel_l2_error  "
        "max_abs_error  max_rel_error  obs_rank  condition"
    )
    for result in results:
        print(
            f"{result['case_name']:<20} "
            f"{result['pod_rank']:>4.0f}  "
            f"{result['absolute_l2_error']:.6e}  "
            f"{result['relative_l2_error']:.6e}  "
            f"{result['maximum_absolute_error']:.6e}  "
            f"{result['maximum_relative_error']:.6e}  "
            f"{result['rank']:>8.0f}  "
            f"{result['condition_number']:.6e}"
        )
    mean_error = np.mean([result["relative_l2_error"] for result in results])
    print(f"Mean relative L2 error: {mean_error:.6e}")


if __name__ == "__main__":
    main()