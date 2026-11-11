import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Simulation.study_landmarks_random import (
    _sauger_positions_nastran,
    aggregate_amount_metrics,
    derive_trial_seed,
    select_best_worst_cases,
)
from Simulation.utils.random_landmark_study import (
    AreaWeightedSurfaceSampler,
    load_random_landmark_study_config,
)
from Simulation.utils.study_pipeline import Landmark, evaluate_landmark_scenarios
from Simulation.utils.surface_resampling import ResampledSnapshotSet, SurfaceGrid


def make_snapshots(case_count: int = 4) -> ResampledSnapshotSet:
    axis_values = np.linspace(0.0, 1.0, 2)
    axis_a, axis_b = np.meshgrid(axis_values, axis_values, indexing="ij")
    coordinates = np.column_stack((axis_a.ravel(), axis_b.ravel()))
    displacement_cases = []
    for case_index in range(case_count):
        factor = float(case_index + 1)
        displacement_cases.append(
            np.column_stack(
                (
                    coordinates[:, 0] * factor,
                    coordinates[:, 1] * (factor + 0.5),
                    (coordinates[:, 0] + coordinates[:, 1]) * factor,
                )
            )
        )
    return ResampledSnapshotSet(
        case_names=tuple(f"case_{index:03d}" for index in range(case_count)),
        grid=SurfaceGrid(
            axis_indices=(0, 2),
            fixed_axis=1,
            fixed_value=0.0,
            coordinates_2d=coordinates,
            shape=(2, 2),
        ),
        displacements=np.asarray(displacement_cases, dtype=np.float64),
        valid_mask=np.ones(4, dtype=bool),
    )


class RandomLandmarkStudyTests(unittest.TestCase):
    def test_loads_attached_config_and_defaults_output_path(self) -> None:
        config_file = (
            Path(__file__).resolve().parents[1]
            / "StudyConfigs"
            / "random_landmarks.json"
        )
        raw_config = json.loads(config_file.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "Results" / "experiment"
            config = load_random_landmark_study_config(
                config_file,
                results_dir=results_dir,
            )

        self.assertEqual(config.amounts, tuple(raw_config["amounts"]))
        self.assertEqual(config.trials_per_amount, raw_config["trials_per_amount"])
        self.assertEqual(
            config.best_worst_selection_metric,
            raw_config["best_worst_selection_metric"],
        )
        self.assertEqual(
            config.output_dir,
            (results_dir / "Study" / "random_landmarks").resolve(),
        )

    def test_config_defaults_to_maximum_absolute_error_and_validates_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "random.json"
            base_config = {
                "name": "random",
                "amounts": [1, 2],
                "trials_per_amount": 2,
                "random_seed": 5,
                "landmarks": [],
            }
            config_file.write_text(json.dumps(base_config), encoding="utf-8")
            config = load_random_landmark_study_config(
                config_file,
                results_dir=Path(directory) / "experiment",
            )
            self.assertEqual(
                config.best_worst_selection_metric,
                "maximum_absolute_error",
            )

            base_config["best_worst_selection_metric"] = "invalid"
            config_file.write_text(json.dumps(base_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "best_worst_selection_metric"):
                load_random_landmark_study_config(
                    config_file,
                    results_dir=Path(directory) / "experiment",
                )

    def test_sampler_is_reproducible_and_stays_on_surface(self) -> None:
        sampler = AreaWeightedSurfaceSampler.from_snapshots(make_snapshots())
        first = sampler.sample(30, random.Random(901))
        repeated = sampler.sample(30, random.Random(901))

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 30)
        for point in first:
            self.assertGreaterEqual(point.x, 0.0)
            self.assertLessEqual(point.x, 1.0)
            self.assertGreaterEqual(point.z, 0.0)
            self.assertLessEqual(point.z, 1.0)

    def test_sampler_rejects_domain_without_sampleable_area(self) -> None:
        snapshots = make_snapshots()
        snapshots = ResampledSnapshotSet(
            case_names=snapshots.case_names,
            grid=snapshots.grid,
            displacements=snapshots.displacements,
            valid_mask=np.array([True, False, False, True]),
        )

        with self.assertRaisesRegex(ValueError, "no sampleable grid-cell area"):
            AreaWeightedSurfaceSampler.from_snapshots(snapshots)

    def test_trial_seeds_are_repeatable_and_distinct(self) -> None:
        seeds = [derive_trial_seed(10, amount, trial) for amount in (1, 2) for trial in (1, 2)]

        self.assertEqual(seeds, [derive_trial_seed(10, amount, trial) for amount in (1, 2) for trial in (1, 2)])
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_sauger_positions_are_converted_to_nastran_coordinates(self) -> None:
        positions = _sauger_positions_nastran(
            {
                "s1_x": "200.0",
                "s1_z": "800.0",
                "sauger_1_displacement": "2.5",
                "s2_x": "500.0",
                "s2_z": "600.0",
            }
        )

        self.assertEqual(
            positions,
            [
                {"sauger_index": 1, "x": -200.0, "y": 2.5, "z": -800.0},
                {"sauger_index": 2, "x": -500.0, "y": None, "z": -600.0},
            ],
        )

    def test_batched_scenarios_evaluate_each_case_for_each_placement(self) -> None:
        snapshots = make_snapshots()
        scenarios = (
            ("amount_01_trial_001", (Landmark(0.25, 0.25),)),
            ("amount_02_trial_001", (Landmark(0.75, 0.75), Landmark(0.25, 0.5))),
        )

        rows = evaluate_landmark_scenarios(
            snapshots=snapshots,
            scenarios=scenarios,
            energy_fraction=0.999,
        )

        self.assertEqual(len(rows), len(snapshots.case_names) * len(scenarios))
        for scenario_id, _ in scenarios:
            self.assertEqual(
                {row["case_name"] for row in rows if row["scenario_id"] == scenario_id},
                set(snapshots.case_names),
            )

    def test_aggregation_reports_trial_spread_and_case_spread(self) -> None:
        trial_rows = [
            {"amount": 1, "status": "success", "mean_relative_l2_error": 0.2},
            {"amount": 1, "status": "success", "mean_relative_l2_error": 0.4},
        ]
        case_rows = [
            {"amount": 1, "case_name": "case_a", "relative_l2_error": 0.1},
            {"amount": 1, "case_name": "case_a", "relative_l2_error": 0.3},
            {"amount": 1, "case_name": "case_b", "relative_l2_error": 0.5},
            {"amount": 1, "case_name": "case_b", "relative_l2_error": 0.7},
        ]

        summary = aggregate_amount_metrics(trial_rows, case_rows, (1,))[0]

        self.assertAlmostEqual(summary["mean_relative_l2_error"], 0.3)
        self.assertAlmostEqual(summary["std_relative_l2_error"], 0.1)
        self.assertAlmostEqual(summary["min_case_mean_relative_l2_error"], 0.2)
        self.assertAlmostEqual(summary["max_case_mean_relative_l2_error"], 0.6)

    def test_best_worst_selection_uses_configured_metric(self) -> None:
        cases = [
            {
                "case_name": "case_a",
                "scenario_id": "scenario_1",
                "maximum_absolute_error": 0.9,
                "deformed_coordinate_relative_l2_error": 0.1,
            },
            {
                "case_name": "case_b",
                "scenario_id": "scenario_2",
                "maximum_absolute_error": 0.2,
                "deformed_coordinate_relative_l2_error": 0.8,
            },
        ]

        best, worst = select_best_worst_cases(
            cases,
            "deformed_coordinate_relative_l2_error",
        )

        self.assertEqual(best["case_name"], "case_a")
        self.assertEqual(worst["case_name"], "case_b")


if __name__ == "__main__":
    unittest.main()