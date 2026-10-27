import unittest
from pathlib import Path

import numpy as np

from Simulation.run_grid_comparison import (
    build_grid_landmarks,
    format_sauger_coordinates,
    select_best_worst_cases,
    summarize_metrics,
)
from Simulation.utils.gappy_pod import ObservationOperator
from Simulation.utils.study_pipeline import (
    Landmark,
    StudyConfig,
    _landmark_operator,
)
from Simulation.utils.surface_resampling import (
    ResampledSnapshotSet,
    SurfaceGrid,
)


class GridComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StudyConfig(
            name="test",
            results_dir=Path("."),
            output_dir=Path("."),
            nx=51,
            nz=51,
            landmarks=(
                Landmark(x=-1000.0, z=-1000.0),
                Landmark(x=0.0, z=0.0),
            ),
        )

    def test_builds_each_square_grid_over_configured_bounds(self) -> None:
        for grid_size in range(2, 12):
            with self.subTest(grid_size=grid_size):
                landmarks = build_grid_landmarks(self.config, grid_size)

                self.assertEqual(len(landmarks), grid_size**2)
                self.assertEqual(
                    len({(point.x, point.z) for point in landmarks}),
                    grid_size**2,
                )
                self.assertEqual(landmarks[0], Landmark(x=-1000.0, z=-1000.0))
                self.assertEqual(landmarks[-1], Landmark(x=0.0, z=0.0))

    def test_grid_landmarks_are_not_snapped_to_interpolation_grid(self) -> None:
        landmarks = build_grid_landmarks(self.config, 5)

        self.assertEqual(landmarks[1].z, -750.0)

    def test_best_and_worst_cases_use_maximum_absolute_error(self) -> None:
        cases = [
            {"case_name": "case_001", "maximum_absolute_error": 0.1},
            {"case_name": "case_002", "maximum_absolute_error": 0.8},
            {"case_name": "case_003", "maximum_absolute_error": 0.4},
        ]

        best, worst = select_best_worst_cases(cases)

        self.assertEqual(best["case_name"], "case_001")
        self.assertEqual(worst["case_name"], "case_002")

    def test_sauger_image_label_uses_nastran_coordinates(self) -> None:
        label = format_sauger_coordinates(
            {
                "s1_x": "200.0",
                "s1_z": "800.0",
                "sauger_1_displacement": "0.75",
                "s2_x": "500.0",
                "s2_z": "600.0",
                "sauger_2_displacement": "-1.25",
            }
        )

        self.assertIn("S1=(-200.0, -800.0)", label)
        self.assertIn("S2=(-500.0, -600.0)", label)
        self.assertIn("s1_y=0.750", label)
        self.assertIn("s2_y=-1.250", label)

    def test_averages_errors_across_held_out_cases(self) -> None:
        summary = summarize_metrics(
            2,
            [
                {
                    "relative_l2_error": 0.2,
                    "absolute_l2_error": 10.0,
                    "maximum_absolute_error": 2.0,
                    "deformed_coordinate_relative_l2_error": 0.1,
                },
                {
                    "relative_l2_error": 0.4,
                    "absolute_l2_error": 30.0,
                    "maximum_absolute_error": 6.0,
                    "deformed_coordinate_relative_l2_error": 0.3,
                },
            ],
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["landmark_count"], 4)
        self.assertAlmostEqual(summary["mean_relative_l2_error"], 0.3)
        self.assertAlmostEqual(summary["min_case_relative_l2_error"], 0.2)
        self.assertAlmostEqual(summary["max_case_relative_l2_error"], 0.4)
        self.assertAlmostEqual(summary["mean_absolute_l2_error"], 20.0)
        self.assertAlmostEqual(summary["mean_maximum_absolute_error"], 4.0)
        self.assertAlmostEqual(
            summary["mean_deformed_coordinate_relative_l2_error"], 0.2
        )

    def test_non_finite_case_errors_are_excluded_from_the_mean(self) -> None:
        summary = summarize_metrics(
            2,
            [
                {"relative_l2_error": 0.25},
                {"relative_l2_error": float("inf")},
            ],
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["mean_relative_l2_error"], 0.25)
        self.assertEqual(summary["finite_case_count_relative_l2_error"], 1)
        self.assertAlmostEqual(summary["min_case_relative_l2_error"], 0.25)
        self.assertAlmostEqual(summary["max_case_relative_l2_error"], 0.25)
        self.assertIsNone(summary["mean_absolute_l2_error"])

    def test_landmark_observation_uses_four_inverse_distance_weights(self) -> None:
        coordinates = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        snapshots = ResampledSnapshotSet(
            case_names=("case",),
            grid=SurfaceGrid(
                axis_indices=(0, 2),
                fixed_axis=1,
                fixed_value=0.0,
                coordinates_2d=coordinates,
                shape=(2, 2),
            ),
            displacements=np.zeros((1, 4, 3), dtype=np.float64),
            valid_mask=np.ones(4, dtype=bool),
        )

        operator, metadata = _landmark_operator(
            snapshots,
            (Landmark(x=0.25, z=0.5),),
        )
        state = np.arange(12, dtype=np.float64)
        observed = operator.observe(state)
        neighbor_ids = operator.point_indices[0]
        neighbor_weights = operator.point_weights[0]
        expected = sum(
            neighbor_weights[index] * state[3 * point_index : 3 * point_index + 3]
            for index, point_index in enumerate(neighbor_ids)
        )

        self.assertIsNotNone(operator.point_indices)
        self.assertEqual(operator.point_indices.shape, (1, 4))
        np.testing.assert_allclose(observed, expected)
        self.assertEqual(len(metadata[0]["neighbors"]), 4)
        self.assertAlmostEqual(sum(neighbor_weights), 1.0)

    def test_weighted_operator_samples_pod_modes_consistently(self) -> None:
        operator = ObservationOperator.from_weighted_point_indices(
            point_count=4,
            point_indices=np.array([[0, 1, 2, 3]]),
            point_weights=np.array([[0.1, 0.2, 0.3, 0.4]]),
        )
        modes = np.arange(24, dtype=np.float64).reshape(12, 2)

        self.assertEqual(operator.observe(modes).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()