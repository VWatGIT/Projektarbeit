import csv
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Simulation.utils.create_parameters import (
    DeviationVariationConfig,
    ParameterStudyConfig,
    SuctionStudyConfig,
    VariationConfig,
    create_parameters,
)
from Simulation.utils.sauger_landmark_study import (
    GeneratedLandmark,
    SaugerCenter,
    generate_landmarks,
    invalid_landmarks,
    load_sauger_centers,
    load_sauger_landmark_study_config,
    trial_seed,
)
from Simulation.run_sauger_landmark_study import (
    aggregate_trials,
    write_true_surface_fem_metrics,
)
from Simulation.utils.study_pipeline import (
    Landmark,
    StudyConfig,
    evaluate_landmark_scenarios,
    evaluate_study,
)
from Simulation.utils.gappy_pod import PODBasis
from Simulation.utils.surface_resampling import ResampledSnapshotSet, SurfaceGrid


class SaugerLandmarkStudyTests(unittest.TestCase):
    def test_pod_uses_zero_reference_and_un_centered_snapshots(self) -> None:
        states = np.array(
            [[10.0, 12.0, 14.0], [1.0, 3.0, 5.0]],
            dtype=np.float64,
        )

        basis = PODBasis.fit(states, rank=2)

        np.testing.assert_array_equal(basis.mean, np.zeros(2))
        np.testing.assert_allclose(
            basis.modes.T @ basis.modes,
            np.eye(2),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            basis.modes @ (basis.modes.T @ states),
            states,
        )

    def test_initial_positions_written_in_both_coordinate_systems(self) -> None:
        config = ParameterStudyConfig(
            name="test_sweep",
            number_of_simulations=2,
            random_seed=3,
            saugers=(SuctionStudyConfig(200.0, 800.0),),
            variation=VariationConfig(
                position=DeviationVariationConfig(2.0, 1.0, 3.0),
                rotation=DeviationVariationConfig(0.0, 0.0, 0.0, False),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            experiment_name = create_parameters(config, directory)
            experiment_directory = Path(directory) / experiment_name
            json_file = experiment_directory / "initial_sauger_positions.json"
            csv_file = experiment_directory / "initial_sauger_positions.csv"

            data = json.loads(json_file.read_text(encoding="utf-8"))
            with csv_file.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue((experiment_directory / "parameters.csv").is_file())

        expected = {
            "sauger_index": 1,
            "inventor_x": 200.0,
            "inventor_z": 800.0,
            "nastran_x": -200.0,
            "nastran_z": -800.0,
        }
        self.assertEqual(data["saugers"][0], expected)
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["nastran_z"]), -800.0)

    def test_seeded_landmarks_have_fixed_radius_and_reproduce(self) -> None:
        centers = (SaugerCenter(1, -200.0, -200.0),)
        seed = trial_seed(100, radius_index=2, trial_index=3, trials_per_radius=20)

        first = generate_landmarks(centers, 5, 2.5, random.Random(seed))
        repeated = generate_landmarks(centers, 5, 2.5, random.Random(seed))

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 5)
        angles = []
        for generated in first:
            distance = math.hypot(
                generated.point.x - centers[0].x,
                generated.point.z - centers[0].z,
            )
            self.assertAlmostEqual(distance, 2.5)
            angles.append(
                math.atan2(
                    generated.point.z - centers[0].z,
                    generated.point.x - centers[0].x,
                )
            )
        self.assertEqual(len(set(angles)), 5)

    def test_position_loader_uses_nastran_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            position_file = Path(directory) / "positions.json"
            position_file.write_text(
                json.dumps(
                    {
                        "saugers": [
                            {
                                "sauger_index": 1,
                                "inventor_x": 200.0,
                                "inventor_z": 800.0,
                                "nastran_x": -200.0,
                                "nastran_z": -800.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            centers = load_sauger_centers(position_file)

        self.assertEqual(centers, (SaugerCenter(1, -200.0, -800.0),))

    def test_config_rejects_non_positive_radii(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "invalid",
                        "results_dir": "results/FEA",
                        "output_dir": "results/Study",
                        "initial_positions_file": "positions.json",
                        "landmarks_per_sauger": 1,
                        "radii": [0],
                        "trials_per_radius": 1,
                        "random_seed": 10,
                        "energy_fraction": 0.999,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "finite positive"):
                load_sauger_landmark_study_config(path)

    def test_radius_aggregation_uses_successful_trial_means(self) -> None:
        trials = [
            {
                "radius": 1.0,
                "status": "success",
                "mean_relative_l2_error": 0.2,
                "mean_absolute_l2_error": 2.0,
                "mean_maximum_absolute_error": 1.0,
                "mean_deformed_coordinate_relative_l2_error": 0.01,
            },
            {
                "radius": 1.0,
                "status": "success",
                "mean_relative_l2_error": 0.4,
                "mean_absolute_l2_error": 4.0,
                "mean_maximum_absolute_error": 3.0,
                "mean_deformed_coordinate_relative_l2_error": 0.03,
            },
            {"radius": 1.0, "status": "failed"},
        ]

        case_metrics = [
            {"radius": 1.0, "case_name": "case_a", "relative_l2_error": 0.1},
            {"radius": 1.0, "case_name": "case_a", "relative_l2_error": 0.3},
            {"radius": 1.0, "case_name": "case_b", "relative_l2_error": 0.5},
            {"radius": 1.0, "case_name": "case_b", "relative_l2_error": 0.7},
        ]
        summary = aggregate_trials(trials, (1.0,), case_metrics)[0]

        self.assertEqual(summary["successful_trials"], 2)
        self.assertEqual(summary["failed_trials"], 1)
        self.assertAlmostEqual(summary["mean_relative_l2_error"], 0.3)
        self.assertAlmostEqual(summary["std_relative_l2_error"], 0.1)
        self.assertAlmostEqual(summary["min_case_mean_relative_l2_error"], 0.2)
        self.assertAlmostEqual(summary["max_case_mean_relative_l2_error"], 0.6)

    def test_true_surface_vs_fem_metrics_are_averaged_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "true_vs_fem.json"

            report = write_true_surface_fem_metrics(
                output_file,
                self._snapshots(case_count=2),
            )
            written = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(report, written)
        self.assertEqual(report["comparison"], "true_surface_vs_fem_surface")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(set(report["metrics"]), set((
            "relative_l2_error",
            "absolute_l2_error",
            "maximum_absolute_error",
            "deformed_coordinate_relative_l2_error",
        )))
        self.assertAlmostEqual(report["metrics"]["relative_l2_error"], 1.0)

    def test_out_of_domain_landmark_is_reported(self) -> None:
        snapshots = self._snapshots()
        landmarks = (
            GeneratedLandmark(1, 1, Landmark(0.5, 0.5)),
            GeneratedLandmark(1, 2, Landmark(2.0, 0.5)),
        )

        invalid = invalid_landmarks(snapshots, landmarks)

        self.assertEqual(invalid, (landmarks[1],))

    def test_metrics_only_batch_matches_single_study_without_files(self) -> None:
        snapshots = self._snapshots(case_count=4)
        landmark_set = (Landmark(0.35, 0.45), Landmark(0.7, 0.2))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "study_output"
            config = StudyConfig(
                name="test",
                results_dir=Path(directory),
                output_dir=output_dir,
                nx=2,
                nz=2,
                landmarks=landmark_set,
                rank=1,
            )
            single_metrics = evaluate_study(
                config,
                snapshots=snapshots,
                persist=False,
            )
            batch_metrics = evaluate_landmark_scenarios(
                snapshots,
                (("scenario", landmark_set),),
                rank=1,
            )

        self.assertFalse(output_dir.exists())
        self.assertEqual(len(single_metrics), len(batch_metrics))
        for single, batch in zip(single_metrics, batch_metrics):
            self.assertEqual(single["case_name"], batch["case_name"])
            for metric in (
                "relative_l2_error",
                "absolute_l2_error",
                "maximum_absolute_error",
                "deformed_coordinate_relative_l2_error",
            ):
                self.assertAlmostEqual(single[metric], batch[metric])

    @staticmethod
    def _snapshots(case_count: int = 3) -> ResampledSnapshotSet:
        coordinates = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        displacements = np.empty((case_count, 4, 3), dtype=np.float64)
        for case_index in range(case_count):
            displacements[case_index, :, 0] = np.arange(4) * (case_index + 1) * 0.1
            displacements[case_index, :, 1] = case_index * 0.05
            displacements[case_index, :, 2] = np.arange(4) * 0.03
        return ResampledSnapshotSet(
            case_names=tuple(f"case_{index}" for index in range(case_count)),
            grid=SurfaceGrid(
                axis_indices=(0, 2),
                fixed_axis=1,
                fixed_value=0.0,
                coordinates_2d=coordinates,
                shape=(2, 2),
            ),
            displacements=displacements,
            valid_mask=np.ones(4, dtype=bool),
        )


if __name__ == "__main__":
    unittest.main()