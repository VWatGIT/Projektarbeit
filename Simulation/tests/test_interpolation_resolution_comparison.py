import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyNastran.bdf.bdf import BDF

from Simulation import study_interpolation_resolution as study


class InterpolationResolutionComparisonTests(unittest.TestCase):
    def test_comparison_writes_both_graphs_and_case_means(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            simulation = Path(directory) / "Simulation"
            coordinates = np.array([
                [0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2],
                [1, 0, 0], [2, 0, 1], [1, 0, 2], [0, 0, 1],
            ], dtype=float)
            model = BDF(debug=False)
            for node_id, position in enumerate(coordinates, 1):
                model.add_grid(node_id, position)
            model.add_pshell(1, mid1=1, t=1.)
            model.add_mat1(1, E=1., G=None, nu=0.3)
            model.add_cquad8(1, 1, list(range(1, 9)))

            for displacement, experiment in study.DISPLACEMENT_EXPERIMENTS.items():
                root = simulation / "Results" / experiment
                root.mkdir(parents=True)
                (root / "initial_sauger_positions.json").write_text(json.dumps({
                    "saugers": [
                        {"sauger_index": i, "nastran_x": float(i), "nastran_z": 0.}
                        for i in range(1, 4)
                    ],
                }), encoding="utf-8")
                cases = []
                for case in range(1, 4):
                    folder = root / f"case_{case}"
                    folder.mkdir()
                    model.write_bdf(str(folder / "case.nas"))
                    x, z = coordinates[:, 0], coordinates[:, 2]
                    field = displacement * np.column_stack((
                        (x + 1) * (case + 1),
                        (z + 2) * case**2,
                        (x + 1) * (z + 1) * (case + 2),
                    ))
                    np.savez(
                        folder / "case.npz",
                        node_ids=np.arange(1, 9),
                        original_coordinates=coordinates,
                        displacement=field,
                    )
                    cases.append({"status": "success", "npz_file": f"case_{case}/case.npz"})
                (root / "simulation_status.json").write_text(
                    json.dumps({"cases": cases}), encoding="utf-8",
                )

            output = simulation / "Comparison" / "test"
            arguments = [
                "--min-resolution", "2", "--max-resolution", "3",
                "--resolution-step", "1", "--output-dir", str(output),
            ]
            with patch.object(study, "__file__", str(simulation / "study_interpolation_resolution.py")):
                study.main(arguments)

            self.assertEqual(
                {path.name for path in output.glob("*.png")},
                {"three_sauger_interpolation_linear.png", "three_sauger_interpolation_elements.png"},
            )
            with (output / "three_sauger_interpolation_cases.csv").open(newline="", encoding="utf-8") as file:
                cases = list(csv.DictReader(file))
            with (output / "three_sauger_interpolation_summary.csv").open(newline="", encoding="utf-8") as file:
                summaries = list(csv.DictReader(file))
            self.assertEqual(len(cases), 3 * 2 * 2 * 3)
            self.assertEqual(len(summaries), 3 * 2 * 2)
            for summary in summaries:
                matching = [
                    row for row in cases
                    if all(row[key] == summary[key] for key in (
                        "experiment", "surface_resampling", "interpolation_resolution",
                    ))
                ]
                self.assertEqual(len(matching), 3)
                self.assertEqual(int(summary["valid_grid_points"]), int(summary["interpolation_resolution"]) ** 2)
                self.assertAlmostEqual(
                    float(summary["mean_maximum_absolute_error"]),
                    sum(float(row["maximum_absolute_error"]) for row in matching) / len(matching),
                )

            experiment = study.DISPLACEMENT_EXPERIMENTS[1]
            output = simulation / "Results" / experiment / "Study" / "test"
            arguments = [
                "--experiment", experiment, "--compare-resampling",
                "--min-resolution", "2", "--max-resolution", "3",
                "--resolution-step", "1", "--output-dir", str(output),
            ]
            with (
                patch.object(study, "__file__", str(simulation / "study_interpolation_resolution.py")),
                patch("matplotlib.pyplot.close"),
            ):
                study.main(arguments)
            import matplotlib.pyplot as plt

            figure = plt.gcf()
            self.addCleanup(plt.close, figure)
            lines = {line.get_label(): line for line in figure.axes[0].lines}
            self.assertEqual(set(lines), {"linear", "elements"})
            self.assertTrue((output / "interpolation_resampling_comparison.png").is_file())
            with (output / "interpolation_resampling_summary.csv").open(newline="", encoding="utf-8") as file:
                summaries = list(csv.DictReader(file))
            with (output / "interpolation_resampling_cases.csv").open(newline="", encoding="utf-8") as file:
                cases = list(csv.DictReader(file))
            self.assertEqual(len(summaries), 4)
            self.assertEqual(len(cases), 12)
            for method, line in lines.items():
                records = sorted(
                    (row for row in summaries if row["surface_resampling"] == method),
                    key=lambda row: int(row["interpolation_resolution"]),
                )
                np.testing.assert_array_equal(line.get_xdata(), [2, 3])
                np.testing.assert_allclose(
                    line.get_ydata(),
                    [float(row["mean_maximum_absolute_error"]) for row in records],
                )
                for summary in records:
                    matching = [
                        row for row in cases
                        if row["surface_resampling"] == method
                        and row["interpolation_resolution"] == summary["interpolation_resolution"]
                    ]
                    self.assertEqual(len(matching), 3)
                    self.assertAlmostEqual(
                        float(summary["mean_maximum_absolute_error"]),
                        sum(float(row["maximum_absolute_error"]) for row in matching) / 3,
                    )

    def test_comparison_rejects_an_explicit_single_method(self) -> None:
        with self.assertRaises(SystemExit):
            study.parse_args(["--surface-resampling", "elements"])

    def test_single_experiment_mode_remains_available(self) -> None:
        args = study.parse_args(["--experiment", "example", "--surface-resampling", "elements"])
        self.assertEqual(args.experiment, "example")
        self.assertEqual(args.surface_resampling, "elements")

    def test_single_experiment_comparison_rejects_conflicting_options(self) -> None:
        with self.assertRaises(SystemExit):
            study.parse_args(["--compare-resampling"])
        with self.assertRaises(SystemExit):
            study.parse_args([
                "--experiment", "example", "--compare-resampling",
                "--surface-resampling", "linear",
            ])

    def test_comparison_rejects_different_case_sets(self) -> None:
        config = study.StudyConfig(
            name="test", results_dir=Path("experiment"), output_dir=Path("output"),
            nx=2, nz=2, landmarks=(),
        )
        first = (
            [{"case_name": "a", "maximum_absolute_error": 1.},
             {"case_name": "b", "maximum_absolute_error": 2.}],
            {}, ("a", "b"), 4,
        )
        second = ([{"case_name": "a"}, {"case_name": "c"}], {}, ("a", "c"), 4)
        with patch.object(study, "_evaluate_resolution", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "case names differ"):
                study._compare_resampling(config, range(2, 3), 3)


if __name__ == "__main__":
    unittest.main()
