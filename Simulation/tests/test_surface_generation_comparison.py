import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from Simulation.compare_surface_generation import (
    CL9_DISPLACEMENT_EXPERIMENTS,
    CLAMP_EXPERIMENTS,
    DISPLACEMENT_EXPERIMENTS,
    load_baseline,
    load_grid_errors,
    main,
    plot_clamps,
    plot_grid_sizes,
)


class SurfaceGenerationComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("Simulation.compare_surface_generation.load_resampled_snapshots")
        loader = patcher.start()
        self.addCleanup(patcher.stop)
        loader.return_value.state_matrix.return_value = np.array([[1.0, -3.0], [4.0, 0.0]])
        self.loader = loader

    def write_study(self, root: Path, name: str, rows: list[dict]) -> Path:
        path = root / name / "Study" / "landmarks_grid" / "grid_comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        metadata = {"per_grid": [{
            "best": {"interpolation_grid": {"nx": 51, "nz": 51}},
            "worst": {"interpolation_grid": {"nx": 51, "nz": 51}},
        }]}
        (path.parent / "global_best_worst_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return path

    def row(self, size: int, maximum: float, finite_count: int = 2) -> dict:
        return {
            "grid_size": size,
            "landmark_count": size**2,
            "case_count": 2,
            "finite_case_count_relative_l2_error": finite_count,
            "max_case_relative_l2_error": maximum,
            "finite_case_count_absolute_l2_error": finite_count,
            "max_case_absolute_l2_error": maximum * 10,
            "finite_case_count_maximum_absolute_error": finite_count,
            "max_case_maximum_absolute_error": maximum * 5,
            "mean_maximum_absolute_error": maximum * 2,
            "mean_relative_l2_error": 0.001,
        }

    def test_reads_real_reference_maximum(self) -> None:
        root = Path(__file__).resolve().parents[1] / "Results"
        path = root / "cl1_sf25_d5_r0" / "Study" / "landmarks_grid" / "grid_comparison.csv"
        if not path.is_file():
            self.skipTest("Reference study results are not present")
        with path.open(newline="", encoding="utf-8-sig") as file:
            row = next(row for row in csv.DictReader(file) if row["grid_size"] == "3")
        errors = load_grid_errors(root, "cl1_sf25_d5_r0", "landmarks_grid")
        self.assertEqual(errors[3], float(row["max_case_relative_l2_error"]))
        errors = load_grid_errors(root, "cl1_sf25_d5_r0", "landmarks_grid", "absolute")
        self.assertEqual(errors[3], float(row["max_case_absolute_l2_error"]))
        for metric, column in (
            ("absolute_max", "max_case_maximum_absolute_error"),
            ("absolute_mean", "mean_maximum_absolute_error"),
        ):
            errors = load_grid_errors(root, "cl1_sf25_d5_r0", "landmarks_grid", metric)
            self.assertEqual(errors[3], float(row[column]))

    def test_baseline_uses_valid_states_and_study_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_study(root, "test", [self.row(3, 0.4)])
            baseline = load_baseline(root, "test", "landmarks_grid")
            self.loader.assert_called_once_with(root / "test", nx=51, nz=51)
            self.assertEqual(baseline.case_count, 2)
            self.assertEqual(baseline.values["relative"], 1.0)
            self.assertEqual(baseline.values["absolute_max"], 4.0)
            self.assertEqual(baseline.values["absolute_mean"], 3.5)
            self.assertAlmostEqual(baseline.values["absolute"], np.sqrt(17))
            self.loader.return_value.state_matrix.return_value = np.zeros((2, 2))
            self.assertEqual(load_baseline(root, "test", "landmarks_grid").values["relative"], 0.0)
            self.loader.return_value.state_matrix.return_value = np.array([[float("nan"), 1.0]])
            with self.assertRaisesRegex(ValueError, "no finite valid"):
                load_baseline(root, "test", "landmarks_grid")

    def test_baseline_requires_matching_metadata_and_case_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_study(root, "test", [self.row(3, 0.4)])
            with self.assertRaisesRegex(ValueError, "study has 2 cases.*has 3"):
                load_grid_errors(root, "test", "landmarks_grid", expected_cases=3)
            metadata = {
                "per_grid": [{
                    "best": {"interpolation_grid": {"nx": 51, "nz": 51}},
                    "worst": {"interpolation_grid": {"nx": 51, "nz": 50}},
                }]
            }
            (path.parent / "global_best_worst_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_baseline(root, "test", "landmarks_grid")

    def test_baseline_uses_recorded_element_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_study(root, "test", [self.row(3, 0.4)])
            metadata_file = path.parent / "global_best_worst_metadata.json"
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            metadata["surface_resampling"] = "elements"
            metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
            load_baseline(root, "test", "landmarks_grid")
            self.loader.assert_called_once_with(root / "test", nx=51, nz=51, method="elements")

    def test_reads_maximum_and_rejects_nonfinite_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_study(root, "test", [self.row(3, 0.4)])
            self.assertEqual(load_grid_errors(root, "test", "landmarks_grid"), {3: 0.4})
            self.assertEqual(
                load_grid_errors(root, "test", "landmarks_grid", "absolute"), {3: 4.0}
            )
            self.assertEqual(
                load_grid_errors(root, "test", "landmarks_grid", "absolute_max"), {3: 2.0}
            )
            self.assertEqual(
                load_grid_errors(root, "test", "landmarks_grid", "absolute_mean"), {3: 0.8}
            )
            self.write_study(root, "test", [self.row(3, 0.4, finite_count=1)])
            with self.assertRaisesRegex(ValueError, "not finite for all cases"):
                load_grid_errors(root, "test", "landmarks_grid")
            with self.assertRaisesRegex(ValueError, "absolute_l2_error"):
                load_grid_errors(root, "test", "landmarks_grid", "absolute")
            for metric in ("absolute_max", "absolute_mean"):
                with self.assertRaisesRegex(ValueError, "maximum_absolute_error"):
                    load_grid_errors(root, "test", "landmarks_grid", metric)

    def test_plot_data_uses_grid_three_and_landmark_counts(self) -> None:
        with patch("matplotlib.pyplot.subplots") as subplots:
            figure, axis = MagicMock(), MagicMock()
            subplots.return_value = (figure, axis)
            plot_clamps(
                Path("unused.png"),
                {1: {9: {3: 0.3}, 1: {3: 0.1}, 3: {3: 0.2}},
                 5: {1: {3: 0.4}, 3: {3: 0.5}, 9: {3: 0.6}}},
                baselines={1: {1: 1.1, 3: 1.2, 9: 1.3},
                           5: {1: 2.1, 3: 2.2, 9: 2.3}},
            )
            self.assertEqual(axis.plot.call_count, 4)
            self.assertEqual(axis.plot.call_args_list[0].args, ([1, 3, 9], [0.1, 0.2, 0.3]))
            self.assertEqual(axis.plot.call_args_list[0].kwargs["label"], "1 mm")
            self.assertEqual(axis.plot.call_args_list[1].args, ([1, 3, 9], [1.1, 1.2, 1.3]))
            self.assertEqual(axis.plot.call_args_list[1].kwargs["linestyle"], "--")
            self.assertEqual(axis.plot.call_args_list[2].args, ([1, 3, 9], [0.4, 0.5, 0.6]))
            self.assertEqual(axis.plot.call_args_list[2].kwargs["label"], "5 mm")
            self.assertEqual(axis.plot.call_args_list[3].args, ([1, 3, 9], [2.1, 2.2, 2.3]))
            axis.legend.assert_called_once_with(title="Displacement")
            figure.savefig.assert_called_once()

        with patch("matplotlib.pyplot.subplots") as subplots:
            figure, axis = MagicMock(), MagicMock()
            subplots.return_value = (figure, axis)
            plot_grid_sizes(Path("unused.png"), {1: {2: 0.4, 3: 0.2}, 10: {2: 0.8, 3: 0.6}},
                            baselines={1: 1.0, 10: 2.0})
            self.assertEqual(axis.plot.call_args_list[0].args, ([4, 9], [0.4, 0.2]))
            self.assertEqual(axis.plot.call_args_list[1].args, ([4, 9], [0.8, 0.6]))
            self.assertEqual(axis.plot.call_args_list[0].kwargs["label"], "1 mm")
            self.assertEqual([call.args[0] for call in axis.axhline.call_args_list], [1.0, 2.0])
            self.assertEqual(axis.axhline.call_args_list[0].kwargs["linestyle"], "--")

        with patch("matplotlib.pyplot.subplots") as subplots:
            figure, axis = MagicMock(), MagicMock()
            subplots.return_value = (figure, axis)
            plot_grid_sizes(Path("unused.png"), {1: {3: 4.0}, 5: {3: 7.0}}, 9, "absolute",
                            baselines={1: 4.0, 5: 7.0})
            self.assertEqual(axis.plot.call_args_list[1].args, ([9], [7.0]))
            axis.set_title.assert_called_once_with("Grid-size comparison (9 clamps)")
            axis.set_ylabel.assert_called_once_with(
                "Maximum absolute L2 displacement error (model units)"
            )
            axis.yaxis.set_major_formatter.assert_not_called()
        for metric, label in (
            ("absolute_max", "Maximum case-wise max. absolute error (model units)"),
            ("absolute_mean", "Mean case-wise max. absolute error (model units)"),
        ):
            with self.subTest(metric=metric), patch("matplotlib.pyplot.subplots") as subplots:
                figure, axis = MagicMock(), MagicMock()
                subplots.return_value = (figure, axis)
                plot_clamps(Path("unused.png"), {1: {3: {3: 1.5}}}, metric,
                            baselines={1: {3: 2.0}})
                self.assertEqual(axis.plot.call_args_list[0].args, ([3], [1.5]))
                self.assertEqual(axis.plot.call_args_list[1].args, ([3], [2.0]))
                axis.set_ylabel.assert_called_once_with(label)
                axis.yaxis.set_major_formatter.assert_not_called()

    def test_cli_writes_all_plots_and_reports_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, output = root / "results", root / "comparison"
            args = ["--results-dir", str(results), "--output-dir", str(output)]
            names = ({name for group in CLAMP_EXPERIMENTS.values() for name in group.values()}
                     | set(DISPLACEMENT_EXPERIMENTS.values())
                     | set(CL9_DISPLACEMENT_EXPERIMENTS.values()))
            for name in names:
                self.write_study(results, name, [self.row(2, 0.5), self.row(3, 0.2)])
            main(args)
            self.assertTrue((output / "clamps_d1_d5_grid3.png").is_file())
            self.assertTrue((output / "cl3_displacement_by_grid.png").is_file())
            self.assertTrue((output / "cl9_displacement_by_grid.png").is_file())
            for suffix in ("maximum_absolute", "mean_maximum_absolute"):
                for stem in ("clamps_d1_d5_grid3", "cl3_displacement_by_grid", "cl9_displacement_by_grid"):
                    self.assertTrue((output / f"{stem}_{suffix}.png").is_file())
            self.assertFalse((output / "clamps_d1_d5_grid3_absolute_l2.png").exists())
            main(args + ["--plot", "clamps", "--metric", "absolute"])
            self.assertTrue((output / "clamps_d1_d5_grid3_absolute_l2.png").is_file())
            missing = results / CLAMP_EXPERIMENTS[5][9] / "Study" / "landmarks_grid" / "grid_comparison.csv"
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                main(args)
            main(args + ["--plot", "grids", "--grid-clamps", "3"])
            with self.assertRaises(FileNotFoundError):
                main(args + ["--plot", "grids", "--grid-clamps", "9"])

    def test_inconsistent_grids_and_missing_three_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ({name for group in CLAMP_EXPERIMENTS.values() for name in group.values()}
                     | set(DISPLACEMENT_EXPERIMENTS.values())
                     | set(CL9_DISPLACEMENT_EXPERIMENTS.values()))
            for name in names:
                self.write_study(root, name, [self.row(2, 0.5), self.row(3, 0.2)])
            self.write_study(root, DISPLACEMENT_EXPERIMENTS[10], [self.row(3, 0.2)])
            args = ["--results-dir", str(root), "--output-dir", str(root / "output")]
            with self.assertRaisesRegex(ValueError, "same grid sizes"):
                main(args)
            self.assertFalse((root / "output").exists())
            self.write_study(root, CLAMP_EXPERIMENTS[1][9], [self.row(2, 0.5)])
            with self.assertRaisesRegex(ValueError, "1 mm:.*3x3"):
                main(args + ["--plot", "clamps"])

    def test_clamp_cli_maps_both_displacement_studies_for_each_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for displacement, experiments in CLAMP_EXPERIMENTS.items():
                for count, name in experiments.items():
                    self.write_study(root, name, [self.row(3, displacement + count)])
            args = ["--results-dir", str(root), "--output-dir", str(root / "output"),
                    "--plot", "clamps"]
            with patch("Simulation.compare_surface_generation.plot_clamps") as plot:
                main(args)
            self.assertEqual(plot.call_count, 3)
            for call, multiplier in zip(plot.call_args_list, (1, 5, 2)):
                series = call.args[1]
                for displacement in (1, 5):
                    for count in (1, 3, 9):
                        self.assertEqual(series[displacement][count][3],
                                         (displacement + count) * multiplier)


if __name__ == "__main__":
    unittest.main()
