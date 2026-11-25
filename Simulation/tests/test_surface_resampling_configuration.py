import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Simulation import study_interpolation_resolution, study_pod_modes, study_pod_surfaces
from Simulation.study_landmarks_offset import run_study as run_offset_study
from Simulation.study_landmarks_random import run_random_landmark_study
from Simulation.utils.random_landmark_study import load_random_landmark_study_config
from Simulation.utils.sauger_landmark_study import load_sauger_landmark_study_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "StudyConfigs"


class SurfaceResamplingConfigurationTests(unittest.TestCase):
    def test_existing_study_configs_expose_resampling(self) -> None:
        configs = list(CONFIG_DIR.glob("*.json"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn(config["surface_resampling"], ("linear", "elements"))

    def test_random_and_offset_settings_are_validated(self) -> None:
        cases = (
            (
                load_random_landmark_study_config,
                {"name": "random", "amounts": [1], "trials_per_amount": 1, "random_seed": 1},
            ),
            (
                load_sauger_landmark_study_config,
                {"name": "offset", "landmarks_per_sauger": 1, "radii": [1], "trials_per_radius": 1, "random_seed": 1},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for loader, data in cases:
                with self.subTest(loader=loader.__name__):
                    path.write_text(json.dumps(data), encoding="utf-8")
                    self.assertEqual(loader(path, results_dir=directory).surface_resampling, "linear")
                    data["surface_resampling"] = "elements"
                    path.write_text(json.dumps(data), encoding="utf-8")
                    self.assertEqual(loader(path, results_dir=directory).surface_resampling, "elements")
                    data["surface_resampling"] = "invalid"
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "surface_resampling"):
                        loader(path, results_dir=directory)

    def test_pod_and_resolution_settings_reach_the_resampler(self) -> None:
        cases = (
            (study_pod_surfaces, "config_pod_surfaces.json"),
            (study_pod_modes, "config_pod_modes.json"),
            (study_interpolation_resolution, "config_interpolation_resolution.json"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for module, filename in cases:
                with self.subTest(study=module.__name__):
                    data = json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))
                    data["surface_resampling"] = "elements"
                    path.write_text(json.dumps(data), encoding="utf-8")
                    args = module.parse_args(["--settings", str(path), "--experiment", "test"])
                    with patch.object(module, "load_resampled_snapshots", side_effect=RuntimeError("resampler")) as loader:
                        with self.assertRaisesRegex(RuntimeError, "resampler"):
                            module.run_study(args)
                    self.assertEqual(loader.call_args.kwargs["method"], "elements")
                    override = module.parse_args([
                        "--settings", str(path), "--experiment", "test",
                        "--surface-resampling", "linear",
                    ])
                    self.assertEqual(override.surface_resampling, "linear")
                    del data["surface_resampling"]
                    path.write_text(json.dumps(data), encoding="utf-8")
                    legacy = module.parse_args(["--settings", str(path), "--experiment", "test"])
                    self.assertEqual(legacy.surface_resampling, "linear")
                    data["surface_resampling"] = "invalid"
                    path.write_text(json.dumps(data), encoding="utf-8")
                    invalid = module.parse_args(["--settings", str(path), "--experiment", "test"])
                    with self.assertRaisesRegex(ValueError, "surface_resampling"):
                        module.run_study(invalid)

    def test_random_and_offset_studies_forward_selected_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps({
                "name": "random", "amounts": [1], "trials_per_amount": 1,
                "random_seed": 1, "surface_resampling": "elements",
            }), encoding="utf-8")
            config = load_random_landmark_study_config(path, results_dir=root)
            with patch("Simulation.study_landmarks_random.load_resampled_snapshots", side_effect=RuntimeError("resampler")) as loader:
                with self.assertRaisesRegex(RuntimeError, "resampler"):
                    run_random_landmark_study(config)
            self.assertEqual(loader.call_args.kwargs["method"], "elements")

            path.write_text(json.dumps({
                "name": "offset", "landmarks_per_sauger": 1, "radii": [1],
                "trials_per_radius": 1, "random_seed": 1, "surface_resampling": "elements",
            }), encoding="utf-8")
            with (
                patch("Simulation.study_landmarks_offset.load_sauger_centers", return_value=(object(),)),
                patch("Simulation.study_landmarks_offset._parameter_sauger_count", return_value=1),
                patch("Simulation.study_landmarks_offset._load_case_parameters", return_value={}),
                patch("Simulation.study_landmarks_offset.load_resampled_snapshots", side_effect=RuntimeError("resampler")) as loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "resampler"):
                    run_offset_study(path, results_dir=root)
            self.assertEqual(loader.call_args.kwargs["method"], "elements")


if __name__ == "__main__":
    unittest.main()
