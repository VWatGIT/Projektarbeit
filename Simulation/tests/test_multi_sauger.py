import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Simulation.utils.create_parameters import (
    create_parameters,
    generate_cases,
    load_parameter_cases,
    load_study_config,
    parameter_columns,
)
from Simulation.utils.inventor_automation import SuctionPosition


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "Simulation"
    / "SweepConfigs"
    / "Test_3_Sauger.json"
)


def config_data() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class MultiSaugerTests(unittest.TestCase):
    def test_three_saugers_round_trip(self) -> None:
        data = config_data()
        data["number_of_simulations"] = 1
        data["saugers"] = data["saugers"][:2]
        data["saugers"].append({"nominal_x": 600.0, "nominal_z": 700.0})
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            config_path = temp_dir / "config.json"
            parameter_path = temp_dir / "parameters.csv"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            config = load_study_config(config_path)
            experiment_name = create_parameters(config, temp_dir)
            parameters = load_parameter_cases(
                temp_dir / experiment_name / "parameters.csv"
            )

            self.assertEqual(len(config.saugers), 3)
            self.assertEqual(list(parameters.columns), parameter_columns(3))
            self.assertEqual(len(parameters), 1)

    def test_position_deviation_y_generates_displacement(self) -> None:
        data = config_data()
        data["number_of_simulations"] = 3
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            cases = generate_cases(load_study_config(config_path))

            displacements = [case["sauger_1_displacement"] for case in cases]
            self.assertTrue(all(-10.0 <= value <= 10.0 for value in displacements))
            self.assertTrue(any(value != 0.0 for value in displacements))

    def test_disabled_variation_uses_zero_deviations(self) -> None:
        data = config_data()
        data["variation"]["position"]["vary_position"] = False
        data["variation"]["rotation"]["vary_rotation"] = False
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            case = generate_cases(load_study_config(config_path))[0]

            first_sauger = data["saugers"][0]
            self.assertEqual(
                (case["s1_x"], case["s1_z"]),
                (first_sauger["nominal_x"], first_sauger["nominal_z"]),
            )
            self.assertEqual(case["sauger_1_displacement"], 0.0)
            self.assertEqual(case["sauger_1_rotation_x"], 0.0)
            self.assertEqual(case["sauger_1_rotation_y"], 0.0)
            self.assertEqual(case["sauger_1_rotation_z"], 0.0)

    def test_empty_sauger_array_is_rejected(self) -> None:
        data = config_data()
        data["saugers"] = []
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_study_config(config_path)

    def test_inventor_arguments_are_generated_for_all_saugers(self) -> None:
        class NameValueMap:
            def __init__(self) -> None:
                self.values: dict[str, float] = {}

            def Add(self, name: str, value: float) -> None:
                self.values[name] = value

        name_value_map = NameValueMap()
        inventor = SimpleNamespace(
            TransientObjects=SimpleNamespace(
                CreateNameValueMap=lambda: name_value_map,
            )
        )

        SuctionPosition(
            positions=((100.0, 200.0), (300.0, 400.0), (500.0, 600.0))
        ).as_arguments(inventor)

        self.assertEqual(
            name_value_map.values,
            {
                "sauger_count": 3,
                "s1_x": 100.0,
                "s1_z": 200.0,
                "s2_x": 300.0,
                "s2_z": 400.0,
                "s3_x": 500.0,
                "s3_z": 600.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
