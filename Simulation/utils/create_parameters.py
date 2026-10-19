from pathlib import Path
import csv
import json
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SuctionStudyConfig:
    """Sampling configuration for one suction."""

    nominal_x: float
    nominal_z: float


@dataclass(frozen=True)
class DeviationVariationConfig:
    """Global x/y/z deviation variation."""

    deviation_x: float
    deviation_y: float
    deviation_z: float
    varies: bool = True

    def sample(
        self,
        deviation: float,
        random_generator: random.Random,
    ) -> float:
        if not self.varies:
            return 0.0
        return random_generator.uniform(-deviation, deviation)


@dataclass(frozen=True)
class VariationConfig:
    """Shared variation settings for all suctions."""

    position: DeviationVariationConfig
    rotation: DeviationVariationConfig


@dataclass(frozen=True)
class ParameterStudyConfig:
    """Definition of a reproducible parameter sweep."""

    name: str
    number_of_simulations: int
    random_seed: int
    saugers: tuple[SuctionStudyConfig, ...]
    variation: VariationConfig


def load_study_config(config_file: str | Path) -> ParameterStudyConfig:
    """Load a parameter-study definition from JSON."""
    path = Path(config_file).resolve()
    with path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    suction_data = data["saugers"]
    variation_data = data["variation"]
    position_data = variation_data["position"]

    if not isinstance(suction_data, list) or not suction_data:
        raise ValueError("saugers must be a non-empty JSON array.")

    def suction(values: dict[str, Any]) -> SuctionStudyConfig:
        return SuctionStudyConfig(
            nominal_x=float(values["nominal_x"]),
            nominal_z=float(values["nominal_z"]),
        )

    variation = VariationConfig(
        position=DeviationVariationConfig(
            deviation_x=float(position_data["deviation_x"]),
            deviation_y=float(position_data["deviation_y"]),
            deviation_z=float(position_data["deviation_z"]),
            varies=read_boolean(
                position_data.get("vary_position", True),
                "variation.position.vary_position",
            ),
        ),
        rotation=DeviationVariationConfig(
            deviation_x=float(variation_data["rotation"]["deviation_x"]),
            deviation_y=float(variation_data["rotation"]["deviation_y"]),
            deviation_z=float(variation_data["rotation"]["deviation_z"]),
            varies=read_boolean(
                variation_data["rotation"].get("vary_rotation", True),
                "variation.rotation.vary_rotation",
            ),
        ),
    )
    config = ParameterStudyConfig(
        name=str(data["name"]),
        number_of_simulations=int(data["number_of_simulations"]),
        random_seed=int(data["random_seed"]),
        saugers=tuple(suction(values) for values in suction_data),
        variation=variation,
    )
    if config.number_of_simulations < 1:
        raise ValueError("number_of_simulations must be positive.")
    for name, variation_settings in (
        ("position", config.variation.position),
        ("rotation", config.variation.rotation),
    ):
        if any(
            deviation < 0
            for deviation in (
                variation_settings.deviation_x,
                variation_settings.deviation_y,
                variation_settings.deviation_z,
            )
        ):
            raise ValueError(
                f"variation.{name} deviations must be non-negative."
            )
    return config


def read_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean.")
    return value


def generate_cases(config: ParameterStudyConfig) -> list[dict[str, float | int]]:
    """Generate concrete cases from a study definition."""
    if config.number_of_simulations < 1:
        raise ValueError("number_of_simulations must be positive.")

    random_generator = random.Random(config.random_seed)

    def position(suction: SuctionStudyConfig) -> tuple[float, float, float]:
        position_variation = config.variation.position
        return (
            suction.nominal_x
            + position_variation.sample(
                position_variation.deviation_x,
                random_generator,
            ),
            suction.nominal_z
            + position_variation.sample(
                position_variation.deviation_z,
                random_generator,
            ),
            position_variation.sample(
                position_variation.deviation_y,
                random_generator,
            ),
        )

    cases: list[dict[str, float | int]] = []
    for simulation_id in range(1, config.number_of_simulations + 1):
        case: dict[str, float | int] = {"simulation_id": simulation_id}
        for index, suction in enumerate(config.saugers, start=1):
            x, z, displacement_y = position(suction)
            case[f"s{index}_x"] = x
            case[f"s{index}_z"] = z
            case[f"sauger_{index}_displacement"] = displacement_y
            for axis, deviation in (
                ("x", config.variation.rotation.deviation_x),
                ("y", config.variation.rotation.deviation_y),
                ("z", config.variation.rotation.deviation_z),
            ):
                case[f"sauger_{index}_rotation_{axis}"] = config.variation.rotation.sample(
                    deviation,
                    random_generator,
                )
        cases.append(case)
    return cases


def parameter_columns(sauger_count: int) -> list[str]:
    """Return the ordered CSV columns for a given sauger count."""
    columns = ["simulation_id"]
    for index in range(1, sauger_count + 1):
        columns.extend((f"s{index}_x", f"s{index}_z"))
        columns.append(f"sauger_{index}_displacement")
        columns.extend(
            f"sauger_{index}_rotation_{axis}"
            for axis in ("x", "y", "z")
        )
    return columns


def create_parameters(
    config: ParameterStudyConfig,
    results_dir: str | Path,
) -> str:
    """Generate a study CSV and return its experiment directory name."""
    experiment_name = config.name.strip()
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError(
            "The study name must be a non-empty directory name without path separators."
        )

    destination = Path(results_dir).resolve() / experiment_name / "parameters.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(config)
    columns = parameter_columns(len(config.saugers))

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    column: (
                        f"{case[column]:.6f}" # Format floats with 6 decimal places, unnecessary? trying to fix inventor COM ilogic rule error # Now done, could be removed
                        if column != "simulation_id"
                        else case[column]
                    )
                    for column in columns
                }
            )

    print(
        f"Created {destination} with {len(cases)} simulations "
        f"for study {config.name!r}."
    )
    return experiment_name


def detect_sauger_count(columns: list[str]) -> int: # TODO maybe add a simple sauger parameter to config file?
    """Infer and validate the contiguous sauger schema in CSV columns."""
    indices = set()
    for column in columns:
        prefix, _, suffix = column.partition("_")
        if prefix.startswith("s") and prefix[1:].isdigit() and suffix == "x":
            indices.add(int(prefix[1:]))
    if not indices or indices != set(range(1, max(indices) + 1)):
        raise ValueError("CSV must contain contiguous sauger coordinate columns.")
    return max(indices)


def load_parameter_cases(csv_file: Path) -> pd.DataFrame:
    """Read and validate the parameter CSV file."""

    if not csv_file.is_file():
        raise FileNotFoundError(
            f"Parameter CSV not found:\n{csv_file}"
        )

    parameters = pd.read_csv(
        csv_file,
        skipinitialspace=True,
    )

    # Remove whitespace around column names.
    parameters.columns = parameters.columns.str.strip()

    try:
        sauger_count = detect_sauger_count(parameters.columns.tolist())
    except ValueError as error:
        raise ValueError(f"Invalid sauger CSV schema: {error}") from error

    required_columns = parameter_columns(sauger_count)
    missing_columns = [
        column for column in required_columns if column not in parameters.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    parameters = parameters[required_columns].copy()

    # Check for empty values before converting data types.
    if parameters.isna().any().any():
        raise ValueError("The parameter CSV contains empty values.")

    integer_columns = ["simulation_id"]
    displacement_columns = [
        f"sauger_{index}_displacement"
        for index in range(1, sauger_count + 1)
    ]
    rotation_columns = [
        f"sauger_{index}_rotation_{axis}"
        for index in range(1, sauger_count + 1)
        for axis in ("x", "y", "z")
    ]
    position_columns = [
        coordinate
        for index in range(1, sauger_count + 1)
        for coordinate in (f"s{index}_x", f"s{index}_z")
    ]

    for column in integer_columns:
        parameters[column] = pd.to_numeric(
            parameters[column],
            errors="raise",
        ).astype(int)

    for column in displacement_columns:
        parameters[column] = pd.to_numeric(
            parameters[column],
            errors="raise",
        ).astype(float)

    for column in rotation_columns:
        parameters[column] = pd.to_numeric(
            parameters[column],
            errors="raise",
        ).astype(float)

    for column in position_columns:
        parameters[column] = pd.to_numeric(
            parameters[column],
            errors="raise",
        ).astype(float)

    if parameters["simulation_id"].duplicated().any():
        duplicates = parameters.loc[
            parameters["simulation_id"].duplicated(keep=False),
            "simulation_id",
        ].unique().tolist()

        raise ValueError(
            f"Duplicate simulation IDs found: {duplicates}"
        )

    return parameters


if __name__ == "__main__":

    CONFIG_FILE = Path(__file__).parents[1] / "parameter_config.json"
    DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "Versuche"


    study_config = load_study_config(CONFIG_FILE)
    create_parameters(study_config, DEFAULT_OUTPUT_DIR)