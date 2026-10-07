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
    position_deviation_x: float
    position_deviation_z: float


@dataclass(frozen=True)
class RangeConfig:
    """Inclusive bounds for a uniformly sampled value."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class ParameterStudyConfig:
    """Definition of a reproducible parameter sweep."""

    name: str
    number_of_simulations: int
    random_seed: int
    sauger_1: SuctionStudyConfig
    sauger_2: SuctionStudyConfig
    displacement_y: RangeConfig
    sauger_1_id: int = 1
    sauger_2_id: int = 2


def load_study_config(config_file: str | Path) -> ParameterStudyConfig:
    """Load a parameter-study definition from JSON."""
    path = Path(config_file).resolve()
    with path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    suction_data = data["suction_positions"]
    displacement_data = data["displacement_y"]

    def suction(name: str) -> SuctionStudyConfig:
        values = suction_data[name]
        return SuctionStudyConfig(
            nominal_x=float(values["nominal_x"]),
            nominal_z=float(values["nominal_z"]),
            position_deviation_x=float(values["deviation_x"]),
            position_deviation_z=float(values["deviation_z"]),
        )

    config = ParameterStudyConfig(
        name=str(data["name"]),
        number_of_simulations=int(data["number_of_simulations"]),
        random_seed=int(data["random_seed"]),
        sauger_1=suction("s1"),
        sauger_2=suction("s2"),
        displacement_y=RangeConfig(
            minimum=float(displacement_data["min"]),
            maximum=float(displacement_data["max"]),
        ),
        sauger_1_id=int(data.get("sauger_1_id", 26)),
        sauger_2_id=int(data.get("sauger_2_id", 27)),
    )
    if config.number_of_simulations < 1:
        raise ValueError("number_of_simulations must be positive.")
    if config.displacement_y.minimum > config.displacement_y.maximum:
        raise ValueError("displacement_y.min must not exceed displacement_y.max.")
    for name, suction_config in (
        ("s1", config.sauger_1),
        ("s2", config.sauger_2),
    ):
        if (
            suction_config.position_deviation_x < 0
            or suction_config.position_deviation_z < 0
        ):
            raise ValueError(
                f"{name} position deviations must be non-negative."
            )
    return config


def generate_cases(config: ParameterStudyConfig) -> list[dict[str, float | int]]:
    """Generate concrete cases from a study definition."""
    if config.number_of_simulations < 1:
        raise ValueError("number_of_simulations must be positive.")

    random_generator = random.Random(config.random_seed)

    def position(suction: SuctionStudyConfig) -> tuple[float, float]:
        return (
            random_generator.uniform(
                suction.nominal_x - suction.position_deviation_x,
                suction.nominal_x + suction.position_deviation_x,
            ),
            random_generator.uniform(
                suction.nominal_z - suction.position_deviation_z,
                suction.nominal_z + suction.position_deviation_z,
            ),
        )

    cases: list[dict[str, float | int]] = []
    for simulation_id in range(1, config.number_of_simulations + 1):
        s1_x, s1_z = position(config.sauger_1)
        s2_x, s2_z = position(config.sauger_2)
        cases.append(
            {
                "simulation_id": simulation_id,
                "sauger_1_id": config.sauger_1_id,
                "sauger_2_id": config.sauger_2_id,
                "s1_x": s1_x,
                "s1_z": s1_z,
                "s2_x": s2_x,
                "s2_z": s2_z,
                "sauger_1_displacement": random_generator.uniform(
                    config.displacement_y.minimum,
                    config.displacement_y.maximum,
                ),
                "sauger_2_displacement": random_generator.uniform(
                    config.displacement_y.minimum,
                    config.displacement_y.maximum,
                ),
            }
        )
    return cases


def create_parameters(
    config: ParameterStudyConfig,
    output_file: str | Path,
) -> None:
    """Generate and write the concrete cases for a study."""
    destination = Path(output_file).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(config)
    columns = REQUIRED_COLUMNS

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    column: (
                        f"{case[column]:.6f}" # Format floats with 6 decimal places, unnecessary? trying to fix inventor COM ilogic rule error
                        if column in FLOAT_COLUMNS
                        else case[column]
                    )
                    for column in columns
                }
            )

    print(
        f"Created {destination} with {len(cases)} simulations "
        f"for study {config.name!r}."
    )


REQUIRED_COLUMNS = [
    "simulation_id",
    "sauger_1_id",
    "sauger_2_id",
    "s1_x",
    "s1_z",
    "s2_x",
    "s2_z",
    "sauger_1_displacement",
    "sauger_2_displacement",
]

FLOAT_COLUMNS = [
    "s1_x",
    "s1_z",
    "s2_x",
    "s2_z",
    "sauger_1_displacement",
    "sauger_2_displacement",
]


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

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in parameters.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    parameters = parameters[REQUIRED_COLUMNS].copy()

    # Check for empty values before converting data types.
    if parameters.isna().any().any():
        raise ValueError("The parameter CSV contains empty values.")

    integer_columns = [
        "simulation_id",
        "sauger_1_id",
        "sauger_2_id",
    ]

    displacement_columns = [
        "sauger_1_displacement",
        "sauger_2_displacement",
    ]
    position_columns = [
        "s1_x",
        "s1_z",
        "s2_x",
        "s2_z",
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
    DEFAULT_OUTPUT_FILE = Path(__file__).parents[1] / "Versuche" / "parameters.csv"


    study_config = load_study_config(CONFIG_FILE)
    create_parameters(study_config, DEFAULT_OUTPUT_FILE)