from pathlib import Path
import csv
import random
import pandas as pd


OUTPUT_FILE = Path("Projektarbeit\Simulation\Versuche\parameters.csv")
NUMBER_OF_SIMULATIONS = 10
RANDOM_SEED = 42

SAUGER_1_ID = 26 # not necessary if the WP
SAUGER_2_ID = 27

SAUGER_1_NOMINAL_X = 200 # Controlled by i-logic rule
SAUGER_1_NOMINAL_Z = 300
SAUGER_2_NOMINAL_X = 400
SAUGER_2_NOMINAL_Z = 500
MAX_POSITION_DEVIATION = 20

MIN_DISPLACEMENT_Y = -100 # Controlled by ForcedDisplacment in Nastran input file
MAX_DISPLACEMENT_Y = 0 # Controlled by ForcedDisplacment in Nastran input file

MIN_ROTATION_X = 0 # Controlled by ForcedDisplacment in Nastran input file
MAX_ROTATION_X = 0 # Controlled by ForcedDisplacment in Nastran input file
MIN_ROTATION_Y = 0 # Controlled by ForcedDisplacment in Nastran input file
MAX_ROTATION_Y = 0 # Controlled by ForcedDisplacment in Nastran input file
MIN_ROTATION_Z = 0 # Controlled by ForcedDisplacment in Nastran input file
MAX_ROTATION_Z = 0 # Controlled by ForcedDisplacment in Nastran input file


# TODO implement displacement and rotation ranges for all axes in the i-logic rule and Nastran input file.

def create_parameters():
    random_generator = random.Random(RANDOM_SEED)

    with OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
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
            )

            for simulation_id in range(1, NUMBER_OF_SIMULATIONS + 1):
                writer.writerow(
                    [
                        simulation_id,
                        SAUGER_1_ID,
                        SAUGER_2_ID,
                        random_generator.uniform(
                            SAUGER_1_NOMINAL_X - MAX_POSITION_DEVIATION,
                            SAUGER_1_NOMINAL_X + MAX_POSITION_DEVIATION,
                        ),
                        random_generator.uniform(
                            SAUGER_1_NOMINAL_Z - MAX_POSITION_DEVIATION,
                            SAUGER_1_NOMINAL_Z + MAX_POSITION_DEVIATION,
                        ),
                        random_generator.uniform(
                            SAUGER_2_NOMINAL_X - MAX_POSITION_DEVIATION,
                            SAUGER_2_NOMINAL_X + MAX_POSITION_DEVIATION,
                        ),
                        random_generator.uniform(
                            SAUGER_2_NOMINAL_Z - MAX_POSITION_DEVIATION,
                            SAUGER_2_NOMINAL_Z + MAX_POSITION_DEVIATION,
                        ),
                        random_generator.uniform(
                            MIN_DISPLACEMENT_Y,
                            MAX_DISPLACEMENT_Y,
                        ),
                        random_generator.uniform(
                            MIN_DISPLACEMENT_Y,
                            MAX_DISPLACEMENT_Y,
                        ),
                    ]
                )

    print(
        f"Created {OUTPUT_FILE.resolve()} with "
        f"{NUMBER_OF_SIMULATIONS} simulations."
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
    create_parameters()