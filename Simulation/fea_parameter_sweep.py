from pathlib import Path

from pyNastran.bdf.bdf import BDF

from utils.NastranScripting.model import (
    change_forced_displacement_by_position,
    load_model,
    save_model,
)
from utils.NastranScripting.solver import run_nastran
from utils.create_parameters import (
    create_parameters,
    load_parameter_cases,
    load_study_config,
)
from utils.postprocess import export_displacement_field_to_numpy
from utils.inventor_automation import SuctionPosition, generate_case



# ============================================================
# KONFIGURATION
# ============================================================
#SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2023\Nastran\Nastran.exe") # Uncomment depending on Inventor/Nastran Version 
SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2025\Nastran\Nastran.exe") 

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATE_MODEL = REPO_ROOT / "Projektarbeit" / "Simulation"/ "Inventor" / "NastranExports" /"CFK_Platte.nas"
RESULTS_DIR = REPO_ROOT / "Projektarbeit" / "Simulation" / "Results"

ExperimentName = "TestTest02" # Change this before starting a new FEA parameter sweep, otherwise the old results will be overwritten

# Maybe first check parameters before creating the parameter file
create_parameters(
    config=load_study_config(
        REPO_ROOT
        / "Projektarbeit"
        / "Simulation"
        / "parameter_sweep_config.json"
    ),
    output_file=RESULTS_DIR / ExperimentName / "parameters.csv",
)


PARAMETER_FILE = RESULTS_DIR / ExperimentName / "parameters.csv"
OUTPUT_DIR = RESULTS_DIR / ExperimentName / "FEA"



# Inventor's suction parameters use a different coordinate system from the
# generated Nastran GRID coordinates. --> Results from the dimensioning in the Inventor Sketch
NASTRAN_POSITION_SCALE = -1.0


#Solver_Arguments = ["INMEMORYIO=AUS", "BUFFEREDDECKREAD=OFF"] # TODO check out potentially useful solver arguments

# ============================================================
# HAUPTPROGRAMM
# ============================================================

def suction_position_in_nastran_coordinates(
    position_x: float,
    position_z: float,
) -> tuple[float, float]:
    """Convert an Inventor suction position to Nastran x/z coordinates."""

    return (
        NASTRAN_POSITION_SCALE * position_x,
        NASTRAN_POSITION_SCALE * position_z,
    )


def main() -> None:
    parameters = load_parameter_cases(PARAMETER_FILE)

    for case in parameters.itertuples(index=False):
        simulation_id = int(case.simulation_id)

        case_name = f"simulation_{simulation_id:03d}"
        case_directory = RESULTS_DIR / f"{case_name}"
        output_model = case_directory / f"{case_name}.nas"
        
        pch_file = case_directory / f"{case_name}.pch"
        npy_file = case_directory / f"{case_name}.npz" # Displacement field

        print(f"\nRunning {case_name}")

        generate_case(
            project_root=REPO_ROOT / "Projektarbeit", # uses project root to find .vb rules and the Inventor model
            position=SuctionPosition(
                s1_x=float(case.s1_x),
                s1_z=float(case.s1_z),
                s2_x=float(case.s2_x),
                s2_z=float(case.s2_z),
            ),
        )

        # Load the freshly generated model for this simulation.
        model = load_model(PLATE_MODEL)

        sauger_1_nastran_x, sauger_1_nastran_z = (
            suction_position_in_nastran_coordinates(
                float(case.s1_x),
                float(case.s1_z),
            )
        )
        change_forced_displacement_by_position(
            model=model,
            position_x=sauger_1_nastran_x,
            position_z=sauger_1_nastran_z,
            new_value=float(case.sauger_1_displacement),
        )

        sauger_2_nastran_x, sauger_2_nastran_z = (
            suction_position_in_nastran_coordinates(
                float(case.s2_x),
                float(case.s2_z),
            )
        )
        change_forced_displacement_by_position(
            model=model,
            position_x=sauger_2_nastran_x,
            position_z=sauger_2_nastran_z,
            new_value=float(case.sauger_2_displacement),
        )

        save_model(
            model=model,
            output_file=output_model,
        )

        run_nastran(
            solver_path=SOLVER_PATH,
            input_file=output_model,
        )

        export_displacement_field_to_numpy(
            bdf_file=output_model,
            pch_file=pch_file,
            output_file=npy_file,
            subcase_id=1,
        )


if __name__ == "__main__":
    main()