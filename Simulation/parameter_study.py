from pathlib import Path
import subprocess

from pyNastran.bdf.bdf import BDF

from NastranScripting.model import (
    change_forced_displacement_by_position,
    load_model,
    save_model,
)
from NastranScripting.solver import run_nastran
from NastranScripting.parameters import load_parameter_cases
from NastranScripting.postprocess import export_displacement_field_to_numpy
from inventor_automation import SuctionPosition, generate_case



# ============================================================
# KONFIGURATION
# ============================================================
#SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2023\Nastran\Nastran.exe") # Uncomment depending on Inventor/Nastran Version 
SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2025\Nastran\Nastran.exe") 

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATE_MODEL = REPO_ROOT / "Projektarbeit" / "Inventor" / "NastranExports" /"CFK_Platte.nas"
VERSUCHE_DIR = REPO_ROOT / "Projektarbeit" / "Simulation" / "Versuche"
PARAMETER_FILE = VERSUCHE_DIR / "parameters.csv"
RESULTS_DIR = VERSUCHE_DIR / "Results"

# Inventor's suction parameters use a different coordinate system from the
# generated Nastran GRID coordinates. --> Results from the dimensioning in the Inventor Sketch
NASTRAN_POSITION_SCALE = -1.0


#Solver_Arguments = ["INMEMORYIO=AUS", "BUFFEREDDECKREAD=OFF"]

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
            project_root=REPO_ROOT / "Projektarbeit",
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