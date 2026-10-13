import math
from pathlib import Path
import json

from pyNastran.bdf.bdf import BDF

from utils.NastranScripting.model import (
    change_forced_value_by_position,
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
from utils.result_status import write_simulation_status



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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status_cases = []
    successful_count = 0
    failed_count = 0

    for case in parameters.itertuples(index=False):
        simulation_id = int(case.simulation_id)

        case_name = f"simulation_{simulation_id:03d}"
        case_directory = OUTPUT_DIR / f"{case_name}"
        output_model = case_directory / f"{case_name}.nas"
        
        pch_file = case_directory / f"{case_name}.pch"
        npy_file = case_directory / f"{case_name}.npz" # Displacement field

        print(f"\nRunning {case_name}")
        npy_file.unlink(missing_ok=True)
        pch_file.unlink(missing_ok=True)
        status = {
            "simulation_id": simulation_id,
            "case_name": case_name,
            "status": "failed",
            "npz_file": str(npy_file.relative_to(OUTPUT_DIR)),
            "returncode": None,
            "error": None,
        }

        try:
            generate_case(
                project_root=REPO_ROOT / "Projektarbeit",
                position=SuctionPosition(
                    s1_x=float(case.s1_x),
                    s1_z=float(case.s1_z),
                    s2_x=float(case.s2_x),
                    s2_z=float(case.s2_z),
                ),
            )

            model = load_model(PLATE_MODEL)
            sauger_1_nastran_x, sauger_1_nastran_z = suction_position_in_nastran_coordinates(
                float(case.s1_x), float(case.s1_z)
            )
            change_forced_displacement_by_position(
                model=model,
                position_x=sauger_1_nastran_x,
                position_z=sauger_1_nastran_z,
                new_value=float(case.sauger_1_displacement),
            )
            for component, column in ((4, "sauger_1_rotation_x"), (5, "sauger_1_rotation_y"), (6, "sauger_1_rotation_z")):
                change_forced_value_by_position(
                    model=model,
                    position_x=sauger_1_nastran_x,
                    position_z=sauger_1_nastran_z,
                    component=component,
                    new_value=math.radians(float(getattr(case, column))),
                )

            sauger_2_nastran_x, sauger_2_nastran_z = suction_position_in_nastran_coordinates(
                float(case.s2_x), float(case.s2_z)
            )
            change_forced_displacement_by_position(
                model=model,
                position_x=sauger_2_nastran_x,
                position_z=sauger_2_nastran_z,
                new_value=float(case.sauger_2_displacement),
            )
            for component, column in ((4, "sauger_2_rotation_x"), (5, "sauger_2_rotation_y"), (6, "sauger_2_rotation_z")):
                change_forced_value_by_position(
                    model=model,
                    position_x=sauger_2_nastran_x,
                    position_z=sauger_2_nastran_z,
                    component=component,
                    new_value=math.radians(float(getattr(case, column))),
                )

            save_model(model=model, output_file=output_model)
            solver_result = run_nastran(solver_path=SOLVER_PATH, input_file=output_model)
            status["returncode"] = solver_result.returncode
            if not solver_result.successful:
                raise RuntimeError(
                    "Nastran failed. See "
                    f"{solver_result.log_file}"
                )

            export_displacement_field_to_numpy(
                bdf_file=output_model,
                pch_file=pch_file,
                output_file=npy_file,
                subcase_id=1,
            )
            if not npy_file.is_file():
                raise RuntimeError(f"Expected result file was not created: {npy_file}")

            status["status"] = "success"
            successful_count += 1
            print(f"{case_name} succeeded")
        except Exception as error:
            failed_count += 1
            status["error"] = str(error)
            print(f"{case_name} failed: {error}")

        status_cases.append(status)
        write_simulation_status(OUTPUT_DIR, status_cases)

    print(
        f"\nSweep finished: {successful_count} successful, "
        f"{failed_count} failed, {len(status_cases)} total."
    )


if __name__ == "__main__":
    main()