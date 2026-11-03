import math
import time
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
    detect_sauger_count,
    load_parameter_cases,
    load_study_config,
)
from utils.coordinate_transforms import suction_position_in_nastran_coordinates
from utils.postprocess import export_displacement_field_to_numpy
from utils.inventor_automation import SuctionPosition, generate_case
from utils.result_status import write_simulation_status



# ============================================================
# KONFIGURATION
# ============================================================
#SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2023\Nastran\Nastran.exe") # Uncomment depending on Inventor/Nastran Version 
SOLVER_PATH = Path(r"C:\Program Files\Autodesk\Inventor Nastran 2025\Nastran\Nastran.exe") 

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "Projektarbeit" / "Simulation" / "Results"
SWEEP_CONFIGS_DIR = REPO_ROOT / "Projektarbeit" / "Simulation" / "SweepConfigs"
SWEEP_CONFIG_PATH = SWEEP_CONFIGS_DIR / "S2_V5_N100.json" # Change this to the relevant one


study_config = load_study_config(SWEEP_CONFIG_PATH)
sauger_count = len(study_config.saugers)
MODEL_STEM = f"CFK_Platte_{sauger_count}"
INVENTOR_MODEL = REPO_ROOT / "Projektarbeit" / "Simulation" / "Inventor" / f"{MODEL_STEM}.ipt"
PLATE_MODEL = REPO_ROOT / "Projektarbeit" / "Simulation" / "Inventor" / "NastranExports" / f"{MODEL_STEM}.nas"
experiment_name = create_parameters(
    config=study_config,
    results_dir=RESULTS_DIR,
)


EXPERIMENT_DIR = RESULTS_DIR / experiment_name
PARAMETER_FILE = EXPERIMENT_DIR / "parameters.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "FEA"



#Solver_Arguments = ["INMEMORYIO=AUS", "BUFFEREDDECKREAD=OFF"] # TODO check out potentially useful solver arguments

# ============================================================
# HAUPTPROGRAMM
# ============================================================

def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def main() -> None:
    parameters = load_parameter_cases(PARAMETER_FILE)
    parameter_sauger_count = detect_sauger_count(parameters.columns.tolist())
    if parameter_sauger_count != sauger_count:
        raise ValueError(
            "The generated parameter file contains a different number of saugers "
            "than the loaded sweep configuration."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status_cases = []
    successful_count = 0
    failed_count = 0
    total_cases = len(parameters)
    sweep_start = time.monotonic()
    for position, case in enumerate(parameters.itertuples(index=False), start=1):
        simulation_id = int(case.simulation_id)

        case_name = f"simulation_{simulation_id:03d}"
        case_directory = OUTPUT_DIR / f"{case_name}"
        output_model = case_directory / f"{case_name}.nas"
        
        pch_file = case_directory / f"{case_name}.pch"
        npy_file = case_directory / f"{case_name}.npz" # Displacement field

        elapsed = time.monotonic() - sweep_start
        completed = position - 1
        eta_text = (
            format_duration(elapsed / completed * (total_cases - completed))
            if completed
            else "?"
        )
        print(
            f"\n[{position}/{total_cases}] {completed / total_cases:.0%} done | "
            f"elapsed {format_duration(elapsed)} | ETA ~{eta_text} | "
            f"{successful_count} ok, {failed_count} failed"
        )
        print(f"Running {case_name}")
        npy_file.unlink(missing_ok=True)
        pch_file.unlink(missing_ok=True)
        status = {
            "simulation_id": simulation_id,
            "case_name": case_name,
            "status": "failed",
            "npz_file": str(npy_file.relative_to(EXPERIMENT_DIR)),
            "returncode": None,
            "error": None,
        }

        try:
            generate_case(
                project_root=REPO_ROOT / "Projektarbeit",
                inventor_model_path=INVENTOR_MODEL,
                position=SuctionPosition(
                    positions=tuple(
                        (
                            float(getattr(case, f"s{index}_x")),
                            float(getattr(case, f"s{index}_z")),
                        )
                        for index in range(1, parameter_sauger_count + 1)
                    ),
                ),
            )

            model = load_model(PLATE_MODEL)
            for index in range(1, parameter_sauger_count + 1):
                position_x, position_z = suction_position_in_nastran_coordinates(
                    float(getattr(case, f"s{index}_x")),
                    float(getattr(case, f"s{index}_z")),
                )
                change_forced_displacement_by_position(
                    model=model,
                    position_x=position_x,
                    position_z=position_z,
                    new_value=float(getattr(case, f"sauger_{index}_displacement")),
                )
                for component, axis in ((4, "x"), (5, "y"), (6, "z")):
                    rotation_column = f"sauger_{index}_rotation_{axis}"
                    change_forced_value_by_position(
                        model=model,
                        position_x=position_x,
                        position_z=position_z,
                        component=component,
                        new_value=math.radians(float(getattr(case, rotation_column))),
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
        write_simulation_status(EXPERIMENT_DIR, status_cases)

    print(
        f"\nSweep finished: {successful_count} successful, "
        f"{failed_count} failed, {len(status_cases)} total "
        f"in {format_duration(time.monotonic() - sweep_start)}."
    )


if __name__ == "__main__":
    main()