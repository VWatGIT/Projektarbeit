import argparse
import csv
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
SURFACE_GENERATION_CONFIGS_DIR = REPO_ROOT / "Projektarbeit" / "Simulation" / "SurfaceGenerationConfigs"


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


def resolve_config_path(argument: str) -> Path:
    path = Path(argument)
    if path.suffix != ".json":
        path = SURFACE_GENERATION_CONFIGS_DIR / f"{argument}.json"
    return path.resolve()


def read_batch_file(batch_file: Path) -> list[str]:
    with batch_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "config" not in reader.fieldnames:
            raise ValueError(f"{batch_file} must contain a 'config' column.")
        return [row["config"].strip() for row in reader if row["config"].strip()]


def run_surface_generation(config_path: Path) -> bool:
    study_config = load_study_config(config_path)
    sauger_count = len(study_config.saugers)
    model_stem = f"CFK_Platte_{sauger_count}"
    inventor_model = REPO_ROOT / "Projektarbeit" / "Simulation" / "Inventor" / f"{model_stem}.ipt"
    plate_model = REPO_ROOT / "Projektarbeit" / "Simulation" / "Inventor" / "NastranExports" / f"{model_stem}.nas"
    experiment_name = create_parameters(
        config=study_config,
        results_dir=RESULTS_DIR,
    )
    experiment_dir = RESULTS_DIR / experiment_name
    parameter_file = experiment_dir / "parameters.csv"
    output_dir = experiment_dir / "FEA"

    def log(message: str) -> None:
        print(f"[{study_config.name}] {message}")

    parameters = load_parameter_cases(parameter_file)
    parameter_sauger_count = detect_sauger_count(parameters.columns.tolist())
    if parameter_sauger_count != sauger_count:
        raise ValueError(
            "The generated parameter file contains a different number of saugers "
            "than the loaded sweep configuration."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_cases = []
    successful_count = 0
    failed_count = 0
    total_cases = len(parameters)
    sweep_start = time.monotonic()
    for position, case in enumerate(parameters.itertuples(index=False), start=1):
        simulation_id = int(case.simulation_id)

        case_name = f"simulation_{simulation_id:03d}"
        case_directory = output_dir / f"{case_name}"
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
        log(
            f"\n[{position}/{total_cases}] {completed / total_cases:.0%} done | "
            f"elapsed {format_duration(elapsed)} | ETA ~{eta_text} | "
            f"{successful_count} ok, {failed_count} failed"
        )
        log(f"Running {case_name}")
        npy_file.unlink(missing_ok=True)
        pch_file.unlink(missing_ok=True)
        status = {
            "simulation_id": simulation_id,
            "case_name": case_name,
            "status": "failed",
            "npz_file": str(npy_file.relative_to(experiment_dir)),
            "returncode": None,
            "error": None,
        }

        try:
            generate_case(
                project_root=REPO_ROOT / "Projektarbeit",
                inventor_model_path=inventor_model,
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
            model = load_model(plate_model)
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
            log(f"{case_name} succeeded")
        except Exception as error:
            failed_count += 1
            status["error"] = str(error)
            log(f"{case_name} failed: {error}")

        status_cases.append(status)
        write_simulation_status(experiment_dir, status_cases)

    log(
        f"\nSweep finished: {successful_count} successful, "
        f"{failed_count} failed, {len(status_cases)} total "
        f"in {format_duration(time.monotonic() - sweep_start)}."
    )
    return failed_count == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate surface data for one or more configs, one after another."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            "Config names (without .json) or paths. Names are resolved in "
            f"{SURFACE_GENERATION_CONFIGS_DIR}. If omitted, the batch file is used."
        ),
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=SURFACE_GENERATION_CONFIGS_DIR / "batch.csv",
        help="CSV file with a 'config' column listing the configs to run.",
    )
    arguments = parser.parse_args()
    config_names = arguments.configs or read_batch_file(arguments.batch)
    if not config_names:
        raise SystemExit("No configs given and the batch file is empty.")
    config_paths = [resolve_config_path(name) for name in config_names]

    failed = [
        config_path.stem
        for config_path in config_paths
        if not run_surface_generation(config_path)
    ]
    if failed:
        raise SystemExit(f"Surface generation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()