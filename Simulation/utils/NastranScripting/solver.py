from pathlib import Path
import subprocess


def run_nastran(
    solver_path: Path,
    input_file: Path,
    solver_arguments: list[str] | None = None,
) -> None:
    if not solver_path.is_file():
        raise FileNotFoundError(
            f"Nastran-Solver nicht gefunden:\n{solver_path}"
        )

    result_dir = input_file.parent
    log_file = result_dir / "solver_console.log"

    command = [
        str(solver_path),
        input_file.name,
        *(solver_arguments or []),
    ]

    print("Starte Nastran:")
    print(subprocess.list2cmdline(command))

    with log_file.open(
        mode="w",
        encoding="utf-8",
        errors="replace",
    ) as log:
        process = subprocess.run(
            command,
            cwd=result_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Nastran endete mit Rückgabecode {process.returncode}.\n"
            f"Logdatei: {log_file}"
        )

    print(f"Simulation abgeschlossen: {result_dir}")