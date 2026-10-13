from pathlib import Path
import subprocess
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NastranResult:
    """Outcome of one Nastran process invocation."""

    returncode: int
    log_file: Path
    fatal_errors: tuple[str, ...]

    @property
    def successful(self) -> bool:
        return self.returncode == 0 and not self.fatal_errors


def run_nastran(
    solver_path: Path,
    input_file: Path,
    solver_arguments: list[str] | None = None,
) -> NastranResult:
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

    fatal_errors = tuple(
        line.strip()
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if re.match(r"^\s*FATAL ERROR\b", line, re.IGNORECASE)
    )
    result = NastranResult(
        returncode=process.returncode,
        log_file=log_file,
        fatal_errors=fatal_errors,
    )

    if not result.successful:
        details = "; ".join(fatal_errors) or f"return code {process.returncode}"
        print(f"Nastran failed: {details}")
        return result

    print(f"Simulation abgeschlossen: {result_dir}")
    return result