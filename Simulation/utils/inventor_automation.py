"""Minimal Inventor/iLogic automation adapter for geometry parameter studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ILOGIC_ADDIN_ID = "{3BDD8D79-2179-4B11-8A5A-257B1C0263AC}"


@dataclass(frozen=True)
class SuctionPosition:
    """Values passed to the four named Inventor parameters."""

    s1_x: float
    s1_z: float
    s2_x: float
    s2_z: float

    def as_arguments(self, inventor_application: Any) -> Any:
        """Create an Inventor NameValueMap for an external iLogic rule."""
        arguments = inventor_application.TransientObjects.CreateNameValueMap()
        for name, value in zip(
            ("s1_x", "s1_z", "s2_x", "s2_z"),
            (self.s1_x, self.s1_z, self.s2_x, self.s2_z),
        ):
            arguments.Add(name, float(value))
        return arguments


def connect_to_inventor() -> Any:
    """Connect to a running Inventor instance or start one."""
    try:
        from win32com.client import GetActiveObject

        return GetActiveObject("Inventor.Application")
    except Exception:
        from win32com.client import Dispatch

        return Dispatch("Inventor.Application")


def run_external_rule(
    document_path: str | Path,
    rule_path: str | Path,
    position: SuctionPosition,
    visible: bool = True,
) -> Path:
    """Run an iLogic rule and save the Inventor document in place.

    Inventor and its iLogic add-in must be installed on Windows. The rule is
    responsible for any updates or exports it performs before the document is
    saved.
    """
    document_path = Path(document_path).resolve()
    rule_path = Path(rule_path).resolve()
    if not document_path.is_file():
        raise FileNotFoundError(f"Inventor document not found: {document_path}")
    if not rule_path.is_file():
        raise FileNotFoundError(f"iLogic rule not found: {rule_path}")

    inventor = connect_to_inventor()
    original_visibility = inventor.Visible
    original_silent_operation = inventor.SilentOperation
    inventor.Visible = visible
    inventor.SilentOperation = True
    document = None
    try:
        document = inventor.Documents.Open(str(document_path), visible)
        ilogic_addin = inventor.ApplicationAddIns.ItemById(ILOGIC_ADDIN_ID)
        if not ilogic_addin.Activated:
            ilogic_addin.Activate()

        automation = ilogic_addin.Automation
        arguments = position.as_arguments(inventor)
        #print(f"Running iLogic rule {rule_path} on {document_path} with arguments {arguments}") # debugging
        automation.RunExternalRuleWithArguments(
            document,
            str(rule_path),
            arguments,
        )
        document.Update()
        document.Save()
        return document_path
    finally:
        if document is not None:
            document.Close(True)
        inventor.SilentOperation = original_silent_operation
        inventor.Visible = original_visibility


def generate_case(
    project_root: str | Path,
    position: SuctionPosition,
) -> Path:
    """Generate a case and update its Nastran mesh and input file."""
    project_root = Path(project_root).resolve()
    generated_path = run_external_rule(
        document_path=project_root / "Simulation"/ "Inventor" / "CFK_Platte.ipt",
        rule_path=project_root / "Simulation" / "Inventor" / "iLogic" / "ChangeSaugerPositions.vb",
        position=position,
    )
    return run_external_rule(
        document_path=generated_path,
        rule_path=project_root / "Simulation" / "Inventor" / "iLogic" / "UpdateMeshAndNas.vb",
        position=position,
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    s1_x = 100
    s1_z = 400
    s2_x = 500
    s2_z = 800

    output = generate_case(
        project_root=project_root,
        position=SuctionPosition(
            s1_x=s1_x,
            s1_z=s1_z,
            s2_x=s2_x,
            s2_z=s2_z,
        )
    )