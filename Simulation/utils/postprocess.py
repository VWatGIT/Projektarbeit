from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pyNastran.bdf.bdf import read_bdf


_SUBCASE_PATTERN = re.compile(
    r"\$SUBCASE\s+ID\s*=\s*(\d+)",
    re.IGNORECASE,
)

_NASTRAN_EXPONENT_PATTERN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$"
)


def _nastran_float(value: str) -> float:
    """
    Converts standard and compact Nastran floating-point values.

    Examples:
        1.25E-3
        1.25D-3
        1.25-3
    """
    value = value.strip().replace("D", "E")

    try:
        return float(value)
    except ValueError:
        match = _NASTRAN_EXPONENT_PATTERN.match(value)

        if match is None:
            raise

        return float(f"{match.group(1)}E{match.group(2)}")

# Originally i wanted to use the OP2 format, but couldnt get it to work
def export_displacement_field_to_numpy( 
    bdf_file: str | Path,
    pch_file: str | Path,
    output_file: str | Path,
    subcase_id: int | None = 1,
) -> None:
    """
    Exports original node coordinates and translational displacements
    to a compressed NumPy NPZ file.

    The .nas and .bdf extensions represent the same Nastran deck format.

    For nonlinear analyses, a node may occur multiple times in the PCH.
    The last occurrence is retained, which normally corresponds to the
    final output/load step.

    Saved arrays:
        node_ids                 Shape: (n,)
        original_coordinates     Shape: (n, 3)
        displacement             Shape: (n, 3)
        displacement_magnitude   Shape: (n,)
        deformed_coordinates     Shape: (n, 3)
    """
    bdf_file = Path(bdf_file)
    pch_file = Path(pch_file)
    output_file = Path(output_file)

    if not bdf_file.is_file():
        raise FileNotFoundError(
            f"NAS/BDF model not found:\n{bdf_file}"
        )

    if not pch_file.is_file():
        raise FileNotFoundError(
            f"PCH result file not found:\n{pch_file}"
        )

    # --------------------------------------------------------
    # Read original GRID coordinates
    # --------------------------------------------------------
    model = read_bdf(
        str(bdf_file),
        xref=True,
        punch=False,
        debug=False,
    )

    original_coordinates_by_id = {
        int(node_id): np.asarray(
            node.get_position(),
            dtype=np.float64,
        )
        for node_id, node in model.nodes.items()
    }

    if not original_coordinates_by_id:
        raise ValueError(
            f"No GRID nodes found in:\n{bdf_file}"
        )

    # --------------------------------------------------------
    # Read translational displacement results from PCH
    # --------------------------------------------------------
    displacement_by_id: dict[int, np.ndarray] = {}

    current_subcase: int | None = None
    in_displacement_section = False
    displacement_section_found = False

    result_section_names = (
        "DISPLACEMENT",
        "VELOCITY",
        "ACCELERATION",
        "SPC FORCE",
        "MPC FORCE",
        "LOAD VECTOR",
        "ELEMENT FORCE",
        "ELEMENT STRESS",
        "ELEMENT STRAIN",
    )

    with pch_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            # Process PCH header lines.
            if stripped.startswith("$"):
                upper_line = stripped.upper()

                subcase_match = _SUBCASE_PATTERN.search(stripped)

                if subcase_match:
                    current_subcase = int(
                        subcase_match.group(1)
                    )

                if "DISPLACEMENT" in upper_line:
                    in_displacement_section = True
                    displacement_section_found = True
                    continue

                if any(
                    section_name in upper_line
                    for section_name in result_section_names
                    if section_name != "DISPLACEMENT"
                ):
                    in_displacement_section = False

                continue

            if not in_displacement_section:
                continue

            if (
                subcase_id is not None
                and current_subcase is not None
                and current_subcase != subcase_id
            ):
                continue

            fields = stripped.replace(",", " ").split()

            # Typical PCH displacement row:
            #
            # GRID_ID  G  Ux  Uy  Uz
            #
            # The following continuation row contains rotations and
            # is intentionally ignored.
            if len(fields) < 5:
                continue

            if fields[1].upper() != "G":
                continue

            try:
                node_id = int(fields[0])

                displacement = np.array(
                    [
                        _nastran_float(fields[2]),
                        _nastran_float(fields[3]),
                        _nastran_float(fields[4]),
                    ],
                    dtype=np.float64,
                )
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"Invalid displacement record in "
                    f"{pch_file}, line {line_number}:\n"
                    f"{line.rstrip()}"
                ) from error

            # Repeated values are overwritten intentionally so that
            # the final nonlinear result state is retained.
            displacement_by_id[node_id] = displacement

    if not displacement_section_found:
        raise ValueError(
            "No displacement section was found in the PCH file. "
            "Ensure the model contains:\n"
            "DISPLACEMENT(PUNCH) = ALL"
        )

    if not displacement_by_id:
        raise ValueError(
            "No GRID displacement results were found for "
            f"subcase {subcase_id} in:\n{pch_file}"
        )

    # --------------------------------------------------------
    # Match PCH results to model GRID IDs
    # --------------------------------------------------------
    model_node_ids = set(original_coordinates_by_id)
    result_node_ids = set(displacement_by_id)

    common_node_ids = sorted(
        model_node_ids.intersection(result_node_ids)
    )

    if not common_node_ids:
        raise ValueError(
            "No matching GRID IDs were found between the "
            "NAS/BDF model and the PCH results."
        )

    node_ids = np.asarray(
        common_node_ids,
        dtype=np.int64,
    )

    original_coordinates = np.vstack(
        [
            original_coordinates_by_id[node_id]
            for node_id in common_node_ids
        ]
    )

    displacement = np.vstack(
        [
            displacement_by_id[node_id]
            for node_id in common_node_ids
        ]
    )

    displacement_magnitude = np.linalg.norm(
        displacement,
        axis=1,
    )

    deformed_coordinates = (
        original_coordinates + displacement
    )

    # --------------------------------------------------------
    # Save NumPy output
    # --------------------------------------------------------
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_file,
        node_ids=node_ids,
        original_coordinates=original_coordinates,
        displacement=displacement,
        displacement_magnitude=displacement_magnitude,
        deformed_coordinates=deformed_coordinates,
        subcase_id=np.asarray(
            -1 if subcase_id is None else subcase_id,
            dtype=np.int64,
        ),
    )

    missing_node_ids = model_node_ids - result_node_ids

    # print(f"Model GRID nodes: {len(model_node_ids)}")
    # print(f"PCH result nodes: {len(result_node_ids)}")
    # print(f"Exported nodes: {len(common_node_ids)}")
    # print(f"Nodes without PCH results: {len(missing_node_ids)}")
    # print(
    #     "Maximum displacement magnitude: "
    #     f"{displacement_magnitude.max():.6g}"
    # )
    print(f"Saved NumPy result: {output_file}")


if __name__ == "__main__":
    import pyvista as pv


    # op2_file_PATH = Path(r"C:\Users\val83626\Documents\GitLab\verformungsmodellierung_valentin\Projektarbeit\Simulation\Versuche\Results\simulation_001\simulation_001.OP2")
    # pch_file_PATH = Path(r"C:\Users\val83626\Documents\GitLab\verformungsmodellierung_valentin\Projektarbeit\Simulation\Versuche\Results\simulation_001\simulation_001.pch")
    # bdf_file_PATH = Path(r"C:\Users\val83626\Documents\GitLab\verformungsmodellierung_valentin\Projektarbeit\Inventor\base_model.nas")
    # out_file_PATH = Path(r"C:\Users\val83626\Documents\GitLab\verformungsmodellierung_valentin\Projektarbeit\Simulation\Versuche\Results\simulation_001\out.vtp")


    data = np.load(
        r"C:\Users\val83626\Documents\GitLab\Projektarbeit\Simulation\Results\baseline\Study\ReconstructedSurfaces\simulation_001_reconstruction.npz"
    )

    original = data["original_coordinates"]
    displacement = data["displacement"]
    magnitude = data["displacement_magnitude"]

    deformation_scale = 1.0
    displayed = original + deformation_scale * displacement

    cloud = pv.PolyData(displayed)
    cloud["Displacement magnitude"] = magnitude
    cloud["Displacement"] = displacement

    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud,
        scalars="Displacement magnitude",
        cmap="turbo",
        point_size=6,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()