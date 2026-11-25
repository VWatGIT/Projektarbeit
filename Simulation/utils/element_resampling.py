"""Interpolate planar quadratic Nastran elements using their shape functions."""

from pathlib import Path

import numpy as np
from pyNastran.bdf.bdf import read_bdf

from .surface_resampling import ResampledSurface, SurfaceGrid


def _shape_functions(kind: str, natural: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s = natural[:, 0], natural[:, 1]
    if kind == "CTRIA6":
        bary = np.column_stack((1 - r - s, r, s))
        derivatives = np.array([[-1., -1.], [1., 0.], [0., 1.]])
        values = np.column_stack(
            (bary * (2 * bary - 1), 4 * bary[:, 0] * bary[:, 1],
             4 * bary[:, 1] * bary[:, 2], 4 * bary[:, 2] * bary[:, 0])
        )
        gradients = np.empty((len(r), 6, 2))
        gradients[:, :3] = (4 * bary - 1)[:, :, None] * derivatives
        for i, (a, b) in enumerate(((0, 1), (1, 2), (2, 0)), start=3):
            gradients[:, i] = 4 * (
                bary[:, a, None] * derivatives[b] + bary[:, b, None] * derivatives[a]
            )
        return values, gradients

    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    a = 1 + r[:, None] * signs[:, 0]
    b = 1 + s[:, None] * signs[:, 1]
    c = r[:, None] * signs[:, 0] + s[:, None] * signs[:, 1] - 1
    corners = 0.25 * a * b * c
    edges = np.column_stack((
        0.5 * (1 - r**2) * (1 - s),
        0.5 * (1 + r) * (1 - s**2),
        0.5 * (1 - r**2) * (1 + s),
        0.5 * (1 - r) * (1 - s**2),
    ))
    gradients = np.empty((len(r), 8, 2))
    gradients[:, :4, 0] = 0.25 * signs[:, 0] * b * (c + a)
    gradients[:, :4, 1] = 0.25 * signs[:, 1] * a * (c + b)
    gradients[:, 4:, 0] = np.column_stack((
        -r * (1 - s), 0.5 * (1 - s**2), -r * (1 + s), -0.5 * (1 - s**2)
    ))
    gradients[:, 4:, 1] = np.column_stack((
        -0.5 * (1 - r**2), -(1 + r) * s, 0.5 * (1 - r**2), -(1 - r) * s
    ))
    return np.column_stack((corners, edges)), gradients


def _control_points(kind: str, coordinates: np.ndarray) -> np.ndarray:
    """Bound the quadratic geometry by its Bernstein control points."""
    if kind == "CTRIA6":
        return np.vstack((
            coordinates[:3],
            2 * coordinates[3] - 0.5 * (coordinates[0] + coordinates[1]),
            2 * coordinates[4] - 0.5 * (coordinates[1] + coordinates[2]),
            2 * coordinates[5] - 0.5 * (coordinates[2] + coordinates[0]),
        ))
    center = 0.5 * coordinates[4:].sum(axis=0) - 0.25 * coordinates[:4].sum(axis=0)
    samples = np.array([
        [coordinates[0], coordinates[7], coordinates[3]],
        [coordinates[4], center, coordinates[6]],
        [coordinates[1], coordinates[5], coordinates[2]],
    ])
    control = samples.copy()
    control[1] = 2 * samples[1] - 0.5 * (samples[0] + samples[2])
    control[:, 1] = 2 * control[:, 1] - 0.5 * (control[:, 0] + control[:, 2])
    return control.reshape(-1, 2)


def _interpolate_element(
    kind: str,
    coordinates: np.ndarray,
    displacement: np.ndarray,
    points: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "CTRIA6":
        starts = ((1 / 3, 1 / 3), (0.05, 0.05), (0.9, 0.05), (0.05, 0.9))
    else:
        starts = ((0., 0.), (-0.75, -0.75), (0.75, -0.75), (0.75, 0.75), (-0.75, 0.75))
    found = np.zeros(len(points), dtype=bool)
    interpolated = np.zeros((len(points), 3))
    scale = max(float(np.ptp(coordinates, axis=0).max()) ** 2, 1e-30)
    for start in starts:
        remaining = np.flatnonzero(~found)
        if not remaining.size:
            break
        targets = points[remaining]
        natural = np.tile(start, (len(remaining), 1))
        for _ in range(30):
            values, gradients = _shape_functions(kind, natural)
            residual = values @ coordinates - targets
            jacobian = np.einsum("ni,pnj->pij", coordinates, gradients)
            determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
            solvable = np.abs(determinant) > np.finfo(float).eps * scale
            if not np.any(solvable) or np.all(np.linalg.norm(residual[solvable], axis=1) <= tolerance):
                break
            delta = np.zeros_like(natural)
            delta[solvable, 0] = (
                jacobian[solvable, 1, 1] * residual[solvable, 0]
                - jacobian[solvable, 0, 1] * residual[solvable, 1]
            ) / determinant[solvable]
            delta[solvable, 1] = (
                -jacobian[solvable, 1, 0] * residual[solvable, 0]
                + jacobian[solvable, 0, 0] * residual[solvable, 1]
            ) / determinant[solvable]
            delta /= np.maximum(1., np.linalg.norm(delta, axis=1))[:, None]
            natural -= delta

        values, _ = _shape_functions(kind, natural)
        r, s = natural[:, 0], natural[:, 1]
        if kind == "CTRIA6":
            inside = (r >= -1e-9) & (s >= -1e-9) & (r + s <= 1 + 1e-9)
        else:
            inside = (np.abs(r) <= 1 + 1e-9) & (np.abs(s) <= 1 + 1e-9)
        inside &= np.linalg.norm(values @ coordinates - targets, axis=1) <= tolerance
        selected = remaining[inside]
        found[selected] = True
        interpolated[selected] = values[inside] @ displacement
    return found, interpolated


def _validate_geometry(kind: str, coordinates: np.ndarray, element_id: int, nas_file: Path) -> None:
    if kind == "CTRIA6":
        samples = np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5], [1 / 3, 1 / 3]])
    else:
        samples = np.array([[r, s] for r in (-1., 0., 1.) for s in (-1., 0., 1.)])
    _, gradients = _shape_functions(kind, samples)
    jacobian = np.einsum("ni,pnj->pij", coordinates, gradients)
    determinants = np.linalg.det(jacobian)
    scale = max(float(np.ptp(coordinates, axis=0).max()) ** 2, 1e-30)
    if np.any(np.abs(determinants) <= np.finfo(float).eps * scale) or (
        np.any(determinants > 0) and np.any(determinants < 0)
    ):
        raise ValueError(f"{nas_file}: element {element_id} has a singular or folded geometry.")


def resample_elements(
    nas_file: Path,
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    displacement: np.ndarray,
    grid: SurfaceGrid,
    source_case: str | None,
) -> ResampledSurface:
    if not nas_file.is_file():
        raise FileNotFoundError(f"Nastran model not found: {nas_file}")
    model = read_bdf(str(nas_file), xref=True, punch=False, debug=False)
    unsupported = sorted({element.type for element in model.elements.values()} - {"CQUAD8", "CTRIA6"})
    if unsupported:
        raise ValueError(f"{nas_file}: unsupported element types: {', '.join(unsupported)}")
    if not model.elements:
        raise ValueError(f"{nas_file}: no supported surface elements found.")
    ids = np.asarray(node_ids)
    if ids.ndim != 1 or len(ids) != len(coordinates) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{nas_file}: result node IDs must be unique and match the coordinates.")
    lookup = {int(node_id): i for i, node_id in enumerate(ids)}
    if coordinates.shape != displacement.shape or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates and displacement must have matching shape (n, 3).")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(displacement)):
        raise ValueError(f"{nas_file}: result coordinates and displacements must be finite.")

    output = np.full((grid.point_count, 3), np.nan)
    valid = np.zeros(grid.point_count, dtype=bool)
    for element in model.elements.values():
        element_ids = element.node_ids
        if any(node_id not in lookup for node_id in element_ids):
            raise ValueError(f"{nas_file}: element {element.eid} has nodes missing from displacement results.")
        indices = [lookup[node_id] for node_id in element_ids]
        model_coordinates = np.array([model.nodes[node_id].get_position() for node_id in element_ids])
        if not np.allclose(coordinates[indices], model_coordinates, rtol=1e-9, atol=1e-8):
            raise ValueError(f"{nas_file}: element {element.eid} coordinates differ from the result archive.")
        xy = coordinates[indices][:, grid.axis_indices]
        _validate_geometry(element.type, xy, element.eid, nas_file)
        controls = _control_points(element.type, xy)
        tolerance = max(float(np.ptp(controls, axis=0).max()), 1.) * 1e-9
        candidates = np.flatnonzero(
            ~valid
            & np.all(grid.coordinates_2d >= controls.min(axis=0) - tolerance, axis=1)
            & np.all(grid.coordinates_2d <= controls.max(axis=0) + tolerance, axis=1)
        )
        if not candidates.size:
            continue
        inside, interpolated = _interpolate_element(
            element.type, xy, displacement[indices], grid.coordinates_2d[candidates], tolerance
        )
        selected = candidates[inside]
        output[selected] = interpolated[inside]
        valid[selected] = True
    if not valid.any():
        raise ValueError(f"{nas_file}: no interpolation grid points lie inside the supported elements.")
    return ResampledSurface(grid.coordinates_2d, output, valid, source_case)
