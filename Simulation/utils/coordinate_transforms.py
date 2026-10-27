"""Coordinate conversions shared by Inventor automation and FEA studies."""

from __future__ import annotations


NASTRAN_POSITION_SCALE = -1.0


def suction_position_in_nastran_coordinates(
    position_x: float,
    position_z: float,
) -> tuple[float, float]:
    """Convert an Inventor suction position to Nastran x/z coordinates."""
    return (
        NASTRAN_POSITION_SCALE * position_x,
        NASTRAN_POSITION_SCALE * position_z,
    )