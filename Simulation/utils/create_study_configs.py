"""Create study configurations containing a rectangular landmark grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def grid_values(start: float, stop: float, count: int) -> list[float]:
	"""Return evenly spaced values including both bounds."""
	if count < 1:
		raise ValueError("Grid counts must be positive.")
	if count == 1:
		return [float(start)]
	step = (stop - start) / (count - 1)
	return [float(start + index * step) for index in range(count)]


def create_study_config(
	*,
	name: str,
	x_min: float,
	x_max: float,
	x_count: int,
	z_min: float,
	z_max: float,
	z_count: int,
	nx: int,
	nz: int,
	energy_fraction: float | None,
	rank: int | None,
) -> dict[str, object]:
	"""Build a study configuration with nameless x/z grid landmarks."""
	if x_min > x_max or z_min > z_max:
		raise ValueError("Grid minimums must not exceed their maximums.")
	if nx < 1 or nz < 1:
		raise ValueError("Interpolation grid dimensions must be positive.")
	if energy_fraction is not None and not 0 < energy_fraction <= 1:
		raise ValueError("energy_fraction must be greater than 0 and at most 1.")
	if rank is not None and rank < 1:
		raise ValueError("rank must be positive.")
	if rank is not None and energy_fraction is not None:
		raise ValueError("Configure either rank or energy_fraction, not both.")


		
	landmarks = [
		{"x": x, "z": z}
		for x in grid_values(x_min, x_max, x_count)
		for z in grid_values(z_min, z_max, z_count)
	]
	config: dict[str, object] = {
		"name": name,
		"interpolation_grid": {"nx": nx, "nz": nz},
		"energy_fraction": 0.999 if energy_fraction is None else energy_fraction,
		"landmark_grid": {"l_total": x_count * z_count, "lx": x_count, "lz": z_count},
		"landmarks": landmarks,
	}

	return config


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create a nameless rectangular landmark-grid study config."
	)
	parser.add_argument("--name", default="study_config")
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--x-min", type=float, default=-1000.0)
	parser.add_argument("--x-max", type=float, default=0.0)
	parser.add_argument("--x-count", type=int, default=11)
	parser.add_argument("--z-min", type=float, default=-1000.0)
	parser.add_argument("--z-max", type=float, default=0.0)
	parser.add_argument("--z-count", type=int, default=11)
	parser.add_argument("--nx", type=int, default=51) # specify interpolation grid size
	parser.add_argument("--nz", type=int, default=51) # specify interpolation grid size
	selection = parser.add_mutually_exclusive_group()
	selection.add_argument("--energy-fraction", type=float)
	selection.add_argument("--rank", type=int)
	return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> Path:
	args = parse_args(arguments)
	output = args.output

	if output is None:
		output = (
			Path(__file__).resolve().parents[1]
			/ "StudyConfigs"
			/ f"{args.name}.json"
		)
	elif output.suffix.lower() != ".json":
		output = output / f"{args.name}.json"

	config = create_study_config(
		name=args.name,
		x_min=args.x_min,
		x_max=args.x_max,
		x_count=args.x_count,
		z_min=args.z_min,
		z_max=args.z_max,
		z_count=args.z_count,
		nx=args.nx,
		nz=args.nz,
		energy_fraction=args.energy_fraction,
		rank=args.rank,
	)
	output = output.resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", encoding="utf-8") as file:
		json.dump(config, file, indent=2)
		file.write("\n")
	print(f"Created {output} with {len(config['landmarks'])} landmarks.")
	return output






if __name__ == "__main__":
	main()
