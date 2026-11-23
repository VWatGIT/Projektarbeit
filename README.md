Using python 3.12 because pynastran requires 3.9 - 3.12

Parameter sweep configuration:

- Add saugers to the `saugers` array with `nominal_x` and `nominal_z` coordinates. The array may contain any number of saugers.
- Use `variation.position` for the global x/y/z settings: `vary_position`, `deviation_x`, `deviation_y`, and `deviation_z`. The y deviation produces displacement around zero.
- Use `variation.rotation` for all rotation axes: `vary_rotation`, `deviation_x`, `deviation_y`, and `deviation_z`. Values are sampled independently for each sauger and symmetrically around zero.
- Set `vary_position` or `vary_rotation` to `false` to use zero deviations.
- Generated CSV columns use `s1_x`, `s1_z`, `sauger_1_displacement`, and `sauger_1_rotation_x/y/z`, followed by the same pattern for each additional sauger.

Study landmark grids:

Use `Simulation/utils/create_study_configs.py` to create a study config without manually listing landmarks. For example:

```powershell
python Simulation/utils/create_study_configs.py --name landmarks_grid --x-min -800 --x-max -200 --x-count 4 --z-min -800 --z-max -200 --z-count 4 --energy-fraction 0.999
```

The generated landmarks contain only `x` and `z`. Customize the interpolation grid with `--nx` and `--nz`, reconstruction rank with `--rank` or `--energy-fraction`, paths with `--results-dir` and `--output-dir`, and the output file with `--output`.

Surface-generation comparison:

After running the grid study for `cl1_sf25_d1_r0`, `cl1_sf25_d5_r0`, `cl3_sf25_d5_r0`,
`cl9_sf25_d5_r0`, `cl3_sf25_d1_r0`, `cl3_sf25_d10_r0`,
and `cl9_sf25_d1_r0`, run:

```powershell
python .\Simulation\compare_surface_generation.py
```

The script reads `Study\landmarks_grid\grid_comparison.csv` under each
experiment in `Simulation\Results` and writes nine PNGs to
`Simulation\Comparison`: three error summaries for each of the clamp, cl3
grid-size, and cl9 grid-size comparisons. The relative L2 plots show the
maximum held-out-case relative L2 error as a percentage. For each held-out
case, the absolute error is its largest residual across all points and
displacement components. The absolute plots show either the maximum or the
arithmetic mean of those per-case maxima, in model displacement units. The
latter is not the mean absolute residual over all points.
The clamp comparison shows 1 mm and 5 mm displacement as separate curves
using a 3x3 landmark grid.
The grid-size comparisons show d1, d5, and d10 for cl3, and d1 and d5 for cl9.
Within each comparison, studies must contain the same grid sizes and finite
errors for every case.
Dashed lines show the flat-surface baseline (zero estimated displacement),
computed from the current successful FEM cases resampled using each grid
study's recorded interpolation resolution. The baseline uses the same metric
and case aggregation as its reconstruction curve; the relative L2 baseline
is 100% for nonzero fields. Grid-size baselines are horizontal; clamp-count
baselines vary by clamp count. The script checks case counts against the
study, but the grid summary does not record every case identity: rerun the
study if FEM results or case membership have changed.
Use `--plot clamps` or `--plot grids` to generate only one comparison,
`--metric relative`, `--metric absolute_max`, or `--metric absolute_mean` to
select a summary (`--metric absolute` remains available for absolute L2),
`--grid-clamps 3` or `--grid-clamps 9` to select a grid-size comparison,
`--study` to select a different grid-study directory, and `--results-dir` or
`--output-dir` to override the default paths. Missing or invalid inputs are
reported rather than silently omitted.

Also use Inventor 2023 + Nastran 2023

Nastran Setup:
Inventor Nastran Environment -> Default settings -> NAstran File Settings -> Auto Update Nastran File
Parameters -> Output Control Directives -> RSLTFILETYPE: NASTRANBINARY



Notes to self:
- Later compare improvement of Gappy-POD surface with just working with the CAD/wrong surface
- with 0 displacement, the Gappy-POD should interpolate perfectly using a single Landmark

Error:
eTotal ≈ eRegistration + ePOD + eGappy
​



POD in Data Centers:
https://www.sciencedirect.com/science/article/pii/S1359431124029788


POD-Zusammenfassung:
https://www.sciencedirect.com/science/article/pii/S305051782600078X




Mehr Sauger -> lokale Deformation (hochfrequent)
Weniger Sauger -> Globale Deformation (niederfrequent)



Anzahl der Moden ist bahängig von

    - Anzahl der Sauger
    - Displacment Größenordnung
    - Materialverhalten (Eigensteifigkeit)