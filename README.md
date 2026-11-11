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