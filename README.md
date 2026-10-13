Using python 3.12 because pynastran requires 3.9 - 3.12

Parameter sweep variation controls:

- Set `vary_position` to `false` for `s1` or `s2` to use its nominal coordinates exactly.
- Set `vary_displacement` to `false` for `displacement_y` or `vary_rotation` to `false` for a rotation axis, and provide its `fixed_value`.
- Omitted flags retain the default varying behavior.

Also use Inventor 2023 + Nastran 2023

Nastran Setup:
Inventor Nastran Environment -> Default settings -> NAstran File Settings -> Auto Update Nastran File
Parameters -> Output Control Directives -> RSLTFILETYPE: NASTRANBINARY



Notes to self:
Later compare improvement of Gappy-POD surface with just working with the CAD/wrong surface

Error:
eTotal ≈ eRegistration + ePOD + eGappy
​



POD in Data Centers:
https://www.sciencedirect.com/science/article/pii/S1359431124029788


POD-Zusammenfassung:
https://www.sciencedirect.com/science/article/pii/S305051782600078X


