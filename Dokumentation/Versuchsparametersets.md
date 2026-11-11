
## Simulation

### FEA Surface Generation

Plate Specs: 
- Plate Dimensions: 1m x 1m x 0.02m
- Clamp Diameter: 0.05m

Skript Name: *fea_surface_generation.py*
All surfaces are generated with a grid size of:
- **Plate-mesh-size = 50 mm**
- **Clamp-mesh-size = 20mm**

--> Finer Grid is better, but computationally more expensive.
--> Should do Experiments to explore the influence of mesh size vs accuracy 


Testing for: 
- Influence of the Amount of Displacement (use cl3 -> 1mm, 5mm, 10mm) 
- Influence of Number of Clamps on Absolute Error (Draw 3 Graphs in one Plot, using the 1,5,10 from cl1, cl3, cl9 respectively)

| ID  | (Config-)Name   | N_Clamps | N_Surfaces | Displacement [mm] | Rotation | Comment    |
| --- | --------------- | -------- | ---------- | ----------------- | -------- | ---------- |
| 1   | cl1_sf25_d1_r0  | 1        | 25         | 1                 | 0        |            |
| 2   | cl1_sf25_d5_r0  | 1        | 25         | 5                 | 0        |            |
| 3   | cl1_sf25_d10_r0 | 1        | 25         | 10                | 0        |            |
| 4   | cl3_sf25_d1_r0  | 3        | 25         | 1                 | 0        |            |
| 5   | cl3_sf25_d5_r0  | 3        | 25         | 5                 | 0        |            |
| 6   | cl3_sf25_d10_r0 | 3        | 25         | 10                | 0        |            |
| 7   | cl9_sf25_d1_r0  | 9        | 25         | 1                 | 0        |            |
| 8   | cl9_sf25_d5_r0  | 9        | 25         | 5                 | 0        |            |
| 9   | cl9_sf25_d10_r0 | 9        | 25         | 10                | 0        |            |
| 10  | cl9_sf5_d5_r0   | 9        | 100        | 1                 | 0        | Necessary? |

All generated sets of surfaces should be compared/evaluated with:

- 3x3 Landmark grid 
- 0.999 Energy Fraction
- 51x51 Interpolation Resolution?

Need 2 Additional Experiments:
- Influence of FEA-Mesh density
- influence of interpolation grid density
### Gappy-POD Studies

Do this on one of the the 3 (or 9) Clamp FEA Surface Result Sets 

Testing for: 
- Influence of Energy Fraction in calculation of POD basis
- Influence of Surfaces used in calculation of POD basis
- Influence of Number of Landmarks
- Influence of Position of Landmarks

| ID  | Testing Effect of  | Script-Name                 | Comment                                                 | What does it Show? |
| --- | ------------------ | --------------------------- | ------------------------------------------------------- | ------------------ |
| 1   | Energy_Fraction    | *study_pod_modes.py*        | using 3x3 landmark grid                                 |                    |
| 2   | Training Surfaces  | *study_pod_surfaces.py*     | using 3x3 landmark grid                                 |                    |
| 3   | Landmarks (Grid)   | *study_landmarks_grid.py*   | varying the size of the landmark grid                   |                    |
| 4   | Landmarks (Random) | *study_landmarks_random.py* | varying the amount of landmarks at random positions     |                    |
| 5   | Landmarks (Offset) | *study_landmarks_offset.py* | Varying Landmark amount and Offset from Sauger Position |                    |
|     |                    |                             |                                                         |                    |

