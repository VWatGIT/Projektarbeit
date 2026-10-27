# Error Metrics for Landmark Studies

## Evaluation Context

Each study uses leave-one-out Gappy POD. One FEM simulation is held out, a POD basis is fitted to the remaining simulations, and the held-out displacement field is reconstructed from weighted landmark observations. Errors are evaluated over all valid points on the fixed resampled surface grid and all three displacement components, unless stated otherwise.

POD snapshots are displacement fields relative to the undeformed surface. The basis is fitted directly to the un-centered displacement snapshots, with a zero reference field; reconstructed displacement is a linear combination of modes with no mean-deformation offset. The physical reconstructed surface is the undeformed coordinates plus that displacement.

For each landmark-grid size, the comparison CSV and plot report the arithmetic mean of the per-case metrics across the same held-out FEM simulations. The grid sizes change the number of landmark observations; the interpolation grid and FEM case set stay fixed.

The grid-comparison plot shades the minimum-to-maximum range of finite held-out-case errors at each grid size around the mean curve. Its summary CSV records those bounds as `min_case_*` and `max_case_*` columns. This range describes case-to-case variation; it is not a confidence interval.

For a sauger-centered offset study, case metrics are first averaged across held-out FEM cases within each landmark placement. The radius summary then reports the arithmetic mean and population standard deviation of those placement-level means across successful trials at that radius. Failed out-of-domain placements are recorded but excluded from the aggregate metrics.

In the offset plot, error bars show plus or minus one population standard deviation across trial means. The shaded band shows the minimum-to-maximum range of held-out-case means, where each case's metric is first averaged across successful trials at that radius. The `summary.csv` includes both spread measures for each plotted metric.

## Full-Field Error Metrics

Let $u_{p,c}^{\mathrm{ref}}$ be the FEM reference displacement and $u_{p,c}^{\mathrm{rec}}$ the reconstructed displacement at valid grid point $p$ and component $c \in \{x,y,z\}$.

### Absolute L2 Displacement Error

`absolute_l2_error` is the Euclidean norm of the displacement residual over the complete valid surface:

$$
E_{\mathrm{abs}} =
\sqrt{\sum_{p,c}\left(u_{p,c}^{\mathrm{ref}} - u_{p,c}^{\mathrm{rec}}\right)^2}
$$

It has displacement units (for example, millimetres). It is a whole-field norm, not the mean error per grid point.

### Relative L2 Displacement Error

`relative_l2_error` normalizes the absolute L2 error by the norm of the reference displacement field:

$$
E_{\mathrm{rel}} =
\frac{\left\|u^{\mathrm{ref}} - u^{\mathrm{rec}}\right\|_2}
     {\left\|u^{\mathrm{ref}}\right\|_2}
$$

It is dimensionless and is displayed as a percentage in the plot. The code computes this ratio separately for each held-out case and then averages those ratios. If the reference field has exactly zero norm, the implementation returns the absolute error as a fallback; in that edge case the fallback is not dimensionless.

### Maximum Absolute Displacement Error

`maximum_absolute_error` is the largest absolute residual among all valid grid points and displacement components:

$$
E_{\max,\mathrm{abs}} =
\max_{p,c}\left|u_{p,c}^{\mathrm{ref}} - u_{p,c}^{\mathrm{rec}}\right|
$$

It has displacement units and identifies the worst single point-component error. The comparison plot shows the mean of this per-case maximum across held-out simulations.

### Maximum Relative Displacement Error

`maximum_relative_error` is the largest component-wise relative residual:

$$
E_{\max,\mathrm{rel}} =
\max_{p,c}\frac{\left|u_{p,c}^{\mathrm{ref}} - u_{p,c}^{\mathrm{rec}}\right|}
                      {\left|u_{p,c}^{\mathrm{ref}}\right|}
$$

Components with a zero reference and zero residual contribute zero. A zero reference with nonzero residual produces infinity. This metric is computed in the per-study metrics files but is not included in the grid-comparison plot because it can be dominated by near-zero reference components.

## Deformed-Surface Error

`deformed_coordinate_relative_l2_error` compares the FEM-deformed coordinates with the reconstructed-deformed coordinates. Let $X_p$ be an original coordinate and $u_p$ a displacement:

$$
E_{\mathrm{deformed}} =
\frac{\left\|(X + u^{\mathrm{ref}}) - (X + u^{\mathrm{rec}})\right\|_2}
     {\left\|X + u^{\mathrm{ref}}\right\|_2}
$$

The original coordinates cancel in the numerator, so that numerator equals the absolute displacement residual. The denominator is the norm of the reference deformed coordinates, over valid points and all three spatial coordinates. A small value can therefore reflect the large scale of the plate coordinates relative to its displacement, rather than a small displacement-relative error. Because the denominator uses absolute coordinates, this metric depends on the coordinate origin. The implementation floors the denominator at $10^{-12}$ to avoid division by zero.

The offset study also writes `true_surface_vs_fem_surface_metrics.json`. It compares the undeformed resampled coordinates (`true_surface`) with those coordinates plus each FEM displacement field (`fem_surface`), then reports the arithmetic mean of the four plotted metrics over all FEM cases. For the displacement-relative metric, the estimate is zero displacement, so the error is 1.0 whenever the reference displacement norm is nonzero. The deformed-coordinate-relative metric instead normalizes by the norm of the FEM-deformed coordinates.

## Reconstruction Diagnostics

These are saved with each study but are not the full-field error curves in the comparison plot:

- `observation_residual`: norm of the mismatch between the landmark observations and the POD reconstruction at those observations. It measures the fit at sampled locations, not accuracy over the whole surface.
- `least_squares_residual`: residual reported by the least-squares solve used to find POD coefficients. It describes the coefficient fit to the observations, not the full-field displacement error.
- `condition_number` and `rank`: diagnostics for the sampled POD-mode system. They help identify poorly conditioned or underdetermined reconstructions; they are not error magnitudes.

## Implementation References

- `utils/gappy_pod.py`: `reconstruction_errors()` and `relative_l2_error()`.
- `utils/study_pipeline.py`: leave-one-out evaluation and deformed-coordinate error.
- `run_grid_comparison.py`: metric selection and averaging across held-out cases.
- `run_sauger_landmark_study.py`: per-placement and per-radius aggregation for random offsets.