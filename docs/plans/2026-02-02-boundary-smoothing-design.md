# K_ref Boundary Smoothing Design

## Problem

`SegmentedMultiKRefSurface` interpolates across independently-fit B-spline
surfaces using Catmull-Rom in log(K_ref) space. Three edge cases produce
artificial kinks:

1. **< 4 K_ref points:** falls back to piecewise linear (C0 only)
2. **Edge clamping:** duplicating edge indices degrades Catmull-Rom to quadratic
3. **Outside K_ref range:** hard switch to nearest surface (discontinuous derivative)

## Fixes

### 1. Monotone Hermite for < 4 points

Replace the piecewise linear fallback in `interp_across_krefs()` with
Fritsch-Carlson monotone Hermite interpolation. For 3 points this gives C1
continuity while preserving monotonicity. For 2 points, keep linear.

Inline implementation (~20 lines), no new utility file.

### 2. Virtual edge points for Catmull-Rom

Replace index clamping (`i0=i1` or `i3=i2`) with linearly extrapolated
virtual points:

- Left edge: `x_v = 2*x[1] - x[2]`, `y_v = 2*y[1] - y[2]`
- Right edge: `x_v = 2*x[2] - x[1]`, `y_v = 2*y[2] - y[1]`

Preserves cubic character at boundary intervals.

### 3. Smooth extrapolation outside K_ref range

Let Catmull-Rom naturally extrapolate (up to ~1 K_ref spacing) instead of
hard-switching to nearest surface. Beyond the extrapolation limit, clamp
to nearest.

### 4. Boundary disagreement metric (test-only)

Free function in the test file that evaluates adjacent surfaces at shared
K_ref boundaries across a grid of (tau, sigma, rate) and reports max/mean
normalized price difference. Diagnostic for deciding if overlap blending
(future Option B) is needed.

## Files

- `src/option/table/segmented_multi_kref_surface.cpp` -- fixes 1, 2, 3
- `tests/segmented_multi_kref_surface_test.cc` -- fix 4, smoothness tests

## Testing

- Numerical derivative checks: verify C1 continuity across K_ref boundaries
- Boundary disagreement metric: quantify cross-surface mismatch
- Regression: existing tests pass
- IV solver integration: Newton convergence unchanged or improved
