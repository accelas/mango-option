# 3D Dimensionless Interpolation

## Summary

Add 3D dimensionless interpolation surfaces alongside the existing 4D approach.
The dimensionless coordinate transform collapses (σ, r) into κ = 2r/σ², reducing
the interpolation from 4D (ln S/K, τ, σ, r) to 3D (x, τ', ln κ). Both B-spline
and Chebyshev-Tucker backends are supported. Reference code lives in the
`experiment/chebyshev-tensor` worktree.

**Trade-off:** Fewer PDE solves at build time (N_κ vs N_σ × N_r), but σ/r coupling
limits accuracy to ~312 bps RMS.

## New Components

### 1. `DimensionlessTransform3D`

**File:** `src/option/table/transforms/dimensionless_3d.hpp`

Satisfies the `CoordinateTransform` concept with `kDim = 3`.

Coordinate mapping:
- x = ln(S/K)
- τ' = σ²τ/2
- ln κ = ln(2r/σ²)

Vega weights (chain rule through dimensionless coords):
- [0, στ, −2/σ]

Stateless struct, ~20 lines.

### 2. 3D Type Aliases

Compose with the existing layer stack:

```
// B-spline 3D
BSpline3DTransformLeaf = TransformLeaf<SharedBSplineInterp<3>, DimensionlessTransform3D>
BSpline3DLeaf          = EEPLayer<BSpline3DTransformLeaf, AnalyticalEEP>
BSpline3DPriceTable    = PriceTable<BSpline3DLeaf>

// Chebyshev 3D
Chebyshev3DTransformLeaf = TransformLeaf<ChebyshevInterpolant<3, TuckerTensor<3>>, DimensionlessTransform3D>
Chebyshev3DLeaf          = EEPLayer<Chebyshev3DTransformLeaf, AnalyticalEEP>
Chebyshev3DPriceTable    = PriceTable<Chebyshev3DLeaf>
```

All use K_ref = 1 (surface is moneyness-normalized). SplitSurface variants
compose on top for discrete dividends.

### 3. Builder Pipeline

**Files:**
- `src/option/table/dimensionless/dimensionless_builder.hpp`
- `src/option/table/dimensionless/dimensionless_builder.cpp`

**Input:**

```cpp
struct DimensionlessAxes {
    std::vector<double> log_moneyness;  // x = ln(S/K)
    std::vector<double> tau_prime;       // τ' = σ²τ/2
    std::vector<double> ln_kappa;        // ln κ = ln(2r/σ²)
};
```

**Build process:**

1. For each ln_kappa[i], compute κ = exp(ln_kappa[i])
2. Solve one PDE with σ_eff = √2, r_eff = κ, q = 0
3. Snapshot at all τ' grid points along the log-moneyness grid
4. Apply EEP decomposition: subtract dimensionless European price
5. Collect into 3D tensor eep[x][τ'][ln_κ]
6. Fit surface — B-spline (PriceTableSurfaceND<3>) or Chebyshev (ChebyshevInterpolant<3, TuckerTensor<3>>)

**Cost:** N_κ PDE solves (e.g., 10–15 vs N_σ × N_r = 20+ for 4D).

**Adaptive refinement** reuses `run_refinement()` from `adaptive_refinement.hpp`.
Segments the ln κ axis into ~3-unit chunks to avoid oscillations (same as
experiment).

### 4. Factory Integration

New path variant for `make_interpolated_iv_solver`:

```cpp
struct DimensionlessIVPath {
    double maturity;
    double kappa_min = 0.01;   // min 2r/σ² (low rate, high vol)
    double kappa_max = 10.0;   // max 2r/σ² (high rate, low vol)
    std::vector<Dividend> discrete_dividends = {};
};
```

The factory dispatches on path type:
- StandardIVPath → 4D (existing)
- SegmentedIVPath → 4D segmented (existing)
- DimensionlessIVPath → 3D builder → BSpline3DPriceTable or Chebyshev3DPriceTable

Interpolation backend (B-spline vs Chebyshev) selected via config enum.

## Testing

1. **Unit tests** — DimensionlessTransform3D coord mapping and vega weights
2. **Builder tests** — construct 3D surfaces, verify against direct PDE solves
   (both B-spline and Chebyshev backends, ~10–20s each)
3. **IV accuracy test** — end-to-end: build → InterpolatedIVSolver → IV sweep →
   compare against FDM IV. Validates full chain including vega chain rule.
4. **Benchmark** — head-to-head 3D vs 4D build cost and query accuracy

Expected accuracy ceiling: ~312 bps RMS from σ/r coupling.

## File Plan

```
src/option/table/
├── transforms/
│   ├── standard_4d.hpp              # existing
│   └── dimensionless_3d.hpp         # NEW
├── dimensionless/
│   ├── dimensionless_builder.hpp    # NEW
│   ├── dimensionless_builder.cpp    # NEW
│   └── dimensionless_european.hpp   # NEW — European pricing in dim-less coords
├── bspline/
│   └── bspline_3d_surface.hpp       # NEW — 3D type aliases
├── chebyshev/
│   └── chebyshev_3d_surface.hpp     # NEW — 3D type aliases

tests/
├── dimensionless_transform_test.cc  # NEW
├── dimensionless_builder_test.cc    # NEW
├── dimensionless_iv_test.cc         # NEW

benchmarks/
├── dimensionless_vs_4d.cc           # NEW
```

## Reference

Experiment code: `.worktrees/chebyshev-tensor/src/option/table/dimensionless/`
