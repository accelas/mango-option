# Fix AdaptiveGridBuilder refinement metric

## Problem

The `AdaptiveGridBuilder` declares convergence at 2 bps while actual
interpolation IV errors reach 30–3800 bps.  Three defects cause this:

1. **Wrong vega.**  The metric uses European BS vega, which overstates
   sensitivity for deep ITM American puts.  The builder under-refines
   exactly where errors are worst.

2. **Skipped samples.**  When vega falls below `vega_floor`, the metric
   returns `nullopt` and the sample vanishes from convergence checks.
   The builder cannot see errors in low-vega regions.

3. **Ignored maturity knots.**  `build_standard_adaptive` extracts
   min/max from `StandardIVPath.maturity_grid` and creates a 5-point
   linspace.  The user's maturity grid is discarded.

## Changes

### A. `compute_error_metric` — new signature

```cpp
// Before: std::optional<double>, 8 params, internal bs_vega
// After:  double, 2 params, caller passes vega
double compute_error_metric(double price_error, double vega) const
{
    double vega_clamped = std::max(std::abs(vega), params_.vega_floor);
    double iv_error = price_error / vega_clamped;

    // Cap when price is already within tolerance.
    // Prevents FD noise from driving runaway refinement.
    double price_tol = params_.target_iv_error * params_.vega_floor;
    if (price_error <= price_tol) {
        iv_error = std::min(iv_error, params_.target_iv_error);
    }
    return iv_error;
}
```

Return type changes from `std::optional<double>` to `double`.
No sample is ever skipped.

### B. Standard path — American vega via finite difference

In `run_refinement`, after the existing FD solve at σ, add two
bump solves:

```cpp
double eps = std::max(1e-4, 0.01 * sigma);
auto fd_up = solve_fn(spot, strike, tau, sigma + eps, rate);
auto fd_dn = solve_fn(spot, strike, tau, sigma - eps, rate);

double vega;
if (fd_up.has_value() && fd_dn.has_value()) {
    vega = (fd_up.value() - fd_dn.value()) / (2.0 * eps);
} else {
    // FD bump failed — use floor directly so low-vega regions
    // still drive refinement (do NOT fall back to BS vega).
    vega = 0.0;  // clamped to vega_floor inside compute_error_metric
}
```

Adaptive ε avoids nonlinearity bias at high σ and FD noise at low σ.
When bump solves fail (extreme parameters), vega falls to 0 and the
metric clamps to `vega_floor` — the sample still counts.

### C. Segmented path — keep BS vega

`build_segmented` runs 2–3 independent probe refinement loops.
Adding FD vega triples probe cost (10–40 s extra).  Not justified
when the dominant dividend error comes from the Catmull-Rom
K_ref interpolation, not the per-probe grid.

The segmented path continues to use BS vega but removes the
`nullopt` skip.  Its `compute_error_fn` lambda becomes:

```cpp
auto compute_error_fn = [this](double interp, double ref,
                                double spot, double strike, double tau,
                                double sigma, double rate,
                                double div_yield) -> double {
    double price_error = std::abs(interp - ref);
    double vega = bs_vega(spot, strike, tau, sigma, rate, div_yield);
    return compute_error_metric(price_error, vega);
};
```

### D. Seed maturity grid with user knots

In `build_standard_adaptive` (`iv_solver_factory.cpp`), pass
`path.maturity_grid` to the adaptive builder as the initial tau grid.

`run_refinement` gains an optional parameter for the initial tau grid.
When provided, it replaces the 5-point linspace.  The builder still
refines tau by inserting midpoints when tau is the worst dimension.

Only tau seeding changes.  Moneyness, vol, and rate continue to
start from linspace within domain bounds.

### E. Caller updates

All callers of `compute_error_metric`:

| Caller | Vega source | Change |
|--------|------------|--------|
| `run_refinement` (standard) | FD American vega | New: 2 extra PDE solves per sample |
| `run_refinement` (segmented compute_error_fn) | BS vega | Remove nullopt only |
| `build_segmented` final validation | BS vega | Remove nullopt only |

All `if (!err.has_value()) continue;` become unconditional.

### F. No public API changes

`AdaptiveGridParams`, `IVSolverFactoryConfig`, and
`make_interpolated_iv_solver()` are unchanged.

## Cost

Standard path: +640 PDE solves per build (5 iterations × 64 samples
× 2 vega bumps).  At 5–20 ms each, adds 3–13 s to a 30–120 s build.

Segmented path: no extra PDE solves (BS vega retained).

## Verification

Re-run `benchmarks/interp_iv_safety` before and after.  Target:
vanilla T≥1y errors drop from 10–150 bps to ≤5 bps; short-maturity
errors improve but may remain elevated (fundamental vega limitation).

## Files to change

- `src/option/table/adaptive_grid_builder.hpp` — signature
- `src/option/table/adaptive_grid_builder.cpp` — metric, validation loop, tau seeding
- `src/option/iv_solver_factory.cpp` — pass maturity_grid to builder
