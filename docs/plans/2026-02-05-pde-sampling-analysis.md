# PDE Sampling Pipeline Analysis

> Codex analysis of why interpolated IV errors remain 500-1500 bps despite adaptive grid metric fixes.

## Context

After implementing the adaptive grid metric fix (FD American vega, no nullopt skip, all-dimension seeding), benchmark errors remained high:
- Vanilla σ=15%: 649 bps RMS
- Vanilla σ=30%: 1448 bps RMS
- Dividend σ=15%: 695 bps RMS
- Dividend σ=30%: 501 bps RMS

## How PDE Sampling Works

The price table extraction has two interpolation stages:

1. **Maturity snapshots**: `set_snapshot_times()` captures PDE solution at specific maturities during time-stepping

2. **Spatial resampling**: For each maturity, `CubicSpline` interpolates the PDE's spatial grid (200-400 points) onto our moneyness grid (typically 5-20 points)

3. **4D B-spline fitting**: The sampled tensor is fit to a 4D B-spline surface

## Key Findings

### 1. Information Loss is Significant

Yes, it can be **material**—especially for American options where the value surface has **high curvature / kinks near the exercise boundary** in log-moneyness. Two smoothers back-to-back:
1. Natural cubic spline down to ~5–20 samples
2. 4D B-spline fit on those samples

That's a **strong low-pass filter**. If the surface has localized curvature (short τ, low vol, deep ITM/OTM), 20 samples is often not enough to preserve the structure that drives IV.

Where it shows up in code: the moneyness grid used for sampling is `axes.grids[0]` and is often small. The PDE grid `x_grid` is not retained beyond the spline evaluation. (`src/option/table/price_table_builder.cpp`)

### 2. CubicSpline Boundary Issues

Yes. The spline is **natural** (second derivative forced to zero at both ends) and **extrapolates** outside the x-grid using the last polynomial segment. This combination is well-known to distort boundary behavior for financial payoffs because:
- Near the boundaries, option values are **not** naturally zero-curvature in log-moneyness
- American early-exercise regions add non-smoothness that natural splines can overshoot
- Extrapolation can amplify any mismatch near the boundaries

See `CubicSpline::eval()` and boundary config in `src/math/cubic_spline_solver.hpp`. In `price_table_builder.cpp`, there is **no clamp** before calling `spline.eval(log_moneyness[i])`.

### 3. Using PDE Grid Directly as Moneyness Knots

It would reduce information loss **a lot**, but at a heavy cost:
- 4D spline grid size explodes if moneyness goes from 20 → 400 points
- Memory and fit time scale badly with that dimension

More realistic alternatives:
- Keep B-spline grid **adaptive but denser in moneyness**, e.g., 40–80 points
- Use a **nonuniform moneyness grid** aligned with where the PDE curvature is high (exercise boundary / ATM)

### 4. Other Sampling Strategies

Top candidates in order of cost/benefit:

1. **Increase moneyness grid density in adaptive refinement**
   - Let the adaptive loop refine moneyness more aggressively and/or raise `max_points_per_dim` for moneyness
   - Lowest friction change, likely to reduce bulk of error

2. **Switch from natural cubic to monotone / clamped spline**
   - Natural spline can overshoot; monotone cubic Hermite (Fritsch-Carlson) or clamped boundary conditions using known asymptotic slopes should behave better at edges
   - Would directly address boundary oscillation without needing more m-points
   - Today, `CubicSpline` only supports NATURAL; clamped is "coming soon" (`src/math/cubic_spline_solver.hpp`)

3. **Clamp spline evaluation to x-domain**
   - Even if moneyness is supposed to be covered, clamping `x` to `[x_min, x_max]` avoids edge extrapolation from floating error or expanded bounds
   - Low effort, reduces tail oscillation risk

4. **Use PDE-grid-aware resampling**
   - Instead of "sample → B-spline", do a **projection/fit** of the PDE grid onto the B-spline basis, minimizing error, rather than point sampling
   - Avoids aliasing, preserves more information for fixed number of knots

5. **Local refinement near the exercise boundary**
   - Worst interpolation error typically comes from the moving early-exercise boundary
   - Identify on PDE grid (where V ≈ intrinsic), then densify moneyness grid around it

## Pipeline-Specific Findings

- `price_table_builder.cpp` and `price_table_extraction.cpp` are slightly inconsistent: The former assumes snapshots are indexed directly by `j`, while the latter recomputes step indices from `dt`. If the latter is ever used, it could introduce **time-alignment error**, but currently seems unused.

- The adaptive loop **does** refine moneyness based on FD validation error, but if `max_points_per_dim` is low, it stops at too-coarse grids even if error remains.

## Implementation: Minimum Moneyness Density

Added `min_moneyness_points = 40` to `AdaptiveGridParams` to ensure moneyness grid always has sufficient density:

```cpp
// In adaptive_grid_types.hpp
struct AdaptiveGridParams {
    // ...
    /// Minimum moneyness grid points (default: 40)
    /// Moneyness requires higher density than other dimensions due to
    /// exercise boundary curvature and PDE → B-spline sampling loss.
    size_t min_moneyness_points = 40;
};
```

## Next Steps

1. Run benchmark with increased moneyness density (40 points minimum)
2. If errors improve significantly, confirms sampling loss as root cause
3. Consider adding boundary clamping for additional improvement
4. Consider monotone spline for boundary/kink behavior
