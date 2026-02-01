# FDM Discrete Dividend Support

## Goal

Add discrete dividend handling to the FDM American option solver. Discrete cash
dividends shift the spot price at known dates. The PDE solver must land exactly
on each dividend date and apply a jump condition that shifts the solution in
log-moneyness space via cubic spline interpolation.

## Architecture

### TimeDomain with mandatory time points

`TimeDomain` gains a factory that merges uniform steps with mandatory dividend
times:

```
TimeDomain::with_mandatory_points(t_start, t_end, dt, mandatory_times)
```

This produces a non-uniform sequence of time points. Between mandatory times,
sub-steps are roughly dt-sized. The solve loop reads dt per step from the
sequence rather than using a single constant.

**Boundary policy:** Dividend times use calendar time `t_cal` where `t_cal = 0`
is the valuation date and `t_cal = T` is expiry. Dividends at `t_cal <= 0` (at
or before valuation — already paid) and `t_cal >= T` (at or after expiry — not
relevant) are silently ignored. Only dividends strictly within `(0, T)` are
included as mandatory points. Internally, calendar times are converted to
time-to-expiry via `tau = T - t_cal` for the PDE solver, which marches in tau
from 0 to T.

### PDESolver variable-dt loop

The solve loop changes from uniform `t_next = t + dt` to reading time points
directly from the stored sequence: `t_next = time_points_[step+1]` with
`local_dt = t_next - t`. This avoids float drift from repeated addition.

The TR-BDF2 stages already accept dt as a parameter and recompute weights
(`stage1_weight(dt)`, `stage2_weight(dt)`) fresh each call. No matrices or
coefficients are cached across steps, so variable step sizes work without
changing the stage implementations.

### Snapshot indexing for non-uniform grids

`convert_times_to_indices` in `grid.hpp` currently assumes uniform dt and
computes `step_exact = (t - t_start) / dt`. For non-uniform grids, this is
replaced with a binary search through the stored time points vector via
`std::lower_bound`. The function falls back to the existing arithmetic path
when the time domain is uniform (no stored time points).

### Dividend event callback

`make_dividend_event(amount, strike, option_type)` creates the temporal event
callback. The option type is required for correct fallback when `S - D <= 0`:

- **Put:** intrinsic = 1.0 (deep ITM, S_adj floored at 0)
- **Call:** intrinsic = 0.0 (worthless when spot ≤ 0)

At each dividend time, the callback:

1. Builds a cubic spline of u(x) from the current grid solution.
2. For each grid point x[i], computes x' = ln(exp(x[i]) - D/K).
3. Sets u[i] = spline(x') if exp(x[i]) > D/K, else option-type-aware
   intrinsic value.
4. If the spline build fails (degenerate data), the callback leaves u unchanged
   and returns early. The PDESolver re-applies obstacle conditions, so the
   worst case is one dividend event missed — acceptable since spline failure
   requires pathologically degenerate grid data.
5. PDESolver re-applies obstacle and boundary conditions automatically.

### Grid expansion for spline safety

The spline evaluates shifted points `x' = ln(exp(x) - D/K)`. For the lowest
grid point x_min, the shifted value is `ln(exp(x_min) - D/K)` which is more
negative than x_min. To keep shifted points within the interpolation domain,
`estimate_grid_for_option` extends `x_min` to `ln(exp(x_min) + d_max)` where
`d_max = max(D_i/K)` across all dividends. This computes the exact lower bound
needed rather than using an approximate linear shift.

### Registration in AmericanOptionSolver

After creating the PDE solver, before calling solve():

```cpp
for (auto& [t_cal, amount] : params_.discrete_dividends) {
    double tau_div = params_.maturity - t_cal;
    if (tau_div > 0.0 && tau_div < params_.maturity) {
        pde_solver.add_temporal_event(tau_div,
            make_dividend_event(amount, params_.strike, params_.type));
    }
}
```

The time domain is constructed with dividend times as mandatory points so that
the solver lands exactly on each dividend date.

### Batch support

- **Regular batch** (OpenMP parallel): each option gets its own solver with
  dividend events registered per option. No changes needed beyond single-option
  support.
- **Normalized chain**: keeps rejecting discrete dividends. The shared PDE
  across strikes makes dividend shifts ambiguous (D/K differs per strike).

## Interpolation

Cubic spline (natural boundary conditions) for the solution shift at dividend
dates. The Thomas solver infrastructure already exists in the codebase.

## Scope

- Single-option solver (`solve_american_option_auto`)
- Regular batch solver (`solve_regular_batch`)
- Normalized chain keeps existing rejection
- `convert_times_to_indices` updated for non-uniform grids
