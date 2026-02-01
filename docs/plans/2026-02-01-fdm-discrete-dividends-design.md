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

### PDESolver variable-dt loop

The solve loop changes from uniform `t_next = t + dt` to
`t_next = time_points[step+1]` with `local_dt = t_next - t`. The TR-BDF2 stages
already accept dt as a parameter, so variable step sizes work without changing
the stage implementations.

### Dividend event callback

At each dividend time, a temporal event fires and:

1. Builds a cubic spline of u(x) from the current grid solution.
2. For each grid point x[i], computes x' = ln(exp(x[i]) - D/K).
3. Sets u[i] = spline(x') if exp(x[i]) > D/K, else intrinsic value.
4. PDESolver re-applies obstacle and boundary conditions automatically.

### Registration in AmericanOptionSolver

After creating the PDE solver, before calling solve():

```cpp
for (auto& [t_cal, amount] : params_.discrete_dividends) {
    double tau_div = params_.maturity - t_cal;
    pde_solver.add_temporal_event(tau_div, make_dividend_callback(amount, K));
}
```

The time domain is constructed with dividend times as mandatory points so that
the solver lands exactly on each dividend date.

### Batch support

- **Regular batch** (OpenMP parallel): each option gets its own solver with
  dividend events registered per option. No changes needed beyond single-option
  support.
- **Normalized chain**: keeps rejecting discrete dividends. The shared PDE across
  strikes makes dividend shifts ambiguous.

## Interpolation

Cubic spline (natural boundary conditions) for the solution shift at dividend
dates. The Thomas solver infrastructure already exists in the codebase.

## Scope

- Single-option solver (`solve_american_option_auto`)
- Regular batch solver (`solve_regular_batch`)
- Normalized chain keeps existing rejection
