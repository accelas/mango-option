# Probe-Based PDE Grid Estimation for IVSolver

## Problem

Users of the FDM-based IV solver must currently configure `GridAccuracyParams` with parameters like `tol` (truncation error target), `n_sigma`, and `alpha`. These are numerical method details that most users don't understand. They just want accurate results.

The current heuristic `dx = σ√tol` is a reasonable approximation but:
1. Doesn't empirically verify grid adequacy
2. Re-estimates grid every Brent iteration (wasteful)
3. Exposes implementation details in the API

## Solution

Replace heuristic grid estimation with Richardson-style probe solves that empirically verify grid adequacy. Calibrate once at high-volatility, cache the grid per `solve()` call, and reuse for all Brent iterations within that call.

## Design

### User-Facing API Change

**Current** (confusing):
```cpp
IVSolverFDMConfig config{
    .grid = GridAccuracyParams{.tol = 1e-3, .n_sigma = 5.0, .alpha = 2.0}
};
```

**Proposed** (simple):
```cpp
IVSolverFDMConfig config{
    .target_price_error = 0.01  // $0.01 absolute price error
};
```

**API Rules:**
- `target_price_error` is a `double` with default `0.01` (not optional)
- **Precedence:** If `target_price_error > 0`, use probe-based calibration; `grid` field is ignored
- If `target_price_error == 0`, fall back to existing `grid` behavior (heuristic or explicit)
- Users who need manual control can set `target_price_error = 0` and pass explicit `PDEGridConfig` via `grid`

**Fallback on probe failure:** If probe doesn't converge (returns `converged=false`), IVSolver uses **default heuristic** (`GridAccuracyParams{}`) regardless of what `grid` field contains. This is because `grid` is explicitly ignored when `target_price_error > 0`. Users who want explicit grid control must set `target_price_error = 0`.

**Units:** `target_price_error` is in absolute price units (same as option price). For a $100 stock, `0.01` means $0.01 accuracy.

### Probe Algorithm

```cpp
struct ProbeResult {
    GridSpec<double> grid;
    TimeDomain time_domain;
    double estimated_error;
    size_t probe_iterations;
    bool converged;  // False if max iterations reached
};

std::expected<ProbeResult, ValidationError> probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx = 100)
{
    for (int attempt = 0; attempt < 3; ++attempt) {
        // Solve at Nx (CFL sets Nt automatically)
        auto [grid1, td1] = make_grid(params, Nx);
        double P1 = solve_pde(params, grid1, td1);

        // Solve at 2Nx
        auto [grid2, td2] = make_grid(params, 2 * Nx);
        double P2 = solve_pde(params, grid2, td2);

        // Compute delta consistently at the SAME physical point (spot)
        // Uses value_at() which does linear interpolation in log-moneyness space
        double h = max(0.01 * spot, 0.01);  // 1% bump with floor

        // Use INTERSECTION of both grid domains to avoid extrapolation
        // Clamp h symmetrically around spot (keep center at spot)
        double h_max = min(spot - domain_lo, domain_hi - spot);
        h = min(h, h_max);

        // Guard: if h too small, skip delta check (use price only)
        if (h < 1e-6) { delta_diff = 0; }
        else {
            double delta1 = (value_at(sol1, spot + h) - value_at(sol1, spot - h)) / (2 * h);
            double delta2 = (value_at(sol2, spot + h) - value_at(sol2, spot - h)) / (2 * h);
            delta_diff = |delta1 - delta2|;
        }

        // Composite acceptance criterion using max of both prices
        double price_ref = std::max({std::abs(P1), std::abs(P2), 0.10});
        double price_tol = std::max(target_error, 0.001 * price_ref);
        double delta_tol = 0.01;

        if (|P1 - P2| <= price_tol && |delta1 - delta2| <= delta_tol) {
            return ProbeResult{grid2, td2, |P1 - P2|, attempt + 1, true};
        }

        Nx *= 2;
    }

    // Return finest grid with converged=false (caller decides how to proceed)
    return ProbeResult{grid_finest, td_finest, estimated_error, 3, false};
}
```

**Key details:**
- Delta computed via finite difference at spot, clamped symmetrically to intersection of both grid domains
- If h becomes too small (< 1e-6) after clamping, skip delta check and use price convergence only
- Error criterion uses `max(|P1|, |P2|, floor)` for consistent relative scale
- Returns `converged` flag so caller can decide fallback behavior
- Nt floor (min 50 steps) ensures temporal resolution for short maturities
- Note: TR-BDF2 is L-stable (unconditionally stable for diffusion), so no CFL constraint needed; Nt floor is purely for accuracy

### IVSolver Integration

**Current flow** (wasteful):
```
Brent iteration 1: estimate_pde_grid(σ=0.15) → solve
Brent iteration 2: estimate_pde_grid(σ=0.18) → solve
...
(8-12 iterations, re-estimates every time)
```

**Proposed flow** (calibrate once per solve() call):
```
solve(query):
  σ_high = config.sigma_upper  // Use configured upper bound
  cached_grid = probe_grid_adequacy(params_at_σ_high)

  Brent iterations (all use cached_grid):
    iteration 1: solve with cached_grid
    iteration 2: solve with cached_grid
    ...
```

**Cache scoping:** Grid is cached per `solve()` call, NOT on the solver instance. This ensures different options (with different spot/strike/maturity/dividends) get their own calibrated grids. The cache is a local variable in `solve()`, not a member variable.

**σ_high selection:** Use `config.sigma_upper` (the configured upper bound of the Brent search). This is the worst case for grid requirements. Do NOT use `2 * initial_guess` which could exceed intended bounds.

Calibrate at σ_high because higher volatility requires wider domain and finer grid. A grid adequate for σ_high is safe for all σ ≤ σ_high.

### Edge Cases

| Case | Problem | Solution |
|------|---------|----------|
| Short maturity (T < 0.05) | Too few time steps | Nt floor (min 50) for temporal accuracy |
| Coincidental agreement | P(Nx) ≈ P(2Nx) by chance | Check delta convergence at same physical point |
| Order reduction | TR-BDF2 degrades near free boundary | Use finer grid (2Nx) as result |
| Discrete dividends | Discontinuities | Inherit dividend-aware domain widening |
| Boundary placement | Domain too narrow | Verify n_sigma covers moneyness |
| Probe failure | Max iterations reached | Return `converged=false`, IVSolver uses default heuristic |
| Price converges, delta doesn't | Grid adequate for price but not greeks | Require both to converge |

### Performance

- Calibration: ~50-100ms one-time per `solve()` call (4-6 PDE solves)
- Per Brent iteration: ~5-15ms (single solve with cached grid)
- Total IV solve: ~100-150ms (vs current ~190ms with per-iteration re-estimation)

Net result: ~20% faster AND empirically verified accuracy.

**Note:** Performance gains apply when Brent search runs multiple iterations. Single-iteration solves pay calibration cost without amortization. Benchmarks should measure both scenarios.

## Scope

**In scope:**
- `IVSolverFDMConfig.target_price_error` field (double, default 0.01)
- `probe_grid_adequacy()` function with `converged` flag
- IVSolver grid caching per `solve()` call at σ_high
- Edge case handling (Nt floor for accuracy, consistent delta via interpolation)
- USDT probes for calibration tracing
- Fallback to default heuristic when probe doesn't converge

**Out of scope (future work):**
- Single-option `solve_american_option()` — heuristic adequate
- Batch solver — follow-up after IVSolver proves approach
- `AdaptiveGridBuilder` — already has probe system for interpolation

## Testing

- Unit tests for `probe_grid_adequacy()` convergence
- Unit test: probe fails (max iterations) returns `converged=false`
- Unit test: price converges but delta doesn't → keeps iterating
- Integration tests verifying cached grid is per-`solve()` (not shared across calls)
- Integration test: IVSolver falls back to heuristic when probe fails
- Edge case tests (short maturity, dividends, deep ITM/OTM)
- Benchmark comparing old vs new IVSolver performance (multi-iteration and single-iteration cases)
