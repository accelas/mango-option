# Probe-Based PDE Grid Estimation for IVSolver

## Problem

Users of the FDM-based IV solver must currently configure `GridAccuracyParams` with parameters like `tol` (truncation error target), `n_sigma`, and `alpha`. These are numerical method details that most users don't understand. They just want accurate results.

The current heuristic `dx = σ√tol` is a reasonable approximation but:
1. Doesn't empirically verify grid adequacy
2. Re-estimates grid every Brent iteration (wasteful)
3. Exposes implementation details in the API

## Solution

Replace heuristic grid estimation with Richardson-style probe solves that empirically verify grid adequacy. Calibrate once at high-volatility, cache the grid, and reuse for all Brent iterations.

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
    .target_price_error = 0.01  // $0.01 per $100 notional
};
```

Users who need manual control can still pass explicit `PDEGridConfig`.

### Probe Algorithm

```cpp
ProbeResult probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx = 100)
{
    for (int attempt = 0; attempt < 3; ++attempt) {
        // Solve at Nx (CFL sets Nt automatically)
        auto [grid1, td1] = make_grid(params, Nx);
        double P1 = solve_pde(params, grid1, td1);
        double delta1 = compute_delta(solution);

        // Solve at 2Nx
        auto [grid2, td2] = make_grid(params, 2 * Nx);
        double P2 = solve_pde(params, grid2, td2);
        double delta2 = compute_delta(solution);

        // Composite acceptance criterion
        double price_tol = std::max(target_error, 0.001 * std::max(P1, 0.10));
        double delta_tol = 0.01;

        if (|P1 - P2| <= price_tol && |delta1 - delta2| <= delta_tol) {
            return {grid2, td2, |P1 - P2|};  // Return finer grid
        }

        Nx *= 2;  // Double and retry
    }

    return {grid_finest, td_finest, estimated_error};  // Fallback
}
```

### IVSolver Integration

**Current flow** (wasteful):
```
Brent iteration 1: estimate_pde_grid(σ=0.15) → solve
Brent iteration 2: estimate_pde_grid(σ=0.18) → solve
...
(8-12 iterations, re-estimates every time)
```

**Proposed flow** (calibrate once):
```
Calibration: probe_grid_adequacy(σ_high) → cached_grid
Brent iteration 1: solve with cached_grid
Brent iteration 2: solve with cached_grid
...
(all iterations reuse cached grid)
```

Calibrate at σ_high because higher volatility requires wider domain and finer grid. A grid adequate for σ_high is safe for all σ ≤ σ_high.

### Edge Cases

| Case | Problem | Solution |
|------|---------|----------|
| Short maturity (T < 0.05) | Too few time steps | Nt floor guardrail (min 50) |
| Coincidental agreement | P(Nx) ≈ P(2Nx) by chance | Check delta convergence too |
| Order reduction | TR-BDF2 degrades near free boundary | Use finer grid (2Nx) as result |
| Discrete dividends | Discontinuities | Inherit dividend-aware domain widening |
| Boundary placement | Domain too narrow | Verify n_sigma covers moneyness |
| Probe failure | Max iterations reached | Return finest grid + USDT warning |

### Performance

- Calibration: ~50-100ms one-time (4-6 PDE solves)
- Per Brent iteration: ~5-15ms (single solve with cached grid)
- Total IV solve: ~100-150ms (vs current ~190ms)

Net result: ~20% faster AND empirically verified accuracy.

## Scope

**In scope:**
- `IVSolverFDMConfig.target_price_error` field
- `probe_grid_adequacy()` function
- IVSolver grid caching at σ_high
- Edge case handling
- USDT probes for calibration tracing

**Out of scope (future work):**
- Single-option `solve_american_option()` — heuristic adequate
- Batch solver — follow-up after IVSolver proves approach
- `AdaptiveGridBuilder` — already has probe system for interpolation

## Testing

- Unit tests for `probe_grid_adequacy()` convergence
- Integration tests verifying cached grid reuse
- Edge case tests (short maturity, dividends, deep ITM/OTM)
- Benchmark comparing old vs new IVSolver performance
