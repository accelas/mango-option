# IV Solver API Quick Reference

One-page cheat sheet for using mango-iv's three IV solving approaches.

## Choose Your Solver

```
Single option? → IVSolver (FDM) [143ms]
Few options (5-50)? → BatchIVSolver (FDM parallel) [107 IV/s on 32 cores]
Many queries (100s)? → IVSolverInterpolated [~20µs, requires pre-computed table]
```

---

## 1. Single Option (FDM)

```cpp
#include "src/option/iv_solver.hpp"

mango::IVSolver solver({
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = false
});

auto result = solver.solve();
if (result.converged) {
    std::cout << "IV: " << result.implied_vol << "\n";
}
```

---

## 2. Batch Options (FDM Parallel)

```cpp
#include "src/option/iv_solver.hpp"

std::vector<mango::IVParams> batch = { ... };
mango::IVConfig config{...};
auto results = mango::solve_implied_vol_batch(batch, config);
```

---

## 3. Fast Queries (Pre-computed Table)

### Step 1: Build Table (one-time)

```cpp
#include "src/option/price_table_4d_builder.hpp"

// From market chain
auto builder = mango::PriceTable4DBuilder::from_chain(chain);

// Pre-compute (200 PDEs, ~24 sec on 16 cores)
auto result_or_error = builder.precompute(
    mango::OptionType::PUT, 101, 1000);
auto surface = result_or_error.value().surface;

// Save for later
surface.workspace()->save("table.bin", "SPY", 0);
```

### Step 2: Query IV (instant)

```cpp
#include "src/option/iv_solver_interpolated.hpp"

// Load table (milliseconds)
auto ws = mango::PriceTableWorkspace::load("table.bin");
auto surface = mango::PriceTableSurface(
    std::make_shared<mango::PriceTableWorkspace>(
        ws.value()));

// Create solver
mango::IVSolverInterpolated solver(surface);

// Solve (20 microseconds)
auto iv = solver.solve({
    .market_price = 10.45,
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05
});

if (iv.converged) {
    std::cout << "IV: " << iv.implied_vol << "\n";
}
```

---

## Input/Output Types

### FDM Solver

**Input (IVParams):**
- `spot_price` - Current underlying price
- `strike` - Strike price
- `time_to_maturity` - Years to expiration
- `risk_free_rate` - Risk-free rate
- `market_price` - Observed market price
- `is_call` - Call (true) or put (false)

**Output (IVResult):**
- `converged` - Success flag
- `implied_vol` - Solved volatility (if converged)
- `iterations` - Number of Brent iterations
- `final_error` - |V(σ) - Market_Price|
- `failure_reason` - Error description (optional)
- `vega` - ∂V/∂σ (optional)

**Config (IVConfig):**
- `root_config` - Brent parameters (max_iter, tolerance)
- `grid_n_space` - PDE spatial grid (default 101)
- `grid_n_time` - PDE time steps (default 1000)
- `grid_s_max` - Max spot price (default 200.0)

### Interpolation Solver

**Input (IVQuery):**
- `market_price` - Observed option price
- `spot` - Current spot price
- `strike` - Strike (can differ from K_ref)
- `maturity` - Time to expiration (years)
- `rate` - Risk-free rate
- `option_type` - CALL or PUT

**Config (IVSolverConfig):**
- `max_iterations` - Newton max iterations (default 50)
- `tolerance` - Price error tolerance (default 1e-6)
- `vega_epsilon` - FD step (default 1e-4)
- `sigma_min` - Min vol (default 0.01)
- `sigma_max` - Max vol (default 3.0)

**Output:** Same IVResult as FDM

---

## Performance Summary

| Method | Speed | Setup | Queries |
|--------|-------|-------|---------|
| IVSolver | 143ms | - | Single |
| BatchIVSolver | 107/sec | - | Batch |
| IVSolverInterpolated | 20µs | 24s | 50K+/sec |

---

## Critical Notes

**Strike Handling in Interpolation Solver:**
- Surface built with moneyness m = S/K_ref
- For query strike ≠ K_ref:
  - Compute m = S/K_ref (use K_ref from surface)
  - Price scales: V(K) = V(K_ref) × (K/K_ref)

**Bounds Estimation (Both Solvers):**
- High time value (>50%): σ_upper = 300%
- Moderate (20-50%): σ_upper = 200%
- Low (<20%): σ_upper = 150%
- Always: σ_lower = 1%

**Thread Safety:**
- IVSolver: Thread-safe (creates own workspace)
- BatchIVSolver: Uses MANGO_PRAGMA_PARALLEL_FOR
- IVSolverInterpolated: Read-only surface, thread-safe

---

## Error Checking

```cpp
auto result = solver.solve();

if (!result.converged) {
    std::string msg = *result.failure_reason;
    
    if (msg.find("out of bounds") != std::string::npos) {
        // Outside table domain → use FDM
    } else if (msg.find("intrinsic") != std::string::npos) {
        // Arbitrage violation
    } else {
        // Convergence failure
    }
}
```

---

## Files to Include

```cpp
// Single option or batch
#include "src/option/iv_solver.hpp"

// Pre-computed table
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/iv_solver_interpolated.hpp"
```

---

## Links to Detailed Docs

See companion files:
- **IV_API_EXPLORATION.md** - Complete API reference with all details
- **IV_API_USAGE_EXAMPLES.md** - 8 working code examples with explanations
