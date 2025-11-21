# IV Solver API Usage Examples

Quick reference for using the three IV solving approaches in mango-iv.

## 1. FDM-Based IV Solver (Single Option)

Ground truth approach for individual IV calculations.

```cpp
#include "src/option/iv_solver.hpp"

// Define option and market conditions
mango::IVParams params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = false
};

// Configure solver (optional, uses defaults if not specified)
mango::IVConfig config{
    .root_config = mango::RootFindingConfig{
        .max_iter = 100,
        .tolerance = 1e-6
    },
    .grid_n_space = 101,
    .grid_n_time = 1000,
    .grid_s_max = 200.0
};

// Solve for implied volatility
mango::IVSolver solver(params, config);
mango::IVResult result = solver.solve();

if (result.converged) {
    std::cout << "Implied Volatility: " << result.implied_vol << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Error: " << result.final_error << "\n";
    if (result.vega.has_value()) {
        std::cout << "Vega: " << result.vega.value() << "\n";
    }
} else {
    std::cerr << "Failed to converge: " << *result.failure_reason << "\n";
}
```

**Performance:** ~143ms per calculation
**Use Case:** Single option pricing, validation, research

---

## 2. Batch FDM IV Solver (Multiple Options in Parallel)

Fast parallel IV calculation for option chains.

```cpp
#include "src/option/iv_solver_fdm.hpp"
#include <vector>

// Prepare batch of options
std::vector<mango::IVQuery> queries = {
    {.option = {.spot = 100.0, .strike = 95.0, .maturity = 1.0,
                .rate = 0.05, .type = OptionType::PUT},
     .market_price = 8.5},
    {.option = {.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                .rate = 0.05, .type = OptionType::PUT},
     .market_price = 10.45},
    {.option = {.spot = 100.0, .strike = 105.0, .maturity = 1.0,
                .rate = 0.05, .type = OptionType::PUT},
     .market_price = 12.8},
};

// Shared configuration for all options
mango::IVSolverFDMConfig config{
    .root_config = {.max_iter = 100, .tolerance = 1e-6},
    .grid_n_space = 101,
    .grid_n_time = 1000,
    .grid_s_max = 200.0
};

// Create solver and solve in parallel
mango::IVSolverFDM solver(config);
std::vector<mango::IVResult> results(queries.size());
auto status = solver.solve_batch(queries, results);

if (!status) {
    std::cerr << "Batch solve failed: " << status.error() << "\n";
    return;
}

// Process results
for (size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    std::cout << "Option " << i << ": ";
    if (result.converged) {
        std::cout << "IV = " << result.implied_vol << "\n";
    } else {
        std::cout << "FAILED - " << *result.failure_reason << "\n";
    }
}
```

**Performance:** ~107 IVs/sec on 32 cores (15.3× parallel speedup)
**Use Case:** Volatility surface construction, option chains, risk calculation

---

## 3. Price Table Pre-computation Workflow

Build fast lookup table for thousands of IV calculations.

### Step 1: Create Builder from Market Chain

```cpp
#include "src/option/price_table_4d_builder.hpp"

// Option A: From raw market chain
mango::OptionChain spy_chain{
    .ticker = "SPY",
    .spot = 450.0,
    .strikes = {400, 425, 450, 475, 500},  // Can have duplicates
    .maturities = {0.1, 0.25, 0.5, 1.0},
    .implied_vols = {0.15, 0.20, 0.25, 0.30},
    .rates = {0.03, 0.04, 0.05},
    .dividend_yield = 0.015
};

auto builder_result = mango::PriceTable4DBuilder::from_chain(spy_chain);
if (!builder_result) {
    std::cerr << "Failed to create builder: " << builder_result.error() << "\n";
    return;
}
auto builder = builder_result.value();

// Option B: From explicit grids
auto builder2 = mango::PriceTable4DBuilder::create(
    {0.8, 0.9, 1.0, 1.1, 1.2},      // moneyness
    {0.1, 0.25, 0.5, 1.0},          // maturity
    {0.15, 0.20, 0.25, 0.30},       // volatility
    {0.03, 0.04, 0.05},             // rate
    450.0  // K_ref (usually ATM strike)
);
```

### Step 2: Pre-compute Price Table

```cpp
// Configure PDE solver for pre-computation
mango::PriceTableConfig pde_config{
    .option_type = mango::OptionType::PUT,
    .n_space = 101,
    .n_time = 1000,
    .dividend_yield = 0.015
};

// Pre-compute (200 PDE solves × 100 space points)
auto result_or_error = builder.precompute(mango::OptionType::PUT, 101, 1000);

if (!result_or_error) {
    std::cerr << "Pre-computation failed: " << result_or_error.error() << "\n";
    return;
}

auto result = result_or_error.value();
std::cout << "Pre-computed " << result.n_pde_solves << " options in "
          << result.precompute_time_seconds << " seconds\n";
std::cout << "Max fitting residual: " << result.fitting_stats.max_residual_overall << "\n";
```

**Time:** ~24 seconds on 16 cores for 200 solves
**Memory:** ~2.4 MB for typical 4D table

### Step 3: Create IV Solver from Price Table

```cpp
// Get the pre-computed surface
auto surface = result.surface;

// Create IV solver
mango::IVSolverInterpolated iv_solver(surface);

// Configure Newton solver (optional, uses defaults)
mango::IVSolverConfig iv_config{
    .max_iterations = 50,
    .tolerance = 1e-6
};
```

---

## 4. Interpolation-Based IV Solver (Fast Queries)

Ultra-fast IV calculation using pre-computed table.

```cpp
#include "src/option/iv_solver_interpolated.hpp"

// Use the pre-computed surface from Step 3 above
mango::IVSolverInterpolated iv_solver(surface);

// Query for IV
mango::IVQuery query{
    .market_price = 10.45,
    .spot = 100.0,          // S
    .strike = 100.0,        // K (can differ from K_ref)
    .maturity = 1.0,        // T
    .rate = 0.05,           // r
    .option_type = mango::OptionType::PUT
};

auto iv_result = iv_solver.solve(query);

if (iv_result.converged) {
    std::cout << "IV: " << iv_result.implied_vol << " (in " << iv_result.iterations << " iterations)\n";
} else {
    std::cout << "Failed: " << *iv_result.failure_reason << "\n";
}
```

**Performance:** ~20µs per query (4,800× faster than FDM)
**Use Case:** Real-time trading, risk dashboards, market data processing

---

## 5. Batch Interpolation-Based IV Solver

Process multiple options with pre-computed table (advanced).

```cpp
// Pre-computed surface from Step 3
auto surface = result.surface;
mango::IVSolverInterpolated iv_solver(surface);

// Batch queries
std::vector<mango::IVQuery> queries = {
    {.market_price = 8.5, .spot = 100.0, .strike = 95.0, .maturity = 1.0, .rate = 0.05},
    {.market_price = 10.45, .spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05},
    {.market_price = 12.8, .spot = 100.0, .strike = 105.0, .maturity = 1.0, .rate = 0.05},
};

// Process in parallel (each thread creates own solver to avoid contention)
std::vector<mango::IVResult> iv_results;
iv_results.reserve(queries.size());

#pragma omp parallel for
for (size_t i = 0; i < queries.size(); ++i) {
    // Create solver in each thread (B-spline evaluation is read-only)
    mango::IVSolverInterpolated local_solver(surface);
    #pragma omp critical
    iv_results.push_back(local_solver.solve(queries[i]));
}

// Report
for (const auto& res : iv_results) {
    std::cout << "IV: " << res.implied_vol << "\n";
}
```

**Performance:** 50,000+ IVs/sec on 32 cores (read-only B-spline surface)

---

## 6. Save/Load Price Table (Persistence)

Avoid re-computation by saving pre-computed tables.

```cpp
// After pre-computation (from Step 2)
auto result = builder.precompute(mango::OptionType::PUT, 101, 1000);
auto surface = result.surface;

// Save to disk
auto save_result = surface.workspace()->save(
    "spy_american_put.bin",  // filename
    "SPY",                    // ticker
    0                         // 0=PUT, 1=CALL
);

if (!save_result) {
    std::cerr << "Save failed: " << save_result.error() << "\n";
    return;
}

std::cout << "Price table saved\n";

// ========== Later, in different program ==========

// Load from disk (very fast)
auto load_result = mango::PriceTableWorkspace::load("spy_american_put.bin");
if (!load_result) {
    std::cerr << "Load failed\n";
    return;
}

auto workspace = load_result.value();
auto surface2 = mango::PriceTableSurface(std::make_shared<mango::PriceTableWorkspace>(std::move(workspace)));

// Use loaded surface for IV calculations
mango::IVSolverInterpolated iv_solver(surface2);
auto iv_result = iv_solver.solve(query);
```

**Load Time:** Milliseconds (mmap + validation)
**Disk Size:** ~2.4 MB for typical 4D table

---

## 7. Error Handling Patterns

### Pattern A: Check convergence field

```cpp
auto result = solver.solve();
if (!result.converged) {
    std::cout << "Solver failed: " << *result.failure_reason << "\n";
    std::cout << "Reason code: " << *result.failure_reason << "\n";
}
```

### Pattern B: Understand convergence failures

```cpp
// Different failure modes
if (result.failure_reason) {
    std::string msg = *result.failure_reason;
    
    if (msg.find("out of bounds") != std::string::npos) {
        // Use FDM solver instead (interpolation surface too small)
        std::cout << "Query outside interpolation bounds, use FDM\n";
    } else if (msg.find("Spot price below intrinsic") != std::string::npos) {
        // Market price violates arbitrage
        std::cout << "Arbitrage opportunity detected\n";
    } else {
        // Generic convergence failure
        std::cout << "Numerical solver diverged\n";
    }
}
```

### Pattern C: Performance fallback

```cpp
// Try fast interpolation first
mango::IVSolverInterpolated fast_solver(surface);
auto fast_result = fast_solver.solve(query);

if (!fast_result.converged) {
    // Fall back to FDM for out-of-bounds queries
    mango::IVSolver fdm_solver(params, fdm_config);
    auto fdm_result = fdm_solver.solve();
    
    if (fdm_result.converged) {
        std::cout << "FDM solution: " << fdm_result.implied_vol << "\n";
    }
}
```

---

## 8. Volatility Surface Construction

Build entire volatility surface from market quotes.

```cpp
#include "src/option/iv_solver.hpp"

// Market data: strikes, maturities, prices for PUT options
std::vector<double> strikes = {90, 95, 100, 105, 110};
std::vector<double> maturities = {0.25, 0.5, 1.0};
std::vector<std::vector<double>> market_prices = {
    {12.3, 10.5, 8.5},   // K=90
    {10.2, 8.8, 7.1},    // K=95
    {8.5, 7.2, 5.8},     // K=100
    {7.1, 6.0, 4.8},     // K=105
    {5.9, 4.9, 3.9},     // K=110
};

double spot = 100.0;
double rate = 0.05;

// Compute IV for each (strike, maturity) pair
std::vector<std::vector<double>> iv_surface(strikes.size(), std::vector<double>(maturities.size()));

for (size_t i = 0; i < strikes.size(); ++i) {
    for (size_t j = 0; j < maturities.size(); ++j) {
        mango::IVParams params{
            .spot_price = spot,
            .strike = strikes[i],
            .time_to_maturity = maturities[j],
            .risk_free_rate = rate,
            .market_price = market_prices[i][j],
            .is_call = false
        };
        
        mango::IVSolver solver(params, config);
        auto result = solver.solve();
        
        if (result.converged) {
            iv_surface[i][j] = result.implied_vol;
        } else {
            iv_surface[i][j] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

// Use surface for visualization, analysis, model calibration, etc.
for (size_t i = 0; i < strikes.size(); ++i) {
    std::cout << "K=" << strikes[i] << ": ";
    for (size_t j = 0; j < maturities.size(); ++j) {
        std::cout << std::fixed << std::setprecision(4) << iv_surface[i][j] << " ";
    }
    std::cout << "\n";
}
```

---

## Decision Tree: Which Solver to Use?

```
Is this a single option calculation?
├─ YES → Use IVSolver (FDM)
│        Simple, accurate, ground truth
│        ~143ms per calculation
│        See Example 1
│
└─ NO → Multiple options?
         ├─ <10 options → Use IVSolverFDM::solve_batch
         │              Fast parallel, no pre-computation needed
         │              ~107 options/sec on 32 cores
         │              See Example 2
         │
         └─ >100 options → Build Price Table
                          ├─ Do you need results in <1 minute? 
                          │  ├─ YES → Use IVSolverFDM::solve_batch
                          │  │        Parallel, reasonably fast
                          │  │
                          │  └─ NO → Pre-compute Table
                          │          Use IVSolverInterpolated (fast)
                          │          ~20µs per query afterward
                          │          1 hour setup, instant queries
                          │          See Examples 3-4
                          │
                          └─ Need to save for later use?
                             └─ YES → Save/Load (Example 6)
```

---

## Performance Summary

| Method | Single IV | Batch | Setup Time | Query Time | Use Case |
|--------|-----------|-------|------------|------------|----------|
| IVSolver (FDM) | ~143ms | ~107/sec | None | ~143ms | Single, ground truth |
| IVSolverFDM::solve_batch | N/A | ~107/sec | None | Batch | Small batches <10 options |
| PriceTable | N/A | N/A | ~24sec | N/A | Setup for fast queries |
| IVSolverInterpolated | ~20µs | 50K+/sec | ~24sec | ~20µs | Many queries, pre-computed |

Choose based on your latency/throughput requirements:
- **Ground truth (slow, accurate):** FDM
- **Fast single option:** FDM is acceptable (~143ms)
- **Batch of 5-50:** Parallel FDM
- **Thousands of queries:** Pre-compute, then interpolate
