<!-- SPDX-License-Identifier: MIT -->
# API Usage Guide

Practical examples and common usage patterns for the mango-option library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For mathematical formulations, see [MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)**
**For workflow guidance, see [../CLAUDE.md](../CLAUDE.md)**

## Table of Contents

1. [Quick Start](#quick-start)
2. [American Option Pricing](#american-option-pricing)
3. [Implied Volatility Calculation](#implied-volatility-calculation)
4. [Price Table Pre-Computation](#price-table-pre-computation)
5. [Custom PDE Solving](#custom-pde-solving)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Batch Processing](#batch-processing)
8. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Minimal American Option Example

```cpp
#include "src/option/american_option.hpp"
#include <iostream>

int main() {
    // Option parameters
    mango::PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };

    // Auto-estimate grid
    auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);

    // Create workspace
    std::pmr::synchronized_pool_resource pool;
    auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

    // Solve
    mango::AmericanOptionSolver solver(params, workspace);
    auto result = solver.solve();

    if (result.has_value()) {
        std::cout << "Price: " << result->price() << "\n";
        std::cout << "Delta: " << result->delta() << "\n";
        std::cout << "Gamma: " << result->gamma() << "\n";
    }

    return 0;
}
```

### Minimal IV Calculation Example

```cpp
#include "src/option/iv_solver_fdm.hpp"
#include <iostream>

int main() {
    // Option specification
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = mango::OptionType::PUT
    };

    // IV query
    mango::IVQuery query{.option = spec, .market_price = 10.45};

    // Solve
    mango::IVSolverFDM solver(mango::IVSolverFDMConfig{});
    auto result = solver.solve_impl(query);

    if (result.has_value()) {
        std::cout << "Implied Vol: " << result->implied_vol << "\n";
        std::cout << "Iterations: " << result->iterations << "\n";
    } else {
        std::cerr << "Error: " << result.error().message << "\n";
    }

    return 0;
}
```

---

## American Option Pricing

### Basic Pricing with Auto Grid

**Recommended for most use cases:**

```cpp
#include "src/option/american_option.hpp"

// Define option
mango::PricingParams params{
    .strike = 100.0,
    .spot = 105.0,          // 5% ITM put
    .maturity = 0.5,        // 6 months
    .volatility = 0.25,
    .rate = 0.03,
    .continuous_dividend_yield = 0.01,
    .type = mango::OptionType::PUT
};

// Auto-estimate grid (accounts for σ, T, moneyness)
auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);

// Create workspace
std::pmr::synchronized_pool_resource pool;
auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

// Solve
mango::AmericanOptionSolver solver(params, workspace);
auto result = solver.solve();

if (result.has_value()) {
    double price = result->price();
    double delta = result->delta();
    double gamma = result->gamma();

    std::cout << "Price: " << price << "\n";
    std::cout << "Delta: " << delta << "\n";
    std::cout << "Gamma: " << gamma << "\n";
}
```

### Custom Grid Configuration

**For advanced users requiring specific grid control:**

```cpp
// Custom grid specification
auto grid_spec = mango::GridSpec<double>::sinh_spaced(
    -3.0,    // x_min (log-moneyness)
    3.0,     // x_max
    201,     // n_points (spatial resolution)
    2.5      // alpha (sinh concentration)
).value();

size_t n_time = 2000;  // Temporal resolution

// Create workspace with custom grid
std::pmr::synchronized_pool_resource pool;
auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

// Pass custom grid to solver
mango::AmericanOptionSolver solver(
    params,
    workspace,
    std::nullopt,           // No snapshots
    grid_spec,              // Custom grid
    n_time                  // Custom time steps
);

auto result = solver.solve();
```

### Greeks Calculation

**Delta and Gamma computed via centered finite differences:**

```cpp
auto result = solver.solve();

if (result.has_value()) {
    // Lazy evaluation (computed on first access, then cached)
    double delta = result->delta();  // ∂V/∂S
    double gamma = result->gamma();  // ∂²V/∂S²

    // Greeks use same CenteredDifference operators as PDE solver
    // Automatically handles uniform and non-uniform (sinh) grids
}
```

### Dividend Handling

**Continuous dividends:**
```cpp
mango::PricingParams params{
    // ...
    .continuous_dividend_yield = 0.03,  // 3% continuous yield
};
```

**Discrete dividends:** Not yet implemented in current API

---

## Implied Volatility Calculation

### FDM-Based IV (Robust, ~143ms)

**Uses Brent's method with nested PDE pricing:**

```cpp
#include "src/option/iv_solver_fdm.hpp"

// Option specification
mango::OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = mango::OptionType::PUT
};

// IV query with market price
mango::IVQuery query{.option = spec, .market_price = 10.45};

// Configure solver (optional)
mango::IVSolverFDMConfig config{
    .root_config = mango::RootFindingConfig{
        .max_iter = 100,
        .tolerance = 1e-6
    }
};

// Solve
mango::IVSolverFDM solver(config);
auto result = solver.solve_impl(query);

if (result.has_value()) {
    std::cout << "Implied Vol: " << result->implied_vol << " ("
              << (result->implied_vol * 100) << "%)\n";
    std::cout << "Iterations: " << result->iterations << "\n";
    std::cout << "Final Error: " << result->final_error << "\n";

    if (result->vega) {
        std::cout << "Vega: " << *result->vega << "\n";
    }
} else {
    const auto& error = result.error();
    std::cerr << "Error [" << static_cast<int>(error.code) << "]: "
              << error.message << "\n";

    if (error.last_vol) {
        std::cerr << "Last volatility tried: " << *error.last_vol << "\n";
    }
}
```

### Batch IV Calculation

**Process multiple IV queries in parallel:**

```cpp
// Create batch of queries
std::vector<mango::IVQuery> queries;
for (const auto& [strike, price] : market_data) {
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = strike,
        .maturity = 0.25,
        .rate = 0.03,
        .dividend_yield = 0.0,
        .type = mango::OptionType::CALL
    };
    queries.push_back({.option = spec, .market_price = price});
}

// Solve batch (OpenMP parallel)
mango::IVSolverFDM solver(config);
auto batch = solver.solve_batch_impl(queries);

std::cout << "Succeeded: " << (batch.results.size() - batch.failed_count) << "\n";
std::cout << "Failed: " << batch.failed_count << "\n";

// Process results
for (size_t i = 0; i < batch.results.size(); ++i) {
    if (batch.results[i].has_value()) {
        std::cout << "Strike " << queries[i].option.strike
                  << ": IV = " << batch.results[i]->implied_vol << "\n";
    } else {
        std::cerr << "Strike " << queries[i].option.strike
                  << ": " << batch.results[i].error().message << "\n";
    }
}
```

### Custom Grid for IV

**Override auto-estimation with manual grid:**

```cpp
mango::IVSolverFDMConfig config{
    .use_manual_grid = true,
    .grid_n_space = 201,
    .grid_n_time = 2000,
    .grid_x_min = -3.0,
    .grid_x_max = 3.0,
    .grid_alpha = 2.5
};

mango::IVSolverFDM solver(config);
auto result = solver.solve_impl(query);
```

---

## Price Table Pre-Computation

### Building a 4D Price Surface

**Pre-compute American option prices across parameter space:**

```cpp
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"

// Define 4D parameter grids
std::vector<double> moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};  // m = S/K
std::vector<double> maturity_grid = {0.027, 0.1, 0.25, 0.5, 1.0, 2.0};      // τ (years)
std::vector<double> vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};         // σ
std::vector<double> rate_grid = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};         // r

double K_ref = 100.0;  // Reference strike price

// Create PDE grid specification
auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 101).value();

// Create builder and axes using factory method
auto factory_result = mango::PriceTableBuilder<4>::from_vectors(
    moneyness_grid,
    maturity_grid,
    vol_grid,
    rate_grid,
    K_ref,
    grid_spec,
    1000,  // n_time (time steps)
    mango::OptionType::PUT
);

if (!factory_result.has_value()) {
    std::cerr << "Factory creation failed: " << factory_result.error() << "\n";
    return;
}

auto [builder, axes] = std::move(factory_result.value());

// Build price table (parallelized with OpenMP)
// This takes ~15-20 minutes for 300K grid on 32 cores
auto result = builder.build(axes);

if (!result.has_value()) {
    std::cerr << "Build failed: " << result.error() << "\n";
    return;
}

// Access the surface (shared_ptr)
auto surface = result->surface;

// Query price and partials (~500ns each)
double m = 1.05;      // Moneyness (S/K)
double tau = 0.25;    // Time to maturity (years)
double sigma = 0.20;  // Volatility
double r = 0.05;      // Risk-free rate

double price = surface->value({m, tau, sigma, r});
double delta = surface->partial(0, {m, tau, sigma, r});  // ∂price/∂m
double vega = surface->partial(2, {m, tau, sigma, r});   // ∂price/∂σ
double gamma = surface->partial(1, {m, tau, sigma, r});  // ∂price/∂τ (often called theta)
```

### Factory Methods

**Three convenience factories for common use cases:**

```cpp
// 1. from_vectors: Explicit moneyness values
auto result1 = mango::PriceTableBuilder<4>::from_vectors(
    moneyness_grid, maturity_grid, vol_grid, rate_grid,
    K_ref, grid_spec, n_time, mango::OptionType::PUT);

// 2. from_strikes: Auto-computes moneyness from spot and strikes
auto result2 = mango::PriceTableBuilder<4>::from_strikes(
    spot, strikes, maturities, volatilities, rates,
    grid_spec, n_time, mango::OptionType::CALL);

// 3. from_chain: Extracts all parameters from OptionChain
auto result3 = mango::PriceTableBuilder<4>::from_chain(
    option_chain, grid_spec, n_time, mango::OptionType::PUT);

// All return std::expected<std::pair<builder, axes>, std::string>
```

### Automatic Grid Profiles

**Use profiles to auto-estimate both table grids and PDE grid/time steps:**

```cpp
auto result = mango::PriceTableBuilder<4>::from_chain_auto_profile(
    option_chain,
    mango::PriceTableGridProfile::High,
    mango::GridAccuracyProfile::High,
    mango::OptionType::PUT);

auto [builder, axes] = result.value();
auto surface_result = builder.build(axes);
```

**Python convenience wrapper** (see [PYTHON_GUIDE.md](PYTHON_GUIDE.md) for full API):

```python
import mango_option as mo

chain = mo.OptionChain()
chain.spot = 100.0
chain.strikes = [90, 95, 100, 105, 110]
chain.maturities = [0.25, 0.5, 1.0]
chain.implied_vols = [0.15, 0.20, 0.25]
chain.rates = [0.02, 0.03]
chain.dividend_yield = 0.0

surface = mo.build_price_table_surface_from_chain(
    chain,
    option_type=mo.OptionType.PUT,
    grid_profile=mo.PriceTableGridProfile.HIGH,
    pde_profile=mo.GridAccuracyProfile.HIGH,
)
```

**Real data benchmark (SPY, auto-grid profiles, interpolation-only timing):**

| Profile | Grid (m×τ×σ×r) | PDE solves | interp IV (µs) | interp IV/s | FDM IV (µs) | FDM IV/s | max err (bps) | avg err (bps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | 8×8×14×6 | 84 | 4.68 | 214k | 5275 | 190 | 90.5 | 52.5 |
| Medium | 10×10×20×8 | 160 | 4.30 | 233k | 5416 | 185 | 144.7 | 38.1 |
| High (default) | 12×12×30×10 | 300 | 3.83 | 261k | 5280 | 189 | 61.7 | 19.5 |
| Ultra | 15×15×43×12 | 516 | 3.85 | 260k | 5271 | 190 | 35.2 | 13.1 |

### Batch Queries on Price Surface

**Evaluate many points efficiently:**

```cpp
auto surface = result->surface;

std::vector<std::array<double, 4>> queries = {
    {1.00, 0.25, 0.20, 0.05},  // ATM, 3M, 20% vol, 5% rate
    {0.95, 0.50, 0.25, 0.03},  // ITM, 6M, 25% vol, 3% rate
    {1.10, 1.00, 0.15, 0.02},  // OTM, 1Y, 15% vol, 2% rate
};

for (const auto& coords : queries) {
    double price = surface->value(coords);
    double vega = surface->partial(2, coords);  // ∂price/∂σ
    std::cout << "m=" << coords[0] << ", price=" << price << ", vega=" << vega << "\n";
}
```

### Build Diagnostics

**Access performance and quality metrics:**

```cpp
auto result = builder.build(axes);

if (result.has_value()) {
    std::cout << "PDE solves: " << result->n_pde_solves << "\n";
    std::cout << "Build time: " << result->precompute_time_seconds << "s\n";

    const auto& stats = result->fitting_stats;
    std::cout << "Max B-spline residual: " << stats.max_residual_overall << "\n";
    std::cout << "Max condition number: " << stats.condition_max << "\n";
    std::cout << "Failed slices: " << stats.failed_slices_total << "\n";
}
```

---

## Custom PDE Solving

### Heat Equation Example

**Solve simple diffusion PDE:**

```cpp
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"
#include "src/pde/core/boundary_conditions.hpp"

// Define derived solver class using CRTP
class HeatSolver : public mango::PDESolver<HeatSolver> {
public:
    HeatSolver(std::shared_ptr<mango::Grid<double>> grid,
               mango::PDEWorkspace workspace,
               double diffusion_coeff)
        : PDESolver(grid, workspace)
        , D_(diffusion_coeff)
        , grid_(grid)
    {}

    // Implement CRTP interface
    auto left_boundary() const {
        return mango::DirichletBC([](double t) { return 0.0; });
    }

    auto right_boundary() const {
        return mango::DirichletBC([](double t) { return 0.0; });
    }

    auto spatial_operator() const {
        mango::operators::LaplacianPDE pde(D_);
        return mango::operators::make_spatial_operator(pde, *grid_);
    }

private:
    double D_;
    std::shared_ptr<mango::Grid<double>> grid_;
};

// Use the solver
int main() {
    // Create grid
    auto grid_spec = mango::GridSpec<double>::uniform(0.0, 1.0, 101).value();
    mango::TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

    auto grid = std::make_shared<mango::Grid<double>>(grid_spec.generate(), time);

    // Create workspace
    std::pmr::synchronized_pool_resource pool;
    auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

    // Create solver
    HeatSolver solver(grid, workspace, 0.1);  // D = 0.1

    // Initial condition: u(x, 0) = sin(π·x)
    solver.initialize([](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::sin(M_PI * x[i]);
        }
    });

    // Solve
    auto result = solver.solve();

    if (result.has_value()) {
        auto solution = grid->solution();
        std::cout << "Solution at t=" << time.t_end() << ":\n";
        for (size_t i = 0; i < solution.size(); i += 10) {
            std::cout << "u[" << i << "] = " << solution[i] << "\n";
        }
    }

    return 0;
}
```

---

## Error Handling Patterns

### Checking for Errors

**std::expected provides type-safe error handling:**

```cpp
auto result = solver.solve_impl(query);

// Pattern 1: if/else
if (result.has_value()) {
    std::cout << "IV: " << result->implied_vol << "\n";
} else {
    std::cerr << "Error: " << result.error().message << "\n";
}

// Pattern 2: value_or
double iv = result.has_value() ? result->implied_vol : 0.20;  // Fallback

// Pattern 3: Monadic transform
auto vol_pct = result.transform([](const auto& r) {
    return r.implied_vol * 100.0;
});
```

### Error Code Handling

**Detailed error diagnostics:**

```cpp
auto result = solver.solve_impl(query);

if (!result.has_value()) {
    const auto& error = result.error();

    switch (error.code) {
        case mango::IVErrorCode::NegativeSpot:
            std::cerr << "Invalid spot price\n";
            break;

        case mango::IVErrorCode::ArbitrageViolation:
            std::cerr << "Arbitrage violation: " << error.message << "\n";
            break;

        case mango::IVErrorCode::MaxIterationsExceeded:
            std::cerr << "Failed to converge after " << error.iterations << " iterations\n";
            std::cerr << "Final error: " << error.final_error << "\n";
            if (error.last_vol) {
                std::cerr << "Last volatility: " << *error.last_vol << "\n";
            }
            break;

        default:
            std::cerr << "Error [" << static_cast<int>(error.code) << "]: "
                      << error.message << "\n";
    }
}
```

### Monadic Validation Chains

**Compose validations with .and_then():**

```cpp
// Inside IVSolverFDM
auto validate_query(const IVQuery& query) const
    -> std::expected<std::monostate, IVError>
{
    return validate_positive_parameters(query)
        .and_then([&](auto) { return validate_arbitrage_bounds(query); })
        .and_then([&](auto) { return validate_grid_params(); });
}

// Short-circuit on first error
auto validation_result = validate_query(query);
if (!validation_result.has_value()) {
    return std::unexpected(validation_result.error());  // Propagate error
}
```

---

## Batch Processing

### American Option Batch

**Not yet implemented - use manual loop with OpenMP:**

```cpp
std::vector<mango::PricingParams> params_batch = { /* ... */ };
std::vector<std::expected<mango::AmericanOptionResult, mango::SolverError>> results(params_batch.size());

// Shared grid for all options (requires compute_global_grid_for_batch)
auto [grid_spec, time_domain] = mango::compute_global_grid_for_batch(params_batch);

#pragma omp parallel for
for (size_t i = 0; i < params_batch.size(); ++i) {
    std::pmr::synchronized_pool_resource pool;
    auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

    mango::AmericanOptionSolver solver(params_batch[i], workspace);
    results[i] = solver.solve();
}
```

### IV Batch (Built-in)

**Use solve_batch_impl() for parallel IV calculation:**

```cpp
std::vector<mango::IVQuery> queries = load_market_data();

mango::IVSolverFDM solver(config);
auto batch = solver.solve_batch_impl(queries);

// Results and statistics
std::cout << "Succeeded: " << (batch.results.size() - batch.failed_count) << "\n";
std::cout << "Failed: " << batch.failed_count << "\n";

// Process individual results
for (size_t i = 0; i < batch.results.size(); ++i) {
    if (batch.results[i].has_value()) {
        process_success(batch.results[i].value());
    } else {
        handle_error(batch.results[i].error());
    }
}
```

---

## Advanced Topics

### ThreadWorkspaceBuffer for Parallel Operations

**Zero-allocation parallel workloads with 64-byte alignment:**

```cpp
#include "src/support/thread_workspace.hpp"
#include "src/math/bspline_collocation_workspace.hpp"

// Example: Parallel B-spline fitting with zero allocations per iteration
const size_t n_axis = 100;
const size_t n_slices = 1000;

MANGO_PRAGMA_PARALLEL
{
    // Allocate once per thread (64-byte aligned for AVX-512)
    mango::ThreadWorkspaceBuffer buffer(
        mango::BSplineCollocationWorkspace<double>::required_bytes(n_axis));

    // Create workspace once per thread
    auto ws = mango::BSplineCollocationWorkspace<double>::from_bytes(
        buffer.bytes(), n_axis).value();

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < n_slices; ++i) {
        // Reuse workspace - solver overwrites arrays each iteration
        // Zero allocations in this hot path
        solver.fit_with_workspace(values[i], ws, config);
    }
}
```

**For PDE solving in parallel:**

```cpp
#include "src/pde/core/american_pde_workspace.hpp"

MANGO_PRAGMA_PARALLEL
{
    // Per-thread buffer for PDE workspace
    mango::ThreadWorkspaceBuffer buffer(
        mango::AmericanPDEWorkspace::required_bytes(n_space));

    auto ws = mango::AmericanPDEWorkspace::from_bytes(
        buffer.bytes(), n_space).value();

    MANGO_PRAGMA_FOR_STATIC
    for (size_t i = 0; i < batch_size; ++i) {
        solver.solve_with_workspace(options[i], ws);
    }
}
```

### Multi-Sinh Grids

**Concentrate resolution at multiple locations:**

```cpp
// Define clusters for ATM and deep ITM
std::vector<mango::MultiSinhCluster<double>> clusters = {
    {.center_x = 0.0, .alpha = 2.5, .weight = 2.0},   // ATM (higher weight)
    {.center_x = -0.2, .alpha = 2.0, .weight = 1.0}   // 20% ITM
};

auto grid_spec = mango::GridSpec<double>::multi_sinh_spaced(
    -3.0, 3.0, 201, clusters
).value();

// Use for price table or batch solving
```

**Guidance:**
- Use multi-sinh only when strikes differ by >20% (Δx ≥ 0.18)
- Auto-merge clusters closer than 0.3/α_avg
- Each cluster adds ~10% computational cost

### Grid Accuracy Tuning

**Control spatial/temporal resolution tradeoffs:**

```cpp
mango::GridAccuracyParams accuracy{
    .n_sigma = 5.0,                 // Domain half-width (±5 std devs)
    .alpha = 2.5,                   // Sinh concentration
    .tol = 1e-3,                    // Target spatial error
    .c_t = 0.75,                    // CFL safety factor
    .min_spatial_points = 100,
    .max_spatial_points = 1200,
    .max_time_steps = 5000
};

auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params, accuracy);
```

**Tolerance guidelines:**
- `tol = 1e-2`: Fast mode (~100-150 points, ~5ms)
- `tol = 1e-3`: Standard mode (~300-400 points, ~50ms)
- `tol = 1e-6`: High accuracy (~1200 points, ~300ms)

---

## Related Documentation

- **Software Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Mathematical Foundations:** [MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)
- **Workflow:** [../CLAUDE.md](../CLAUDE.md)
- **USDT Tracing:** [TRACING.md](TRACING.md)
