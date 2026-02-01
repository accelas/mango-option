# API Usage Guide

Practical examples and common usage patterns for the mango-option library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For mathematical formulations, see [MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)**

## Table of Contents

1. [Quick Start](#quick-start)
2. [American Option Pricing](#american-option-pricing)
3. [Implied Volatility Calculation](#implied-volatility-calculation)
4. [Price Table Pre-Computation](#price-table-pre-computation)
5. [Discrete Dividend IV](#discrete-dividend-iv)
6. [Custom PDE Solving](#custom-pde-solving)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Batch Processing](#batch-processing)
9. [Advanced Topics](#advanced-topics)

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
    auto result = solver.solve(query);

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

**Discrete dividends via IV solver factory:**
```cpp
#include "src/option/iv_solver_factory.hpp"

// The make_iv_solver factory dispatches on the path variant
mango::IVSolverConfig config{
    .option_type = mango::OptionType::PUT,
    .spot = 100.0,
    .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
    .vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
    .rate_grid = {0.02, 0.03, 0.05, 0.07},
    .path = mango::SegmentedIVPath{
        .maturity = 1.0,
        .discrete_dividends = {{0.25, 1.50}, {0.50, 1.50}},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    },
};

auto solver = mango::make_iv_solver(config);
auto result = solver->solve(query);
```

See [Discrete Dividend IV](#discrete-dividend-iv) for the full workflow.

---

## Implied Volatility Calculation

### FDM-Based IV (Robust, ~19ms)

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
    },
    // Control PDE grid accuracy (higher accuracy = lower IV error, slower)
    .grid_accuracy = mango::GridAccuracyParams{.tol = 1e-3}
};

// Solve
mango::IVSolverFDM solver(config);
auto result = solver.solve(query);

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
auto batch = solver.solve_batch(queries);

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

**Control PDE accuracy via GridAccuracyParams (recommended):**

```cpp
mango::IVSolverFDMConfig config{
    .grid_accuracy = mango::GridAccuracyParams{
        .tol = 1e-4,                // Tighter truncation error
        .min_spatial_points = 200,
        .max_spatial_points = 800
    }
};

mango::IVSolverFDM solver(config);
auto result = solver.solve(query);
```

**Override auto-estimation with manual grid (advanced):**

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
auto result = solver.solve(query);
```

---

## Price Table Pre-Computation

### Building a 4D Price Surface

**Pre-compute American option prices across parameter space:**

```cpp
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/american_price_surface.hpp"

// Define 4D parameter grids
std::vector<double> moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};  // m = S/K
std::vector<double> maturity_grid = {0.027, 0.1, 0.25, 0.5, 1.0, 2.0};      // τ (years)
std::vector<double> vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};         // σ
std::vector<double> rate_grid = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};         // r

double K_ref = 100.0;  // Reference strike price

// Create builder (always stores EEP for ~5x better interpolation accuracy)
auto factory_result = mango::PriceTableBuilder<4>::from_vectors(
    moneyness_grid, maturity_grid, vol_grid, rate_grid,
    K_ref,
    mango::GridAccuracyParams{},  // auto-estimate PDE grid
    mango::OptionType::PUT);

if (!factory_result.has_value()) {
    std::cerr << "Factory creation failed: " << factory_result.error() << "\n";
    return;
}

auto [builder, axes] = std::move(factory_result.value());

// Build price table (parallelized with OpenMP)
auto result = builder.build(axes);

if (!result.has_value()) {
    std::cerr << "Build failed: " << result.error() << "\n";
    return;
}

// Wrap in AmericanPriceSurface for full price reconstruction
auto aps = mango::AmericanPriceSurface::create(
    result->surface, mango::OptionType::PUT).value();

// Query American option prices (~500ns)
double price = aps.price(spot, strike, tau, sigma, rate);

// Greeks via interpolation (see accuracy note below)
double delta = aps.delta(spot, strike, tau, sigma, rate);
double vega  = aps.vega(spot, strike, tau, sigma, rate);
```

### Interpolated Greek Accuracy

`AmericanPriceSurface` computes Greeks by combining B-spline partial derivatives of the EEP surface with exact Black-Scholes Greeks for the European component:

| Greek | Method | Accuracy |
|---|---|---|
| **Price** | B-spline interpolation + analytical European | Best: ~$0.005 RMSE (O(h⁴) B-spline) |
| **Delta** | B-spline ∂EEP/∂m + analytical European delta | Good: one derivative lowers B-spline order to O(h³) |
| **Vega** | B-spline ∂EEP/∂σ + analytical European vega | Good: same as delta |
| **Theta** | B-spline ∂EEP/∂τ + analytical European theta | Good: same as delta |
| **Gamma** | B-spline ∂²EEP/∂m² + analytical European gamma | Good: O(h²) analytical second derivative |

Gamma uses the analytical B-spline second derivative with a log-moneyness chain rule correction: ∂²f/∂m² = (g″(x) − g′(x)) / m².

**Measured accuracy** (interpolated vs PDE solver, σ=0.20, r=0.05, q=0.02):

| Greek | Max Abs Error | Max Rel Error | Notes |
|---|---|---|---|
| **Price** | $0.086 | 1.9% | |
| **Delta** | 0.0087 | 2.8% | |
| **Gamma** | 0.0024 | 7.3% | Worst at short τ; < 1.3% for τ ≥ 0.5 |
| **Theta** | $0.15 | 3.3% | |

Accuracy degrades at short maturities (τ < 0.5yr) where Greeks have sharper curvature. When in doubt, use the PDE solver directly (`AmericanOptionSolver`) for authoritative Greeks.

### Factory Methods

**Three convenience factories for common use cases:**

```cpp
// PDE grid: auto-estimated (recommended) or explicit
mango::PDEGridSpec pde_grid = mango::GridAccuracyParams{};

// 1. from_vectors: Explicit moneyness values
auto result1 = mango::PriceTableBuilder<4>::from_vectors(
    moneyness_grid, maturity_grid, vol_grid, rate_grid,
    K_ref, pde_grid, mango::OptionType::PUT);

// 2. from_strikes: Auto-computes moneyness from spot and strikes
auto result2 = mango::PriceTableBuilder<4>::from_strikes(
    spot, strikes, maturities, volatilities, rates,
    pde_grid, mango::OptionType::CALL);

// 3. from_grid: Extracts all parameters from OptionGrid
auto result3 = mango::PriceTableBuilder<4>::from_grid(
    option_grid, pde_grid, mango::OptionType::PUT);

// All return std::expected<std::pair<builder, axes>, PriceTableError>
```

### Automatic Grid Profiles

**Use profiles to auto-estimate both table grids and PDE grid/time steps:**

```cpp
auto result = mango::PriceTableBuilder<4>::from_grid_auto_profile(
    option_grid,
    mango::PriceTableGridProfile::High,
    mango::GridAccuracyProfile::High,
    mango::OptionType::PUT);

auto [builder, axes] = result.value();
auto surface_result = builder.build(axes);
```

**Python convenience wrapper** (see [PYTHON_GUIDE.md](PYTHON_GUIDE.md) for full API):

```python
import mango_option as mo

chain = mo.OptionGrid()
chain.spot = 100.0
chain.strikes = [90, 95, 100, 105, 110]
chain.maturities = [0.25, 0.5, 1.0]
chain.implied_vols = [0.15, 0.20, 0.25]
chain.rates = [0.02, 0.03]
chain.dividend_yield = 0.0

surface = mo.build_price_table_surface_from_grid(
    chain,
    option_type=mo.OptionType.PUT,
    grid_profile=mo.PriceTableGridProfile.HIGH,
    pde_profile=mo.GridAccuracyProfile.HIGH,
)
```

**Real data benchmark (SPY 7-day puts, auto-grid profiles with EEP decomposition):**

| Profile | PDE solves | ATM (bps) | Near-OTM (bps) | Deep-OTM (bps) | Price RMSE |
|---|---:|---:|---:|---:|---:|
| Low | 100 | 10.0 | 2.7 | 20.7 | $0.014 |
| Medium | 240 | 4.4 | 3.0 | 22.5 | $0.008 |
| High (default) | 495 | 0.1 | 2.8 | 22.2 | $0.005 |
| Ultra | 812 | 0.2 | 3.3 | 22.6 | $0.005 |

IV error in bps varies with vega: a constant ~$0.005 price error maps to <1 bps near-ATM but 20+ bps for deep OTM short-dated options where vega is tiny. Price RMSE is the stable metric.

### Using Price Surface with IVSolverInterpolated

```cpp
#include "src/option/iv_solver_interpolated.hpp"

// Create IV solver from AmericanPriceSurface
auto iv_solver = mango::IVSolverInterpolated::create(std::move(aps)).value();

// Solve IV — internally uses EEP reconstruction + Newton iteration
auto iv_result = iv_solver.solve(iv_query);
```

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

## Discrete Dividend IV

### Factory-Based Solver Creation

The `make_iv_solver` factory builds an interpolated IV solver that automatically handles discrete dividends. When no dividends are specified, it takes the standard single-surface path. When dividends are present, it builds a segmented surface with backward chaining.

```cpp
#include "src/option/iv_solver_factory.hpp"

// Standard path (no discrete dividends):
mango::IVSolverConfig config{
    .option_type = mango::OptionType::PUT,
    .spot = 100.0,
    .dividend_yield = 0.02,
    .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},
    .vol_grid = {0.10, 0.15, 0.20, 0.30, 0.40},
    .rate_grid = {0.02, 0.03, 0.05, 0.07},
    .path = mango::StandardIVPath{
        .maturity_grid = {0.1, 0.25, 0.5, 1.0},
    },
};

// Segmented path (with discrete dividends):
mango::IVSolverConfig div_config{
    .option_type = mango::OptionType::PUT,
    .spot = 100.0,
    .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
    .vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
    .rate_grid = {0.02, 0.03, 0.05, 0.07},
    .path = mango::SegmentedIVPath{
        .maturity = 1.0,
        .discrete_dividends = {{0.25, 1.50}, {0.50, 1.50}},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    },
};

auto solver = mango::make_iv_solver(config);
```

### Solving IV Queries

The solver exposes the same interface regardless of whether dividends are present:

```cpp
mango::IVQuery query;
query.spot = 100.0;
query.strike = 95.0;
query.maturity = 0.5;
query.rate = mango::RateSpec{0.05};
query.type = mango::OptionType::PUT;
query.market_price = 7.5;

auto result = solver->solve(query);

if (result.has_value()) {
    std::cout << "IV: " << result->implied_vol << "\n";
}

// Batch solving works the same way
auto batch = solver->solve_batch(queries);
```

### Configuration Notes

- **No dividends:** Use `StandardIVPath` with a `maturity_grid` (vector of maturities). The factory takes the standard single-surface path.
- **With dividends:** Use `SegmentedIVPath` with a single `maturity`, `discrete_dividends`, and optional `kref_config`. The factory segments the time axis at each dividend date and chains surfaces backward.
- **`kref_config`** is optional. When omitted, it defaults to three reference strikes at {0.8S, S, 1.2S}. Cash dividends break price homogeneity in strike, so multiple K_ref surfaces are built and interpolated to maintain accuracy.

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
auto result = solver.solve(query);

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
auto result = solver.solve(query);

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

**Use solve_batch() for parallel IV calculation:**

```cpp
std::vector<mango::IVQuery> queries = load_market_data();

mango::IVSolverFDM solver(config);
auto batch = solver.solve_batch(queries);

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
- **USDT Tracing:** [TRACING.md](TRACING.md)
