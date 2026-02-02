# API Usage Guide

Practical examples and common usage patterns for the mango-option library.

**For software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**
**For mathematical formulations, see [MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)**

## Table of Contents

1. [Quick Start](#quick-start)
2. [American Option Pricing](#american-option-pricing)
3. [Implied Volatility Calculation](#implied-volatility-calculation)
4. [Price Table Pre-Computation](#price-table-pre-computation)
5. [Discrete Dividends](#discrete-dividends)
6. [Batch Processing](#batch-processing)
7. [Custom PDE Solving](#custom-pde-solving)
8. [Error Handling Patterns](#error-handling-patterns)
9. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Minimal American Option Example

```cpp
#include "src/option/american_option.hpp"
#include <iostream>

int main() {
    // Option parameters
    mango::PricingParams params(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);  // volatility

    // Solve (auto grid, auto workspace)
    auto result = mango::solve_american_option(params);

    if (result.has_value()) {
        std::cout << "Price: " << result->value_at(100.0) << "\n";
        std::cout << "Delta: " << result->delta() << "\n";
        std::cout << "Gamma: " << result->gamma() << "\n";
    }

    return 0;
}
```

### Minimal IV Calculation Example

```cpp
#include "src/option/iv_solver.hpp"
#include <iostream>

int main() {
    // Option specification
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT
    };

    // IV query
    mango::IVQuery query(spec, 10.45);

    // Solve
    mango::IVSolver solver(mango::IVSolverConfig{});
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
mango::PricingParams params(
    mango::OptionSpec{
        .spot = 105.0,          // 5% ITM put
        .strike = 100.0,
        .maturity = 0.5,        // 6 months
        .rate = 0.03,
        .dividend_yield = 0.01,
        .option_type = mango::OptionType::PUT},
    0.25);  // volatility

// Solve (auto grid, auto workspace)
auto result = mango::solve_american_option(params);

if (result.has_value()) {
    double price = result->value_at(105.0);
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

// Create workspace
size_t n = grid_spec.n_points();
std::pmr::synchronized_pool_resource pool;
std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

// Pass custom grid to solver via PDEGridConfig
auto solver = mango::AmericanOptionSolver::create(
    params, workspace,
    mango::PDEGridConfig{.grid_spec = grid_spec, .n_time = n_time}
).value();

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

---

## Implied Volatility Calculation

### FDM-Based IV (Robust, ~19ms)

**Uses Brent's method with nested PDE pricing:**

```cpp
#include "src/option/iv_solver.hpp"

// Option specification
mango::OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .option_type = mango::OptionType::PUT
};

// IV query with market price
mango::IVQuery query(spec, 10.45);

// Configure solver (optional)
mango::IVSolverConfig config{
    .root_config = mango::RootFindingConfig{
        .max_iter = 100,
        .tolerance = 1e-6
    },
    // Control PDE grid accuracy (higher accuracy = lower IV error, slower)
    .grid = mango::GridAccuracyParams{.tol = 1e-3}
};

// Solve
mango::IVSolver solver(config);
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

### Custom Grid for IV

**Control PDE accuracy via GridAccuracyParams (recommended):**

```cpp
mango::IVSolverConfig config{
    .grid = mango::GridAccuracyParams{
        .tol = 1e-4,                // Tighter truncation error
        .min_spatial_points = 200,
        .max_spatial_points = 800
    }
};

mango::IVSolver solver(config);
auto result = solver.solve(query);
```

**Override auto-estimation with explicit grid (advanced):**

```cpp
auto grid_spec = mango::GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.5).value();

mango::IVSolverConfig config{
    .grid = mango::PDEGridConfig{
        .grid_spec = grid_spec,
        .n_time = 2000
    }
};

mango::IVSolver solver(config);
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

| Profile | PDE solves | ATM (bps) | Near-OTM (bps) | Deep-OTM (bps) | Near-ITM (bps) | Deep-ITM (bps)† | Price RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 100 | 10.4 | 2.8 | 20.7 | 13.9 | 2006 | $0.016 |
| Medium | 240 | 2.7 | 2.9 | 22.7 | 1.1 | 2070 | $0.005 |
| High (default) | 495 | 0.4 | 3.3 | 22.8 | 1.3 | 2023 | $0.005 |
| Ultra | 812 | 0.3 | 2.9 | 22.0 | 0.8 | 2002 | $0.004 |

†Deep-ITM and deep-OTM options share the same low-vega characteristic: vega is near zero, so even a tiny price error (< $0.01) maps to thousands of bps in IV space. The actual price-relative error remains small — deep-ITM price RMSE is < $0.001 across all profiles. **Price RMSE is the stable metric** across all moneyness regimes.

### Using Price Surface with InterpolatedIVSolver

```cpp
#include "src/option/interpolated_iv_solver.hpp"

// Create IV solver from AmericanPriceSurface
auto iv_solver = mango::InterpolatedIVSolver::create(std::move(aps)).value();

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

## Discrete Dividends

A cash dividend at time t creates a discontinuity: at the ex-dividend instant, the spot drops by the dividend amount. Pricing must enforce the jump condition V(t⁻, S) = V(t⁺, S − D) at each dividend date.

### Pricing with Discrete Dividends

Pass discrete dividends to `PricingParams`:

```cpp
#include "src/option/american_option.hpp"

mango::PricingParams params(
    mango::OptionSpec{
        .spot = 100.0,
        .strike = 100.0,
        .maturity = 1.0,
        .rate = 0.05,
        .dividend_yield = 0.01,   // continuous yield (enters PDE coefficients)
        .option_type = mango::OptionType::PUT},
    0.20,  // volatility
    {      // discrete dividends
        mango::Dividend{.calendar_time = 0.25, .amount = 1.50},
        mango::Dividend{.calendar_time = 0.50, .amount = 1.50}
    });

auto result = mango::solve_american_option(params);
```

The solver handles dividends via temporal events:

1. `estimate_pde_grid()` inserts mandatory time steps at each dividend date and widens the spatial domain to accommodate post-dividend spot shifts.
2. The PDE solver marches backward from T to 0. At each dividend time τ, a temporal event fires.
3. The temporal event rebuilds a cubic spline from the current solution, then evaluates it at shifted log-moneyness x′ = ln(exp(x) − D/K) for each grid point — applying the jump condition.
4. The solver continues to the next dividend or t = 0.

Continuous and discrete dividends combine naturally: the continuous yield enters the Black-Scholes PDE coefficients, while discrete dividends operate through the temporal event mechanism.

### Continuous Yield Approximation

When discrete dividend timing is unimportant, approximate with a continuous yield:

```cpp
double total_div = 3.00;    // sum of discrete dividends
double T = 1.0;             // maturity
double S = 100.0;           // spot
double equiv_yield = total_div / (S * T);  // ~3% continuous yield

mango::PricingParams params(
    mango::OptionSpec{
        .spot = S, .strike = 100.0, .maturity = T,
        .rate = 0.05, .dividend_yield = equiv_yield,
        .option_type = mango::OptionType::PUT},
    0.20);  // no discrete dividends
```

This solves a single smooth PDE without temporal events. The tradeoff: continuous yield spreads the dividend effect uniformly across maturity, which misprices options with maturities near dividend dates.

### Segmented Surfaces for IV (Discrete Dividends)

When computing implied volatility for many options that share the same dividend schedule — the typical case when calibrating a vol surface from market data — the FDM solver is too slow per query. The segmented surface builder pre-computes price surfaces that can be queried in microseconds.

**Why segmentation?** A single B-spline surface cannot fit the discontinuity at a dividend date. The builder splits the maturity axis into segments separated by dividend dates and solves each independently.

**How segments connect.** The builder works backward from expiry:

1. **Segment 0** (nearest to expiry, τ ∈ [0, τ₁]): Built with standard EEP (Early Exercise Premium) decomposition. The initial condition is the option payoff.

2. **Segment k** (τ ∈ [τ_k, τ_{k+1}]): Built in raw-price mode. Its initial condition comes from the previous segment's surface, evaluated at the post-dividend spot: S_adj = S − D_k. This embeds the dividend jump into the initial condition, so no spot adjustment is needed at query time for these segments.

The result is a `SegmentedPriceSurface` — an ordered list of segments that together cover [0, T]. At query time, the surface finds the segment covering the requested τ and evaluates it directly.

**Why multiple K_ref values?** Cash dividends break the scale invariance that American options normally have in strike. A single reference-strike surface cannot accurately interpolate across strikes far from K_ref. The builder constructs surfaces at several reference strikes and interpolates across them with Catmull-Rom splines in log(K_ref). The result is a `SegmentedMultiKRefSurface`.

### Building a Segmented IV Solver

The `make_interpolated_iv_solver` factory handles all the segmented construction:

```cpp
#include "src/option/iv_solver_factory.hpp"

mango::IVSolverFactoryConfig config{
    .option_type = mango::OptionType::PUT,
    .spot = 100.0,
    .dividend_yield = 0.01,
    .moneyness_grid = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3},
    .vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40},
    .rate_grid = {0.02, 0.03, 0.05, 0.07},
    .path = mango::SegmentedIVPath{
        .maturity = 1.0,
        .discrete_dividends = {
            mango::Dividend{.calendar_time = 0.25, .amount = 1.50},
            mango::Dividend{.calendar_time = 0.50, .amount = 1.50}},
        .kref_config = {.K_refs = {80.0, 100.0, 120.0}},
    },
};

auto solver = mango::make_interpolated_iv_solver(config);
```

The factory dispatches on the `path` variant:

- **`StandardIVPath`** — no discrete dividends. Builds a single `AmericanPriceSurface` with a maturity grid. Use this for continuous-dividend options.
- **`SegmentedIVPath`** — discrete dividends present. Builds a `SegmentedMultiKRefSurface` with backward chaining. Use this when dividends are known.

### Standard Path (No Discrete Dividends)

```cpp
mango::IVSolverFactoryConfig config{
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

auto solver = mango::make_interpolated_iv_solver(config);
```

### Querying the Solver

Both paths produce an `AnyIVSolver` with the same interface:

```cpp
mango::IVQuery query(
    mango::OptionSpec{
        .spot = 100.0, .strike = 95.0, .maturity = 0.5,
        .rate = 0.05, .dividend_yield = 0.0,
        .option_type = mango::OptionType::PUT},
    7.5);  // market price

auto result = solver->solve(query);

if (result.has_value()) {
    std::cout << "IV: " << result->implied_vol << "\n";
}
```

For batch IV queries, see [Batch Processing § IV Batch](#iv-batch).

### Configuration Notes

- **`kref_config`** is optional. When omitted, the builder selects reference strikes automatically at log-spaced intervals around the spot. Explicit K_refs are useful when you know the strike range of interest.
- **Continuous yield** applies inside each segment's PDE. Discrete dividends operate at segment boundaries. Both can be used together.

### Choosing an Approach

| Scenario | Approach | Latency |
|---|---|---|
| Price one option with dividends | `solve_american_option(params)` | ~5–20ms |
| Price one option without dividends | `solve_american_option(params)` | ~5–20ms |
| IV surface (no dividends) | `make_interpolated_iv_solver` + `StandardIVPath` | ~3.5μs/query |
| IV surface (with dividends) | `make_interpolated_iv_solver` + `SegmentedIVPath` | ~3.5μs/query |

Use the FDM solver for individual prices or when you need per-option control (custom grids, yield curves). Use the segmented surface builder when you need to evaluate many IV queries against the same dividend schedule — the upfront build cost is amortized across thousands of queries. The continuous yield approximation avoids temporal events entirely, at the cost of accuracy near dividend dates.

---

## Batch Processing

### American Option Batch

**Parallel batch solver with automatic optimization:**

```cpp
#include "src/option/american_option_batch.hpp"

// Build batch of options
std::vector<mango::PricingParams> batch;
for (double K : {90.0, 95.0, 100.0, 105.0, 110.0}) {
    batch.emplace_back(
        mango::OptionSpec{
            .spot = 100.0, .strike = K, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);  // volatility
}

mango::BatchAmericanOptionSolver solver;
auto results = solver.solve_batch(batch);

std::cout << "Succeeded: " << (results.results.size() - results.failed_count) << "\n";

for (size_t i = 0; i < results.results.size(); ++i) {
    if (results.results[i].has_value()) {
        std::cout << "Strike " << batch[i].strike
                  << ": " << results.results[i]->value() << "\n";
    }
}
```

**Shared grid mode** solves all options on the same spatial grid (required for price table construction):

```cpp
auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);
```

### Chain Solving (Normalized Batch Optimization)

The solver exploits a scale-invariance property of the Black-Scholes PDE: the normalized PDE with S = K = 1 has the same solution shape for all strikes sharing the same PDE parameters. The solver groups options by (σ, r, q, maturity, option type), solves one normalized PDE per group, then rescales each solution for the option's actual S/K ratio — yielding up to **19,000× speedup** over solving each option independently.

This optimization is automatic — `solve_batch()` routes eligible batches to the normalized path. A batch with mixed maturities, volatilities, or rates produces multiple groups, each solved with a single normalized PDE.

**Eligibility requirements:**
- `use_shared_grid = true`
- No discrete dividends (these break scale invariance)
- Positive spot and strike values
- Grid spacing and domain width within stability constraints
- No `SetupCallback` (per-option callbacks are incompatible with shared PDE solves)

**Disabling chain solving:**

```cpp
// Force per-option PDE solves (for benchmarking or debugging)
mango::BatchAmericanOptionSolver solver;
solver.set_use_normalized(false);
auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);
```

Providing a `SetupCallback` also disables the normalized path, since per-option configuration is incompatible with solving a single shared PDE.

### Grid Accuracy and Snapshots

**Fluent API for batch configuration:**

```cpp
mango::BatchAmericanOptionSolver solver;
solver.set_grid_accuracy(mango::GridAccuracyParams{.tol = 1e-4})
      .set_snapshot_times(std::span{maturity_grid});

auto results = solver.solve_batch(batch, /*use_shared_grid=*/true);
```

### IV Batch

**Use solve_batch() for parallel IV calculation:**

```cpp
std::vector<mango::IVQuery> queries;
for (const auto& [strike, price] : market_data) {
    mango::OptionSpec spec{
        .spot = 100.0,
        .strike = strike,
        .maturity = 0.25,
        .rate = 0.03,
        .dividend_yield = 0.0,
        .option_type = mango::OptionType::CALL
    };
    queries.push_back(mango::IVQuery(spec, price));
}

mango::IVSolver solver(config);
auto batch = solver.solve_batch(queries);

std::cout << "Succeeded: " << (batch.results.size() - batch.failed_count) << "\n";
std::cout << "Failed: " << batch.failed_count << "\n";

for (size_t i = 0; i < batch.results.size(); ++i) {
    if (batch.results[i].has_value()) {
        std::cout << "Strike " << queries[i].strike
                  << ": IV = " << batch.results[i]->implied_vol << "\n";
    }
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
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), &pool);
    auto workspace = mango::PDEWorkspace::from_buffer(buffer, n).value();

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
// Inside IVSolver
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

auto [grid_spec, time_domain] = mango::estimate_pde_grid(params, accuracy);
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
