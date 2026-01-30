# Simple API Guide

The `mango::simple` namespace provides an opinionated, high-level API for American option pricing and implied volatility calculation. It wraps the lower-level PDE solver infrastructure into convenience functions that handle grid estimation, workspace allocation, and error conversion automatically.

## Quick Start

### Pricing

```cpp
#include "src/simple/pricing.hpp"

// Price an ATM put
auto result = mango::simple::price(
    100.0,  // spot
    100.0,  // strike
    1.0,    // maturity (years)
    0.20,   // volatility (20%)
    0.05,   // risk-free rate (5%)
    0.02,   // dividend yield (2%)
    mango::OptionType::PUT);

if (result.has_value()) {
    std::cout << "Price: " << *result << "\n";
} else {
    std::cerr << "Error: " << result.error() << "\n";
}
```

### Implied Volatility

```cpp
auto iv = mango::simple::implied_vol(
    100.0,  // spot
    100.0,  // strike
    1.0,    // maturity
    8.50,   // market price
    0.05,   // rate
    0.02,   // dividend yield
    mango::OptionType::PUT);

if (iv.has_value()) {
    std::cout << "IV: " << *iv << "\n";
}
```

Both functions return `std::expected<double, std::string>`. Errors are human-readable strings.

## Price Tables

For repeated queries (e.g., fitting a vol surface), pre-compute a price table. This builds a 4D B-spline surface over (moneyness, maturity, volatility, rate), enabling ~500ns price lookups instead of ~5ms PDE solves.

### Build

```cpp
#include "src/simple/price_table.hpp"

// Default config: 21×15×15×7 grid, moneyness [0.5, 2.0], etc.
auto table = mango::simple::build_price_table();

// Or customize:
mango::simple::PriceTableConfig config{
    .type = mango::OptionType::PUT,
    .strike_ref = 100.0,
    .n_moneyness = 31,
    .n_maturity = 20,
    .n_volatility = 20,
    .n_rate = 9,
    .moneyness_min = 0.5,
    .moneyness_max = 2.0,
    .maturity_min = 0.01,
    .maturity_max = 2.0,
    .vol_min = 0.05,
    .vol_max = 1.0,
    .rate_min = -0.01,
    .rate_max = 0.10,
};
auto table = mango::simple::build_price_table(config);
```

### Query

```cpp
if (table.has_value()) {
    // moneyness = S/K, tau, sigma, rate
    double price = table->value(1.0, 0.5, 0.20, 0.05);
}
```

### Save and Load

Tables persist as Apache Arrow IPC files for zero-copy loading:

```cpp
// Save
table->save("put_table.arrow");

// Load (later session)
auto loaded = mango::simple::load_price_table("put_table.arrow");
```

## Fast Implied Volatility

Create an interpolation-based IV solver from a price table for ~30us IV calculations (vs ~143ms with FDM):

```cpp
auto solver = mango::simple::make_iv_solver(*table);
if (solver.has_value()) {
    mango::IVQuery query;
    query.spot = 100.0;
    query.strike = 100.0;
    query.maturity = 0.5;
    query.rate = 0.05;
    query.dividend_yield = 0.02;
    query.type = mango::OptionType::PUT;
    query.market_price = 8.50;

    auto iv = solver->solve_impl(query);
    if (iv.has_value()) {
        std::cout << "IV: " << iv->implied_vol << "\n";
    }
}
```

## Market Data Integration

The simple namespace also includes market data adapters. See `ChainBuilder` for constructing option chains from yfinance, Databento, or IBKR data:

```cpp
#include "src/simple/simple.hpp"

// Build chain from yfinance JSON
auto chain = mango::simple::ChainBuilder()
    .spot(415.0)
    .ticker("SPY")
    .add_expiry(expiry_slice)
    .build();
```

## When to Use Lower-Level APIs

The simple API is suitable for most use cases. Reach for the lower-level APIs when you need:

- **Custom grid control**: `AmericanOptionSolver` with manual `GridSpec` and `TimeDomain`
- **Yield curve rates**: `PricingParams` with `RateSpec` variant (simple API uses scalar rates)
- **Discrete dividends**: `PricingParams::discrete_dividends` schedule
- **Batch processing**: `AmericanOptionBatch` for solving many options on a shared grid
- **Custom B-spline grids**: `PriceTableBuilder::from_chain_auto_profile()` for adaptive grid estimation
- **Solution snapshots**: `AmericanOptionResult::at_time()` for time-stepping data
- **Greeks**: `AmericanOptionResult::delta()`, `gamma()`, `theta()`
