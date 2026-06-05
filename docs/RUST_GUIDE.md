# Rust Guide

Safe Rust bindings for American option pricing and FDM implied volatility solving.

## Crates

Two crates live under `crates/`:

- **`mango-option`** — safe, idiomatic Rust API. Use this crate. Callers write no `unsafe`.
- **`mango-option-sys`** — raw FFI bindings to the C ABI shim. Internal; do not use directly.

The binding is built in-tree via `rules_rust` (Bazel). It is not published to crates.io.

## Build

```bash
bazel build //crates/mango-option:mango_option
```

## Test

```bash
bazel test //crates/mango-option:integration_test
```

## Pricing an Option

Construct `PricingParams` with an `OptionSpec`, then call `price_american`:

```rust
use mango_option::{
    price_american, Dividend, OptionSpec, OptionType, PricingParams, Rate,
};

let spec = OptionSpec {
    spot: 100.0,
    strike: 100.0,
    maturity: 1.0,
    dividend_yield: 0.02,
    rate: Rate::Const(0.05),
    discrete_dividends: vec![],
    option_type: OptionType::Put,
};

let params = PricingParams { spec, volatility: 0.20 };

let result = price_american(&params)?;

println!("price      : {}", result.value());
println!("price@spot : {}", result.value_at(100.0)?);
println!("delta      : {}", result.delta());
println!("gamma      : {}", result.gamma());
println!("theta      : {}", result.theta());
```

`PriceResult` is `!Send + !Sync` because `value_at` drives mutable caches inside the C++ object; do not share it across threads.

### Greeks reference

| Method | Returns |
|--------|---------|
| `value()` | Option price at the original spot |
| `value_at(spot: f64)` | Option price at an arbitrary spot (returns `Result`) |
| `delta()` | First-order price sensitivity to spot |
| `gamma()` | Second-order price sensitivity to spot |
| `theta()` | Time decay |

## Implied Volatility

Construct an `IvQuery` and call `solve_iv`:

```rust
use mango_option::{
    solve_iv, Dividend, IvConfig, IvQuery, OptionSpec, OptionType, Rate,
};

let spec = OptionSpec {
    spot: 100.0,
    strike: 100.0,
    maturity: 1.0,
    dividend_yield: 0.02,
    rate: Rate::Const(0.05),
    discrete_dividends: vec![],
    option_type: OptionType::Put,
};

let query = IvQuery { spec, market_price: 10.45 };
let iv = solve_iv(&query, &IvConfig::default())?;

println!("implied vol : {}", iv.implied_vol);
println!("iterations  : {}", iv.iterations);
println!("final error : {}", iv.final_error);
if let Some(vega) = iv.vega {
    println!("vega        : {}", vega);
}
```

`IvSuccess` fields:

| Field | Type | Description |
|-------|------|-------------|
| `implied_vol` | `f64` | Solved implied volatility |
| `iterations` | `usize` | Number of Brent iterations |
| `final_error` | `f64` | Residual price error at solution |
| `vega` | `Option<f64>` | Vega estimate, if available |

### IvConfig options

`IvConfig::default()` uses the library's built-in defaults. Override if needed:

```rust
let config = IvConfig {
    max_iter: Some(100),
    brent_tol_abs: Some(1e-8),
};
```

## Rates: constant vs yield curve

Use `Rate::Const(f64)` for a flat term structure:

```rust
rate: Rate::Const(0.05),
```

Use `Rate::Curve(Vec<TenorPoint>)` for a term structure. The first point **must** have `tenor = 0` and `log_discount = 0`:

```rust
use mango_option::{Rate, TenorPoint};

rate: Rate::Curve(vec![
    TenorPoint { tenor: 0.0,  log_discount: 0.0    },  // required anchor
    TenorPoint { tenor: 0.25, log_discount: -0.005 },
    TenorPoint { tenor: 0.5,  log_discount: -0.011 },
    TenorPoint { tenor: 1.0,  log_discount: -0.024 },
]),
```

`log_discount` is ln(P(0, T)), i.e. negative for positive rates.

## Discrete dividends

Both `price_american` and `solve_iv` support discrete cash dividends. Supply them in the `OptionSpec`:

```rust
use mango_option::Dividend;

let spec = OptionSpec {
    spot: 100.0,
    strike: 100.0,
    maturity: 1.0,
    dividend_yield: 0.0,
    rate: Rate::Const(0.05),
    discrete_dividends: vec![
        Dividend { calendar_time: 0.25, amount: 1.50 },
        Dividend { calendar_time: 0.75, amount: 1.50 },
    ],
    option_type: OptionType::Put,
};
```

`calendar_time` is measured in years from today. Dividends must fall within `(0, maturity)`.

## Error handling

All fallible functions return `Result<_, Error>`. `Error` carries:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `ErrorKind` | Error category |
| `message` | `String` | Diagnostic from the C++ side |

`ErrorKind` variants:

| Variant | When |
|---------|------|
| `Validation` | Invalid parameters (e.g. negative spot, maturity ≤ 0) |
| `Arbitrage` | Market price violates no-arbitrage bounds |
| `NoConvergence` | PDE or Newton solver did not converge |
| `Bracketing` | Root-finder could not bracket the solution |
| `Solver` | Other solver-level failure |

`Error` implements `std::error::Error` and `Display`:

```rust
match price_american(&params) {
    Ok(result) => { /* use result */ }
    Err(e) => eprintln!("pricing failed [{:?}]: {}", e.kind, e.message),
}
```

## API summary

### Types

| Type | Description |
|------|-------------|
| `OptionType` | `Call` or `Put` |
| `OptionSpec` | Option contract: spot, strike, maturity, dividend_yield, rate, discrete_dividends, option_type |
| `TenorPoint` | Yield-curve point: `tenor`, `log_discount` |
| `Dividend` | Discrete dividend: `calendar_time`, `amount` |
| `Rate` | `Const(f64)` or `Curve(Vec<TenorPoint>)` |
| `PricingParams` | `spec: OptionSpec` + `volatility: f64` |
| `PriceResult` | Pricing result with `value`, `value_at`, `delta`, `gamma`, `theta` |
| `IvQuery` | IV input: `spec: OptionSpec` + `market_price: f64` |
| `IvConfig` | IV solver config: `max_iter`, `brent_tol_abs` |
| `IvSuccess` | IV result: `implied_vol`, `iterations`, `final_error`, `vega` |
| `Error` | Error with `kind: ErrorKind` and `message: String` |
| `ErrorKind` | `Validation`, `Arbitrage`, `NoConvergence`, `Bracketing`, `Solver` |

### Functions

| Function | Signature |
|----------|-----------|
| `price_american` | `(&PricingParams) -> Result<PriceResult, Error>` |
| `solve_iv` | `(&IvQuery, &IvConfig) -> Result<IvSuccess, Error>` |
