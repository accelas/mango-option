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

## Interpolation path

The interpolation path trades a one-time build cost for very fast repeated
queries. You pre-compute a B-spline price surface over a grid of
(moneyness, maturity, vol, rate), then either solve implied volatility against it
(`InterpIvSolver`) or query prices and greeks directly (`PriceTable`). Both
handles wrap an immutable surface and are `Send + Sync`, so you can share them
across threads behind an `Arc`.

Everything is driven by a single `FactoryConfig`:

```rust
use mango_option::{FactoryConfig, InterpSolverConfig, IvGrid, OptionType};

let config = FactoryConfig {
    option_type: OptionType::Put,
    spot: 100.0,
    dividend_yield: 0.0,
    grid: IvGrid {
        // S/K moneyness (NOT log). Each axis needs >= 4 points.
        moneyness: vec![0.8, 0.9, 1.0, 1.1, 1.2],
        vol: vec![0.10, 0.20, 0.30, 0.40],
        rate: vec![0.01, 0.03, 0.05, 0.07],
    },
    // The maturity axis also needs >= 4 points.
    maturity_grid: vec![0.25, 0.5, 0.75, 1.0],
    solver: InterpSolverConfig::default(),
    adaptive: None,
    discrete_dividends: None,
};
```

> **Grid-size constraint.** The cubic B-spline builder needs **at least 4 points
> on every axis** — `grid.moneyness`, `grid.vol`, `grid.rate`, and
> `maturity_grid`. A shorter axis fails the build with
> `ErrorKind::Validation` (an `InvalidGridSize` on the C++ side). `IvGrid::default()`
> already satisfies this; supply your own `maturity_grid`.

### Interpolated IV solver

Build an `InterpIvSolver` from the config, then call `solve` for a single query:

```rust
use mango_option::{InterpIvSolver, IvQuery, OptionSpec, OptionType, Rate};

let solver = InterpIvSolver::new(&config)?;

let spec = OptionSpec {
    spot: 100.0,
    strike: 100.0,
    maturity: 1.0,
    dividend_yield: 0.0,
    rate: Rate::Const(0.03),
    discrete_dividends: vec![],
    option_type: OptionType::Put,
};

let query = IvQuery { spec, market_price: 5.40 };
let iv = solver.solve(&query)?;

println!("implied vol : {}", iv.implied_vol);
println!("iterations  : {}", iv.iterations);
if iv.used_rate_approximation {
    // A yield-curve query was collapsed to a single flat rate to hit the
    // surface; treat the result as approximate (see note below).
    eprintln!("note: rate approximated to a flat value");
}
```

`solve` returns the same `IvSuccess` as the FDM path, with one extra field:

| Field | Type | Description |
|-------|------|-------------|
| `implied_vol` | `f64` | Solved implied volatility |
| `iterations` | `usize` | Newton iterations |
| `final_error` | `f64` | Residual price error at solution |
| `vega` | `Option<f64>` | Vega estimate, if available |
| `used_rate_approximation` | `bool` | `true` when a `Rate::Curve` query was collapsed to a flat rate to query the surface |

`used_rate_approximation` is always `false` for `Rate::Const` queries. It is set
only when the solver had to flatten a yield curve to a single rate to land on the
surface's rate axis; in that case the implied vol is an approximation rather than
a curve-exact solve.

### Batch solving

`solve_batch` solves a slice of queries and returns a `BatchResult`:

```rust
let queries = vec![query_a, query_b, query_c];
let batch = solver.solve_batch(&queries);

println!("failed: {}", batch.failed);
for (i, r) in batch.results.iter().enumerate() {
    match r {
        Ok(iv) => println!("[{i}] iv = {}", iv.implied_vol),
        Err(e) => println!("[{i}] failed: {:?}", e.kind),
    }
}
```

`BatchResult` has two fields:

| Field | Type | Description |
|-------|------|-------------|
| `results` | `Vec<Result<IvSuccess, Error>>` | One result per input query, in order |
| `failed` | `usize` | Number of failed slots |

> **Batch error detail.** A failed slot in `results` carries only an error
> **category** (`Error::kind`) — its `message` is a generic placeholder, not the
> full diagnostic. If you need the precise reason a particular query failed,
> re-run it through the single `solve`, which surfaces the full C++ message.

### Price table

A `PriceTable` exposes the surface directly for pricing and greeks. Queries take
a `PricingParams` (the same type as `price_american`):

```rust
use mango_option::{PriceTable, PricingParams};

let table = PriceTable::new(&config)?;

let pp = PricingParams { spec, volatility: 0.25 };

let price = table.price(&pp);      // infallible f64
let vega  = table.vega(&pp);       // infallible f64
let delta = table.delta(&pp)?;     // Result<f64, Error>
let gamma = table.gamma(&pp)?;
let theta = table.theta(&pp)?;
let rho   = table.rho(&pp)?;

println!("price={price} vega={vega} delta={delta}");
println!("option type    : {:?}", table.option_type());
println!("dividend yield : {}", table.dividend_yield());
```

`PriceTable` query methods:

| Method | Returns | Notes |
|--------|---------|-------|
| `price(&PricingParams)` | `f64` | Infallible; extrapolates out of domain |
| `vega(&PricingParams)` | `f64` | Infallible; extrapolates out of domain |
| `delta` / `gamma` / `theta` / `rho` | `Result<f64, Error>` | `Err` when out of domain or numerically unstable |
| `option_type()` | `OptionType` | The surface's option type |
| `dividend_yield()` | `f64` | The surface's continuous dividend yield |

> **Extrapolation caveat.** `price` and `vega` perform **no implicit bounds
> check** — outside the grid domain they extrapolate the B-spline and can return
> meaningless values (they only return `NaN` on an internal failure, never as an
> out-of-bounds signal). If you need an explicit domain check, call
> `validate(&params)` first; it returns `Err(ErrorKind::Validation)` for
> out-of-domain points:

```rust
table.validate(&pp)?;          // Err if pp is outside the surface domain
let price = table.price(&pp);  // now safe to trust
```

### Deriving an IV solver from a table

If you already built a `PriceTable`, derive an `InterpIvSolver` from it instead
of rebuilding the surface. Pass `None` to use the default Newton config, or
`Some(&InterpSolverConfig { .. })` to override it:

```rust
let table = PriceTable::new(&config)?;
let solver = table.iv_solver(None)?;     // reuses the table's surface
let iv = solver.solve(&query)?;
```

### Adaptive grid refinement

Set `adaptive: Some(AdaptiveGridParams { .. })` to have the builder refine the
grid until it meets a target IV error, instead of using the fixed grid as-is:

```rust
use mango_option::AdaptiveGridParams;

let mut config = config;
config.adaptive = Some(AdaptiveGridParams {
    target_iv_error: 1e-3,
    max_iter: 2,
    validation_samples: 16,
    min_moneyness_points: 20,
    ..AdaptiveGridParams::default()
});
let solver = InterpIvSolver::new(&config)?;
```

This raises build time but tightens interpolation accuracy. `AdaptiveGridParams::default()`
mirrors the C++ defaults; override only the fields you care about.

### Discrete dividends

For discrete cash dividends on the interpolation path, set
`discrete_dividends: Some(DiscreteDividendConfig { .. })`. The surface is built
across multiple reference strikes (`MultiKRef`) so it can reconstruct prices for
the segmented dividend geometry:

```rust
use mango_option::{DiscreteDividendConfig, Dividend, MultiKRef};

let mut config = config;
config.discrete_dividends = Some(DiscreteDividendConfig {
    maturity: 1.0,
    dividends: vec![Dividend { calendar_time: 0.5, amount: 2.0 }],
    kref_config: MultiKRef {
        k_refs: vec![90.0, 100.0, 110.0],
        ..MultiKRef::default()
    },
});
let solver = InterpIvSolver::new(&config)?;
```

Queries against this solver must carry the **same** `discrete_dividends` in their
`OptionSpec`. `MultiKRef::default()` supplies sensible `k_ref_count` /
`k_ref_span` defaults; an empty `k_refs` lets the builder pick reference strikes
automatically.

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
| `IvSuccess` | IV result: `implied_vol`, `iterations`, `final_error`, `vega`, `used_rate_approximation` |
| `Error` | Error with `kind: ErrorKind` and `message: String` |
| `ErrorKind` | `Validation`, `Arbitrage`, `NoConvergence`, `Bracketing`, `Solver` |

#### Interpolation path

| Type | Description |
|------|-------------|
| `FactoryConfig` | B-spline surface build config: `option_type`, `spot`, `dividend_yield`, `grid`, `maturity_grid`, `solver`, `adaptive`, `discrete_dividends` |
| `IvGrid` | Grid axes: `moneyness` (S/K), `vol`, `rate` (each >= 4 points) |
| `InterpSolverConfig` | Newton config: `max_iter`, `tolerance`, `sigma_min`, `sigma_max`, `vega_threshold` |
| `AdaptiveGridParams` | Adaptive refinement: `target_iv_error`, `max_iter`, `max_points_per_dim`, … |
| `MultiKRef` | Reference strikes: `k_refs`, `k_ref_count`, `k_ref_span` |
| `DiscreteDividendConfig` | `maturity`, `dividends`, `kref_config` |
| `InterpIvSolver` | Interpolated IV solver (`new`, `solve`, `solve_batch`); `Send + Sync` |
| `BatchResult` | Batch output: `results: Vec<Result<IvSuccess, Error>>`, `failed: usize` |
| `PriceTable` | Reusable price surface (`new`, `validate`, `price`, `vega`, greeks, `option_type`, `dividend_yield`, `iv_solver`); `Send + Sync` |

### Functions

| Function | Signature |
|----------|-----------|
| `price_american` | `(&PricingParams) -> Result<PriceResult, Error>` |
| `solve_iv` | `(&IvQuery, &IvConfig) -> Result<IvSuccess, Error>` |

### Constructors / methods (interpolation path)

| Item | Signature |
|------|-----------|
| `InterpIvSolver::new` | `(&FactoryConfig) -> Result<InterpIvSolver, Error>` |
| `InterpIvSolver::solve` | `(&IvQuery) -> Result<IvSuccess, Error>` |
| `InterpIvSolver::solve_batch` | `(&[IvQuery]) -> BatchResult` |
| `PriceTable::new` | `(&FactoryConfig) -> Result<PriceTable, Error>` |
| `PriceTable::validate` | `(&PricingParams) -> Result<(), Error>` |
| `PriceTable::price` / `vega` | `(&PricingParams) -> f64` (infallible, extrapolates) |
| `PriceTable::delta` / `gamma` / `theta` / `rho` | `(&PricingParams) -> Result<f64, Error>` |
| `PriceTable::iv_solver` | `(Option<&InterpSolverConfig>) -> Result<InterpIvSolver, Error>` |
