# Python Guide

Python bindings for American option pricing, implied volatility, and price table interpolation.

## Build

```bash
bazel build //src/python:mango_option
```

The module is built as `mango_option.so` in `bazel-bin/src/python/`.

## Pricing a Single Option

```python
import mango_option as mo

params = mo.PricingParams()
params.spot = 100.0
params.strike = 100.0
params.maturity = 1.0
params.volatility = 0.20
params.rate = 0.05
params.dividend_yield = 0.02
params.option_type = mo.OptionType.PUT

result = mo.american_option_price(params)

print(result.value_at(100.0))  # price at spot=100
print(result.delta())           # first-order sensitivity
print(result.gamma())           # second-order sensitivity
print(result.theta())           # time decay
```

The solver automatically estimates a sinh-spaced grid clustered near the strike. For more control, pass an accuracy profile:

```python
result = mo.american_option_price(params, accuracy=mo.GridAccuracyProfile.HIGH)
```

Profiles: `LOW`, `MEDIUM`, `HIGH`, `ULTRA`. Higher accuracy uses finer grids and more time steps.

### Yield Curves

Pass a `YieldCurve` instead of a flat rate:

```python
curve = mo.YieldCurve.flat(0.05)

# Or from discount factors:
curve = mo.YieldCurve.from_discounts(
    tenors=[0.25, 0.5, 1.0, 2.0],
    discounts=[0.9876, 0.9753, 0.9512, 0.9048],
)

params.rate = curve
result = mo.american_option_price(params)
```

### Discrete Dividends

```python
params.discrete_dividends = [(0.25, 2.0), (0.75, 2.0)]  # (time, amount) pairs
result = mo.american_option_price(params)
```

Python accepts `(time, amount)` pairs or `Dividend` objects and uses the C++
discrete-dividend event path.

## Batch Pricing

`BatchAmericanOptionSolver` prices many options in parallel with OpenMP. When options share the same maturity, volatility, rate, dividend yield, and option type, the solver automatically uses a normalized chain optimization that solves one PDE and reuses it for all strikes.

```python
# Build a batch of puts at different strikes
batch = []
for K in [90, 95, 100, 105, 110]:
    p = mo.PricingParams()
    p.spot = 100.0
    p.strike = K
    p.maturity = 1.0
    p.volatility = 0.20
    p.rate = 0.05
    p.dividend_yield = 0.02
    p.option_type = mo.OptionType.PUT
    batch.append(p)

solver = mo.BatchAmericanOptionSolver()

# use_shared_grid=True enables the normalized chain optimization
results, failed_count = solver.solve_batch(batch, use_shared_grid=True)

for i, (success, result, error) in enumerate(results):
    if success:
        print(f"K={batch[i].strike}: price={result.value_at(100.0):.4f}")
    else:
        print(f"K={batch[i].strike}: failed ({error.code})")
```

### Per-Option Grids

When options have different maturities, use `use_shared_grid=False` (the default). Each option gets its own auto-estimated grid:

```python
results, failed = solver.solve_batch(batch, use_shared_grid=False)
```

## Implied Volatility

### FDM Solver

Solves for IV by repeatedly pricing with the PDE solver and root-finding:

```python
config = mo.IVSolverConfig()
config.root_config.tolerance = 1e-8
config.batch_parallel_threshold = 64

solver = mo.IVSolver(config)

query = mo.IVQuery()
query.spot = 100.0
query.strike = 100.0
query.maturity = 1.0
query.rate = 0.05
query.dividend_yield = 0.02
query.option_type = mo.OptionType.PUT
query.market_price = 10.0

success, result, error = solver.solve(query)
if success:
    print(f"IV = {result.implied_vol:.4f}")
```

### Interpolated Solver (Fast)

Build a price surface and solve IV via Brent's method on the interpolant. Orders of magnitude faster than FDM (~4us vs ~5ms per query).

```python
config = mo.IVSolverFactoryConfig()
config.option_type = mo.OptionType.PUT
config.spot = 100.0
config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
config.grid.vol = [0.10, 0.20, 0.30, 0.40]
config.grid.rate = [0.01, 0.03, 0.05, 0.07]

# Backend selection: BSplineBackend (default), ChebyshevBackend, or DimensionlessBackend
backend = mo.BSplineBackend()
backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
config.backend = backend

iv_solver = mo.make_interpolated_iv_solver(config)

# Solve single query
success, result, error = iv_solver.solve(query)

# Solve batch (parallelized with OpenMP)
queries = [query1, query2, query3]
results, failed_count = iv_solver.solve_batch(queries)
```

#### Backend Selection

```python
# B-spline backend (default, best accuracy)
backend = mo.BSplineBackend()
backend.maturity_grid = [0.1, 0.25, 0.5, 1.0, 2.0]
config.backend = backend

# Chebyshev backend
backend = mo.ChebyshevBackend()
backend.maturity = 2.0
backend.num_pts = [5, 5, 4, 4]
config.backend = backend

# Dimensionless 3D backend
backend = mo.DimensionlessBackend()
backend.maturity = 2.0
backend.chebyshev_pts = [5, 5, 4]
config.backend = backend
```

#### Discrete Dividends

```python
div_config = mo.DiscreteDividendConfig()
div_config.maturity = 1.0
div_config.discrete_dividends = [mo.Dividend(0.25, 1.50), mo.Dividend(0.50, 1.50)]
div_config.kref_config.K_refs = [80.0, 100.0, 120.0]
config.discrete_dividends = div_config

iv_solver = mo.make_interpolated_iv_solver(config)
```

#### Adaptive Grid

```python
adaptive = mo.AdaptiveGridParams()
adaptive.target_iv_error = 0.001  # 10 bps
config.adaptive = adaptive

iv_solver = mo.make_interpolated_iv_solver(config)
```

### Reusable Price Tables

Build a reusable interpolation surface once, then use it for pricing, Greeks,
fast implied volatility, and persistence. With `BSplineBackend` and no discrete
dividends, this builds the standard 4D B-spline table over log-moneyness,
maturity, volatility, and rate.

```python
import mango_option as mo

config = mo.PriceTableConfig()
config.option_type = mo.OptionType.PUT
config.spot = 100.0
config.dividend_yield = 0.02
config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
config.grid.vol = [0.10, 0.20, 0.30, 0.40]
config.grid.rate = [0.01, 0.03, 0.05, 0.07]

backend = mo.BSplineBackend()
backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
config.backend = backend

table = mo.make_price_table(config)
assert table.surface_type == "bspline_4d"

params = mo.PricingParams()
params.spot = 100.0
params.strike = 100.0
params.maturity = 0.5
params.volatility = 0.20
params.rate = 0.05
params.dividend_yield = 0.02
params.option_type = mo.OptionType.PUT

print(table.price(params))
print(table.delta(params))
print(table.gamma(params))

query = mo.IVQuery()
query.spot = 100.0
query.strike = 100.0
query.maturity = 0.5
query.rate = 0.05
query.dividend_yield = 0.02
query.option_type = mo.OptionType.PUT
query.market_price = table.price(params)

iv = table.solve_iv(query)
solver = table.make_iv_solver()
success, result, error = solver.solve(query)  # Back-compatible tuple API

table.save("spy_puts.parquet")
loaded = mo.PriceTable.load("spy_puts.parquet")
```

`bspline_4d` is the main high-throughput interpolation path. Use it when the
surface domain is described by log-moneyness, maturity, volatility, and rate.
Use segmented B-spline surfaces for discrete cash dividends and dimensionless
3D surfaces only when those model constraints are intentional.

## Python Conversions

The binding accepts Python-native values for common inputs:

- rates: `float`, `int`, or `YieldCurve`
- grids and vector fields: lists or tuples of numbers
- dividend schedules: `Dividend` objects or `(time, amount)` pairs
- optional config fields: assign `None` to clear
- persistence paths: strings or `pathlib.Path`

The core binding does not require numpy, pandas, pyarrow, or dataframe objects.
Price-table persistence is implemented by the native C++ Arrow/Parquet backend,
so deployable wheels must still provide the corresponding native shared
libraries.

## Errors

New price-table APIs raise typed exceptions:

- `ValidationError`
- `PriceTableError`
- `SolverException`
- `TypeConversionError`

Existing IV solver methods keep the back-compatible `(success, result, error)`
return shape.

## API Reference

### Enums

| Enum | Values |
|------|--------|
| `OptionType` | `CALL`, `PUT` |
| `GridAccuracyProfile` | `LOW`, `MEDIUM`, `HIGH`, `ULTRA` |
| `PriceTableGridProfile` | `LOW`, `MEDIUM`, `HIGH`, `ULTRA` |
| `SolverErrorCode` | `ConvergenceFailure`, `LinearSolveFailure`, `InvalidConfiguration`, `Unknown` |

### Classes

| Class | Purpose |
|-------|---------|
| `PricingParams` | Option contract parameters (spot, strike, maturity, volatility, rate, dividend_yield, option_type, discrete_dividends) |
| `AmericanOptionResult` | Pricing result with `value_at(spot)`, `delta()`, `gamma()`, `theta()` |
| `BatchAmericanOptionSolver` | Parallel batch pricing with normalized chain optimization |
| `GridAccuracyParams` | Fine-grained grid control (tol, n_sigma, alpha, spatial/time limits) |
| `YieldCurve` | Term structure via `flat(rate)` or `from_discounts(tenors, discounts)` |
| `IVQuery` | IV solver input (inherits OptionSpec + market_price) |
| `IVSolverConfig` | IV solver config with `root_config` and `batch_parallel_threshold` |
| `IVSolver` | PDE-based IV solver |
| `InterpolatedIVSolver` | Fast interpolation IV solver (created via `make_interpolated_iv_solver`) |
| `IVSolverFactoryConfig` | Configuration for IV solver factory (grid, backend, solver params) |
| `PriceTableConfig` | Alias for `IVSolverFactoryConfig` when building reusable price tables |
| `InterpolatedIVSolverConfig` | Interpolated IV config (`max_iter`, `tolerance`, `sigma_min`, `sigma_max`, `vega_threshold`) |
| `IVGrid` | Grid specification (moneyness, vol, rate arrays) |
| `BSplineBackend` | B-spline interpolation backend config (maturity_grid) |
| `ChebyshevBackend` | Chebyshev interpolation backend config (maturity, num_pts) |
| `DimensionlessBackend` | Dimensionless 3D interpolation backend config (maturity, interpolant, chebyshev_pts) |
| `DiscreteDividendConfig` | Discrete dividend config (maturity, dividends, kref_config) |
| `AdaptiveGridParams` | Adaptive grid refinement parameters |
| `MultiKRefConfig` | Multi-reference-strike configuration for discrete dividends |
| `OptionGrid` | Container for chain data (spot, strikes, maturities, vols, rates) |
| `PriceTable` | Reusable price table with `price`, `delta`, `gamma`, `vega`, IV solving, and save/load |
| `SolverError` | Error detail with `code`, `iterations`, `residual` |

### Functions

| Function | Description |
|----------|-------------|
| `american_option_price(params, accuracy=None)` | Price a single American option with auto-grid |
| `make_interpolated_iv_solver(config)` | Create fast IV solver from `IVSolverFactoryConfig` |
| `make_price_table(config)` | Build a reusable `PriceTable` from `PriceTableConfig` |
