# Python Guide

Python bindings for American option pricing, implied volatility, and price table interpolation.

## Build

```bash
bazel build //python:mango_option
```

The module is built as `mango_option.so` in `bazel-bin/python/`.

## Pricing a Single Option

```python
import mango_option as mo

params = mo.AmericanOptionParams()
params.spot = 100.0
params.strike = 100.0
params.maturity = 1.0
params.volatility = 0.20
params.rate = 0.05
params.dividend_yield = 0.02
params.type = mo.OptionType.PUT

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

Note: The discrete dividends field is plumbed through to the C++ solver. The solver's handling of discrete dividends is a work in progress.

## Batch Pricing

`BatchAmericanOptionSolver` prices many options in parallel with OpenMP. When options share the same maturity, volatility, rate, dividend yield, and type, the solver automatically uses a normalized chain optimization that solves one PDE and reuses it for all strikes.

```python
# Build a batch of puts at different strikes
batch = []
for K in [90, 95, 100, 105, 110]:
    p = mo.AmericanOptionParams()
    p.spot = 100.0
    p.strike = K
    p.maturity = 1.0
    p.volatility = 0.20
    p.rate = 0.05
    p.dividend_yield = 0.02
    p.type = mo.OptionType.PUT
    batch.append(p)

solver = mo.BatchAmericanOptionSolver()
solver.set_grid_accuracy(mo.GridAccuracyProfile.LOW)

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

### Fine-Grained Grid Control

```python
accuracy = mo.GridAccuracyParams()
accuracy.tol = 1e-6            # tighter truncation error
accuracy.min_spatial_points = 300
accuracy.max_time_steps = 10000
solver.set_grid_accuracy_params(accuracy)
```

## Implied Volatility

### FDM Solver

Solves for IV by repeatedly pricing with the PDE solver and root-finding:

```python
config = mo.IVSolverFDMConfig()
solver = mo.IVSolverFDM(config)

query = mo.IVQuery(
    spot=100.0, strike=100.0, maturity=1.0,
    rate=0.05, dividend_yield=0.02,
    type=mo.OptionType.PUT, market_price=10.0,
)

success, result, error = solver.solve_impl(query)
if success:
    print(f"IV = {result.implied_vol:.4f}")
```

### Interpolated Solver (Fast)

Build a price table surface first, then solve IV via Newton-Raphson on the interpolant. Orders of magnitude faster than FDM (~4us vs ~5ms per query).

```python
# 1. Build price table from option chain
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

# 2. Create interpolated solver
iv_solver = mo.IVSolverInterpolated.create(surface)

# 3. Solve single query
success, result, error = iv_solver.solve_impl(query)

# 4. Solve batch (parallelized with OpenMP)
queries = [query1, query2, query3]
results, failed_count = iv_solver.solve_batch(queries)
```

### Saving and Loading Price Tables

Price tables can be saved to Arrow IPC format for reuse:

```python
# Save
workspace.save("spy_puts.arrow", ticker="SPY", option_type=0)  # 0=PUT

# Load
workspace = mo.PriceTableWorkspace.load("spy_puts.arrow")
```

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
| `AmericanOptionParams` | Option contract parameters (spot, strike, maturity, volatility, rate, dividend_yield, type, discrete_dividends) |
| `AmericanOptionResult` | Pricing result with `value_at(spot)`, `delta()`, `gamma()`, `theta()` |
| `BatchAmericanOptionSolver` | Parallel batch pricing with normalized chain optimization |
| `GridAccuracyParams` | Fine-grained grid control (tol, n_sigma, alpha, spatial/time limits) |
| `YieldCurve` | Term structure via `flat(rate)` or `from_discounts(tenors, discounts)` |
| `IVQuery` | IV solver input (inherits OptionSpec + market_price) |
| `IVSolverFDM` | PDE-based IV solver |
| `IVSolverInterpolated` | Fast B-spline interpolation IV solver |
| `OptionChain` | Container for chain data (spot, strikes, maturities, vols, rates) |
| `PriceTableSurface4D` | 4D B-spline surface with `value(m, tau, sigma, r)` and `partial(axis, ...)` |
| `PriceTableWorkspace` | Serializable price table data (save/load Arrow IPC) |
| `SolverError` | Error detail with `code`, `iterations`, `residual` |

### Functions

| Function | Description |
|----------|-------------|
| `american_option_price(params, accuracy=None)` | Price a single American option with auto-grid |
| `build_price_table_surface_from_chain(chain, ...)` | Build 4D B-spline surface from OptionChain |
| `build_price_table_surface_from_chain_auto_profile(spot, strikes, ...)` | Same, with positional args instead of OptionChain |
