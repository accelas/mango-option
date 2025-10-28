# IV Calculation and Option Pricing - Quick Reference

## Three Main APIs

### 1. Implied Volatility (European Options)

**File**: `src/implied_volatility.h`

```c
#include "src/implied_volatility.h"

// Simple usage (auto-determines bounds)
IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = true
};

IVResult result = implied_volatility_calculate_simple(&params);

if (result.converged) {
    printf("IV: %.4f, Iterations: %d\n", result.implied_vol, result.iterations);
} else {
    printf("Error: %s\n", result.error);
}
```

**Key Points**:
- Works for calls and puts
- Uses Brent's method for root finding
- Auto-bounds: [0.0001, adaptive upper bound]
- Validates input for arbitrage
- Returns vega for sensitivity

---

### 2. American Option Pricing

**File**: `src/american_option.h`

```c
#include "src/american_option.h"

// Setup option parameters
OptionData option = {
    .strike = 100.0,
    .volatility = 0.2,
    .risk_free_rate = 0.05,
    .time_to_maturity = 1.0,
    .option_type = OPTION_PUT,
    .n_dividends = 0,
    .dividend_times = nullptr,
    .dividend_amounts = nullptr
};

// Setup grid
AmericanOptionGrid grid = {
    .x_min = -0.7,      // ln(50%) - covers 50% of strike
    .x_max = 0.7,       // ln(200%) - covers 200% of strike
    .n_points = 141,
    .dt = 0.001,
    .n_steps = 1000
};

// Price the option
AmericanOptionResult result = american_option_price(&option, &grid);

if (result.status == 0) {
    // Get value at spot = strike
    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    printf("Option value: %.4f\n", value);
    
    // Cleanup
    pde_solver_destroy(result.solver);
} else {
    printf("Pricing failed\n");
}
```

**Key Points**:
- Uses finite difference method (TR-BDF2 scheme)
- Log-price transformation: x = ln(S/K)
- Obstacle condition enforces American constraint
- Supports dividend events
- Returns PDESolver for full solution access

**Grid Guidelines**:
- Typical: 141 points, 1000 steps → 21 ms/option
- Coarse: 101 points, 500 steps → 8 ms/option
- Fine: 201 points, 2000 steps → 80 ms/option

---

### 3. Batch Processing

**File**: `src/american_option.h`

```c
// Create array of options
OptionData options[100];
for (int i = 0; i < 100; i++) {
    options[i] = { /* configure */ };
}

// Create result array
AmericanOptionResult results[100] = {0};

// Batch price (OpenMP parallel)
int status = american_option_price_batch(options, &grid, 100, results);

// Process results
for (int i = 0; i < 100; i++) {
    if (results[i].status == 0) {
        // Use result
        pde_solver_destroy(results[i].solver);
    }
}
```

**Benefits**:
- OpenMP parallel processing
- 10-60x wall-time speedup
- Enables vectorized IV recovery

---

## Black-Scholes Pricing

**File**: `src/implied_volatility.h`

```c
// Price European option directly
double price = black_scholes_price(
    100.0,   // spot
    100.0,   // strike
    1.0,     // time_to_maturity
    0.05,    // risk_free_rate
    0.2,     // volatility
    true     // is_call
);

// Get sensitivity (vega)
double vega = black_scholes_vega(100.0, 100.0, 1.0, 0.05, 0.2);
```

---

## PDE Solver (Advanced)

**File**: `src/pde_solver.h`

For custom PDEs: ∂u/∂t = L(u)

```c
#include "src/pde_solver.h"

// Define callbacks
void my_initial_condition(const double *x, size_t n, double *u0, void *data) {
    for (size_t i = 0; i < n; i++) {
        u0[i] = sin(x[i]);  // Your IC
    }
}

void my_spatial_operator(const double *x, double t, const double *u,
                        size_t n, double *Lu, void *data) {
    // Compute L(u) for all points
}

double my_left_bc(double t, void *data) { return 0.0; }
double my_right_bc(double t, void *data) { return 0.0; }

// Setup callbacks
PDECallbacks callbacks = {
    .initial_condition = my_initial_condition,
    .left_boundary = my_left_bc,
    .right_boundary = my_right_bc,
    .spatial_operator = my_spatial_operator,
    .obstacle = nullptr,
    .jump_condition = nullptr,
    .temporal_event = nullptr,
    .user_data = nullptr
};

// Create grid and time domain
SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);
TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = 0.001, .n_steps = 1000};

// Create and solve
BoundaryConfig bc = pde_default_boundary_config();
TRBDF2Config tr = pde_default_trbdf2_config();

PDESolver *solver = pde_solver_create(&grid, &time, &bc, &tr, &callbacks);
pde_solver_initialize(solver);
pde_solver_solve(solver);

// Access solution
const double *solution = pde_solver_get_solution(solver);
double value_at_x = pde_solver_interpolate(solver, 0.5);

pde_solver_destroy(solver);
```

**Features**:
- General-purpose PDE solver
- TR-BDF2 time-stepping (implicit, L-stable)
- Vectorized callbacks
- Obstacle conditions (variational inequalities)
- Cubic spline interpolation

---

## Performance Notes

### IV Calculation
- **Time**: <1 µs per call (typical)
- **Bottleneck**: Usually the option pricing, not IV solver
- **Iterations**: 8-15 typically

### American Option (Single)
- **Time**: 21.7 ms (default grid)
- **Grid**: 141 points × 1000 steps
- **vs QuantLib**: 2.1x slower (reasonable for research code)
- **Speedup available**: 10-200x with full optimization

### American Option (Batch)
- **Core count scaling**: ~linear up to 16 cores
- **Wall time**: ~2-5 ms per option (16 cores)
- **Enables**: Vectorized IV recovery

---

## Memory Characteristics

| Configuration | Memory | Notes |
|---|---|---|
| IV calc | <1 KB | Just parameters |
| American (141×1000) | ~1.3 MB | Solver workspace |
| American (201×2000) | ~3.2 MB | Fine grid |
| Batch (100 options) | ~130 MB | Parallel allocation |

---

## Validation & Testing

### IV Tests (44 cases)
```bash
bazel test //tests:implied_volatility_test
```
- ATM, OTM, ITM scenarios
- Extreme volatility (5% to 300%)
- Zero/negative rates
- Error cases

### American Option Tests (29 cases)
```bash
bazel test //tests:american_option_test
```
- Single/multiple dividends
- Monotonicity in vol/maturity
- Intrinsic value bounds
- Grid resolution sensitivity

### QuantLib Comparison
```bash
bazel build //tests:quantlib_benchmark
./bazel-bin/tests/quantlib_benchmark
```

---

## Tracing & Diagnostics

The library includes USDT probes for zero-overhead tracing.

```bash
# List available probes
readelf -x .stapsdt.base ./binary | grep ivcalc

# Simple trace with bpftrace
sudo bpftrace -e 'usdt::ivcalc:convergence_iter { 
    printf("Module %d, step %d, error %g\n", arg0, arg1, arg2);
}'

# Or use provided scripts
sudo ./scripts/ivcalc-trace monitor ./binary --preset=convergence
```

---

## Common Patterns

### Pattern: IV Surface Generation

```c
// Generate IV surface for calls across strikes and maturities
for (int i = 0; i < n_strikes; i++) {
    for (int j = 0; j < n_maturities; j++) {
        IVParams params = {
            .spot_price = 100.0,
            .strike = strikes[i],
            .time_to_maturity = maturities[j],
            .risk_free_rate = 0.05,
            .market_price = market_prices[i][j],
            .is_call = true
        };
        
        IVResult result = implied_volatility_calculate_simple(&params);
        if (result.converged) {
            iv_surface[i][j] = result.implied_vol;
        }
    }
}
```

### Pattern: Early Exercise Premium

```c
// European vs American comparison
OptionData european = option;  // Copy option

// Get European price (Black-Scholes)
double european_price = black_scholes_price(
    european.strike, european.strike,
    european.time_to_maturity, european.risk_free_rate,
    european.volatility, european.option_type == OPTION_CALL
);

// Get American price (PDE)
AmericanOptionResult american_result = american_option_price(&option, &grid);
double american_price = american_option_get_value_at_spot(
    american_result.solver, option.strike, option.strike
);

// Early exercise premium
double premium = american_price - european_price;
```

### Pattern: Sensitivity Analysis (Greeks)

```c
// Delta (price change per $1 spot move)
double ds = 0.01;
double price_up = american_option_get_value_at_spot(solver, spot + ds, strike);
double price_down = american_option_get_value_at_spot(solver, spot - ds, strike);
double delta = (price_up - price_down) / (2 * ds);

// Gamma (delta change per $1 spot move)
double gamma = (price_up + price_down - 2*price) / (ds * ds);
```

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| IV fails to converge | Bounds too tight | Use `implied_volatility_calculate()` with wider bounds |
| IV returns error "arbitrage" | Invalid market price | Check price vs intrinsic value |
| American solver fails | Grid too coarse | Increase `n_points` and/or reduce `dt` |
| American solver slow | Grid too fine | Try coarser grid first |
| Memory error on batch | Too many options | Reduce batch size or grid resolution |
| Results inconsistent | Floating point | Use deterministic grids, fixed seed |

---

## Files Quick Index

| Purpose | File |
|---|---|
| IV calculation | `src/implied_volatility.h/.c` |
| American options | `src/american_option.h/.c` |
| PDE solver | `src/pde_solver.h/.c` |
| Black-Scholes | `src/implied_volatility.c` |
| Root finding | `src/brent.h` |
| Interpolation | `src/cubic_spline.h/.c` |
| Linear solver | `src/tridiagonal.h` |
| Tracing | `src/ivcalc_trace.h` |

---

## Further Reading

- **Overview**: `docs/PROJECT_OVERVIEW.md` - Problem domain and project motivation
- **Architecture**: `docs/ARCHITECTURE.md` - Detailed technical architecture
- **Benchmarks**: `tests/BENCHMARK.md` - Performance comparisons
- **Optimization Plan**: `docs/FASTVOL_ANALYSIS_AND_PLAN.md` - Future improvements
- **Examples**: `examples/example_*.c` - Usage examples
- **Tests**: `tests/*_test.cc` - Test suite
- **Tracing**: `TRACING.md`, `TRACING_QUICKSTART.md` - USDT tracing guide

