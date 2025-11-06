# IV Calculation and Option Pricing - Quick Reference (C++20)

## Three Main APIs

### 1. American Implied Volatility

**File**: `src/iv_solver.hpp`

```cpp
#include "src/iv_solver.hpp"

// Setup option parameters (designated initializers)
mango::IVParams params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 6.08,  // American put market price
    .is_call = false
};

// Create solver and solve
mango::IVSolver solver(params);
mango::IVResult result = solver.solve();

if (result.converged) {
    std::cout << "American IV: " << result.implied_vol
              << " (" << result.implied_vol * 100 << "%), "
              << "Iterations: " << result.iterations << "\n";
} else {
    std::cerr << "Error: " << *result.failure_reason << "\n";
}
```

**Key Points**:
- **FDM-based**: Each Brent iteration solves full American option PDE (~21ms)
- **Performance**: ~145ms per calculation (5-8 Brent iterations)
- **Auto-bounds**: Automatic bound estimation based on intrinsic value analysis
- **Validates input** for arbitrage
- Works for calls and puts

---

### 2. American Option Pricing

**File**: `src/american_option.hpp`

```cpp
#include "src/american_option.hpp"

// Create American option pricer
mango::AmericanOption pricer(
    100.0,                     // strike
    0.2,                       // volatility
    0.05,                      // risk-free rate
    1.0,                       // time to maturity
    mango::OptionType::Put     // option type
);

// Price at spot = 100
double value = pricer.price(100.0);

std::cout << "Option value: " << value << "\n";
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

### 3. Custom PDE Solving

**Files**: `src/pde_solver.hpp`, `src/operators/*.hpp`

```cpp
#include "src/pde_solver.hpp"
#include "src/operators/laplacian_pde.hpp"
#include "src/boundary_conditions.hpp"

// Create spatial grid
std::vector<double> grid(101);
for (size_t i = 0; i < grid.size(); ++i) {
    grid[i] = i / 100.0;  // [0, 1]
}

// Setup components
mango::TimeDomain time{0.0, 1.0, 0.001};
auto left_bc = mango::DirichletBC(0.0);
auto right_bc = mango::DirichletBC(0.0);
auto spatial_op = mango::LaplacianOperator(0.1);  // diffusion coeff

// Create solver (template-based, compile-time optimization)
mango::TRBDF2Config config;
mango::RootFindingConfig root_config;
mango::PDESolver solver(grid, time, config, root_config,
                        left_bc, right_bc, spatial_op);

// Initialize with lambda
solver.initialize([](std::span<const double> x, std::span<double> u) {
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::sin(M_PI * x[i]);
    }
});

// Solve
bool converged = solver.solve();

// Access solution (zero-copy view)
auto solution = solver.solution();
```

**Benefits**:
- Template-based zero-cost abstractions
- Compile-time type checking via concepts
- std::span for zero-copy array views
- Lambda-based callbacks for flexibility

---

## Modern C++20 Features Used

The library leverages C++20 features for performance and expressiveness:

```cpp
// Concepts for type constraints
template<typename T>
concept BoundaryCondition = requires(T bc, double t) {
    { bc.value(t) } -> std::convertible_to<double>;
    { bc.type() } -> std::same_as<BoundaryType>;
};

// std::span for zero-copy array views
void process(std::span<const double> data);

// Spaceship operator for comparisons
auto operator<=>(const Event& other) const {
    return time <=> other.time;
}

// Requires expressions for compile-time introspection
if constexpr (requires { op.set_grid(grid, dx); }) {
    op.set_grid(grid, dx);
}

// std::optional for optional return values
std::optional<std::string> error_message;
```

**Key Benefits**:
- **Zero-cost abstractions**: Templates compile away
- **Type safety**: Concepts catch errors at compile-time
- **Performance**: No runtime overhead for type checks
- **Expressiveness**: Clear, concise code

---

## Creating Custom Operators

**File**: `src/operators/spatial_operator.hpp`

Implement the SpatialOperator concept:

```cpp
// Example: Custom advection-diffusion operator
class AdvectionDiffusion {
    double diffusion_;
    double velocity_;

public:
    AdvectionDiffusion(double D, double v) : diffusion_(D), velocity_(v) {}

    // Satisfies SpatialOperator concept
    void operator()(std::span<const double> x, double t,
                    std::span<const double> u,
                    std::span<double> Lu) const {
        const size_t n = x.size();
        const double dx = x[1] - x[0];

        // Interior points
        for (size_t i = 1; i < n - 1; ++i) {
            double laplacian = (u[i-1] - 2*u[i] + u[i+1]) / (dx*dx);
            double advection = (u[i+1] - u[i-1]) / (2*dx);
            Lu[i] = diffusion_ * laplacian - velocity_ * advection;
        }

        // Boundaries handled by BC
        Lu[0] = Lu[n-1] = 0.0;
    }
};

// Use in solver
auto op = AdvectionDiffusion(0.1, 0.5);
mango::PDESolver solver(grid, time, config, root_config,
                        left_bc, right_bc, op);
```

**Features**:
- Template-based operator interface
- Compile-time type checking via concepts
- Zero virtual function overhead
- std::span for safe array access

---

## Performance Notes

### American IV Calculation
- **FDM-based**: ~145ms per call (Brent's method with full PDE solve per iteration)
- **Table-based**: ~11.8ms per call (Newton's method with interpolation, 22.5× faster)
- **Bottleneck (FDM)**: American option pricing in each Brent iteration (~21ms × 5-8 iterations)
- **Brent iterations**: 5-8 typically

### American Option (Single)
- **Time**: 21.7 ms (default grid)
- **Grid**: 141 points × 1000 steps
- **vs QuantLib**: 2.1x slower (reasonable for research code)
- **Speedup available**: 10-200x with full optimization

### American Option (Batch)
- **Core count scaling**: ~linear up to 16 cores
- **Wall time**: ~2-5 ms per option (16 cores)
- **Enables**: Vectorized IV recovery

### Greeks Calculation
- **Via interpolation**: ~500ns per query (vega, gamma precomputed)
- **Via FDM**: ~65ms per query (finite differences on PDE solution)
- **Speedup**: ~130,000× faster with table-based approach

### Validation Framework
- **With reference table**: ~100μs per 1000 samples (~1000× faster than FDM)
- **With FDM**: ~2-5s per 1000 samples (101-point PDEs)
- **Accuracy**: <0.01bp mean error for in-bounds cases
- **Target**: <1bp IV error for 95% of validation points

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

### American IV Tests (9 cases)
```bash
bazel test //tests:implied_volatility_test
```
- American put/call IV recovery (ATM, OTM, ITM)
- Input validation and arbitrage detection
- Grid configuration validation
- Convergence with default settings

### American Option Tests (42 cases)
```bash
bazel test //tests:american_option_test
```
- Single/multiple dividends
- Monotonicity in vol/maturity
- Intrinsic value bounds
- Grid resolution sensitivity

### Benchmarks

**QuantLib Comparison** (vs industry standard):
```bash
bazel run //benchmarks:quantlib_benchmark
```

**Batch Processing Performance** (thread scaling, throughput):
```bash
# Run all batch benchmarks with summary
bazel run //benchmarks:batch_benchmark

# Run specific benchmark
bazel run //benchmarks:batch_benchmark -- --benchmark_filter="ThreadScaling"

# Save results to file
bazel run //benchmarks:batch_benchmark | tee results.txt
```

Key metrics from batch benchmarks:
- Sequential vs Batch: 4.5x-11.7x speedup
- Thread scaling: 91% efficiency at 8 threads
- Sustained throughput: 2,000+ options/second

---

## Tracing & Diagnostics

The library includes USDT probes for zero-overhead tracing.

```bash
# List available probes
readelf -x .stapsdt.base ./binary | grep mango

# Simple trace with bpftrace
sudo bpftrace -e 'usdt::mango:convergence_iter { 
    printf("Module %d, step %d, error %g\n", arg0, arg1, arg2);
}'

# Or use provided scripts
sudo ./scripts/mango-trace monitor ./binary --preset=convergence
```

---

## Common Patterns

### Pattern: American IV Surface Generation

```c
// Generate American IV surface for puts across strikes and maturities
// Note: ~145ms per IV calc, so this will be slow for large surfaces
for (int i = 0; i < n_strikes; i++) {
    for (int j = 0; j < n_maturities; j++) {
        IVParams params = {
            .spot_price = 100.0,
            .strike = strikes[i],
            .time_to_maturity = maturities[j],
            .risk_free_rate = 0.05,
            .market_price = market_prices[i][j],
            .is_call = false
        };

        IVResult result = calculate_iv_simple(&params);
        if (result.converged) {
            iv_surface[i][j] = result.implied_vol;
        }
    }
}
// Typical: 50 strikes × 20 maturities × 145ms = ~2.4 minutes
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
| IV fails to converge | Bounds too tight or FDM issues | Use `calculate_iv()` with custom bounds and grid |
| IV returns error "arbitrage" | Invalid market price | Check price vs intrinsic value |
| IV calculation slow | FDM bottleneck | Normal for American IV (~145ms), use coarser grid for testing |
| American solver fails | Grid too coarse | Increase `n_points` and/or reduce `dt` |
| American solver slow | Grid too fine | Try coarser grid first (101 points, 500 steps) |
| Memory error on batch | Too many options | Reduce batch size or grid resolution |
| Results inconsistent | Floating point | Use deterministic grids, fixed seed |

---

## Files Quick Index

| Purpose | File |
|---|---|
| American IV calculation | `src/iv_solver.{hpp,cpp}` |
| American options | `src/american_option.hpp` |
| PDE solver | `src/pde_solver.hpp` |
| Spatial operators | `src/operators/*.hpp` |
| Boundary conditions | `src/boundary_conditions.hpp` |
| Root finding | `src/root_finding.hpp` |
| Interpolation | `src/cubic_spline_solver.hpp` |
| Linear solver | `src/thomas_solver.hpp` |
| Newton solver | `src/newton_workspace.hpp` |
| Grid management | `src/grid.hpp` |
| Workspace | `src/workspace.hpp` |

---

## Further Reading

- **Overview**: `docs/PROJECT_OVERVIEW.md` - Problem domain and project motivation
- **Architecture**: `docs/ARCHITECTURE.md` - Detailed technical architecture
- **Benchmarks**: `benchmarks/BENCHMARK.md` - Performance comparisons
- **Examples**: `examples/example_*.c` - Usage examples
- **Tests**: `tests/*_test.cc` - Test suite
- **Tracing**: `TRACING.md`, `TRACING_QUICKSTART.md` - USDT tracing guide

