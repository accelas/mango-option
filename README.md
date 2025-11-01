# mango-iv

**Research-grade numerical library for option pricing and implied volatility calculation**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C23](https://img.shields.io/badge/C-23-blue)]() [![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## What is mango-iv?

**mango-iv** is a C23-based library that solves two fundamental problems in quantitative finance:

1. **American Option Pricing**: Calculate fair prices using finite-difference PDE solver
2. **American Option Implied Volatility**: Invert market prices to extract implied volatility

The library combines **performance**, **flexibility**, and **correctness** with a focus on research and prototyping use cases for American options.

### Key Features

- **American Option Pricing** - PDE-based solver using TR-BDF2 method (~22ms)
- **American Option Implied Volatility** - FDM-based IV calculation using Brent's method (~145ms)
- **Let's Be Rational** - Fast European IV estimation for bound calculation (~781ns)
- **General PDE Solver** - Callback-based framework for custom parabolic PDEs
- **Cubic Spline Interpolation** - Off-grid solution evaluation
- **Price Table Pre-computation** - Fast lookups via interpolation (future: ~7.5µs IV)
- **Adaptive Grid Presets** - Non-uniform spacing with 4-23× memory reduction
- **Zero-Overhead Tracing** - USDT probes for production-safe diagnostics
- **Batch Processing** - OpenMP parallelization for multiple calculations
- **SIMD Vectorization** - Automatic vectorization via OpenMP pragmas

---

## Quick Start

### Prerequisites

- **Bazel** (build system)
- **GCC** or **Clang** with C23 support
- **GoogleTest** (automatically fetched by Bazel)

Optional:
- `systemtap-sdt-dev` (for USDT tracing support)
- `libquantlib0-dev` (for QuantLib benchmarks)

### Build and Test

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...

# Run examples
bazel run //examples:example_implied_volatility
bazel run //examples:example_american_option
bazel run //examples:example_heat_equation
```

---

## Usage Examples

### Calculate American Option Implied Volatility

```c
#include "src/implied_volatility.h"

IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 6.08,  // American put price
    .is_call = false
};

IVResult result = calculate_iv_simple(&params);

if (result.converged) {
    printf("Implied volatility: %.4f (%.1f%%)\n",
           result.implied_vol, result.implied_vol * 100);
    printf("Iterations: %d\n", result.iterations);
} else {
    printf("Failed: %s\n", result.error);
}
```

**Output:**
```
Implied volatility: 0.1998 (19.98%)
Iterations: 5
```

### Price an American Put Option

```c
#include "src/american_option.h"

double price = american_option_price(
    100.0,  // spot price
    100.0,  // strike
    1.0,    // time to maturity (years)
    0.05,   // risk-free rate
    0.25,   // volatility
    0.0,    // dividend yield
    false,  // is_call (false = put)
    141,    // spatial grid points
    1000    // time steps
);

printf("American put price: $%.4f\n", price);
```

**Output:**
```
American put price: $6.0842
```

### Solve a Custom PDE

```c
#include "src/pde_solver.h"

// Define callbacks for heat equation: ∂u/∂t = D·∂²u/∂x²
void heat_operator(const double *x, double t, const double *u,
                   size_t n, double *Lu, void *user_data) {
    const double dx = x[1] - x[0];
    const double D = *(double*)user_data;  // diffusion coefficient

    Lu[0] = Lu[n-1] = 0.0;  // boundaries

    #pragma omp simd
    for (size_t i = 1; i < n - 1; i++) {
        Lu[i] = D * (u[i-1] - 2.0*u[i] + u[i+1]) / (dx * dx);
    }
}

// Setup and solve
SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);
TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = 0.001, .n_steps = 1000};
double diffusion = 0.1;

PDECallbacks callbacks = {
    .initial_condition = my_ic_func,
    .left_boundary = my_left_bc,
    .right_boundary = my_right_bc,
    .spatial_operator = heat_operator,
    .user_data = &diffusion
};

BoundaryConfig bc = pde_default_boundary_config();
TRBDF2Config trbdf2 = pde_default_trbdf2_config();

PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
pde_solver_initialize(solver);
pde_solver_solve(solver);

const double *solution = pde_solver_get_solution(solver);
// Use solution...

pde_solver_destroy(solver);
```

### Use Grid Presets for Memory-Efficient Price Tables

Grid presets provide optimized non-uniform spacing that concentrates grid points where option prices have high curvature (ATM, short maturities):

```c
#include "src/grid_presets.h"
#include "src/price_table.h"

// Get adaptive balanced preset (15K points, ~7.5× memory reduction)
GridConfig config = grid_preset_get(
    GRID_PRESET_ADAPTIVE_BALANCED,
    0.7, 1.3,      // moneyness range
    0.027, 2.0,    // maturity range (1 week to 2 years)
    0.10, 0.80,    // volatility range
    0.0, 0.10,     // rate range
    0.0, 0.0);     // no dividend (4D table)

// Generate all grids
GeneratedGrids grids = grid_generate_all(&config);

// Create price table with adaptive grids
OptionPriceTable *table = price_table_create_ex(
    grids.moneyness, grids.n_moneyness,
    grids.maturity, grids.n_maturity,
    grids.volatility, grids.n_volatility,
    grids.rate, grids.n_rate,
    nullptr, 0,
    OPTION_PUT, AMERICAN,
    COORD_RAW, LAYOUT_M_INNER);

// Precompute all prices
AmericanOptionGrid fdm_grid = {
    .x_min = -0.7,
    .x_max = 0.7,
    .n_points = 101,
    .dt = 0.001,
    .n_steps = 2000
};

price_table_precompute(table, &fdm_grid);

// Fast queries via interpolation
double price = price_table_interpolate_4d(table, 1.05, 0.5, 0.20, 0.05);

price_table_destroy(table);
```

**Available presets:**
- `GRID_PRESET_ADAPTIVE_FAST`: ~5K points, rapid prototyping
- `GRID_PRESET_ADAPTIVE_BALANCED`: ~15K points, production-ready
- `GRID_PRESET_ADAPTIVE_ACCURATE`: ~30K points, high-accuracy
- `GRID_PRESET_UNIFORM`: ~112K points, baseline (no concentration)

See `examples/` for complete working programs.

---

## Performance

### Current Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Let's Be Rational (European IV) | ~781ns | Fast bound estimation for American IV |
| American option (single) | 21.7ms | TR-BDF2, 141 points × 1000 steps |
| American option (batch of 64) | ~1.5ms wall | OpenMP parallelization |
| American implied volatility | ~145ms | FDM-based IV with Brent's method |
| Future: IV via interpolation | ~7.5µs | 40,000× speedup (planned) |

### Validation

- **44 test cases** for implied volatility (edge cases, extreme parameters)
- **42 test cases** for American options (puts, calls, dividends, early exercise)
- **QuantLib comparison**: 0.5% relative error, 2.1x slower (reasonable for research code)

---

## Architecture

```
mango-iv/
├── src/                           # Core library
│   ├── implied_volatility.{h,c}   # American IV calculation (FDM + Brent)
│   ├── lets_be_rational.{h,c}     # European IV estimation (bound calculation)
│   ├── american_option.{h,c}      # American option pricing (FDM)
│   ├── pde_solver.{h,c}           # General PDE solver (TR-BDF2)
│   ├── cubic_spline.{h,c}         # 1D cubic spline interpolation
│   ├── interp_strategy.h          # Interpolation strategy pattern
│   ├── interp_cubic.{h,c}         # Multi-dimensional cubic interpolation
│   ├── interp_cubic_workspace.c   # Workspace management for cubic splines
│   ├── price_table.{h,c}          # 4D/5D option price tables
│   ├── iv_surface.{h,c}           # 2D implied volatility surfaces
│   ├── grid_generation.{h,c}      # Non-uniform grid spacing utilities
│   ├── grid_presets.{h,c}         # Preset grid configurations
│   ├── brent.h                    # Brent's method (root-finding)
│   ├── tridiagonal.h              # Tridiagonal solver
│   └── mango_trace.h             # USDT tracing probes
│
├── examples/                      # Demonstration programs
│   ├── example_implied_volatility.c
│   ├── example_american_option.c
│   ├── example_american_option_dividend.c
│   ├── example_heat_equation.c
│   ├── example_interpolation_engine.c
│   ├── example_precompute_table.c
│   └── test_cubic_4d_5d.c
│
├── tests/                         # Comprehensive test suite
│   ├── implied_volatility_test.cc # American IV calculation tests
│   ├── lets_be_rational_test.cc   # European IV estimation tests
│   ├── american_option_test.cc    # American option tests
│   ├── pde_solver_test.cc         # Core PDE solver tests
│   ├── cubic_spline_test.cc       # 1D spline tests
│   ├── interpolation_test.cc      # Multi-dimensional interpolation tests
│   ├── interpolation_workspace_test.cc
│   ├── cubic_interp_4d_5d_test.cc
│   ├── price_table_test.cc        # Price table tests
│   ├── price_table_slow_test.cc   # Long-running table tests
│   ├── coordinate_transform_test.cc
│   ├── memory_layout_test.cc
│   ├── grid_generation_test.cc    # Grid spacing tests
│   ├── grid_presets_test.cc       # Preset configuration tests
│   ├── brent_test.cc              # Root-finding tests
│   ├── tridiagonal_test.cc        # Linear solver tests
│   └── stability_test.cc          # Numerical stability tests
│
├── docs/                          # Documentation
│   ├── PROJECT_OVERVIEW.md        # Problem domain & motivation
│   ├── ARCHITECTURE.md            # Technical deep-dive
│   └── QUICK_REFERENCE.md         # Developer quick-start
│
├── scripts/                       # Utilities
│   ├── mango-trace               # USDT tracing helper
│   └── tracing/                   # bpftrace scripts
│
├── CLAUDE.md                      # Instructions for Claude Code
├── TRACING.md                     # USDT tracing guide
└── MODULE.bazel                   # Build configuration
```

### Key Design Principles

1. **Modularity** - Clear separation: IV ← American pricing ← PDE solver
2. **Performance-Conscious** - Vectorized, SIMD-friendly, zero-allocation hot paths
3. **Correctness First** - Well-established methods, comprehensive tests, QuantLib validation
4. **Zero-Overhead Diagnostics** - USDT tracing (no printf in library code)
5. **Research-Friendly** - Modern C23, clear code, extensive documentation

---

## USDT Tracing

The library includes **zero-overhead tracing** via USDT (User Statically-Defined Tracing) probes:

```bash
# Monitor all library activity
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_american_option

# Watch convergence behavior
sudo ./scripts/mango-trace monitor ./my_program --preset=convergence

# Debug failures
sudo ./scripts/mango-trace monitor ./my_program --preset=debug
```

Tracing provides:
- **Zero overhead** when not active (single NOP instruction)
- **Dynamic control** - enable/disable at runtime without recompiling
- **Production-safe** - can be left in production code

See [TRACING_QUICKSTART.md](TRACING_QUICKSTART.md) for a 5-minute tutorial.

---

## Documentation

- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Problem domain, motivation, use cases
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed technical architecture
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Developer quick-start guide
- **[TRACING.md](TRACING.md)** - Comprehensive USDT tracing documentation
- **[CLAUDE.md](CLAUDE.md)** - Instructions for Claude Code (AI assistant)

---

## Who Should Use mango-iv?

### ✅ Good Fit

- Researchers building new pricing models
- Students learning computational finance
- Developers needing flexible option pricing
- Quants prototyping trading strategies
- Engineers requiring transparent, well-tested code

### ⚠️ Consider Alternatives

- **High-frequency trading**: Use optimized libraries (QuantLib, proprietary solutions)
- **Production systems**: Consider mature, battle-tested solutions
- **GPU acceleration needed**: mango-iv is CPU-only (for now)
- **Exotic options**: Limited support (barriers, Asians not yet implemented)

---

## Comparison to Other Libraries

| Feature | mango-iv | QuantLib | PyQL | Bloomberg API |
|---------|---------|----------|------|---------------|
| **Language** | C23 | C++17 | Python | C++/Python |
| **License** | Open-source | BSD | BSD | Proprietary |
| **American options** | ✅ PDE (TR-BDF2) | ✅ Multiple methods | ✅ Via QuantLib | ✅ |
| **Implied volatility** | ✅ Brent's | ✅ Newton/Brent | ✅ | ✅ |
| **Custom PDEs** | ✅ Callback-based | ❌ | ❌ | ❌ |
| **USDT tracing** | ✅ Zero-overhead | ❌ | ❌ | ❌ |
| **Batch API** | ✅ OpenMP | ❌ | ❌ | ❌ |
| **Documentation** | ✅ Extensive | ⚠️ Patchy | ⚠️ Limited | ✅ Commercial |
| **Performance** | 2.1x slower | Baseline | Slow (Python) | Fast (optimized) |

**mango-iv's niche:** Research-grade flexibility with production-conscious design.

---

## Roadmap

### Current (v0.1)
- ✅ American option pricing (PDE-based, TR-BDF2)
- ✅ American option implied volatility (FDM + Brent's method)
- ✅ Let's Be Rational (European IV for bound estimation)
- ✅ USDT tracing system
- ✅ Comprehensive test suite
- ✅ QuantLib benchmarks

### Near-Term (v0.2-0.3)
- ✅ Cubic spline interpolation (C² continuous, accurate Greeks)
- ✅ Coordinate transformation support (log-sqrt, log-variance)
- 🚧 Price table pre-computation (40,000x speedup planned)
- 🚧 Greeks calculation via finite differences
- 🚧 Volatility surface calibration

### Future (v1.0+)
- ⭐ GPU acceleration (CUDA/OpenCL)
- ⭐ Exotic options (barriers, Asians, lookbacks)
- ⭐ Monte Carlo simulation
- ⭐ Stochastic volatility models (Heston, SABR)
- ⭐ Python bindings

---

## Contributing

Contributions welcome! Areas of interest:

1. **Performance** - SIMD optimizations, cache-blocking, algorithm improvements
2. **Features** - Exotic options, new PDE schemes, Monte Carlo methods
3. **Testing** - More test cases, edge cases, stress tests
4. **Documentation** - Tutorials, explanations, examples
5. **Benchmarks** - Comparisons with other libraries

Please follow the commit message guidelines in [CLAUDE.md](CLAUDE.md).

---

## License

[To be determined]

---

## References

### Foundational Papers
- **Black, F., & Scholes, M.** (1973). *The Pricing of Options and Corporate Liabilities*
- **Merton, R. C.** (1973). *Theory of Rational Option Pricing*
- **Brent, R. P.** (1973). *Algorithms for Minimization without Derivatives*
- **Ascher, U. M., Ruuth, S. J., & Wetton, B. T. R.** (1995). *Implicit-Explicit Methods for Time-Dependent PDEs*

### Books
- **Hull, J.** (2021). *Options, Futures, and Other Derivatives* (10th ed.)
- **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*

### Software
- [QuantLib](https://www.quantlib.org/) - C++ library for quantitative finance
- [PyQL](https://github.com/enthought/pyql) - Python bindings for QuantLib

---

**For technical details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
