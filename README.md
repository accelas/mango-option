# mango-iv

**Research-grade numerical library for option pricing and implied volatility calculation**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C++20](https://img.shields.io/badge/C++-20-blue)]() [![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## What is mango-iv?

**mango-iv** is a modern C++20 library that solves two fundamental problems in quantitative finance:

1. **American Option Pricing**: Calculate fair prices using finite-difference PDE solver
2. **American Option Implied Volatility**: Invert market prices to extract implied volatility

The library combines **performance**, **flexibility**, and **correctness** with a focus on research and prototyping use cases for American options.

### Key Features

- **American Option Pricing** - PDE-based solver using TR-BDF2 method (~22ms)
- **American Option Implied Volatility** - FDM-based IV calculation using Brent's method (~145ms)
- **General PDE Solver** - Template-based framework for custom parabolic PDEs
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
- **GCC 10+** or **Clang 13+** with C++20 support
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

```cpp
#include "src/iv_solver.hpp"

// Setup option parameters
mango::IVParams params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 6.08,  // American put price
    .is_call = false
};

// Solve for implied volatility
mango::IVSolver solver(params);
mango::IVResult result = solver.solve();

if (result.converged) {
    std::cout << "Implied volatility: " << result.implied_vol
              << " (" << result.implied_vol * 100 << "%)\n";
    std::cout << "Iterations: " << result.iterations << "\n";
} else {
    std::cerr << "Failed: " << *result.failure_reason << "\n";
}
```

**Output:**
```
Implied volatility: 0.1998 (19.98%)
Iterations: 5
```

### Price an American Put Option

```cpp
#include "src/american_option.hpp"

// Create American option pricer
mango::AmericanOption pricer(
    100.0,  // strike
    0.25,   // volatility
    0.05,   // risk-free rate
    1.0,    // time to maturity (years)
    mango::OptionType::Put
);

// Price at spot = 100
double price = pricer.price(100.0);

std::cout << "American put price: $" << price << "\n";
```

**Output:**
```
American put price: $6.0842
```

### Solve a Custom PDE

```cpp
#include "src/pde_solver.hpp"
#include "src/operators/laplacian_pde.hpp"
#include "src/boundary_conditions.hpp"

// Create spatial grid
std::vector<double> grid(101);
for (size_t i = 0; i < grid.size(); ++i) {
    grid[i] = i / 100.0;  // [0, 1]
}

// Setup time domain
mango::TimeDomain time{0.0, 1.0, 0.001};

// Setup boundary conditions (Dirichlet: u = 0 at both ends)
auto left_bc = mango::DirichletBC(0.0);
auto right_bc = mango::DirichletBC(0.0);

// Setup spatial operator (Laplacian: D·∂²u/∂x²)
double diffusion = 0.1;
auto spatial_op = mango::LaplacianOperator(diffusion);

// Create solver
mango::TRBDF2Config config;
mango::RootFindingConfig root_config;
mango::PDESolver solver(grid, time, config, root_config,
                        left_bc, right_bc, spatial_op);

// Initialize with custom initial condition
auto initial_condition = [](std::span<const double> x, std::span<double> u) {
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::sin(M_PI * x[i]);
    }
};
solver.initialize(initial_condition);

// Solve
bool converged = solver.solve();

// Access solution
auto solution = solver.solution();
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
| American option (single) | 21.7ms | TR-BDF2, 141 points × 1000 steps |
| American option (batch of 64) | ~1.5ms wall | OpenMP parallelization |
| American IV (FDM-based) | ~145ms | Brent's method with full PDE solve per iteration |
| **American IV (table-based)** | **~11.8ms** | **Newton's method with interpolation (22.5× faster)** |
| Price table interpolation | ~500ns | 4D cubic spline query (43,400× faster than FDM) |
| Greeks (vega, gamma) | ~500ns | Precomputed during table generation |

### Validation & Accuracy

- **44 test cases** for implied volatility (edge cases, extreme parameters)
- **42 test cases** for American options (puts, calls, dividends, early exercise)
- **QuantLib comparison**: 0.5% relative error, 2.1x slower (reasonable for research code)
- **Interpolation accuracy**: <0.01bp mean error for in-bounds cases with table-based IV
- **Validation framework**: Reference table validation ~1000× faster than FDM validation
- **Adaptive refinement**: Achieves <1bp IV error for 95% of validation points

---

## Architecture

```
mango-iv/
├── src/                           # Core library (C++20)
│   ├── iv_solver.{hpp,cpp}        # American IV calculation
│   ├── american_option.hpp        # American option pricing
│   ├── pde_solver.hpp             # General PDE solver (TR-BDF2)
│   ├── cubic_spline_solver.hpp    # Cubic spline interpolation
│   ├── boundary_conditions.hpp    # Boundary condition types
│   ├── spatial_operators.hpp      # Spatial operator interface
│   ├── operators/                 # Operator implementations
│   │   ├── spatial_operator.hpp   # Base operator interface
│   │   ├── laplacian_pde.hpp      # Laplacian operator
│   │   ├── black_scholes_pde.hpp  # Black-Scholes operator
│   │   └── ...                    # Other operators
│   ├── grid.hpp                   # Grid management
│   ├── workspace.hpp              # Memory workspace
│   ├── thomas_solver.hpp          # Tridiagonal solver
│   ├── newton_workspace.hpp       # Newton solver workspace
│   ├── root_finding.hpp           # Root-finding utilities
│   └── ...                        # Other C++20 modules
│
├── examples/                      # Demonstration programs
│   └── example_newton_solver.cc   # Example PDE solving
│
├── tests/                         # Comprehensive test suite
│   ├── iv_solver_test.cc          # IV solver tests
│   ├── american_option_test.cc    # American option tests
│   ├── pde_solver_test.cc         # Core PDE solver tests
│   ├── cubic_spline_test.cc       # Spline tests
│   ├── boundary_conditions_test.cc # BC tests
│   ├── spatial_operators_test.cc  # Operator tests
│   └── ...                        # Additional test suites
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
2. **Performance-Conscious** - Template-based zero-cost abstractions, SIMD-friendly, compile-time optimization
3. **Type Safety** - Strong typing with concepts, compile-time checks, no void* pointers
4. **Correctness First** - Well-established methods, comprehensive tests, QuantLib validation
5. **Zero-Overhead Diagnostics** - USDT tracing (no printf in library code)
6. **Research-Friendly** - Modern C++20, clear code, extensive documentation

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
| **Language** | C++20 | C++17 | Python | C++/Python |
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
- ✅ USDT tracing system
- ✅ Comprehensive test suite
- ✅ QuantLib benchmarks
- ✅ Cubic spline interpolation (C² continuous, accurate Greeks)
- ✅ Coordinate transformation support (log-sqrt, log-variance)

### Near-Term (v0.2-0.3)
- ✅ Price table pre-computation (43,400× speedup achieved)
- ✅ Table-based IV calculation (22.5× faster than FDM)
- ✅ Greeks calculation (vega, gamma via precomputed derivatives)
- ✅ Adaptive grid refinement (<1bp IV error for 95% of points)
- ✅ Unified grid architecture (20,000× memcpy reduction)
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
