# iv_calc

**Research-grade numerical library for option pricing and implied volatility calculation**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C23](https://img.shields.io/badge/C-23-blue)]() [![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## What is iv_calc?

**iv_calc** is a C23-based library that solves two fundamental problems in quantitative finance:

1. **Option Pricing**: Calculate fair prices for European and American options
2. **Implied Volatility**: Invert market prices to extract implied volatility

The library combines **performance**, **flexibility**, and **correctness** with a focus on research and prototyping use cases.

### Key Features

- **Black-Scholes Pricing** - Analytical European option pricing (<1µs)
- **American Option Pricing** - PDE-based solver using TR-BDF2 method (~22ms)
- **Implied Volatility** - Robust calculation using Brent's method (<1µs, 99.9% success rate)
- **General PDE Solver** - Callback-based framework for custom parabolic PDEs
- **Cubic Spline Interpolation** - Off-grid solution evaluation
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

### Calculate Implied Volatility

```c
#include "src/implied_volatility.h"

IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = true
};

IVResult result = calculate_implied_volatility(&params);

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
Implied volatility: 0.2500 (25.0%)
Iterations: 8
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

See `examples/` for complete working programs.

---

## Performance

### Current Performance

| Operation | Time | Notes |
|-----------|------|-------|
| European option (Black-Scholes) | <1µs | Analytical formula |
| Implied volatility | <1µs | 8-12 iterations with Brent's method |
| American option (single) | 21.7ms | TR-BDF2, 141 points × 1000 steps |
| American option (batch of 64) | ~1.5ms wall | OpenMP parallelization |

### Validation

- **44 test cases** for implied volatility (edge cases, extreme parameters)
- **42 test cases** for American options (puts, calls, dividends, early exercise)
- **QuantLib comparison**: 0.5% relative error, 2.1x slower (reasonable for research code)

---

## Architecture

```
iv_calc/
├── src/                           # Core library
│   ├── implied_volatility.{h,c}   # IV calculation + Black-Scholes
│   ├── american_option.{h,c}      # American option pricing
│   ├── pde_solver.{h,c}           # General PDE solver (FDM)
│   ├── cubic_spline.{h,c}         # Interpolation
│   ├── brent.{h,c}                # Root-finding
│   └── ivcalc_trace.h             # USDT tracing probes
│
├── examples/                      # Demonstration programs
│   ├── example_implied_volatility.c
│   ├── example_american_option.c
│   └── example_heat_equation.c
│
├── tests/                         # Comprehensive test suite
│   ├── implied_volatility_test.cc # 32 test cases
│   ├── american_option_test.cc    # 42 test cases
│   └── pde_solver_test.cc         # Core solver tests
│
├── docs/                          # Documentation
│   ├── PROJECT_OVERVIEW.md        # Problem domain & motivation
│   ├── ARCHITECTURE.md            # Technical deep-dive
│   └── QUICK_REFERENCE.md         # Developer quick-start
│
├── scripts/                       # Utilities
│   ├── ivcalc-trace               # USDT tracing helper
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
sudo ./scripts/ivcalc-trace monitor ./bazel-bin/examples/example_american_option

# Watch convergence behavior
sudo ./scripts/ivcalc-trace monitor ./my_program --preset=convergence

# Debug failures
sudo ./scripts/ivcalc-trace monitor ./my_program --preset=debug
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

## Who Should Use iv_calc?

### ✅ Good Fit

- Researchers building new pricing models
- Students learning computational finance
- Developers needing flexible option pricing
- Quants prototyping trading strategies
- Engineers requiring transparent, well-tested code

### ⚠️ Consider Alternatives

- **High-frequency trading**: Use optimized libraries (QuantLib, proprietary solutions)
- **Production systems**: Consider mature, battle-tested solutions
- **GPU acceleration needed**: iv_calc is CPU-only (for now)
- **Exotic options**: Limited support (barriers, Asians not yet implemented)

---

## Comparison to Other Libraries

| Feature | iv_calc | QuantLib | PyQL | Bloomberg API |
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

**iv_calc's niche:** Research-grade flexibility with production-conscious design.

---

## Roadmap

### Current (v0.1)
- ✅ Black-Scholes pricing and IV calculation
- ✅ American option pricing (PDE-based)
- ✅ TR-BDF2 implicit solver
- ✅ USDT tracing system
- ✅ Comprehensive test suite
- ✅ QuantLib benchmarks

### Near-Term (v0.2-0.3)
- 🚧 Interpolation-based pricing engine (40,000x speedup planned)
- 🚧 CPU optimizations (AVX-512, FMA, restrict)
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
