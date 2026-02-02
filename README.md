# mango-option

American option pricing and implied volatility in C++23, with Python bindings.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C++23](https://img.shields.io/badge/C++-23-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

mango-option prices American options by solving the Black-Scholes PDE with a finite difference method. The solver uses TR-BDF2 time stepping with Newton iteration to handle the early exercise constraint. On top of the PDE solver, the library provides implied volatility calculation (either by repeated PDE solves or by interpolating a pre-computed price table) and batch processing with OpenMP parallelization.

The core C++ library exposes a pybind11-based Python module, so you can use it from either language.

**What you get:**

- American option prices and Greeks (delta, gamma, theta, vega) from the PDE solver
- Implied volatility via FDM (~19ms) or B-spline interpolation (~3.5us)
- Pre-computed 4D price tables for sub-microsecond lookups (~476ns, ~1 bps near-ATM)
- Batch pricing with OpenMP (10x speedup on multi-core)
- USDT tracing probes for production monitoring at zero overhead when disabled
- A general-purpose PDE toolkit if you want to solve your own equations

---

## Quick Start

### Prerequisites

- **Bazel** (build system)
- **GCC 14+** or **Clang 19+** with C++23 support
- **GoogleTest** (automatically fetched by Bazel)

Optional:
- `systemtap-sdt-dev` (for USDT tracing)
- `libquantlib0-dev` (for QuantLib benchmarks)

### Build and Test

```bash
git clone https://github.com/your-org/mango-option.git
cd mango-option

bazel build //...   # build everything
bazel test //...    # run all tests
```

---

## Usage

### C++

```cpp
#include "src/option/american_option.hpp"

mango::PricingParams params(
    mango::OptionSpec{
        .spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.02,
        .option_type = mango::OptionType::PUT},
    0.20);  // volatility

auto result = mango::solve_american_option(params);
// result->value_at(100.0), result->delta(), result->gamma(), ...
```

### Python

```python
import mango_option as mo

params = mo.PricingParams()
params.spot, params.strike, params.maturity = 100.0, 100.0, 1.0
params.volatility, params.rate, params.dividend_yield = 0.20, 0.05, 0.02
params.option_type = mo.OptionType.PUT

result = mo.american_option_price(params)
# result.value_at(100.0), result.delta(), result.gamma()
```

The library also supports batch pricing, price table pre-computation, implied volatility solvers, and market data integration. See the [API Guide](docs/API_GUIDE.md) for C++ and the [Python Guide](docs/PYTHON_GUIDE.md) for Python.

---

## Performance

*Benchmarked on AMD Ryzen 9 9955HX (16C/32T, 5.0 GHz), 64 GB RAM, compiled with `-O3 -march=native`.*

### American Option Pricing

| Configuration | Grid | Time/Option | Use Case |
|---|---|---|---|
| Standard (auto) | 101x498 | ~1.35ms | Single option |
| Option chain (shared grid) | 101x498 | ~0.23ms | Multi-strike |

Batch processing (64 options): 10x speedup with OpenMP (~0.13ms/option parallel vs ~1.34ms sequential). Shared-grid chains get another 6x on top of that.

### Implied Volatility

| Method | Time/IV | Near-ATM | Full-chain |
|---|---|---|---|
| FDM-based | ~19ms | Ground truth | Ground truth |
| Interpolated (B-spline) | ~3.5us | ~1 bps IV | ~$0.005 price RMSE |

The interpolation path is 5,400x faster than FDM. You pre-compute a 4D price table (moneyness x maturity x vol x rate), then query it with B-spline interpolation. EEP decomposition (P = P_European + EEP) improves interpolation accuracy by separating the smooth early exercise premium from the closed-form European component.

### Price Table Profiles

The table below shows the accuracy/speed tradeoff across grid density profiles, measured on real SPY option data (7-day puts, strikes from 88% to 107% of spot):

| Profile | PDE solves | ATM IV err (bps) | Near-OTM IV err (bps) | Deep-OTM IV err (bps) | Near-ITM IV err (bps) | Deep-ITM IV err (bps)† | Price RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 100 | 10.4 | 2.8 | 20.7 | 13.9 | 2006 | $0.016 |
| Medium | 240 | 2.7 | 2.9 | 22.7 | 1.1 | 2070 | $0.005 |
| High (default) | 495 | 0.4 | 3.3 | 22.8 | 1.3 | 2023 | $0.005 |
| Ultra | 812 | 0.3 | 2.9 | 22.0 | 0.8 | 2002 | $0.004 |

†Deep-ITM and deep-OTM options share the same low-vega characteristic: vega is near zero, so even a tiny price error (< $0.01) maps to thousands of bps in IV space. The actual price-relative error remains small — deep-ITM price RMSE is < $0.001 across all profiles. **Price RMSE is the stable metric** across all moneyness regimes. Use `from_chain_auto_profile()` with Low/Medium/High/Ultra to control the density/speed tradeoff.

For detailed profiling data, see [docs/PERF_ANALYSIS.md](docs/PERF_ANALYSIS.md).

---

## Documentation

| Document | Covers |
|----------|--------|
| [API Guide](docs/API_GUIDE.md) | Usage examples, patterns, and recipes |
| [Architecture](docs/ARCHITECTURE.md) | Software design, C++23 patterns, module structure |
| [Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md) | PDE formulation, TR-BDF2, Newton iteration, B-splines |
| [Python Guide](docs/PYTHON_GUIDE.md) | Python bindings API reference |
| [Performance Analysis](docs/PERF_ANALYSIS.md) | Instruction-level profiling data |
| [Tracing Guide](docs/TRACING.md) | USDT probe documentation |
---

## Project Structure

```
mango-option/
├── src/
│   ├── pde/
│   │   ├── core/          # Grid, PDESolver, boundary conditions
│   │   └── operators/     # Spatial operators (Black-Scholes, Laplacian)
│   ├── option/            # American option pricing, IV solvers, price tables
│   ├── math/              # Root finding, B-splines, tridiagonal solvers
│   ├── simple/            # Market data integration (yfinance, Databento, IBKR)
│   ├── python/            # Python bindings (pybind11)
│   └── support/           # Memory management, CPU features, utilities
├── tests/                 # 100+ test files with GoogleTest
├── benchmarks/            # Performance benchmarks
├── third_party/           # External dependency configs
├── tools/                 # Build helpers, tracing scripts
└── docs/                  # Documentation
```

---

## Testing

```bash
bazel test //...                                    # all tests
bazel test //tests:pde_solver_test                  # specific suite
bazel test //tests:pde_solver_test --test_output=all  # verbose
```

### Fuzz Testing

Property-based fuzz tests check mathematical invariants (monotonicity, Greek bounds, no crashes on extreme inputs):

```bash
# Requires Earthly: https://earthly.dev/get-earthly
# Earthly provides a containerized build environment to work around
# Clang + libc++ ABI incompatibilities needed for FuzzTest.
earthly +fuzz-test           # quick run (unit test mode)
earthly +fuzz-test-extended  # extended fuzzing (100K iterations)
```

Properties tested: put/call price monotonicity in strike, vega positivity, delta bounds, gamma non-negativity, no crashes on extreme parameters.

---

## Contributing

Contributions welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/descriptive-name`)
3. Make changes and add tests
4. Verify everything builds: `bazel test //...`
5. Commit with an imperative-mood message
6. Push and open a Pull Request

See [CLAUDE.md](CLAUDE.md) for detailed workflow guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- TR-BDF2 scheme: Bank et al. (1985)
- American options: Wilmott, *Derivatives*
- B-splines: de Boor, *A Practical Guide to Splines*
- Finite differences: LeVeque, *Finite Difference Methods*
