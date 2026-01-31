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
- Pre-computed 4D price tables with EEP decomposition for sub-microsecond lookups (~500ns, ~9 bps accuracy)
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

mango::AmericanOptionParams params(
    100.0, 100.0, 1.0, 0.05, 0.02, mango::OptionType::PUT, 0.20);

auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);
std::pmr::synchronized_pool_resource pool;
std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(grid_spec.n_points()), &pool);
auto workspace = mango::PDEWorkspace::from_buffer(buffer, grid_spec.n_points()).value();

mango::AmericanOptionSolver solver(params, workspace);
auto result = solver.solve();
// result->value_at(spot), result->delta(), result->gamma(), ...
```

### Python

```python
import mango_option as mo

params = mo.AmericanOptionParams()
params.spot, params.strike, params.maturity = 100.0, 100.0, 1.0
params.volatility, params.rate, params.dividend_yield = 0.20, 0.05, 0.02
params.type = mo.OptionType.PUT

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

| Method | Time/IV | Accuracy |
|---|---|---|
| FDM-based | ~19ms | Ground truth |
| Interpolated (B-spline) | ~3.5us | 8-10 bps |

The interpolation path is 5,400x faster than FDM. You pre-compute a 4D price table (moneyness x maturity x vol x rate), then query it with B-spline interpolation. Price tables use EEP decomposition — they store the Early Exercise Premium instead of raw prices, removing the American free boundary discontinuity from the interpolated surface.

### Price Table Profiles

The table below shows the accuracy/speed tradeoff across grid density profiles, measured on real SPY option data:

| Profile | Grid (mxTxσxr) | PDE solves | Interp IV | Max err (bps) | Avg err (bps) |
|---|---:|---:|---:|---:|---:|
| Low | 8x8x14x6 | 84 | ~4us | 36 | 9.4 |
| Medium | 10x10x20x8 | 160 | ~4us | 37 | 9.2 |
| High (default) | 12x12x30x10 | 300 | ~4us | 37 | 8.4 |
| Ultra | 15x15x43x12 | 516 | ~4us | 37 | 8.8 |

Use `from_chain_auto_profile()` with Low/Medium/High/Ultra to control this tradeoff. With EEP decomposition, even the Low profile achieves ~9 bps average error — accuracy that previously required the highest grid densities with raw price interpolation. The max error plateaus around ~37 bps from a few deep OTM near-expiry options where the EEP is near zero.

See the [API Guide](docs/API_GUIDE.md#price-table-pre-computation) for usage.

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
