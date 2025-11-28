# mango-option

Modern C++23 library for American option pricing and implied volatility calculation.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C++23](https://img.shields.io/badge/C++-23-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is mango-option?

**mango-option** solves two fundamental problems in quantitative finance:

1. **American Option Pricing** – Finite difference PDE solver with TR-BDF2 time stepping
2. **Implied Volatility Calculation** – Extract implied volatility from market prices

Designed for research, prototyping, and production use with a focus on **correctness**, **performance**, and **flexibility**.

### Key Features

- **Fast American Pricing** – ~1.4ms per option via PDE solver
- **Implied Volatility** – FDM-based (~19ms) or interpolation-based (~3.5µs)
- **Price Tables** – Pre-compute 4D B-spline surfaces for sub-microsecond queries (~470ns)
- **Modern C++23** – std::expected error handling, PMR memory management, SIMD vectorization
- **General PDE Toolkit** – Custom PDEs, boundary conditions, spatial operators
- **Production Ready** – OpenMP batching (15× speedup), USDT tracing, zero-allocation parallel workloads

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
# Clone repository
git clone https://github.com/your-org/mango-option.git
cd mango-option

# Build everything
bazel build //...

# Run all tests
bazel test //...

# Run examples
bazel run //examples:example_newton_solver
```

---

## Usage Examples

### Market Data Integration (Recommended)

```cpp
#include "src/simple/simple.hpp"

using namespace mango::simple;

// Build option chain from yfinance data
auto chain = ChainBuilder<YFinanceSource>{}
    .symbol("SPY")
    .spot(580.50)
    .quote_time("2024-06-21T10:30:00")
    .dividend_yield(0.013)
    .add_put("2024-06-21", {.strike = 575.0, .bid = 0.52, .ask = 0.58})
    .add_put("2024-06-21", {.strike = 580.0, .bid = 2.30, .ask = 2.42})
    .add_call("2024-06-21", {.strike = 580.0, .bid = 2.85, .ask = 2.92})
    .build();

// Set up market context
MarketContext ctx;
ctx.rate = 0.053;  // 5.3% Fed Funds rate
ctx.valuation_time = Timestamp{"2024-06-21T10:30:00"};

// Compute volatility surface
auto surface = compute_vol_surface(chain, ctx);
if (surface.has_value()) {
    for (const auto& smile : surface->smiles) {
        std::cout << "Expiry: " << smile.expiry.to_string() << "\n";
        for (const auto& pt : smile.puts) {
            std::cout << "  K=" << pt.strike.to_double()
                      << " IV=" << pt.iv_mid.value_or(0.0) << "\n";
        }
    }
}
```

### Low-Level API

For direct access to the PDE solver:

```cpp
#include "src/option/american_option.hpp"

mango::AmericanOptionParams params(
    100.0,  // spot
    100.0,  // strike
    1.0,    // maturity
    0.05,   // rate
    0.02,   // dividend_yield
    mango::OptionType::PUT,
    0.20    // volatility
);

auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);
std::pmr::synchronized_pool_resource pool;
std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(grid_spec.n_points()), &pool);
auto workspace = mango::PDEWorkspace::from_buffer(buffer, grid_spec.n_points()).value();

mango::AmericanOptionSolver solver(params, workspace);
auto result = solver.solve();
std::cout << "Price: " << result->value_at(params.spot) << "\n";
```

**For more examples, see [docs/API_GUIDE.md](docs/API_GUIDE.md)**

---

## Documentation

- **[API Guide](docs/API_GUIDE.md)** – Usage examples and patterns
- **[Architecture](docs/ARCHITECTURE.md)** – Software design and C++23 patterns
- **[Mathematical Foundations](docs/MATHEMATICAL_FOUNDATIONS.md)** – PDE formulations and numerical methods
- **[CLAUDE.md](CLAUDE.md)** – Workflow guide for AI assistants
- **[Tracing Guide](docs/TRACING.md)** – USDT probe documentation
- **[Build System](docs/BUILD_SYSTEM.md)** – Bazel configuration details

---

## Performance

*Benchmarked on AMD Ryzen 9 9955HX (16C/32T, 5.0 GHz), 64 GB RAM, compiled with `-O3 -march=native`*

### American Option Pricing

| Configuration | Grid | Time/Option | Use Case |
|---|---|---|---|
| Standard (auto) | 101×498 | ~1.4ms | Single option |
| Option chain (shared grid) | 101×498 | ~0.13ms | Multi-strike |

**Batch processing (64 options):**
- Sequential: ~87ms total (~1.4ms/option)
- Parallel: ~5.7ms total (~0.09ms/option)
- **15× speedup** with parallelization

**Option chain (15 options, shared grid):**
- Total: ~2.0ms (~0.13ms/option)
- **10× speedup** vs individual pricing

### Implied Volatility

| Method | Grid | Time/IV | Accuracy |
|---|---|---|---|
| FDM-based (auto) | 101×498 | ~19ms | Ground truth |
| Interpolated (B-spline) | — | ~3.5µs | <1bp error (95%) |

**Speedup:** 5,400× for interpolated vs FDM

### Price Table Pre-Computation

- **Grid size:** 300K points (50×30×20×10)
- **Pre-compute:** 15-20 min (32 cores)
- **Query:** ~470ns (price), ~2.4µs (vega+gamma)
- **Speedup:** 40,000× vs FDM

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
│   └── support/           # Memory management, CPU features, utilities
├── tests/                 # 86+ test files with GoogleTest
├── examples/              # Example programs
├── benchmarks/            # Performance benchmarks
└── docs/                  # Documentation
```

---

## Testing

```bash
# Run all tests
bazel test //...

# Run specific test suites
bazel test //tests:pde_solver_test
bazel test //tests:american_option_test
bazel test //tests:iv_solver_test

# Run with verbose output
bazel test //tests:pde_solver_test --test_output=all
```

**Test coverage:** 86+ test files

---

## Contributing

This is a research project. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests (`bazel test //...`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Create Pull Request

**See [CLAUDE.md](CLAUDE.md) for detailed workflow guidelines**

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- TR-BDF2 scheme: Bank et al. (1985)
- American options: Wilmott, "Derivatives"
- B-splines: de Boor, "A Practical Guide to Splines"
- Finite differences: LeVeque, "Finite Difference Methods"

---

## Contact

For questions or feedback, please open an issue on GitHub.
