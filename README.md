# mango-iv

Modern C++23 library for American option pricing and implied volatility calculation.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![C++23](https://img.shields.io/badge/C++-23-blue)]() [![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## What is mango-iv?

**mango-iv** solves two fundamental problems in quantitative finance:

1. **American Option Pricing** – Finite difference PDE solver with TR-BDF2 time stepping
2. **Implied Volatility Calculation** – Extract implied volatility from market prices

Designed for research, prototyping, and production use with a focus on **correctness**, **performance**, and **flexibility**.

### Key Features

- **Fast American Pricing** – ~1.3ms per option via PDE solver
- **Implied Volatility** – FDM-based (~15ms) or interpolation-based (~2.1µs)
- **Price Tables** – Pre-compute 4D B-spline surfaces for sub-microsecond queries (~193ns)
- **Modern C++23** – std::expected error handling, PMR memory management, SIMD vectorization
- **General PDE Toolkit** – Custom PDEs, boundary conditions, spatial operators
- **Production Ready** – OpenMP batching (10× speedup), USDT tracing, zero-allocation solves

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
git clone https://github.com/your-org/mango-iv.git
cd mango-iv

# Build everything
bazel build //...

# Run all tests
bazel test //...

# Run examples
bazel run //examples:example_newton_solver
```

---

## Usage Examples

### American Option Pricing

```cpp
#include "src/option/american_option.hpp"

// Option parameters
mango::PricingParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .type = mango::OptionType::PUT
};

// Auto-estimate grid
auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);

// Create workspace
std::pmr::synchronized_pool_resource pool;
auto workspace = mango::PDEWorkspace::create(grid_spec, &pool).value();

// Solve
mango::AmericanOptionSolver solver(params, workspace);
auto result = solver.solve();

if (result.has_value()) {
    std::cout << "Price: " << result->price() << "\n";
    std::cout << "Delta: " << result->delta() << "\n";
}
```

### Implied Volatility Calculation

```cpp
#include "src/option/iv_solver_fdm.hpp"

// Option specification
mango::OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = mango::OptionType::PUT
};

// IV query with market price
mango::IVQuery query{.option = spec, .market_price = 10.45};

// Solve
mango::IVSolverFDM solver(mango::IVSolverFDMConfig{});
auto result = solver.solve_impl(query);

if (result.has_value()) {
    std::cout << "Implied Vol: " << result->implied_vol << "\n";
    std::cout << "Iterations: " << result->iterations << "\n";
} else {
    std::cerr << "Error: " << result.error().message << "\n";
}
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

### American Option Pricing

| Configuration | Grid | Time/Option | Use Case |
|---|---|---|---|
| Standard (auto) | 101×498 | ~1.3ms | Typical case |
| Custom fine | 201×2k | ~8-10ms | High accuracy |

**Batch processing (64 options):**
- Sequential: ~81ms total (~1.26ms/option)
- Parallel: ~7.7ms total (~0.12ms/option)
- **10.4× speedup** with parallelization

### Implied Volatility

| Method | Time/IV | Accuracy |
|---|---|---|
| FDM-based (101×1k) | ~15ms | Ground truth |
| FDM-based (201×2k) | ~61ms | High accuracy |
| Interpolated (B-spline) | ~2.1µs | <1bp error (95%) |

**Speedup:** 7,000× for interpolated vs FDM

### Price Table Pre-Computation

- **Grid size:** 300K points (50×30×20×10)
- **Pre-compute:** 15-20 min (32 cores)
- **Query:** ~193ns (price), ~952ns (vega+gamma)
- **Speedup:** 77,000× vs FDM

---

## Project Structure

```
mango-iv/
├── src/
│   ├── pde/
│   │   ├── core/          # Grid, PDESolver, boundary conditions
│   │   └── operators/     # Spatial operators (Black-Scholes, Laplacian)
│   ├── option/            # American option pricing, IV solvers, price tables
│   ├── math/              # Root finding, B-splines, tridiagonal solvers
│   └── support/           # Memory management, CPU features, utilities
├── tests/                 # 38 test files with GoogleTest
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

**Test coverage:** 38 test files, 13,000+ lines of tests

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

TBD (to be determined)

---

## Acknowledgments

- TR-BDF2 scheme: Bank et al. (1985)
- American options: Wilmott, "Derivatives"
- B-splines: de Boor, "A Practical Guide to Splines"
- Finite differences: LeVeque, "Finite Difference Methods"

---

## Contact

For questions or feedback, please open an issue on GitHub.
