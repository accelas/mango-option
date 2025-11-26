# CLAUDE.md

Guide for Claude Code when working with this repository.

## Project Overview

**mango-iv** is a C++23 library for pricing American options using finite difference methods. The core solver uses TR-BDF2 time-stepping with Newton iteration. The library provides high-level APIs for option pricing, implied volatility calculation (FDM and interpolation-based), and price table pre-computation.

**Key capabilities:**
- American option pricing via PDE solver (~5-20ms per option)
- Implied volatility calculation (~143ms FDM, ~12ms interpolated)
- Price table pre-computation with B-spline interpolation (~500ns per query)
- Batch processing with OpenMP parallelization

**For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
**For mathematical foundations, see [docs/MATHEMATICAL_FOUNDATIONS.md](docs/MATHEMATICAL_FOUNDATIONS.md)**
**For usage examples, see [docs/API_GUIDE.md](docs/API_GUIDE.md)**

## Build System

This project uses Bazel with Bzlmod.

### Common Commands

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...

# Run specific tests
bazel test //tests:pde_solver_test
bazel test //tests:american_option_test
bazel test //tests:iv_solver_test

# Run with verbose output
bazel test //tests:pde_solver_test --test_output=all

# Run examples
bazel run //examples:example_newton_solver

# Clean
bazel clean
```

## Project Structure

```
mango-iv/
├── src/
│   ├── pde/
│   │   ├── core/          # Grid, boundary conditions, PDE solver, time domain
│   │   └── operators/     # Spatial operators (Laplacian, Black-Scholes, centered difference)
│   ├── option/            # American option pricing, IV solvers, price tables
│   ├── math/              # Root finding, cubic splines, Thomas solver, B-splines
│   └── support/           # Memory management (PMR arenas), CPU features, utilities
├── tests/                 # 38 test files with GoogleTest
├── examples/              # Example programs
├── benchmarks/            # Performance benchmarks
└── docs/                  # Architecture, math, API guides
```

**63 source files** organized into:
- **//src/pde/core** - Grid, PDESolver, TimeDomain, boundary conditions
- **//src/pde/operators** - BlackScholesPDE, LaplacianPDE, CenteredDifference
- **//src/option** - AmericanOptionSolver, IVSolverFDM, price tables
- **//src/math** - Root finding, B-splines, tridiagonal solvers
- **//src/support** - PMR arenas, error types, parallel utilities

## Development Workflow

### Pricing American Options

```cpp
#include "src/option/american_option.hpp"

// Define option parameters
mango::PricingParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .type = OptionType::PUT
};

// Auto-estimate grid (recommended)
auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params);
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

**See [docs/API_GUIDE.md](docs/API_GUIDE.md) for more examples**

### Adding Tests

All tests use GoogleTest:

```cpp
#include <gtest/gtest.h>
#include "src/option/american_option.hpp"

TEST(AmericanOptionTest, ATMPut) {
    mango::PricingParams params{...};
    // ...
}
```

Test naming: `*_test.cc` (unit), `*_integration_test.cc` (integration), `*_performance_test.cc` (performance)

### Regression Tests

**For every bug found, add a regression test.** This prevents the bug from reoccurring and documents the fix.

Regression test format:
```cpp
// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Brief description of what went wrong
// Bug: Explanation of the root cause
TEST(ComponentTest, DescriptiveNameForBug) {
    // Setup that triggers the bug
    // Assertion that would have caught it
}
```

Example from yield curve support:
```cpp
// Regression: make_rate_fn must convert time-to-expiry to calendar time
// Bug: Used curve.rate(τ) directly instead of curve.rate(T - τ)
TEST(RateSpecTest, TimeConversionForUpslopingCurve) {
    // Upward sloping curve: rates increase with time
    std::vector<mango::TenorPoint> points = {...};
    auto fn = mango::make_rate_fn(spec, maturity);

    // Near expiry (τ small) → calendar time large → HIGH rate
    EXPECT_NEAR(fn(0.1), 0.05, 1e-10);
    // Far from expiry (τ large) → calendar time small → LOW rate
    EXPECT_NEAR(fn(1.9), 0.04, 1e-10);
}
```

**Why regression tests matter:**
- Documents the bug for future developers
- Prevents reintroduction during refactoring
- Serves as executable specification of correct behavior

### Common Development Patterns

**Pattern 1: American IV Calculation**
```cpp
#include "src/option/iv_solver_fdm.hpp"

mango::IVQuery query{.option = spec, .market_price = 10.45};
mango::IVSolverFDM solver(config);
auto result = solver.solve_impl(query);
```

**Pattern 2: Price Table Pre-computation**
```cpp
#include "src/option/price_table_builder.hpp"
#include "src/option/price_table_surface.hpp"

auto grid_spec = mango::GridSpec<double>::uniform(-3.0, 3.0, 101).value();
auto [builder, axes] = mango::PriceTableBuilder<4>::from_vectors(
    moneyness_grid, maturity_grid, vol_grid, rate_grid, K_ref,
    grid_spec, 1000, mango::OptionType::PUT).value();
auto result = builder.build(axes);
double price = result->surface->value({m, tau, sigma, r});  // ~500ns
```

**See [docs/API_GUIDE.md](docs/API_GUIDE.md) for complete patterns**

## Git Workflow

### Commit Message Guidelines

Follow the seven rules:

1. Separate subject from body with blank line
2. Limit subject to 50 characters
3. Capitalize subject
4. No period at end of subject
5. **Use imperative mood** ("Add feature" not "Added feature")
6. Wrap body at 72 characters
7. **Explain what and why, not how**

Example:
```
Add cubic spline interpolation for off-grid queries

Users need to evaluate PDE solutions at arbitrary points.
Natural cubic splines provide smooth C² interpolation.

Shares tridiagonal solver with TR-BDF2 to avoid duplication.
```

### Pull Request Workflow

**IMPORTANT: Always start new work on a fresh branch**

**Pre-PR Checklist:**
Before creating a pull request, verify:
- [ ] All tests pass: `bazel test //...`
- [ ] All examples compile: `bazel build //examples/...`
- [ ] All benchmarks compile: `bazel build //benchmarks/...`
- [ ] Code builds without warnings
- [ ] Documentation updated if API changed

```bash
# 1. Update main
git checkout main
git pull

# 2. Create feature branch
git checkout -b feature/descriptive-name

# 3. Make changes and commit
# ... edit files ...

# 4. Verify builds before committing
bazel test //...
bazel build //examples/...
bazel build //benchmarks/...

git add <files>
git commit -m "Imperative mood message"

# 5. Push and create PR
git push -u origin feature/descriptive-name
gh pr create --title "Brief description" --body "$(cat <<'EOF'
## Summary
Brief description

## Changes
- Key change 1
- Key change 2

## Testing
- Test results: X/Y passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"

# 5. Merge when tests pass
gh pr merge --squash --delete-branch
git checkout main
git pull
```

**Branch naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `test/` - Adding tests
- `docs/` - Documentation

**Never continue work on existing PR branches**

## USDT Tracing

**Library code MUST NOT use printf/fprintf**

Use USDT probes for all logging and debugging:

```bash
# Monitor execution
sudo ./scripts/mango-trace monitor ./my_program

# Watch convergence
sudo ./scripts/mango-trace monitor ./my_program --preset=convergence

# Debug failures
sudo ./scripts/mango-trace monitor ./my_program --preset=debug
```

**Available presets:** `all`, `convergence`, `debug`, `performance`, `pde`, `iv`

**See [docs/TRACING.md](docs/TRACING.md) and [docs/TRACING_QUICKSTART.md](docs/TRACING_QUICKSTART.md) for details**

## Key C++23 Features

- `std::expected<T, E>` - Type-safe error handling (no exceptions)
- `std::span` - Safe array views
- Concepts (`HasAnalyticalJacobian`, `HasObstacle`)
- `[[gnu::target_clones]]` - Multi-ISA code generation (SSE2/AVX2/AVX-512)
- PMR (`std::pmr::vector`) - Polymorphic memory resources
- CRTP - Compile-time polymorphism for PDESolver
- `std::mdspan` - Multi-dimensional array views (Kokkos reference impl)
- Designated initializers - Struct initialization

## Documentation Structure

- **CLAUDE.md** (this file) - Workflow and project overview
- **docs/ARCHITECTURE.md** - Software architecture, design patterns, performance
- **docs/MATHEMATICAL_FOUNDATIONS.md** - PDE formulations, numerical methods
- **docs/API_GUIDE.md** - Usage examples and patterns
- **docs/TRACING.md** - USDT probe documentation
- **docs/TRACING_QUICKSTART.md** - 5-minute tracing guide

## Quick Reference

| Task | Command |
|------|---------|
| Build all | `bazel build //...` |
| Run all tests | `bazel test //...` |
| Run single test | `bazel test //tests:pde_solver_test` |
| Run example | `bazel run //examples:example_newton_solver` |
| Trace execution | `sudo ./scripts/mango-trace monitor ./program` |
| Create PR | `gh pr create --title "..." --body "..."` |

## Getting Help

- Build errors: Check Bazel logs
- Test failures: Run with `--test_output=all`
- Performance issues: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API usage: See [docs/API_GUIDE.md](docs/API_GUIDE.md)
- Math questions: See [docs/MATHEMATICAL_FOUNDATIONS.md](docs/MATHEMATICAL_FOUNDATIONS.md)
