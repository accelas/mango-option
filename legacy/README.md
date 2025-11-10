# Legacy C Implementation

This folder contains the original C23 implementation of the mango-iv PDE solver and option pricing library. The code has been moved here as the project transitions to the C++23 implementation in `src/cpp/`.

## Contents

- **src/**: Original C source files and headers
  - PDE solver core (`pde_solver.c/h`, `cubic_spline.c/h`)
  - American option pricing (`american_option.c/h`)
  - Implied volatility solver (`implied_volatility.c/h`, `brent.h`)
  - Price table and interpolation (`price_table.c/h`, `interp_cubic.c/h`)
  - Grid utilities (`grid_generation.c/h`, `grid_presets.c/h`)
  - Supporting libraries (`lets_be_rational.c/h`, `validation.c/h`)

- **tests/**: Test files for C implementation (20 files)
  - PDE solver tests
  - Interpolation tests
  - Price table tests
  - IV solver tests

- **examples/**: Example programs demonstrating C API (8 files)
  - Heat equation solver
  - American option pricing (with and without dividends)
  - Implied volatility calculation
  - Price table precomputation
  - Interpolation engine usage

## Status

**This code is no longer actively maintained or tested in CI.** It remains available for:
- Historical reference
- Comparison with C++ implementation
- Understanding the original design decisions

## Migration

The active C++23 implementation provides equivalent (and often enhanced) functionality:

| Legacy C | C++23 Replacement |
|----------|-------------------|
| `src/pde_solver.c` | `src/cpp/pde_solver.hpp` |
| `src/american_option.c` | `src/cpp/american_option.hpp` |
| `src/implied_volatility.c` | `src/cpp/iv_solver.hpp` |
| `src/cubic_spline.c` | `src/cpp/cubic_spline.hpp` (via tridiagonal_solver) |
| `src/price_table.c` | C++23 price table (in development) |
| `src/interp_cubic.c` | C++23 interpolation (in development) |

For new development, please use the C++23 API in `src/cpp/`.

## Building

This code has Bazel build files with all targets marked as `manual`, meaning:
- ✅ Can be built explicitly when needed
- ❌ Won't be built automatically in CI
- ❌ Won't be included in `bazel build //...` or `bazel test //...`

### Building Libraries

```bash
# Build a specific library
bazel build //legacy/src:pde_solver
bazel build //legacy/src:american_option
bazel build //legacy/src:price_table

# Build all legacy libraries (explicit wildcard)
bazel build //legacy/src:all
```

### Running Tests

```bash
# Run a specific test
bazel test //legacy/tests:cubic_spline_test --test_output=all
bazel test //legacy/tests:price_table_test --test_output=all

# Run all legacy tests (explicit wildcard)
bazel test //legacy/tests:all --test_output=all

# Note: Some tests are tagged as "slow"
bazel test //legacy/tests:implied_volatility_test --test_output=all
```

### Running Examples

```bash
# Build and run an example
bazel run //legacy/examples:example_heat_equation
bazel run //legacy/examples:example_american_option
bazel run //legacy/examples:example_precompute_table

# Build all legacy examples
bazel build //legacy/examples:all
```

### Why `manual` tag?

The `manual` tag ensures these targets:
- Don't run in CI (reducing CI time and cost)
- Don't break `bazel build //...` if they have issues
- Can still be built/tested explicitly for reference or debugging
- Preserve the complete build configuration for future reference

## Documentation

Original documentation may reference this C code. The primary documentation has been updated for the C++23 implementation. See:
- `CLAUDE.md` - Updated project guide
- `docs/ARCHITECTURE.md` - Architectural overview
- C++23 API documentation in header files
