# Legacy C Implementation

This folder contains the original C23 implementation of the mango-iv PDE solver and option pricing library. The code has been moved here as the project transitions to the C++20 implementation in `src/cpp/`.

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

The active C++20 implementation provides equivalent (and often enhanced) functionality:

| Legacy C | C++20 Replacement |
|----------|-------------------|
| `src/pde_solver.c` | `src/cpp/pde_solver.hpp` |
| `src/american_option.c` | `src/cpp/american_option.hpp` |
| `src/implied_volatility.c` | `src/cpp/iv_solver.hpp` |
| `src/cubic_spline.c` | `src/cpp/cubic_spline.hpp` (via tridiagonal_solver) |
| `src/price_table.c` | C++20 price table (in development) |
| `src/interp_cubic.c` | C++20 interpolation (in development) |

For new development, please use the C++20 API in `src/cpp/`.

## Building (Not Recommended)

This code is **not** integrated into the Bazel build system. To compile manually:

```bash
# Example: Compile a single example
gcc -std=c23 -O3 -fopenmp-simd \
    legacy/examples/example_heat_equation.c \
    legacy/src/pde_solver.c \
    legacy/src/cubic_spline.c \
    -Ilegacy/src -lm -o heat_equation

# Note: Dependencies must be resolved manually
```

## Documentation

Original documentation may reference this C code. The primary documentation has been updated for the C++20 implementation. See:
- `CLAUDE.md` - Updated project guide
- `docs/ARCHITECTURE.md` - Architectural overview
- C++20 API documentation in header files
