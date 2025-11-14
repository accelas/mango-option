# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mango-iv** is a C23-based PDE (Partial Differential Equation) solver using the finite difference method with TR-BDF2 (Two-stage Runge-Kutta with backward differentiation formula) time-stepping scheme. The project emphasizes flexibility through callback-based architecture, allowing users to define custom initial conditions, boundary conditions, jump conditions, and obstacle constraints.

## Build System

This project uses Bazel with Bzlmod for dependency management.

### Common Commands

```bash
# Build everything
bazel build //...

# Build specific targets (modern C++ implementation)
bazel build //src/option:american_option
bazel build //src/pde/core:pde_solver
bazel build //examples:example_newton_solver

# Build benchmarks
bazel build //benchmarks:market_iv_e2e_benchmark
bazel build //benchmarks:component_performance

# Run all tests
bazel test //...

# Run specific test suites
bazel test //tests:pde_solver_test
bazel test //tests:american_option_solver_test
bazel test //tests:iv_solver_test
bazel test //tests:bspline_4d_test
bazel test //tests:price_table_workspace_test

# Run tests with verbose output
bazel test //tests:iv_solver_test --test_output=all

# Run benchmarks
bazel run //benchmarks:market_iv_e2e_benchmark
bazel run //benchmarks:component_performance

# Clean build artifacts
bazel clean
```

## Project Structure

```
mango-iv/
├── MODULE.bazel           # Bazel module with dependencies (GoogleTest, Benchmark, Apache Arrow via Conan)
├── conanfile.txt          # Conan package manager config for Apache Arrow (persistence)
├── src/
│   ├── BUILD.bazel        # Top-level build configuration
│   ├── pde/               # PDE solver components
│   │   ├── core/          # Core solver (PDESolver, Newton, root-finding, grids)
│   │   ├── memory/        # Memory management (workspaces, allocators)
│   │   └── operators/     # Spatial operators (centered difference, Black-Scholes)
│   ├── option/            # Option pricing and IV solving
│   │   ├── american_option.{hpp,cpp}      # American option solver
│   │   ├── iv_solver_base.hpp             # Base class for IV solvers (C++23 deducing this)
│   │   ├── iv_solver_fdm.{hpp,cpp}        # FDM-based IV solver
│   │   ├── iv_solver_interpolated.{hpp,cpp} # B-spline IV solver
│   │   ├── option_spec.{hpp,cpp}          # Unified option specification API
│   │   ├── price_table_4d_builder.{hpp,cpp} # Price table builder with structured types
│   │   └── price_table_workspace.{hpp,cpp}  # Arrow IPC persistence with CRC64 checksums
│   ├── interpolation/     # B-spline interpolation
│   │   ├── bspline_4d.hpp           # 4D B-spline evaluator
│   │   ├── bspline_fitter_4d.hpp    # 4D B-spline fitting (consolidated)
│   │   └── cubic_spline_solver.hpp  # 1D cubic spline solver
│   └── support/           # Support utilities
│       ├── expected.hpp   # Expected/error handling (tl::expected)
│       ├── error_types.hpp # Error type definitions
│       └── cpu/           # CPU feature detection
├── benchmarks/
│   ├── BUILD.bazel
│   ├── component_performance.cc      # Component benchmarks
│   └── market_iv_e2e_benchmark.cc    # End-to-end IV benchmark
├── examples/
│   ├── BUILD.bazel
│   └── example_newton_solver.cc      # Demonstrates Newton solver
├── legacy/                # Legacy C implementation (archived)
│   ├── src/               # Original C23 implementation
│   ├── examples/          # C examples
│   └── tests/             # C tests
└── tests/
    ├── BUILD.bazel
    ├── pde_solver_test.cc          # Core PDE solver tests
    ├── american_option_solver_test.cc # American option tests
    ├── iv_solver_test.cc           # Unified IV solver tests
    ├── bspline_4d_test.cc          # B-spline interpolation tests
    └── price_table_workspace_test.cc # Arrow IPC persistence tests
```

## Core Architecture

### Callback-Based Design (Vectorized)

The solver uses a vectorized callback architecture for maximum efficiency and flexibility. All callbacks operate on entire arrays to minimize function call overhead and enable SIMD vectorization.

1. **Initial Condition**: `void (*)(const double *x, size_t n, double *u0, void *user_data)`
   - Computes u(x, t=0) for all grid points
   - Vectorized: processes entire array in one call

2. **Boundary Conditions**: `double (*)(double t, void *user_data)` (scalar)
   - Returns boundary value at time t
   - Supports Dirichlet (u=g), Neumann (∂u/∂x=g), Robin (a·u + b·∂u/∂x=g)
   - Separate callbacks for left and right boundaries

3. **Spatial Operator**: `void (*)(const double *x, double t, const double *u, size_t n, double *Lu, void *user_data)`
   - Computes L(u) for PDE ∂u/∂t = L(u)
   - Vectorized: returns Lu for all grid points
   - User implements finite difference stencils (e.g., ∂²u/∂x²)

4. **Diffusion Coefficient** (scalar field): `double diffusion_coeff`
   - Explicit diffusion coefficient D for pure diffusion operators L(u) = D·∂²u/∂x²
   - **Required for Neumann boundary conditions** to ensure accurate ghost point method
   - Set to `NAN` if diffusion is variable/spatially-varying (falls back to estimation)
   - For Black-Scholes: D = σ²/2 (half the variance)
   - Improves numerical stability and removes estimation errors

5. **Jump Condition** (optional): `bool (*)(double x, double *jump_value, void *user_data)`
   - Handles discontinuous coefficients at interfaces
   - Scalar callback for interface location queries

6. **Obstacle Condition** (optional): `void (*)(const double *x, double t, size_t n, double *ψ, void *user_data)`
   - Computes obstacle ψ(x,t) for all grid points
   - Enforces u(x,t) ≥ ψ(x,t) for variational inequalities
   - Vectorized for efficiency

### TR-BDF2 Time Stepping

The solver implements a composite two-stage scheme:
- **Stage 1**: Trapezoidal rule from t_n to t_n + γ·dt (γ ≈ 0.5858)
- **Stage 2**: BDF2 from t_n to t_n+1

This scheme provides:
- L-stability for stiff problems
- Second-order accuracy
- Good damping properties for high-frequency errors

### Cache Blocking (Removed)

**Note:** Cache blocking was previously attempted but has been removed because the implementation was ineffective.

**Why it was removed:**
- The blocked path still passed full array spans to the stencil operators
- Stencil accesses `u[i-1]`, `u[i]`, `u[i+1]` which can span multiple cache blocks
- Added loop/branch overhead without any cache locality benefit
- On large grids, this resulted in slower performance than single-pass evaluation

**What would be needed for true cache blocking:**
- Materialize block-local buffers with halo zones
- Copy block data from global arrays into cache-friendly buffers
- Run stencil on local buffers
- Copy results back to global arrays
- This adds complexity and copy overhead that may not pay off on modern CPUs with large L2/L3 caches

**Current implementation:**
- All grid sizes use single-pass evaluation (direct call to `spatial_op_.apply()`)
- All cache blocking infrastructure has been removed (parameter, methods, tests)
- `TRBDF2Config` no longer has a `cache_blocking_threshold` field

**For developers:**
If cache blocking becomes important in the future, it will need to be re-implemented from scratch with:
1. Profiling to confirm memory bandwidth is the bottleneck (not computation or convergence)
2. True blocking with local buffers and halo zones (not just index ranges)
3. Careful benchmarking to ensure speedup justifies added complexity

### Implicit Solver

Uses fixed-point iteration with under-relaxation (ω = 0.7) to solve implicit systems. Convergence criteria use relative error with default tolerance of 1e-6.

### Memory Management

**Ownership Transfer:**
- `pde_solver_create()` takes ownership of the grid
- After creation, `grid.x` is set to `nullptr` to prevent double-free
- Grid is freed when `pde_solver_destroy()` is called
- `pde_free_grid()` is only needed if grid is created but never passed to a solver

**Single Workspace Buffer:**
- All solver arrays allocated from one contiguous buffer (10n doubles)
- 64-byte alignment for AVX-512 SIMD vectorization
- Better cache locality with sequential memory layout
- Zero malloc overhead during time stepping
- Arrays: u_current, u_next, u_stage, rhs, matrix_{diag,upper,lower}, u_old, Lu, u_temp

### SIMD Vectorization

OpenMP SIMD pragmas on hot loops for automatic vectorization:
```c
#pragma omp simd
for (size_t i = 1; i < n - 1; i++) {
    Lu[i] = D * (u[i-1] - 2.0*u[i] + u[i+1]) / (dx*dx);
}
```

Enables compiler to generate AVX2/AVX-512 instructions for parallel computation.

### CenteredDifference: Automatic ISA Selection

The `CenteredDifference` stencil operator automatically selects the optimal backend based on CPU capabilities:

**Mode Enum:**
- **Mode::Auto** (default): Runtime CPU detection + OS XSAVE check chooses Scalar or SIMD
- **Mode::Scalar**: Force scalar backend (for testing/debugging)
- **Mode::Simd**: Force SIMD backend (for testing/benchmarking)

**Production Usage:**
```cpp
// Always use Mode::Auto in production code
auto spacing = GridSpacing<double>(grid);
auto stencil = CenteredDifference(spacing);  // Auto-selects optimal backend
```

**Test Usage:**
```cpp
// Tests can force specific backends for regression testing
auto scalar = CenteredDifference(spacing, CenteredDifference::Mode::Scalar);
auto simd = CenteredDifference(spacing, CenteredDifference::Mode::Simd);

// Compare results
scalar.compute_second_derivative(u, d2u_scalar, 1, n-1);
simd.compute_second_derivative(u, d2u_simd, 1, n-1);
EXPECT_NEAR(d2u_scalar[i], d2u_simd[i], 1e-14);  // Allow FP rounding
```

**Performance Characteristics:**
- Virtual dispatch overhead: ~5-10ns per call
- Negligible vs computation cost (~5,000ns for 100-point grid)
- Both backends use precomputed arrays on non-uniform grids
- SIMD backend: 3-6x speedup via explicit vectorization

**Architecture:**
- Façade + Backend pattern (similar to strategy pattern)
- ScalarBackend: `#pragma omp simd` for compiler auto-vectorization
- SimdBackend: `std::experimental::simd` + `[[gnu::target_clones]]` for multi-ISA

### Cubic Spline Interpolation

Natural cubic splines allow evaluation of solutions at arbitrary off-grid points:
- Single workspace buffer for all coefficient arrays (4n doubles)
- Single temporary buffer during construction (6n doubles)
- Uses shared tridiagonal solver for efficiency
- Provides both function and derivative evaluation
- Convenience function `pde_solver_interpolate()` for quick queries

## Development Workflow

### Adding New PDE Problems

1. Define callback functions matching the API signatures
2. Create user_data struct to pass problem-specific parameters
3. Set up spatial grid and time domain
4. Configure boundary conditions and TR-BDF2 parameters
5. Create solver, initialize, and solve

**Typical usage pattern:**
```c
// Create grid (will transfer ownership to solver)
SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);

// Setup time domain and callbacks
TimeDomain time = {/* ... */};
PDECallbacks callbacks = {/* ... */};

// Create solver (takes ownership of grid)
PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                      &trbdf2_config, &callbacks);

pde_solver_initialize(solver);
pde_solver_solve(solver);

// Cleanup (frees both solver and grid)
pde_solver_destroy(solver);
// Note: No pde_free_grid() needed - ownership was transferred

// For multiple solvers, create a new grid each time
grid = pde_create_grid(0.0, 1.0, 101);
solver = pde_solver_create(&grid, ...);  // Takes ownership again
```

See `examples/example_heat_equation.c` for complete examples including:
- Basic heat equation
- Jump conditions (discontinuous diffusion coefficients)
- Obstacle conditions (American option pricing)

### Implementing Spatial Operators (Vectorized)

For second-order spatial derivatives (e.g., diffusion, ∂²u/∂x²):
```c
// Vectorized heat equation: L(u) = D·∂²u/∂x²
void heat_operator(const double *x, double t, const double *u,
                   size_t n, double *Lu, void *user_data) {
    const double dx = x[1] - x[0];
    const double D = *(double*)user_data;
    const double dx2_inv = 1.0 / (dx * dx);

    Lu[0] = Lu[n-1] = 0.0;  // Boundaries

    #pragma omp simd
    for (size_t i = 1; i < n - 1; i++) {
        Lu[i] = D * (u[i-1] - 2.0*u[i] + u[i+1]) * dx2_inv;
    }
}
```

For first-order derivatives (e.g., advection, ∂u/∂x):
```c
// Vectorized advection: L(u) = -v·∂u/∂x (upwind for v > 0)
#pragma omp simd
for (size_t i = 1; i < n - 1; i++) {
    Lu[i] = -velocity * (u[i] - u[i-1]) / dx;
}
```

### Adding Tests

- PDE solver tests: `tests/pde_solver_test.cc`
- Spline tests: `tests/cubic_spline_test.cc`
- Stability tests: `tests/stability_test.cc`

All tests use GoogleTest framework with C++ wrappers around C API.

## Key Implementation Details

### Tridiagonal Solver

A shared Thomas algorithm implementation is used by both:
- TR-BDF2 implicit time stepping
- Cubic spline coefficient calculation

### Boundary Condition Application

Applied after each iteration to ensure constraints are satisfied. Order matters:
1. Update interior points
2. Apply boundary conditions
3. Apply obstacle conditions (if present)

### Convergence Issues

If solver fails to converge:
1. Reduce time step (dt)
2. Increase max_iter using `solver.set_trbdf2_config()` (see Advanced Configuration below)
3. Relax tolerance
4. Check spatial operator implementation for errors
5. Verify boundary conditions are consistent

### American Option API Simplification

The `AmericanOptionSolver` constructor has been simplified to hide internal solver configuration details. Most users should never need to modify TR-BDF2 or Newton solver parameters.

**Basic Usage (most common):**
```cpp
#include "src/option/american_option.hpp"

AmericanOptionParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .option_type = OptionType::PUT
};

AmericanOptionGrid grid{
    .n_space = 101,
    .n_time = 1000,
    .x_min = -3.0,
    .x_max = 3.0
};

// Simple construction with defaults
AmericanOptionSolver solver(params, grid);
auto result = solver.solve();
```

**Advanced Configuration (rarely needed):**
If you need to tune internal solver parameters for convergence or accuracy, use setter methods:
```cpp
AmericanOptionSolver solver(params, grid);

// Tune TR-BDF2 time-stepping (for stiff problems)
TRBDF2Config trbdf2_config{
    .max_iter = 50,        // Increase iterations for convergence
    .tolerance = 1e-8      // Higher accuracy
};
solver.set_trbdf2_config(trbdf2_config);

// Tune Newton solver (for early exercise boundary)
RootFindingConfig root_config{
    .max_iter = 200,
    .tolerance = 1e-8
};
solver.set_root_config(root_config);

auto result = solver.solve();
```

**When to use advanced configuration:**
- Solver fails to converge with default settings
- Need higher accuracy than default 1e-6 tolerance
- Debugging numerical issues
- Research/benchmarking (comparing different solver configurations)

**Default values (sensible for most applications):**
- TR-BDF2: 20 iterations, 1e-6 tolerance, gamma = 2 - √2
- Newton: 100 iterations, 1e-6 tolerance, FD epsilon = 1e-7

### Memory Management

All structures use explicit create/destroy patterns:
- `pde_solver_create()` / `pde_solver_destroy()`
- `pde_spline_create()` / `pde_spline_destroy()`
- `pde_create_grid()` / `pde_free_grid()`

## Unified Root-Finding API

The library provides a unified configuration and result interface for all root-finding methods.

### Configuration

```cpp
#include "src/pde/core/root_finding.hpp"

mango::RootFindingConfig config{
    .max_iter = 100,
    .tolerance = 1e-6,
    .jacobian_fd_epsilon = 1e-7,  // Newton-specific
    .brent_tol_abs = 1e-6          // Brent-specific
};
```

### Newton-Raphson Solver

Integrated into PDESolver for implicit time-stepping:

```cpp
mango::PDESolver solver(grid, time, trbdf2_config, root_config,
                       left_bc, right_bc, spatial_op);

solver.initialize(initial_condition);
bool converged = solver.solve();  // Uses Newton for each stage
```

**Memory efficiency:**
- NewtonWorkspace allocates 8n doubles (Jacobian, residual, delta, workspace)
- Borrows 2n doubles from WorkspaceStorage (u_stage, rhs as scratch)
- Total: 13n doubles for entire solver (vs. 15n before)

**Design:**
- Persistent solver instance (created once, reused)
- Quasi-Newton: Jacobian built once per stage
- Compile-time BC dispatch (Dirichlet, Neumann)
- Zero allocation during solve() after construction

### Workspace Management

NewtonWorkspace implements hybrid allocation:
- Owns: Jacobian matrices, residual, delta_u, u_old, tridiag_workspace
- Borrows: Lu (read-only), u_perturb (from u_stage), Lu_perturb (from rhs)

Safe borrowing: u_stage and rhs are unused during Newton iteration.

## Implied Volatility Solver

The library provides **two** implied volatility solvers with a unified API:

1. **IVSolverFDM**: Ground truth solver using Brent's method + nested PDE (~143ms/query)
2. **IVSolverInterpolated**: Fast solver using B-spline price tables (~30µs/query, 4800x speedup)

Both solvers implement a unified interface via C++23 `deducing this` for zero-overhead static polymorphism, supporting single queries and OpenMP-parallelized batch processing.

### Overview

**Unified API:** Both solvers accept `OptionSpec` + `market_price` via `IVQuery` struct
**Performance:** FDM ~143ms, Interpolation ~30µs (4800x speedup)
**Batch Support:** OpenMP-parallelized `solve_batch()` on both solvers
**Thread Safety:** Guaranteed via thread-local solvers (FDM) or lock-free reads (interpolation)
**Status:** Production-ready

### Unified API - Basic Usage

Both solvers share the same `IVQuery` input format:

```cpp
#include "src/option/iv_solver_fdm.hpp"  // or iv_solver_interpolated.hpp
#include "src/option/option_spec.hpp"

// Define option contract specification
mango::OptionSpec spec{
    .spot = 100.0,
    .strike = 100.0,
    .maturity = 1.0,
    .rate = 0.05,
    .dividend_yield = 0.02,
    .type = mango::OptionType::PUT
};

// Query: option + market price
mango::IVQuery query{.option = spec, .market_price = 10.45};

// Solve using FDM (ground truth)
mango::IVSolverFDM fdm_solver(mango::IVSolverFDMConfig{});
mango::IVResult result = fdm_solver.solve(query);

if (result.converged) {
    std::cout << "Implied Volatility: " << result.implied_vol << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
} else {
    std::cerr << "Failed: " << *result.failure_reason << "\n";
}
```

### FDM Solver (IVSolverFDM)

Ground truth solver using Brent's method with nested American option PDE evaluation.

**When to use:**
- High-accuracy IV calculation (ground truth)
- Validating interpolation-based results
- One-off calculations where speed is not critical
- Building pre-computed price tables

**Configuration:**
```cpp
mango::IVSolverFDMConfig config{
    .root_config = mango::RootFindingConfig{
        .max_iter = 100,
        .tolerance = 1e-6
    },
    .grid_n_space = 101,
    .grid_n_time = 1000,
    .grid_s_max = 200.0
};

mango::IVSolverFDM solver(config);
```

**Performance:**
- Single query: ~143ms
- Batch (32 cores): ~107 IVs/sec (15.3x speedup via OpenMP)

### Interpolation Solver (IVSolverInterpolated)

Fast solver using pre-computed 4D B-spline price tables.

**When to use:**
- Real-time market making (thousands of IV queries)
- Live risk calculations
- Intraday volatility surface updates
- Applications requiring sub-millisecond latency

**Setup:**
```cpp
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/price_table_4d_builder.hpp"

// 1. Build price table (one-time, offline)
auto builder = mango::PriceTable4DBuilder::create({m, tau, sigma, r}, K_ref);
auto build_result = builder->precompute(config);

// 2. Create IV solver from table
auto iv_solver = mango::IVSolverInterpolated::create(build_result->surface);

// 3. Query IV (fast!)
mango::IVResult result = iv_solver->solve(query);  // ~30µs
```

**Performance:**
- Single query: ~30µs (4800x faster than FDM)
- Batch: Trivially parallelizable (lock-free reads)
- Memory: ~2.4 MB for 50×30×20×10 grid

### Batch Processing (Both Solvers)

Both solvers support OpenMP-parallelized batch processing via `solve_batch()`:

```cpp
// Prepare batch queries
std::vector<mango::IVQuery> queries = load_market_data();
std::vector<mango::IVResult> results(queries.size());

// FDM batch (uses thread-local solvers for safety)
mango::IVSolverFDM fdm_solver(config);
auto status = fdm_solver.solve_batch(queries, results);

// Interpolation batch (lock-free reads, trivially thread-safe)
auto interp_solver = mango::IVSolverInterpolated::create(surface);
status = interp_solver->solve_batch(queries, results);

// Process results
for (const auto& result : results) {
    if (result.converged) {
        std::cout << "σ = " << result.implied_vol << "\n";
    }
}
```

### Input Validation

Both solvers use centralized validation via `validate_iv_query()` in `option_spec.cpp`:

**Validation checks:**
- **Option spec validation:**
  - Spot price > 0 and finite
  - Strike price > 0 and finite
  - Time to maturity > 0 and finite
  - Rate and dividend yield are finite
- **Market price validation:**
  - Market price > 0 and finite
- **Arbitrage checks:**
  - Call price ≤ spot price (no arbitrage)
  - Put price ≤ strike price (no arbitrage)
  - Market price ≥ intrinsic value (no arbitrage)

All validation errors return descriptive error messages via `expected<void, std::string>`.

### Adaptive Volatility Bounds (FDM Solver)

The FDM solver uses intelligent bounds based on intrinsic value analysis:

| Moneyness | Time Value | Upper Bound | Rationale |
|-----------|-----------|-------------|-----------|
| ATM/OTM | High (>50%) | 300% | High time value suggests high vol |
| Moderate | Medium (20-50%) | 200% | Moderate time value |
| Deep ITM | Low (<20%) | 150% | Low time value, unlikely high vol |
| All | - | 1% (lower) | Minimum realistic volatility |

This adaptive approach reduces Brent iterations compared to arbitrary bounds.

### Performance Characteristics

**FDM Solver:**

| Scenario | Iterations | Time | Notes |
|----------|-----------|------|-------|
| Single query | 10-15 | ~143ms | Ground truth accuracy |
| Batch (32 cores) | - | ~107 IVs/sec | 15.3x speedup via OpenMP |

**Interpolation Solver:**

| Scenario | Time | Notes |
|----------|------|-------|
| Single query | ~30µs | 4800x faster than FDM |
| Batch | ~30µs per query | Trivially parallelizable (lock-free) |
| Table build | 15-20 min | One-time offline cost for 300K grid points |

**Speedup strategy:**
- Use **FDM** for ground truth validation and building price tables
- Use **Interpolation** for production queries requiring sub-millisecond latency
- Interpolation provides ~4800x speedup with minimal accuracy loss (<0.1% typical error)

### USDT Tracing

Monitor IV calculations with USDT probes:

```bash
# Watch IV calculations in real-time
sudo bpftrace -e '
usdt::mango:algo_start /arg0 == 3/ {
    printf("IV calc: S=%.2f K=%.2f T=%.2f Price=%.4f\n",
           arg1, arg2, arg3, arg4);
}
usdt::mango:algo_complete /arg0 == 3/ {
    printf("  Result: σ=%.4f (%d iters)\n", arg1, arg2);
}
usdt::mango:convergence_failed /arg0 == 3/ {
    printf("  FAILED at iter %d\n", arg1);
}' -c './my_program'

# Use predefined scripts
sudo ./scripts/mango-trace monitor ./my_program --preset=convergence
```

**Available MODULE_IMPLIED_VOL probes:**
- `algo_start`: IV calculation begins (spot, strike, maturity, price)
- `algo_complete`: IV calculation completes (implied_vol, iterations)
- `validation_error`: Input validation failures (error_code, param_value)
- `convergence_failed`: Non-convergence diagnostics (iterations, final_error)

### Error Handling

Both solvers return `IVResult` with convergence status and optional error messages:

```cpp
IVResult result = solver.solve(query);

if (!result.converged) {
    // Check failure reason
    if (result.failure_reason.has_value()) {
        std::cerr << "Error: " << *result.failure_reason << "\n";

        // Common failures:
        // - "Market price is not finite"
        // - "Market price 105.0 exceeds upper bound 100.0 (no arbitrage)"
        // - "Market price 5.0 < intrinsic value 10.0 (no arbitrage)"
        // - "Failed to converge after 100 iterations"
    }
}
```

**Batch error handling:**
```cpp
auto status = solver.solve_batch(queries, results);

if (!status) {
    std::cerr << "Batch error: " << status.error() << "\n";
    // Common: "Size mismatch: 100 queries but 50 result slots"
}

// Check individual results
for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].converged) {
        std::cerr << "Query " << i << " failed: "
                  << *results[i].failure_reason << "\n";
    }
}
```

### Related Documentation

- **Unified IV API:** `src/option/iv_solver_base.hpp` - Base class with deducing this
- **FDM Solver:** `src/option/iv_solver_fdm.{hpp,cpp}` - Brent + nested PDE
- **Interpolation Solver:** `src/option/iv_solver_interpolated.{hpp,cpp}` - B-spline IV
- **Option Spec:** `src/option/option_spec.{hpp,cpp}` - Unified input validation
- **Test Suite:** `tests/iv_solver_test.cc` - Comprehensive tests for both solvers

## USDT Tracing System

**CRITICAL: Library code must NEVER use printf/fprintf for debug output.**

This library uses USDT (User Statically-Defined Tracing) probes for all internal logging, debugging, and diagnostics. USDT provides zero-overhead tracing that can be dynamically enabled/disabled at runtime without recompiling.

### Why USDT Instead of printf?

1. **Zero overhead**: Probes compile to single NOP instructions when not actively traced
2. **Production-safe**: Can be left in production code without performance impact
3. **Dynamic control**: Enable/disable at runtime using bpftrace, systemtap, or perf
4. **Structured data**: Captures typed parameters, not formatted strings
5. **Library-appropriate**: Libraries should not pollute stdout/stderr

### When to Use USDT Probes

Use USDT probes for:
- Progress tracking (solver start/progress/complete)
- Convergence monitoring (iteration counts, errors)
- Error conditions (validation failures, convergence failures)
- Performance measurement (timing boundaries)
- Debugging information

**NEVER use printf, fprintf, or any console I/O in library code (`src/` directory).**

### Tracing Tool: bpftrace

This library uses **bpftrace** as the primary tracing tool. bpftrace is:
- Modern, built on eBPF
- Easy to use (awk-like syntax)
- No compilation needed for scripts
- Perfect for our use case (monitoring, debugging, performance)

### Available Probe Categories

See `src/ivcalc_trace.h` for complete probe definitions. The tracing system is designed to work across all modules:

1. **Algorithm Lifecycle** (General): `MANGO_TRACE_ALGO_START`, `MANGO_TRACE_ALGO_PROGRESS`, `MANGO_TRACE_ALGO_COMPLETE`
2. **Convergence Tracking** (General): `MANGO_TRACE_CONVERGENCE_ITER`, `MANGO_TRACE_CONVERGENCE_SUCCESS`, `MANGO_TRACE_CONVERGENCE_FAILED`
3. **Validation/Errors** (General): `MANGO_TRACE_VALIDATION_ERROR`, `MANGO_TRACE_RUNTIME_ERROR`
4. **PDE Solver**: `MANGO_TRACE_PDE_START`, `MANGO_TRACE_PDE_PROGRESS`, `MANGO_TRACE_PDE_COMPLETE`, etc.
5. **Implied Volatility**: `MANGO_TRACE_IV_START`, `MANGO_TRACE_IV_COMPLETE`, `MANGO_TRACE_IV_VALIDATION_ERROR`
6. **American Options**: `MANGO_TRACE_OPTION_START`, `MANGO_TRACE_OPTION_COMPLETE`
7. **Brent's Method**: `MANGO_TRACE_BRENT_START`, `MANGO_TRACE_BRENT_ITER`, `MANGO_TRACE_BRENT_COMPLETE`
8. **Cubic Spline**: `MANGO_TRACE_SPLINE_ERROR`

Each module has access to both general-purpose probes (for common patterns like convergence) and module-specific probes.

### Adding New USDT Probes

When adding new library functionality that needs logging:

1. **Define probe in `src/ivcalc_trace.h`**:
   ```c
   #define MANGO_TRACE_MY_EVENT(module_id, param1, param2) \
       DTRACE_PROBE3(MANGO_PROVIDER, my_event, module_id, param1, param2)
   ```

2. **Use probe in source code**:
   ```c
   #include "ivcalc_trace.h"

   void my_function() {
       // ... code ...
       MANGO_TRACE_MY_EVENT(MODULE_MY_MODULE, value1, value2);
       // ... more code ...
   }
   ```

3. **Update `TRACING.md`** with usage examples and parameter descriptions
4. **Add bpftrace script** in `scripts/tracing/` if the probe is commonly used

### Building with USDT

**USDT is enabled by default!** Just build normally:

```bash
bazel build //src:pde_solver
bazel build //examples:example_heat_equation
bazel build //examples:example_american_option
```

The library gracefully falls back to no-op probes if `systemtap-sdt-dev` is not installed. For best results, install it:

```bash
# Ubuntu/Debian
sudo apt-get install systemtap-sdt-dev

# Then rebuild
bazel clean
bazel build //...
```

### Using bpftrace to Monitor Execution

**Quick Start** - Use the ready-made scripts:

```bash
# Monitor all library activity
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation

# Watch convergence behavior
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=convergence

# Debug failures
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=debug

# Profile performance
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_heat_equation --preset=performance
```

**Available presets:** `all` (default), `convergence`, `debug`, `performance`, `pde`, `iv`

**Available scripts** in `scripts/tracing/`:
- `monitor_all.bt` - High-level overview of all activity
- `convergence_watch.bt` - Real-time convergence monitoring
- `debug_failures.bt` - Alert on errors and failures
- `performance_profile.bt` - Timing and performance analysis
- `pde_detailed.bt` - Deep dive into PDE solver
- `iv_detailed.bt` - Deep dive into IV calculations

**Direct bpftrace usage:**

```bash
# Use pre-made scripts
sudo bpftrace scripts/tracing/monitor_all.bt -c './bazel-bin/examples/example_heat_equation'

# Or write custom one-liners
sudo bpftrace -e 'usdt::mango:convergence_failed {
    printf("Module %d failed at step %d\n", arg0, arg1);
}' -c './my_program'
```

**Helper tool commands:**

```bash
# Check if binary has USDT support
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_heat_equation

# List all available probes
sudo ./scripts/mango-trace list ./bazel-bin/examples/example_heat_equation

# Run specific script
sudo ./scripts/mango-trace run convergence_watch.bt ./my_program
```

For complete documentation, see:
- [TRACING_QUICKSTART.md](TRACING_QUICKSTART.md) - 5-minute getting started guide
- [TRACING.md](TRACING.md) - Comprehensive tracing documentation
- [scripts/tracing/README.md](scripts/tracing/README.md) - Script reference

### Examples vs Library Code

**Examples** (`examples/` directory) may use printf for user-facing output - they are demonstration programs, not library code.

**Library code** (`src/` directory) must use USDT probes exclusively. No printf, fprintf, or stderr allowed.

## C23 Features Used

- `nullptr` keyword
- Modern type declarations
- Designated initializers for structs

## Common Patterns

### Setting Up a Simple Diffusion Problem

```c
SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);
TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = 0.001, .n_steps = 1000};
PDECallbacks callbacks = {
    .initial_condition = my_ic_func,
    .left_boundary = my_left_bc,
    .right_boundary = my_right_bc,
    .spatial_operator = my_spatial_op,
    .diffusion_coeff = 0.1,  // For constant diffusion; set to NAN if variable
    .user_data = &my_data
};
BoundaryConfig bc = pde_default_boundary_config();
TRBDF2Config trbdf2 = pde_default_trbdf2_config();

PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
pde_solver_initialize(solver);
pde_solver_solve(solver);

const double *solution = pde_solver_get_solution(solver);
// Use solution...

pde_solver_destroy(solver);
pde_free_grid(&grid);
```

### Querying Off-Grid Values

```c
double u_at_x = pde_solver_interpolate(solver, 0.123);
```

## Price Table Pre-computation Workflow

The price table module provides fast option pricing through pre-computed 4D B-spline lookup tables with **Apache Arrow IPC persistence** for zero-copy memory-mapped loading.

**Key Features:**
- **Arrow IPC Persistence:** Save/load tables in ~150-300µs using Apache Arrow Feather V2 format
- **Zero-Copy mmap:** Memory-mapped loading avoids deserialization overhead
- **CRC64 Checksums:** Data integrity validation (ECMA polynomial 0x42F0E1EBA9EA3693)
- **Structured API:** Ergonomic types (PriceTableGrid, PriceTableConfig, PriceTableSurface)
- **OpenMP Parallelization:** Fast batch precomputation across multiple cores
- **Sub-microsecond Queries:** ~500ns price interpolation, Greeks included

### Modern C++ API (Recommended)

**1. Build price table with structured types:**
```cpp
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_workspace.hpp"

// Define grid using structured type
mango::PriceTableGrid grid{
    .moneyness_grid = {0.8, 0.9, 1.0, 1.1, 1.2},     // std::vector<double>
    .maturity_grid = {0.027, 0.25, 0.5, 1.0, 2.0},
    .volatility_grid = {0.10, 0.20, 0.30, 0.40},
    .rate_grid = {0.0, 0.02, 0.05},
    .K_ref = 100.0  // Reference strike for moneyness calculation
};

// Or use convenience factory from strikes
auto builder = mango::PriceTable4DBuilder::from_strikes(
    100.0,  // spot
    {80, 90, 100, 110, 120},  // strikes (auto-converts to moneyness)
    {0.027, 0.25, 0.5, 1.0, 2.0},  // maturities
    {0.10, 0.20, 0.30, 0.40},      // volatilities
    {0.0, 0.02, 0.05}              // rates
);

// Alternative: create from grid struct
auto builder_alt = mango::PriceTable4DBuilder::create(grid);
```

**2. Pre-compute prices using structured config:**
```cpp
// Configure precomputation
mango::PriceTableConfig config{
    .option_type = mango::OptionType::PUT,
    .n_pde_space = 101,
    .n_pde_time = 1000,
    .dividend_yield = 0.02
};

// Precompute table (uses OpenMP parallelization)
auto result = builder->precompute(config);

if (!result) {
    std::cerr << "Precomputation failed: " << result.error() << "\n";
    return 1;
}

// Access surface for queries
const auto& surface = result->surface;
```

**3. Save table using Arrow IPC (Feather V2 format):**
```cpp
// Save to disk with CRC64 checksums
auto workspace = std::move(result->workspace);  // Transfer ownership
auto save_status = workspace->save("spx_put_table.arrow");

if (!save_status) {
    std::cerr << "Save failed: " << save_status.error() << "\n";
    return 1;
}

// File format: Apache Arrow IPC with:
// - Schema v1.0: 33 fields (grids, coefficients, metadata)
// - CRC64 checksums for data integrity
// - Zero-copy mmap ready
```

**4. Load table with zero-copy mmap (fast!):**
```cpp
// Load from disk (~150-300µs, no deserialization overhead)
auto workspace = mango::PriceTableWorkspace::load("spx_put_table.arrow");

if (!workspace) {
    std::cerr << "Load failed: " << workspace.error() << "\n";
    // Possible errors: FILE_NOT_FOUND, SCHEMA_MISMATCH, CORRUPTED_*, etc.
    return 1;
}

// Create surface from loaded workspace
auto surface = mango::PriceTableSurface(std::move(*workspace));
```

**5. Query prices and Greeks (sub-microsecond):**
```cpp
// Single queries (~500ns each)
double price = surface.interpolate(1.05, 0.25, 0.20, 0.05);  // moneyness, tau, sigma, r
double vega = surface.interpolate_vega(1.05, 0.25, 0.20, 0.05);   // ∂V/∂σ
double gamma = surface.interpolate_gamma(1.05, 0.25, 0.20, 0.05);  // ∂²V/∂S²

// Batch queries (typical production usage)
for (const auto& query : market_data) {
    double m = query.spot / query.strike;  // Compute moneyness
    double p = surface.interpolate(m, query.maturity, query.vol, query.rate);
    double v = surface.interpolate_vega(m, query.maturity, query.vol, query.rate);
    double g = surface.interpolate_gamma(m, query.maturity, query.vol, query.rate);

    // Use for risk calculations, market making, etc.
    process_greeks(p, v, g);
}
```

### Arrow IPC Persistence Details

**Save/Load Error Codes:**

The workspace implements comprehensive validation during load:

```cpp
enum class LoadError {
    FILE_NOT_FOUND,          // Arrow file doesn't exist
    ARROW_ERROR,             // Arrow library error
    SCHEMA_MISMATCH,         // Wrong schema version or fields
    INVALID_DIMENSIONS,      // Grid dimensions invalid (zero, negative)
    SIZE_MISMATCH,           // Array sizes don't match dimensions
    NON_MONOTONIC,           // Grids not strictly increasing
    ALIGNMENT_ERROR,         // Coefficient arrays not 64-byte aligned
    CORRUPTED_COEFFICIENTS,  // CRC64 checksum mismatch
    CORRUPTED_GRIDS          // CRC64 checksum mismatch
};
```

**Schema v1.0 Fields (33 total):**

| Category | Fields | Description |
|----------|--------|-------------|
| Metadata | mango_version, schema_version, timestamp | Version tracking |
| Grid Dimensions | n_m, n_tau, n_sigma, n_r, K_ref | 4D grid sizes + reference |
| Grid Arrays | moneyness, maturity, volatility, rate | Axis values (double[]) |
| B-spline Knots | knots_m, knots_tau, knots_sigma, knots_r | B-spline knot vectors |
| Coefficients | coefficients, vega_coeffs, gamma_coeffs | 4D tensor data (64-byte aligned) |
| Raw Prices | prices_raw (nullable) | Optional pre-fit price grid |
| Fitting Stats | max_residual_*, condition_number_max | B-spline fitting diagnostics |
| Build Info | n_pde_solves, precompute_time_seconds, pde_n_space, pde_n_time | Build metadata |
| Checksums | crc64_coefficients, crc64_grids | CRC64 data integrity |

**Memory Layout:**

```cpp
// PriceTableWorkspace uses single contiguous allocation (64-byte aligned)
// Example 50×30×20×10 grid:
size_t total_size =
    + 4 * sizeof(size_t)              // Dimensions (n_m, n_tau, n_sigma, n_r)
    + sizeof(double)                   // K_ref
    + (50 + 30 + 20 + 10) * sizeof(double)  // Grid arrays
    + (54 + 34 + 24 + 14) * sizeof(double)  // Knot vectors (n+4 each)
    + (50*30*20*10) * 3 * sizeof(double);   // Coefficients (price, vega, gamma)

// Total: ~2.4 MB for this grid
// All allocations from single arena for cache locality + mmap efficiency
```

### Greeks Computation

All Greeks are computed during precomputation using centered finite differences:

**Vega (∂V/∂σ):**
- Computed via centered FD on volatility axis
- Enables Newton-based IV inversion
- ~500ns query time (same as price)

**Gamma (∂²V/∂S²):**
- Computed via centered FD on moneyness axis
- Properly scaled from ∂²V/∂m² to ∂²V/∂S² using chain rule (γ = ∂²V/∂m² / K_ref²)
- Essential for delta-hedging and convexity analysis
- Accuracy depends on grid spacing (finer grids → better second derivatives)

**Accuracy Guidelines:**
- **Vega (first derivative):** <0.1% error typical with 20+ volatility points
- **Gamma (second derivative):** ~5-10% error with 20 moneyness points, <1% with 50+ points
- Use finer grids for high-accuracy second derivatives

All Greeks preserved during Arrow IPC save/load (coefficients + checksums).

### Performance Characteristics

**Pre-computation (one-time offline cost):**
- 300K grid points (50×30×20×10): ~15-20 minutes on 16 cores
- Throughput: ~300 options/second with OpenMP parallelization
- Memory overhead: ~10 KB per batch (configurable)
- Output: ~2.4 MB Arrow IPC file (with all Greeks + metadata)

**Persistence (Arrow IPC):**
- **Save:** ~1-2ms (includes CRC64 checksum computation)
- **Load:** ~150-300µs (zero-copy mmap, validation only)
- **Verification:** CRC64 checksums validate data integrity on load
- **Format:** Apache Arrow Feather V2 (IPC file format)

**Query performance (production):**
- **Price:** ~500 nanoseconds (B-spline evaluation)
- **Vega:** ~500 nanoseconds (pre-computed coefficients)
- **Gamma:** ~500 nanoseconds (pre-computed coefficients)
- **Speedup vs FDM:** ~286,000x (143ms FDM vs 500ns interpolation)

**Memory usage:**
- 4D table (50×30×20×10): ~2.4 MB resident (mmap shared across processes)
- Single contiguous allocation (64-byte aligned for AVX-512)
- Cache-friendly sequential access pattern

### Conan Dependency Setup

Apache Arrow is required for price table persistence:

```bash
# Install Conan 2.x (one-time)
pip install conan

# Install Arrow and dependencies (~15-30 min first time)
conan install . --output-folder=conan_deps --build=missing

# Build with Arrow support
bazel build --config=arrow //...
```

Arrow dependencies are cached in `~/.conan2/p` for fast subsequent builds.

### USDT Tracing

Monitor pre-computation progress with USDT probes:
```bash
# Watch progress during pre-computation
sudo bpftrace -e 'usdt::mango:algo_progress /arg0 == 4/ {
    printf("Price table: %d%% complete\n", arg2);
}' -c './my_precompute_program'
```

See `benchmarks/market_iv_e2e_benchmark.cc` for complete working examples.

## Numerical Considerations

- Spatial discretization determines maximum stable dt for explicit methods
- TR-BDF2 is L-stable, allowing larger time steps for diffusion-dominated problems
- For advection-dominated problems, consider upwind schemes in spatial operator
- Fine grids (small dx) require smaller convergence tolerance
- Obstacle conditions may slow convergence due to complementarity constraints

## Git Commit Message Guidelines

Follow these seven rules for writing clear, maintainable commit messages:

### The Seven Rules

1. **Separate subject from body with a blank line**
   - The first line is the commit title, treated specially by Git tools
   - A blank line distinguishes it from detailed explanations

2. **Limit the subject line to 50 characters**
   - Forces clarity and ensures readability
   - While 72 is the hard limit before truncation, aim for 50

3. **Capitalize the subject line**
   - Begin with a capital letter for consistency

4. **Do not end the subject line with a period**
   - Trailing punctuation wastes space in the 50-character limit

5. **Use the imperative mood in the subject line**
   - Write commands, not descriptions
   - Complete the sentence: "If applied, this commit will _[your message]_"
   - Examples:
     - ✅ "Add cubic spline interpolation"
     - ✅ "Fix convergence criteria in TR-BDF2 solver"
     - ✅ "Refactor boundary condition application"
     - ❌ "Added cubic spline interpolation"
     - ❌ "Fixing convergence criteria"

6. **Wrap the body at 72 characters**
   - Manual wrapping at 72 allows Git to indent while staying under 80
   - Maintains readability across different tools

7. **Use the body to explain what and why vs. how**
   - Clarify the reasoning and context behind changes
   - Don't mechanically describe code modifications (the diff shows that)
   - Answer: Why is this change necessary? What problem does it solve?

### Example

```
Add cubic spline interpolation for off-grid queries

Users often need to evaluate the PDE solution at points not on the
computational grid. Natural cubic splines provide smooth, continuous
interpolation between grid points.

The implementation shares the tridiagonal solver with the TR-BDF2
scheme to avoid code duplication. Both spline coefficients and
derivative evaluation are supported for gradient-based applications.
```

## Workflow: Creating Pull Requests

**IMPORTANT:** After completing a task and committing changes, create a GitHub Pull Request instead of pushing directly to main.

### **IMPORTANT: Always Start New Work on a Fresh Branch**

Before starting any new task:
1. **Switch to main and pull latest changes**:
   ```bash
   git checkout main
   git pull
   ```

2. **Create a NEW feature branch** from updated main:
   ```bash
   git checkout -b feature/descriptive-name
   ```

**Never continue work on an existing branch that already has an open PR.** Each task should get its own branch from the latest main.

### Standard Workflow

1. **Create a feature branch** (if not already on one):
   ```bash
   git checkout -b feature/descriptive-name
   ```

   Branch naming conventions:
   - `feature/` - New features
   - `fix/` - Bug fixes
   - `test/` - Adding tests
   - `docs/` - Documentation updates

2. **Make changes and commit** following the commit message guidelines above

3. **Push the branch to GitHub**:
   ```bash
   git push -u origin feature/descriptive-name
   ```

4. **Create a Pull Request**:
   ```bash
   gh pr create --title "Brief description" --body "$(cat <<'EOF'
   ## Summary
   Brief description of what this PR does

   ## Changes
   - List key changes
   - Explain technical decisions

   ## Testing
   - Describe how changes were tested
   - Include test results if applicable

   ## Related Issues
   Fixes #issue_number (if applicable)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

5. **Wait for review** (or self-merge if you have permission and tests pass)

### PR Title Guidelines

- Follow the same style as commit messages (imperative mood, 50 chars)
- Examples:
  - "Fix TR-BDF2 Stage 2 coefficient calculation"
  - "Add support for Robin boundary conditions"
  - "Refactor Newton iteration convergence check"

### PR Body Template

```markdown
## Summary
[1-2 sentence overview of what this PR accomplishes]

## Changes
- [Key change 1]
- [Key change 2]
- [Key change 3]

## Testing
[How you tested these changes]
- Bazel test results: X/Y passing
- Manual testing performed
- Performance impact (if any)

## Related Issues
Fixes #[issue_number]
Closes #[issue_number]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### When to Create a PR

- **After completing a logical unit of work** (bug fix, feature, refactoring)
- **After all tests pass** locally
- **Before moving to the next task** (don't accumulate multiple unrelated changes)

### Self-Merging

If you have permission and all checks pass:
```bash
# Merge the PR (after tests pass)
gh pr merge --squash --delete-branch

# Return to main branch
git checkout main
git pull
```

### Example Complete Workflow

```bash
# Start work on a bug fix
git checkout -b fix/trbdf2-stagnation

# Make changes...
# ... edit files ...

# Test locally
bazel test //...

# Commit
git add src/pde_solver.c
git commit -m "Fix TR-BDF2 Stage 2 coefficients

Replaced incorrect formulation with standard coefficients
from Ascher, Ruuth, Wetton (1995).

Fixes #7"

# Push and create PR
git push -u origin fix/trbdf2-stagnation
gh pr create --title "Fix TR-BDF2 Stage 2 coefficients" --body "..."

# After review/tests pass, merge
gh pr merge --squash --delete-branch

# Clean up
git checkout main
git pull
```
