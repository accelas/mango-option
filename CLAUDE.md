# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mango-iv** is a modern C++23 library for pricing American options and solving PDEs using finite difference methods. The core solver uses TR-BDF2 (Two-stage Runge-Kutta with backward differentiation formula) time-stepping with Newton iteration for implicit systems. The library provides high-level option pricing APIs, implied volatility solvers (both FDM and interpolation-based), and price table pre-computation for fast repeated queries.

## Build System

This project uses Bazel with Bzlmod for dependency management.

### Common Commands

```bash
# Build everything
bazel build //...

# Build specific targets
bazel build //src/pde/core:pde_solver
bazel build //src/option:american_option
bazel build //examples:example_newton_solver

# Run all tests
bazel test //...

# Run specific test suites
bazel test //tests:pde_solver_test
bazel test //tests:american_option_test
bazel test //tests:bspline_4d_test
bazel test //tests:iv_solver_test

# Run tests with verbose output
bazel test //tests:pde_solver_test --test_output=all

# Run examples
bazel run //examples:example_newton_solver
bazel run //examples:example_expected_validation

# Run QuantLib benchmarks (requires libquantlib0-dev)
bazel build //benchmarks:quantlib_performance
bazel build //benchmarks:quantlib_accuracy
./bazel-bin/benchmarks/quantlib_performance
./bazel-bin/benchmarks/quantlib_accuracy

# Clean build artifacts
bazel clean
```

## Project Structure

```
mango-iv/
├── MODULE.bazel           # Bazel module with GoogleTest and Benchmark dependencies
├── src/
│   ├── BUILD.bazel        # Build configuration index (see subdirectories)
│   ├── pde/
│   │   ├── core/          # Grid, boundary conditions, PDE solver, time domain
│   │   └── operators/     # Spatial operators (centered difference, Laplacian, Black-Scholes)
│   ├── option/            # American option pricing, IV solvers, price tables
│   ├── bspline/           # B-spline interpolation (4D fitters, basis functions)
│   ├── math/              # Root finding, cubic splines, Thomas solver
│   └── support/           # Memory management (PMR arenas), CPU features, utilities
├── examples/
│   ├── example_newton_solver.cc          # PDE solver with Newton iteration
│   └── example_expected_validation.cpp   # Expected<T, E> error handling demo
├── tests/
│   ├── pde_solver_test.cc                # Core PDE solver tests
│   ├── american_option_test.cc           # American option pricing tests
│   ├── bspline_4d_test.cc                # 4D B-spline interpolation tests
│   ├── iv_solver_test.cc                 # Implied volatility solver tests
│   ├── price_table_*_test.cc             # Price table and workspace tests
│   └── ... (38 test files total)
├── benchmarks/
│   ├── quantlib_performance.cc           # Performance comparison with QuantLib
│   └── quantlib_accuracy.cc              # Accuracy validation vs QuantLib
└── docs/                                 # Architecture, design docs, API guides
```

## Core Architecture

### Modern C++ Template-Based Design

The library uses modern C++23 with template-based spatial operators, compile-time boundary condition dispatch, and `std::expected` for error handling.

**Key API Components:**

1. **PDESolver<BoundaryL, BoundaryR, SpatialOp>** (Template Class)
   - Solves PDEs of the form: ∂u/∂t = L(u, x, t)
   - Template parameters specify boundary conditions and spatial operator at compile-time
   - Uses Newton iteration for implicit TR-BDF2 time stepping
   - Supports obstacle conditions via `ObstacleCallback` function

2. **Boundary Conditions** (Compile-time Dispatch)
   - `DirichletBC`: u = g(t) at boundary
   - `NeumannBC`: ∂u/∂x = g(t) at boundary
   - Generic operator interface: `apply()`, `apply_jacobian()`, `estimate_ghost_value()`

3. **Spatial Operators** (Composable Templates)
   - `operators::SpatialOperator<PDE, Grid, Backend>`: Generic wrapper for PDE formulas
   - `BlackScholesPDE`: L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V
   - `LaplacianPDE`: Simple diffusion L(u) = D·∂²u/∂x²
   - Backend dispatch: `CenteredDifference` (scalar or SIMD)

4. **High-Level American Option API**
   ```cpp
   #include "src/option/american_option.hpp"

   AmericanOptionParams params{.strike = 100.0, .spot = 100.0, ...};
   AmericanOptionGrid grid{.n_space = 101, .n_time = 1000, ...};

   AmericanOptionSolver solver(params, grid);
   auto result = solver.solve();  // Returns Expected<AmericanOptionResult, ErrorCode>
   ```

5. **Obstacle Conditions** (American Options)
   - Implemented via `ObstacleCallback`: `std::function<void(double t, span<const double> x, span<double> psi)>`
   - Enforces u(x,t) ≥ ψ(x,t) via projection after each Newton iteration
   - Used for early exercise boundary in American options

### TR-BDF2 Time Stepping

The solver implements a composite two-stage scheme:
- **Stage 1**: Trapezoidal rule from t_n to t_n + γ·dt (γ ≈ 0.5858)
- **Stage 2**: BDF2 from t_n to t_n+1

This scheme provides:
- L-stability for stiff problems
- Second-order accuracy
- Good damping properties for high-frequency errors

### Newton Iteration for Implicit Systems

The solver uses Newton-Raphson iteration for implicit TR-BDF2 stages:
- **Analytical Jacobian**: Spatial operators provide analytical Jacobian assembly via `assemble_jacobian()` concept
- **Convergence**: Relative error with default tolerance of 1e-6, max 20 iterations
- **Damping**: Not needed (analytical Jacobian ensures quadratic convergence)

### Memory Management

**Modern C++ Ownership:**
- Grid data stored in `std::vector<double>` with RAII semantics
- Workspace allocated once at solver construction, reused across time steps
- PMR (Polymorphic Memory Resource) arenas for advanced use cases (price tables, batch IV)

**Workspace Design:**
- `PDEWorkspace`: Contiguous buffer with 64-byte alignment for SIMD
- Buffers: u_current, u_next, u_stage, rhs, tridiag (diag/upper/lower), u_old, Lu, u_temp
- Hybrid allocation: Newton workspace borrows from PDE workspace to reduce memory footprint

### SIMD Vectorization

Two backends for centered difference operators:
1. **ScalarBackend**: `#pragma omp simd` for compiler auto-vectorization
2. **SimdBackend**: `std::experimental::simd` with `[[gnu::target_clones("default","avx2","avx512f")]]`
   - 3-6× speedup on large grids
   - Automatic ISA selection at runtime

Cox-de Boor B-spline basis evaluation also uses SIMD (4-wide cubic basis functions).

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

### B-Spline Interpolation

The library provides two interpolation strategies:

**1. Cubic Splines** (Thomas solver, legacy):
- Natural cubic splines for 1D off-grid evaluation
- Single workspace buffer (4n doubles), uses tridiagonal solver
- Available via `mango::ThomasCubicSpline` class

**2. B-Spline 4D Interpolation** (Production):
- Separable 4D B-spline fitting with banded LU solver
- 4-diagonal collocation matrix exploits cubic basis structure
- **7.8× speedup** on large grids (50×30×20×10 = 300K points) vs dense solver
- Used for price table pre-computation and fast interpolated IV
- See `src/bspline/bspline_fitter_4d.hpp`

## Development Workflow

### Pricing American Options (High-Level API)

**Most users should use the high-level American option API:**

```cpp
#include "src/option/american_option.hpp"

// Define option parameters
mango::AmericanOptionParams params{
    .strike = 100.0,
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .option_type = OptionType::PUT
};

// Configure PDE grid
mango::AmericanOptionGrid grid{
    .n_space = 101,
    .n_time = 1000,
    .x_min = -3.0,
    .x_max = 3.0
};

// Solve
mango::AmericanOptionSolver solver(params, grid);
auto result = solver.solve();

if (result.has_value()) {
    std::cout << "Price: " << result->price << "\n";
    std::cout << "Delta: " << result->delta << "\n";
    std::cout << "Gamma: " << result->gamma << "\n";
} else {
    std::cerr << "Error: " << static_cast<int>(result.error()) << "\n";
}
```

### Solving Custom PDEs (Low-Level API)

For custom PDE problems beyond American options:

```cpp
#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/laplacian_pde.hpp"
#include "src/pde/operators/operator_factory.hpp"

// 1. Create grid and time domain
auto grid = mango::Grid(0.0, 1.0, 101);
mango::TimeDomain time{.t_start = 0.0, .t_end = 1.0, .n_steps = 1000};

// 2. Define PDE and spatial operator
mango::operators::LaplacianPDE pde(0.1);  // Diffusion coefficient D=0.1
auto spatial_op = mango::operators::make_spatial_operator(pde, grid);

// 3. Define boundary conditions
auto left_bc = mango::DirichletBC([](double t) { return 0.0; });
auto right_bc = mango::DirichletBC([](double t) { return 0.0; });

// 4. Create and run solver
mango::PDESolver solver(grid, time, mango::TRBDF2Config{},
                        left_bc, right_bc, spatial_op);

// Initial condition: u(x, 0) = sin(π·x)
solver.initialize([](std::span<const double> x, std::span<double> u) {
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::sin(M_PI * x[i]);
    }
});

bool success = solver.solve();
auto snapshot = solver.snapshot();  // Final solution
```

### Adding Tests

All tests use GoogleTest with modern C++ APIs:
- PDE solver tests: `tests/pde_solver_test.cc`
- American option tests: `tests/american_option_test.cc`
- B-spline tests: `tests/bspline_4d_test.cc`
- IV solver tests: `tests/iv_solver_test.cc`

**Test naming convention:**
- Unit tests: `*_test.cc`
- Integration tests: `*_integration_test.cc`
- Performance tests: `*_performance_test.cc`

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
1. Reduce time step (increase n_time in grid configuration)
2. Increase spatial resolution (increase n_space in grid configuration)
3. Check spatial operator implementation for errors
4. Verify boundary conditions are consistent
5. For advanced users: use PDESolver directly with custom TRBDF2Config

### American Option API Simplification

The `AmericanOptionSolver` provides a simplified, high-level API that hides internal solver configuration details. The TR-BDF2 time-stepping parameters use sensible defaults that work well for most applications.

**Basic Usage:**
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

// Create workspace with grid configuration
auto workspace = AmericanSolverWorkspace::create(-3.0, 3.0, 101, 1000);

// Simple construction with defaults
AmericanOptionSolver solver(params, workspace.value());
auto result = solver.solve();
```

**Default solver parameters:**
- TR-BDF2: 20 iterations, 1e-6 tolerance, gamma = 2 - √2
- Time stepping: Implicit L-stable scheme
- Obstacle projection: Applied after each Newton iteration

**For advanced configuration:**
If you need to tune TR-BDF2 or Newton solver parameters for convergence or accuracy, use the low-level `PDESolver` API directly instead of `AmericanOptionSolver`. This provides full control over all solver settings.

### Memory Management

Modern C++ RAII patterns - no manual memory management required:
- `PDESolver` and all grid/workspace objects use RAII (automatic cleanup)
- `std::vector`, `std::unique_ptr`, `std::shared_ptr` for owned data
- PMR arenas (`SolverMemoryArena`) for advanced use cases requiring manual control

## Unified Root-Finding API

The library provides a unified configuration and result interface for all root-finding methods.

### Configuration

```cpp
#include "src/math/root_finding.hpp"

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

## PMR (Polymorphic Memory Resource) Usage Patterns

The library implements C++17 PMR (Polymorphic Memory Resource) patterns for efficient memory management in repeated solver operations. PMR enables zero-copy data transfer, arena allocation, and memory pooling for high-performance numerical computing.

### SolverMemoryArena Overview

**Purpose**: Memory efficiency for repeated PDE solves with shared memory arenas

**Three-level hierarchy**: pool → arena → tracker
- **Pool**: `std::pmr::monotonic_buffer_resource` for fast allocation
- **Arena**: `SolverMemoryArena` for workspace coordination
- **Tracker**: Reference counting for active workspaces

**Key features**:
- Factory pattern with `shared_ptr` ownership
- Thread-safe active workspace counting
- Zero-cost reset when no workspaces are active
- 64-byte alignment for AVX-512 SIMD operations

### Creating and Using SolverMemoryArena

**Factory method**: `create_arena()` returns `shared_ptr` for proper lifetime management

**C++ Usage**:
```cpp
#include "src/support/memory/solver_memory_arena.hpp"

// Create 1MB memory arena
auto arena_result = mango::memory::SolverMemoryArena::create(1024 * 1024);
if (!arena_result.has_value()) {
    std::cerr << "Failed to create arena: " << arena_result.error() << "\n";
    return;
}

auto arena = arena_result.value();

// Get arena statistics
auto stats = arena->get_stats();
std::cout << "Total size: " << stats.total_size << "\n";
std::cout << "Used size: " << stats.used_size << "\n";
std::cout << "Active workspaces: " << stats.active_workspace_count << "\n";

// Use with PMR-enabled components
auto collector = PriceTableSnapshotCollector(config, arena);

// Manage workspace lifecycle
arena->increment_active();  // Start using the arena
// ... perform computations ...
arena->decrement_active();  // Done using the arena

// Reset when no workspaces are active (zero-cost)
auto reset_result = arena->try_reset();
if (!reset_result.has_value()) {
    std::cerr << "Cannot reset: " << reset_result.error() << "\n";
}
```

**Python Usage** (via bindings):
```python
import mango_iv

# Create arena using factory method
arena = mango_iv.create_arena(1024 * 1024)

# Get statistics
stats = arena.get_stats()
print(f"Total size: {stats.total_size}")
print(f"Used size: {stats.used_size}")
print(f"Active workspaces: {stats.active_workspace_count}")

# Workspace management
arena.increment_active()
# ... do work ...
arena.decrement_active()

# Memory reset
try:
    arena.try_reset()
    print("Arena reset successful")
except ValueError as e:
    print(f"Cannot reset: {e}")

# Get memory resource for PMR integration
resource = arena.resource()
```

**SolverMemoryArenaStats**:
- `total_size`: Total size of the arena in bytes
- `used_size`: Currently allocated memory
- `active_workspace_count`: Number of active workspaces using the arena

### PriceTableSnapshotCollector Zero-Copy Pattern

**How pmr::vector enables zero-copy**:
- Vectors allocated from arena memory are directly accessible as spans
- No `std::copy` needed between solver workspace and price table
- Memory is reused across multiple price table operations

**Span accessors for workspace borrowing**:
```cpp
PriceTableSnapshotCollector collector(config, arena);

// Zero-copy access to internal buffers
std::span<double> prices = collector.prices_span();
std::span<double> deltas = collector.deltas_span();
std::span<double> gammas = collector.gammas_span();
std::span<double> thetas = collector.thetas_span();

// Direct modification without allocation
for (size_t i = 0; i < prices.size(); ++i) {
    prices[i] = some_computation(i);
}
```

**Constructor with memory arena**:
```cpp
// Constructor accepts shared_ptr to arena
PriceTableSnapshotCollector collector(config, arena);

// All internal pmr::vectors use arena memory:
// - prices_, deltas_, gammas_, thetas_
// - log_moneyness_, spot_values_, inv_spot_, inv_spot_sq_
// - cached_grid_
// - interpolator internals
```

### Integration with Existing Workspaces

**Pass arena.resource() to workspace constructors**:
```cpp
// Create workspace with PMR allocation
BSplineFitter4DWorkspace workspace(max_grid_size, arena->resource());

// NewtonWorkspace with arena memory
NewtonWorkspace newton_workspace(grid_size, arena->resource());

// Price table construction with arena
auto collector = std::make_unique<PriceTableSnapshotCollector>(config, arena);
```

**Shared_ptr lifetime management**:
```cpp
// Arena lifetime managed by shared_ptr
std::shared_ptr<mango::memory::SolverMemoryArena> arena;

{
    auto local_arena = mango::memory::SolverMemoryArena::create(size);
    arena = local_arena.value();

    // Use arena in multiple components
    auto collector1 = PriceTableSnapshotCollector(config1, arena);
    auto collector2 = PriceTableSnapshotCollector(config2, arena);

} // local_arena goes out of scope, but arena remains alive

// Arena still valid here due to shared_ptr
```

**Best practices for repeated solves**:
```cpp
// Create arena once for batch operations
auto arena = mango::memory::SolverMemoryArena::create(arena_size).value();

// Reuse arena across multiple solves
for (const auto& problem : batch_problems) {
    arena->increment_active();

    // Create solver components with arena memory
    auto solver = create_solver_with_arena(problem, arena);
    auto result = solver.solve();

    // Process results directly from arena-allocated buffers
    process_results(result);

    arena->decrement_active();

    // Zero-cost reset between solves
    arena->try_reset();
}
```

### USDT Tracing for PMR Operations

**Available probes for arena operations**:
- `MODULE_MEMORY` (ID: 8) - General memory arena operations
- `MODULE_PRICE_TABLE_COLLECTOR` (ID: 9) - Price table collection with PMR

**Monitoring memory usage and workspace counts**:
```bash
# Monitor arena creation and workspace activity
sudo bpftrace -e '
usdt::mango:algo_start /arg0 == 8/ {
    printf("Arena created: size=%zu bytes\n", arg1);
}
usdt::mango:algo_progress /arg0 == 8 && arg3 == 0/ {
    printf("Active workspaces: %d\n", arg1);
}
usdt::mango:algo_progress /arg0 == 8 && arg3 == 1/ {
    printf("Workspace count decreased: %d remaining\n", arg1);
}
' -c './my_program'

# Monitor price table collection with PMR
sudo bpftrace -e '
usdt::mango:algo_start /arg0 == 9/ {
    printf("Price table collection: moneyness=%d tau=%d total=%d\n",
           arg1, arg2, arg3);
}
usdt::mango:algo_progress /arg0 == 9/ {
    printf("Progress: step=%d/%d message=%s\n", arg1, arg2, str(arg3));
}
' -c './my_precompute_program'
```

**Predefined tracing scripts**:
```bash
# Memory arena monitoring
sudo ./scripts/mango-trace monitor ./my_program --preset=memory

# Memory debugging (catches allocation failures)
sudo ./scripts/mango-trace monitor ./my_program --preset=debug
```

## Implied Volatility Solver

The library provides a complete implied volatility solver for American options using Brent's method with nested PDE evaluation.

### Overview

**Algorithm:** Nested Brent's method + American option PDE solver
**Performance:** ~143ms per IV calculation (43% faster than 250ms target)
**Status:** Production-ready

The IV solver finds the volatility parameter that makes the American option's theoretical price (from PDE solver) match the observed market price.

### Basic Usage

```cpp
#include "src/cpp/iv_solver.hpp"

// Setup option parameters
mango::IVParams params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = false  // American put
};

// Configure solver (optional - uses defaults if not specified)
mango::IVConfig config{
    .root_config = mango::RootFindingConfig{
        .max_iter = 100,
        .tolerance = 1e-6
    },
    .grid_n_space = 101,
    .grid_n_time = 1000,
    .grid_s_max = 200.0
};

// Solve for implied volatility
mango::IVSolver solver(params, config);
mango::IVResult result = solver.solve();

if (result.converged) {
    std::cout << "Implied Volatility: " << result.implied_vol << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Final Error: " << result.final_error << "\n";
} else {
    std::cerr << "Failed to converge: " << *result.failure_reason << "\n";
}
```

### Configuration Options

**Root-Finding Configuration:**
```cpp
mango::RootFindingConfig root_config{
    .max_iter = 100,           // Maximum Brent iterations
    .tolerance = 1e-6,         // Price convergence tolerance
    .brent_tol_abs = 1e-6      // Brent absolute tolerance
};
```

**Grid Configuration:**
```cpp
mango::IVConfig config{
    .root_config = root_config,
    .grid_n_space = 101,       // Spatial grid points
    .grid_n_time = 1000,       // Time steps
    .grid_s_max = 200.0        // Maximum spot price
};
```

### Adaptive Volatility Bounds

The solver uses intelligent bounds based on intrinsic value analysis:

| Moneyness | Time Value | Upper Bound | Rationale |
|-----------|-----------|-------------|-----------|
| ATM/OTM | High (>50%) | 300% | High time value suggests high vol |
| Moderate | Medium (20-50%) | 200% | Moderate time value |
| Deep ITM | Low (<20%) | 150% | Low time value, unlikely high vol |
| All | - | 1% (lower) | Minimum realistic volatility |

This adaptive approach reduces Brent iterations compared to arbitrary bounds.

### Input Validation

The solver validates all inputs and catches arbitrage violations:

**Validation checks:**
- Spot price > 0
- Strike price > 0
- Time to maturity > 0
- Market price > 0
- Call price ≤ spot price (no arbitrage)
- Put price ≤ strike price (no arbitrage)
- Market price ≥ intrinsic value (no arbitrage)

### Performance Characteristics

**Typical performance (100 space points, 1000 time steps):**

| Scenario | Iterations | Time | Notes |
|----------|-----------|------|-------|
| ATM put | 10-12 | ~132ms | Most common case |
| ITM put | 12-15 | ~158ms | Higher time value |
| OTM put | 10-12 | ~139ms | Lower price sensitivity |

**Average:** ~143ms per IV calculation (43% faster than 250ms target)

**Speedup opportunities:**
- For production use requiring many queries, consider interpolation-based IV (~7.5µs)
- FDM-based IV provides ground truth for validation
- See `docs/plans/2025-10-31-interpolation-iv-next-steps.md` for future work

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

```cpp
IVResult result = solver.solve();

if (!result.converged) {
    // Check failure reason
    if (result.failure_reason.has_value()) {
        std::cerr << "Error: " << *result.failure_reason << "\n";
    }

    // Check if it was a convergence issue vs validation error
    if (result.iterations >= config.root_config.max_iter) {
        std::cerr << "Reached max iterations without converging\n";
    }
}
```

### Example: Batch Processing

```cpp
// Process multiple options
std::vector<IVParams> option_params = load_market_data();
std::vector<IVResult> results;

for (const auto& params : option_params) {
    mango::IVSolver solver(params, config);
    results.push_back(solver.solve());
}

// Report statistics
size_t converged = 0;
double total_time = 0.0;

for (const auto& result : results) {
    if (result.converged) {
        converged++;
    }
}

std::cout << "Converged: " << converged << "/" << results.size() << "\n";
```

### Related Documentation

- **Design Document:** `docs/plans/2025-10-31-american-iv-implementation-design.md`
- **Implementation Summary:** `docs/plans/IV_IMPLEMENTATION_SUMMARY.md`
- **Future Work:** `docs/plans/2025-10-31-interpolation-iv-next-steps.md`
- **Test Suite:** `tests/iv_solver_test.cc`

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
bazel build //src/pde/core:pde_solver
bazel build //examples:example_newton_solver
bazel build //examples:example_expected_validation
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
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_newton_solver

# Watch convergence behavior
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_newton_solver --preset=convergence

# Debug failures
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_newton_solver --preset=debug

# Profile performance
sudo ./scripts/mango-trace monitor ./bazel-bin/examples/example_newton_solver --preset=performance
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
sudo bpftrace scripts/tracing/monitor_all.bt -c './bazel-bin/examples/example_newton_solver'

# Or write custom one-liners
sudo bpftrace -e 'usdt::mango:convergence_failed {
    printf("Module %d failed at step %d\n", arg0, arg1);
}' -c './my_program'
```

**Helper tool commands:**

```bash
# Check if binary has USDT support
sudo ./scripts/mango-trace check ./bazel-bin/examples/example_newton_solver

# List all available probes
sudo ./scripts/mango-trace list ./bazel-bin/examples/example_newton_solver

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

## C++23 Features Used

- `std::expected<T, E>` for error handling (no exceptions)
- Designated initializers for structs
- `std::span` for safe array views
- `auto` and structured bindings
- Concepts (`HasAnalyticalJacobian`)
- `std::experimental::simd` for portable SIMD
- `[[gnu::target_clones]]` for multi-ISA code generation
- Three-way comparison operator (`<=>`)
- `constexpr` and `consteval` for compile-time evaluation

## Common Patterns

### Pattern 1: American Option Pricing

```cpp
#include "src/option/american_option.hpp"

mango::AmericanOptionParams params{
    .strike = 100.0, .spot = 100.0, .maturity = 1.0,
    .volatility = 0.20, .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .option_type = OptionType::PUT
};

mango::AmericanOptionGrid grid{.n_space = 101, .n_time = 1000};

mango::AmericanOptionSolver solver(params, grid);
auto result = solver.solve();

if (result.has_value()) {
    std::cout << "Price: " << result->price << "\n";
}
```

### Pattern 2: Implied Volatility (FDM-Based)

```cpp
#include "src/option/iv_solver_fdm.hpp"

mango::IVParams params{
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = false
};

mango::FDMIVSolver solver(params, mango::IVConfig{});
mango::IVResult result = solver.solve();

if (result.converged) {
    std::cout << "Implied Vol: " << result.implied_vol << "\n";
}
```

### Pattern 3: Fast Interpolated IV (with Pre-computed Price Table)

```cpp
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/iv_solver_interpolated.hpp"

// Build price table once (expensive)
auto table = mango::PriceTable4DBuilder()
    .set_moneyness_grid(/* ... */)
    .set_maturity_grid(/* ... */)
    .set_volatility_grid(/* ... */)
    .set_rate_grid(/* ... */)
    .build();

// Solve IV using interpolation (~7.5µs vs ~143ms for FDM)
mango::InterpolatedIVSolver iv_solver(table);
auto iv_result = iv_solver.solve(params);
```

## Price Table Pre-computation Workflow

Pre-compute 4D American option price surfaces (moneyness × maturity × volatility × rate) with B-spline interpolation for ultra-fast repeated queries (~500ns per lookup).

### Typical Workflow

**1. Build the price table:**
```cpp
#include "src/option/price_table_4d_builder.hpp"

// Define 4D grids
auto builder_result = mango::PriceTable4DBuilder::create(
    {0.7, 0.8, ..., 1.3},   // 50 moneyness points (log-spaced recommended)
    {0.027, 0.1, ..., 2.0}, // 30 maturity points
    {0.10, 0.15, ..., 0.80},// 20 volatility points
    {0.0, 0.02, ..., 0.10}, // 10 rate points
    100.0                    // K_ref (reference strike)
);

if (!builder_result.has_value()) {
    std::cerr << "Builder creation failed\n";
    return;
}

auto builder = std::move(builder_result.value());

// Pre-compute prices (200 PDE solves, parallelized with OpenMP)
builder->precompute(OptionType::PUT, 101, 1000);  // n_space, n_time
```

**2. Query prices, deltas, vegas, gammas (sub-microsecond):**
```cpp
auto surface = builder->get_surface();

// Single query (~500ns)
double price = surface.eval(1.05, 0.25, 0.20, 0.05);  // m, tau, sigma, r

// Greeks also available
double delta = surface.eval_delta(1.05, 0.25, 0.20, 0.05);
double vega = surface.eval_vega(1.05, 0.25, 0.20, 0.05);
double gamma = surface.eval_gamma(1.05, 0.25, 0.20, 0.05);

// Batch queries
for (const auto& [m, tau, sigma, r] : market_data) {
    double p = surface.eval(m, tau, sigma, r);
    double v = surface.eval_vega(m, tau, sigma, r);
    // Process results...
}
```

**3. Save/load for persistence:**
```cpp
// Save to binary file (includes grids, coefficients, metadata)
builder->save("spx_american_put_surface.bin");

// Later: fast load (milliseconds vs minutes for pre-computation)
auto loaded = mango::PriceTable4DBuilder::load("spx_american_put_surface.bin");
auto surface = loaded->get_surface();
```

### Greeks Computation

All Greeks are computed during pre-computation using centered finite differences and stored in the B-spline coefficients:

**Delta (∂V/∂S):** First derivative w.r.t. moneyness, scaled to spot price
**Vega (∂V/∂σ):** First derivative w.r.t. volatility
**Gamma (∂²V/∂S²):** Second derivative w.r.t. moneyness, scaled to spot price

Query cost: ~500ns per Greek (same as price lookup, no additional computation)

### Performance Characteristics

**Pre-computation (one-time cost):**
- 300K grid points (50×30×20×10): ~15-20 minutes on 16 cores
- Throughput: ~300 options/second with parallelization
- Memory overhead: ~10 KB per batch (configurable)
- Uses OpenMP for parallel batch processing

**Query performance (amortized benefit):**
- 4D interpolation: ~500 nanoseconds (multilinear)
- 5D interpolation: ~2 microseconds (multilinear)
- Greeks computation: ~5-10 microseconds (requires multiple interpolations)
- Speedup vs FDM: ~40,000x for single query

**Memory usage:**
- 4D table (50×30×20×10): ~2.4 MB
- 5D table adds dividend dimension (proportional scaling)
- Binary format includes grids, prices, and metadata

### Environment Variables

- **MANGO_PRECOMPUTE_BATCH_SIZE**: Batch size for pre-computation (default: 100)
  - Range: 1-100000
  - Larger batches: better throughput, more memory
  - Smaller batches: more frequent progress updates, less memory
  - Recommended: 100-500 for most use cases

### USDT Tracing

Monitor pre-computation progress with USDT probes:
```bash
# Watch progress during pre-computation
sudo bpftrace -e 'usdt::mango:algo_progress /arg0 == 4/ {
    printf("Price table: %d%% complete\n", arg2);
}' -c './my_precompute_program'
```

See `examples/example_precompute_table.c` for a complete working example.

## B-spline Banded Solver Optimization

The B-spline collocation solver uses a banded LU decomposition optimized for the 4-diagonal structure of cubic B-spline basis functions.

### Performance Characteristics

**Micro-benchmark speedup** (1D solver, isolated):
- Small grids (n=50): 7.5× speedup
- Medium grids (n=100): 42× average speedup
- Large grids (n=200): 87× speedup

**End-to-end speedup** (4D separable fitting on realistic grids):
- Small grid (7×4×4×4 = 448 points): 0.56× (overhead dominates)
- Medium grid (20×15×10×8 = 24K points): 1.70× speedup
- Large grid (50×30×20×10 = 300K points): **7.8× speedup**

**Complexity reduction**: O(n³) dense solver → O(n²) banded LU for fixed bandwidth

### Why the Speedup Varies

The end-to-end speedup (7.8×) is less than the micro-benchmark speedup (42×) due to:
1. **Overhead from 4D tensor operations**: Grid extraction, result aggregation
2. **Memory bandwidth constraints**: Separable fitting processes large data arrays
3. **Non-solver costs**: Basis function evaluation, residual computation
4. **Amdahl's law**: Banded solver is ~40% of total runtime on large grids

For production workloads (300K point grids), the **7.8× end-to-end speedup** is the relevant metric.

### Usage

The banded solver is **automatically enabled** for all B-spline fitting operations. No configuration required.

```cpp
#include "src/interpolation/bspline_fitter_4d.hpp"

// Create 4D fitter (banded solver used automatically)
auto fitter_result = mango::BSplineFitter4D::create(
    axis0_grid, axis1_grid, axis2_grid, axis3_grid);

if (fitter_result.has_value()) {
    auto& fitter = fitter_result.value();

    // Fit coefficients (uses banded solver internally)
    auto result = fitter.fit(values_4d);

    if (result.success) {
        // Use result.coefficients with BSpline4D
        // Fitting residuals available in result.max_residual
    }
}
```

### Implementation Details

**BandedMatrixStorage**: Compact storage for 4-diagonal matrices
- Memory: O(4n) vs O(n²) for dense
- Layout: `band_values_[i*4 + k]` for row i, band entry k
- Column tracking: `col_start_[i]` indicates first non-zero column

**banded_lu_solve()**: In-place LU decomposition
- Time complexity: O(n) for fixed bandwidth (k=4)
- Space complexity: O(n) working storage
- Algorithm: Doolittle LU with banded structure exploitation

**Numerical accuracy**: Identical to dense solver (verified to floating-point precision)

### When Speedup Matters

- **Small grids (n < 20)**: Overhead dominates, speedup minimal
- **Medium grids (n = 50-100)**: Noticeable speedup (1.7-2×)
- **Large grids (n > 100)**: Significant speedup (7-8×)

For price table construction with 50×30×20×10 grids, banded solver reduces fitting time from ~46ms to ~6ms.

### Testing and Verification

All tests verify:
- **Numerical correctness**: Banded solver matches dense solver to 1e-14
- **Performance regression**: Speedup tracked across grid sizes
- **Accuracy**: Fitting residuals < 1e-9 on all axes

See `tests/bspline_banded_solver_test.cc` and `tests/bspline_4d_end_to_end_performance_test.cc` for details.

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
