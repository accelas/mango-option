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

# Build specific targets
bazel build //src:pde_solver
bazel build //examples:example_heat_equation

# Run all tests
bazel test //...

# Run specific test suites
bazel test //tests:pde_solver_test
bazel test //tests:cubic_spline_test
bazel test //tests:stability_test

# Run tests with verbose output
bazel test //tests:pde_solver_test --test_output=all

# Run example
bazel run //examples:example_heat_equation

# Run QuantLib comparison benchmark (requires libquantlib0-dev)
bazel build //tests:quantlib_benchmark
./bazel-bin/tests/quantlib_benchmark

# Clean build artifacts
bazel clean
```

## Project Structure

```
mango-iv/
├── MODULE.bazel           # Bazel module with GoogleTest and Benchmark dependencies
├── src/
│   ├── BUILD.bazel        # Build configuration for library
│   ├── pde_solver.h       # Public API header
│   └── pde_solver.c       # TR-BDF2 solver implementation
├── examples/
│   ├── BUILD.bazel
│   └── example_heat_equation.c  # Demonstrates heat equation with callbacks
└── tests/
    ├── BUILD.bazel
    ├── pde_solver_test.cc        # Core PDE solver tests
    ├── cubic_spline_test.cc      # Interpolation tests
    ├── stability_test.cc          # Numerical stability tests
    ├── quantlib_benchmark.cc      # Performance comparison with QuantLib
    └── BENCHMARK.md               # Benchmark documentation
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
2. Increase max_iter in TRBDF2Config
3. Relax tolerance
4. Check spatial operator implementation for errors
5. Verify boundary conditions are consistent

### Memory Management

All structures use explicit create/destroy patterns:
- `pde_solver_create()` / `pde_solver_destroy()`
- `pde_spline_create()` / `pde_spline_destroy()`
- `pde_create_grid()` / `pde_free_grid()`

## Unified Root-Finding API

The library provides a unified configuration and result interface for all root-finding methods.

### Configuration

```cpp
#include "src/cpp/root_finding.hpp"

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

## Slice Solver Workspace (Reusable PDE Solving)

When building price tables or solving many PDEs that differ only in coefficients (volatility, interest rate, dividend yield), you can avoid redundant allocations by using a **SliceSolverWorkspace**.

### Performance Benefits

The workspace eliminates per-solver allocations:
- **Grid buffer**: ~800 bytes per reuse
- **GridSpacing**: ~800 bytes per reuse
- **Total savings**: ~1.6 KB per solver instance

For typical 4D price tables (200 solvers): **~320 KB saved**

### Usage Pattern

```cpp
#include "src/slice_solver_workspace.hpp"
#include "src/american_option.hpp"

// Create workspace once (reused across all solvers)
// Use shared_ptr for proper lifetime management
auto workspace = std::make_shared<SliceSolverWorkspace>(-3.0, 3.0, 101);

// Solve multiple options with different (σ, r, q) parameters
for (auto [sigma, rate] : parameter_grid) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = sigma,
        .rate = rate,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT
    };

    AmericanOptionGrid grid_config{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    // Solver keeps workspace alive via shared_ptr
    AmericanOptionSolver solver(params, grid_config, workspace);
    auto result = solver.solve();
}
```

### Key Points

1. **Lifetime safety**: Uses `std::shared_ptr` to ensure workspace outlives all solvers
2. **Grid parameters must match**: workspace grid (x_min, x_max, n_space) must match grid_config
3. **Thread-safe**: workspace is read-only during solve, safe for OpenMP parallel loops
4. **Backward compatible**: existing code without workspace continues to work
5. **Identical results**: workspace mode produces exactly the same numerical results as standalone mode

### When to Use

✅ **Use workspace mode when:**
- Building price tables (many solves with same grid, different coefficients)
- Parameter sweeps (σ, r, q variations)
- Batch processing of options

❌ **Use standalone mode when:**
- Single option pricing
- Grid parameters vary across solves
- Prototype/debugging code

### Implementation Details

The workspace pre-allocates:
- **Grid buffer**: Spatial grid points (uniform spacing in log-moneyness)
- **GridSpacing**: Precomputed spacing metrics (dx, inverse dx, etc.)

Both are shared via `shared_ptr` across all solver instances using the workspace.

## Price Table Pre-computation Workflow

The price table module provides fast option pricing through pre-computed lookup tables. This is ideal for applications requiring thousands of pricing queries where computation time dominates.

### Typical Workflow

**1. Create the price table structure:**
```c
// Define grid dimensions (example: 4D table for American puts)
double *moneyness = malloc(n_m * sizeof(double));
double *maturity = malloc(n_tau * sizeof(double));
double *volatility = malloc(n_sigma * sizeof(double));
double *rate = malloc(n_r * sizeof(double));

// Generate grids (log-spaced for moneyness, linear for others)
generate_log_spaced(moneyness, n_m, 0.7, 1.3);
generate_linear(maturity, n_tau, 0.027, 2.0);
generate_linear(volatility, n_sigma, 0.10, 0.80);
generate_linear(rate, n_r, 0.0, 0.10);

// Create table (takes ownership of grid arrays)
OptionPriceTable *table = price_table_create(
    moneyness, n_m, maturity, n_tau, volatility, n_sigma,
    rate, n_r, NULL, 0,  // No dividend dimension
    OPTION_PUT, EXERCISE_AMERICAN);

price_table_set_underlying(table, "SPX");
```

**2. Pre-compute all prices:**
```c
// Configure FDM solver grid
AmericanOptionGrid grid = {
    .n_space = 101,
    .n_time = 1000,
    .S_max = 200.0
};

// Optionally tune batch size (default: 100)
setenv("MANGO_PRECOMPUTE_BATCH_SIZE", "200", 1);

// Compute all prices (uses OpenMP parallelization)
int status = price_table_precompute(table, &grid);
if (status != 0) {
    fprintf(stderr, "Pre-computation failed\n");
    return 1;
}
```

**3. Save table for fast loading later:**
```c
price_table_save(table, "spx_american_put.bin");
price_table_destroy(table);

// Later: fast load (milliseconds instead of minutes)
table = price_table_load("spx_american_put.bin");
```

**4. Query prices, vegas, and gammas (sub-microsecond):**
```c
// Single price query
double price = price_table_interpolate_4d(table, 1.05, 0.25, 0.20, 0.05);

// Single vega query (∂V/∂σ)
double vega = price_table_interpolate_vega_4d(table, 1.05, 0.25, 0.20, 0.05);

// Single gamma query (∂²V/∂S²)
double gamma = price_table_interpolate_gamma_4d(table, 1.05, 0.25, 0.20, 0.05);

// Multiple queries (typical usage)
for (size_t i = 0; i < n_queries; i++) {
    double p = price_table_interpolate_4d(table, m[i], tau[i], sigma[i], r[i]);
    double v = price_table_interpolate_vega_4d(table, m[i], tau[i], sigma[i], r[i]);
    double g = price_table_interpolate_gamma_4d(table, m[i], tau[i], sigma[i], r[i]);
    // Process price, vega, and gamma...
}
```

**5. Cleanup:**
```c
price_table_destroy(table);
```

### Vega Interpolation

The price table automatically computes vega (∂V/∂σ) during precomputation using centered finite differences:

```c
// Vega is computed automatically during precomputation
price_table_precompute(table, &grid);  // Computes both prices and vegas

// Query vega at any point (4D table)
double vega = price_table_interpolate_vega_4d(table,
    1.05,   // moneyness
    0.5,    // maturity
    0.20,   // volatility
    0.05);  // rate

// For 5D tables with dividend
double vega_5d = price_table_interpolate_vega_5d(table,
    1.05, 0.5, 0.20, 0.05, 0.02);  // with dividend
```

**Key Points:**
- Vega computed during precomputation (centered finite differences)
- Same interpolation strategy as prices (cubic or multilinear)
- ~8ns per query (same speed as price interpolation)
- More accurate than computing vega at query time
- Enables Newton-based IV inversion
- Binary save/load preserves vega data

### Gamma Interpolation

The price table automatically computes gamma (∂²V/∂S²) during precomputation using centered finite differences on the moneyness axis:

```c
// Gamma is computed automatically during precomputation
price_table_precompute(table, &grid);  // Computes prices, vegas, and gammas

// Query gamma at any point (4D table)
double gamma = price_table_interpolate_gamma_4d(table,
    1.05,   // moneyness
    0.5,    // maturity
    0.20,   // volatility
    0.05);  // rate

// For 5D tables with dividend
double gamma_5d = price_table_interpolate_gamma_5d(table,
    1.05, 0.5, 0.20, 0.05, 0.02);  // with dividend
```

**Key Points:**
- Gamma computed during precomputation (centered finite differences on moneyness)
- Properly scaled from ∂²V/∂m² to ∂²V/∂S² using chain rule (γ = ∂²V/∂m² / K_ref²)
- Same interpolation strategy as prices (cubic or multilinear)
- ~8ns per query (same speed as price interpolation)
- More accurate than computing gamma at query time (avoids numerical errors)
- Essential for delta-hedging strategies and convexity analysis
- Binary save/load preserves gamma data
- Accuracy depends on grid spacing (finer grids → better second derivatives)

**Note on Accuracy:**
Second derivatives are inherently more sensitive to grid spacing than first derivatives. For high-accuracy gamma values, use finer moneyness grids (e.g., 50+ points). Typical relative errors: ~5-10% on moderate grids (20 points), <1% on fine grids (50+ points).

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
