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

4. **Jump Condition** (optional): `bool (*)(double x, double *jump_value, void *user_data)`
   - Handles discontinuous coefficients at interfaces
   - Scalar callback for interface location queries

5. **Obstacle Condition** (optional): `void (*)(const double *x, double t, size_t n, double *ψ, void *user_data)`
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

1. **Algorithm Lifecycle** (General): `IVCALC_TRACE_ALGO_START`, `IVCALC_TRACE_ALGO_PROGRESS`, `IVCALC_TRACE_ALGO_COMPLETE`
2. **Convergence Tracking** (General): `IVCALC_TRACE_CONVERGENCE_ITER`, `IVCALC_TRACE_CONVERGENCE_SUCCESS`, `IVCALC_TRACE_CONVERGENCE_FAILED`
3. **Validation/Errors** (General): `IVCALC_TRACE_VALIDATION_ERROR`, `IVCALC_TRACE_RUNTIME_ERROR`
4. **PDE Solver**: `IVCALC_TRACE_PDE_START`, `IVCALC_TRACE_PDE_PROGRESS`, `IVCALC_TRACE_PDE_COMPLETE`, etc.
5. **Implied Volatility**: `IVCALC_TRACE_IV_START`, `IVCALC_TRACE_IV_COMPLETE`, `IVCALC_TRACE_IV_VALIDATION_ERROR`
6. **American Options**: `IVCALC_TRACE_OPTION_START`, `IVCALC_TRACE_OPTION_COMPLETE`
7. **Brent's Method**: `IVCALC_TRACE_BRENT_START`, `IVCALC_TRACE_BRENT_ITER`, `IVCALC_TRACE_BRENT_COMPLETE`
8. **Cubic Spline**: `IVCALC_TRACE_SPLINE_ERROR`

Each module has access to both general-purpose probes (for common patterns like convergence) and module-specific probes.

### Adding New USDT Probes

When adding new library functionality that needs logging:

1. **Define probe in `src/ivcalc_trace.h`**:
   ```c
   #define IVCALC_TRACE_MY_EVENT(module_id, param1, param2) \
       DTRACE_PROBE3(IVCALC_PROVIDER, my_event, module_id, param1, param2)
   ```

2. **Use probe in source code**:
   ```c
   #include "ivcalc_trace.h"

   void my_function() {
       // ... code ...
       IVCALC_TRACE_MY_EVENT(MODULE_MY_MODULE, value1, value2);
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
sudo ./scripts/ivcalc-trace monitor ./bazel-bin/examples/example_heat_equation

# Watch convergence behavior
sudo ./scripts/ivcalc-trace monitor ./bazel-bin/examples/example_heat_equation --preset=convergence

# Debug failures
sudo ./scripts/ivcalc-trace monitor ./bazel-bin/examples/example_heat_equation --preset=debug

# Profile performance
sudo ./scripts/ivcalc-trace monitor ./bazel-bin/examples/example_heat_equation --preset=performance
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
sudo bpftrace -e 'usdt::ivcalc:convergence_failed {
    printf("Module %d failed at step %d\n", arg0, arg1);
}' -c './my_program'
```

**Helper tool commands:**

```bash
# Check if binary has USDT support
sudo ./scripts/ivcalc-trace check ./bazel-bin/examples/example_heat_equation

# List all available probes
sudo ./scripts/ivcalc-trace list ./bazel-bin/examples/example_heat_equation

# Run specific script
sudo ./scripts/ivcalc-trace run convergence_watch.bt ./my_program
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
