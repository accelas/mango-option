# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**iv_calc** is a C23-based PDE (Partial Differential Equation) solver using the finite difference method with TR-BDF2 (Two-stage Runge-Kutta with backward differentiation formula) time-stepping scheme. The project emphasizes flexibility through callback-based architecture, allowing users to define custom initial conditions, boundary conditions, jump conditions, and obstacle constraints.

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

# Clean build artifacts
bazel clean
```

## Project Structure

```
iv_calc/
├── MODULE.bazel           # Bazel module with GoogleTest dependency
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
    └── stability_test.cc          # Numerical stability tests
```

## Core Architecture

### Callback-Based Design

The solver uses a callback architecture to maximize flexibility:

1. **Initial Condition**: `double (*)(double x, void *user_data)`
   - Defines u(x, t=0)

2. **Boundary Conditions**: `double (*)(double t, void *user_data)`
   - Supports Dirichlet, Neumann, and Robin boundary conditions
   - Separate callbacks for left and right boundaries

3. **Spatial Operator**: `double (*)(const double *x, double t, const double *u, size_t idx, size_t n_points, void *user_data)`
   - Defines the spatial discretization L(u) in du/dt = L(u)
   - User implements finite difference stencils

4. **Jump Condition** (optional): `bool (*)(double x, double *jump_value, void *user_data)`
   - Handles discontinuous coefficients at interfaces

5. **Obstacle Condition** (optional): `double (*)(double x, double t, void *user_data)`
   - Enforces u(x,t) >= ψ(x,t) for variational inequalities

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

### Cubic Spline Interpolation

Natural cubic splines allow evaluation of solutions at arbitrary off-grid points:
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

See `examples/example_heat_equation.c` for complete examples including:
- Basic heat equation
- Jump conditions (discontinuous diffusion coefficients)
- Obstacle conditions (American option pricing)

### Implementing Spatial Operators

For second-order spatial derivatives (e.g., diffusion):
```c
// d²u/dx² using central differences
double d2u_dx2 = (u[idx-1] - 2.0*u[idx] + u[idx+1]) / (dx*dx);
```

For first-order derivatives (e.g., advection):
```c
// du/dx using upwind scheme
double du_dx = (u[idx] - u[idx-1]) / dx;  // for positive velocity
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
