# Boundary Condition Refactoring Proposal

## Current Problem

The boundary condition handling is scattered throughout `pde_solver.c` with repeated if-else chains:

```c
if (solver->bc_config.left_type == BC_DIRICHLET) {
    // Dirichlet handling
} else if (solver->bc_config.left_type == BC_NEUMANN) {
    // Neumann handling
} else if (solver->bc_config.left_type == BC_ROBIN) {
    // Robin handling
}
```

This pattern appears in multiple locations:
- `apply_boundary_conditions()` - line 14, 29
- `evaluate_spatial_operator()` - line 69, 89
- `assemble_jacobian()` - line 116, 123, 148, 198
- `compute_residual()` - line 231, 246

**Issues:**
1. Code duplication and maintenance burden
2. Adding new BC types requires changing multiple locations
3. Hard to test individual BC implementations
4. Messy coupling between BC logic and solver internals

## Proposed Solution: Function Pointer Dispatch

Abstract each boundary condition type with a set of handler functions:

```c
// Boundary condition handler interface
typedef struct {
    // Apply boundary value to solution array
    void (*apply)(PDESolver *solver, double t, double *u, bool is_left);

    // Compute spatial operator contribution at boundary
    void (*eval_operator)(PDESolver *solver, double t, const double *u,
                         double *Lu, bool is_left);

    // Assemble Jacobian row for boundary point
    void (*assemble_jacobian)(PDESolver *solver, double t, const double *u_new,
                             const double *Lu, double coeff_dt,
                             double *diag, double *upper, double *lower,
                             bool is_left);

    // Compute residual for boundary point
    void (*compute_residual)(PDESolver *solver, const double *rhs,
                            const double *u_old, const double *u_new,
                            const double *Lu, double coeff_dt,
                            double *residual, bool is_left);
} BCHandler;
```

### Implementation Strategy

1. **Create handler instances for each BC type:**

```c
// In pde_solver.c
static const BCHandler dirichlet_handler = {
    .apply = apply_dirichlet,
    .eval_operator = eval_operator_dirichlet,
    .assemble_jacobian = assemble_jacobian_dirichlet,
    .compute_residual = compute_residual_dirichlet
};

static const BCHandler neumann_handler = {
    .apply = apply_neumann,
    .eval_operator = eval_operator_neumann,
    .assemble_jacobian = assemble_jacobian_neumann,
    .compute_residual = compute_residual_neumann
};

static const BCHandler robin_handler = {
    .apply = apply_robin,
    .eval_operator = eval_operator_robin,
    .assemble_jacobian = assemble_jacobian_robin,
    .compute_residual = compute_residual_robin
};
```

2. **Add handler pointers to PDESolver:**

```c
struct PDESolver {
    // ... existing fields ...

    const BCHandler *left_bc_handler;
    const BCHandler *right_bc_handler;
};
```

3. **Initialize handlers in pde_solver_create():**

```c
// Select left BC handler
switch (bc_config->left_type) {
    case BC_DIRICHLET:
        solver->left_bc_handler = &dirichlet_handler;
        break;
    case BC_NEUMANN:
        solver->left_bc_handler = &neumann_handler;
        break;
    case BC_ROBIN:
        solver->left_bc_handler = &robin_handler;
        break;
}
```

4. **Simplify call sites:**

```c
// Before (with if-else chain):
if (solver->bc_config.left_type == BC_DIRICHLET) {
    u[0] = g;
} else if (solver->bc_config.left_type == BC_NEUMANN) {
    // Ghost point method
    ...
} else if (solver->bc_config.left_type == BC_ROBIN) {
    ...
}

// After (single dispatch):
solver->left_bc_handler->apply(solver, t, u, true);
```

## Benefits

1. **Separation of Concerns**: Each BC type is self-contained
2. **Extensibility**: Add new BC types by creating new handlers
3. **Testability**: Can test BC implementations independently
4. **Maintainability**: Changes to BC logic isolated to handler functions
5. **Performance**: Function pointers resolved at creation time (no runtime overhead vs if-else)
6. **Readability**: Call sites become simpler and more declarative

## Migration Path

1. Create BC handler structure and function typedefs
2. Implement handler functions for existing BC types (extract from if-else blocks)
3. Add handler pointers to PDESolver
4. Update pde_solver_create() to select handlers
5. Replace if-else chains with handler dispatches
6. Test thoroughly to ensure behavior unchanged
7. Consider adding new BC types (e.g., periodic, mixed)

## Example: Dirichlet Handler Implementation

```c
static void apply_dirichlet(PDESolver *solver, double t, double *u, bool is_left) {
    double g = is_left ?
        solver->callbacks.left_boundary(t, solver->callbacks.user_data) :
        solver->callbacks.right_boundary(t, solver->callbacks.user_data);

    if (is_left) {
        u[0] = g;
    } else {
        u[solver->grid.n_points - 1] = g;
    }
}

static void eval_operator_dirichlet(PDESolver *solver, double t, const double *u,
                                    double *Lu, bool is_left) {
    // Dirichlet BC: boundary point not evolved by PDE
    size_t idx = is_left ? 0 : solver->grid.n_points - 1;
    Lu[idx] = 0.0;
}

static void assemble_jacobian_dirichlet(PDESolver *solver, double t,
                                        const double *u_new, const double *Lu,
                                        double coeff_dt, double *diag,
                                        double *upper, double *lower, bool is_left) {
    size_t idx = is_left ? 0 : solver->grid.n_points - 1;

    // Row i: u_i = g (identity equation)
    diag[idx] = 1.0;
    if (is_left) {
        upper[idx] = 0.0;
    } else {
        lower[idx - 1] = 0.0;
    }
}

static void compute_residual_dirichlet(PDESolver *solver, const double *rhs,
                                       const double *u_old, const double *u_new,
                                       const double *Lu, double coeff_dt,
                                       double *residual, bool is_left) {
    size_t idx = is_left ? 0 : solver->grid.n_points - 1;
    double g = is_left ?
        solver->callbacks.left_boundary(solver->time.t_start + solver->current_step * solver->time.dt,
                                       solver->callbacks.user_data) :
        solver->callbacks.right_boundary(solver->time.t_start + solver->current_step * solver->time.dt,
                                        solver->callbacks.user_data);

    residual[idx] = u_new[idx] - g;
}
```

## Alternative: Strategy Pattern (More OOP-style)

If we want even more flexibility, we could make BC handlers fully dynamic:

```c
typedef struct BCStrategy BCStrategy;

struct BCStrategy {
    BoundaryType type;
    void *config_data;  // BC-specific configuration

    void (*apply)(const BCStrategy *self, PDESolver *solver, double t,
                 double *u, bool is_left);
    // ... other methods ...
};

// Usage:
BCStrategy *create_neumann_strategy(void);
BCStrategy *create_dirichlet_strategy(void);
BCStrategy *create_robin_strategy(double a, double b);
```

This allows runtime creation of BC strategies with different configurations, but adds heap allocation overhead.

## Recommendation

Use the **function pointer dispatch** approach:
- Simpler than full strategy pattern
- No heap allocation overhead
- Sufficient flexibility for our needs
- Clean separation of BC implementations
- Easy to test and maintain

The strategy pattern could be added later if we need more dynamic BC configuration.
