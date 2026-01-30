<!-- SPDX-License-Identifier: MIT -->
# Pure Diffusion Operator Enforcement Design

## Problem Statement

The `evaluate_spatial_operator` function in `src/pde_solver.c` makes a critical assumption when handling Neumann boundary conditions: it assumes the spatial operator is a **pure diffusion operator** of the form:

```
L(u) = D·∂²u/∂x²
```

### Current Implementation Issue

In `src/pde_solver.c:176-195`, the ghost point method for Neumann boundaries estimates the diffusion coefficient `D` from interior points:

```c
// ASSUMPTION: This method assumes a pure diffusion operator L(u) = D·∂²u/∂x²
if (n >= 3 && fabs(u[0] - 2.0*u[1] + u[2]) > 1e-12) {
    double D_estimate = result[1] * dx * dx / (u[0] - 2.0*u[1] + u[2]);
    // Ghost point stencil: L(u)_0 = D * (u_{-1} - 2*u_0 + u_1) / dx²
    result[0] = D_estimate * (2.0*u[1] - 2.0*u[0] - 2.0*dx*g) / (dx * dx);
}
```

This estimation **breaks down** for:
1. **Advection-diffusion operators**: `L(u) = D·∂²u/∂x² - v·∂u/∂x`
2. **Nonlinear operators**: `L(u) = ∇·(D(u)∇u)`
3. **Mixed operators**: `L(u) = D·∂²u/∂x² + f(x,t,u)`
4. **Variable coefficient diffusion** (without explicit D): The estimation may be numerically unstable

### Why This Matters

- **Financial applications** (Black-Scholes): Pure diffusion with σ²/2 as the diffusion coefficient
- **American options**: Requires accurate boundary handling for free boundary problems
- **Correctness**: Silent failures when users provide non-diffusion operators with Neumann BC
- **Robustness**: Numerical instability when denominator approaches zero

## Proposed Solutions

### Option 1: Explicit Diffusion Coefficient Callback (Recommended)

Add a dedicated callback for users to provide the diffusion coefficient explicitly.

#### API Changes

```c
// New callback type: Returns diffusion coefficient D(x,t) at specified locations
typedef void (*DiffusionCoeffFunc)(const double *x, double t, size_t n_points,
                                   double *D, void *user_data);

// Add to PDECallbacks structure
struct PDECallbacks {
    InitialConditionFunc initial_condition;
    BoundaryConditionFunc left_boundary;
    BoundaryConditionFunc right_boundary;
    SpatialOperatorFunc spatial_operator;
    DiffusionCoeffFunc diffusion_coeff;    // NEW: Required for Neumann BC
    JumpConditionFunc jump_condition;
    ObstacleFunc obstacle;
    TemporalEventFunc temporal_event;
    size_t n_temporal_events;
    double *temporal_event_times;
    void *user_data;
};
```

#### Implementation Changes

In `evaluate_spatial_operator`:

```c
if (solver->bc_config.left_type == BC_NEUMANN) {
    double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);

    // Check if diffusion coefficient callback is provided
    if (solver->callbacks.diffusion_coeff == nullptr) {
        // ERROR: Neumann BC requires explicit diffusion coefficient
        fprintf(stderr, "ERROR: Neumann BC requires diffusion_coeff callback\n");
        return;
    }

    // Get diffusion coefficient at boundary
    double D_boundary[3];
    solver->callbacks.diffusion_coeff(&solver->grid.x[0], t, 3, D_boundary,
                                     solver->callbacks.user_data);

    // Use exact D instead of estimation
    // Ghost point stencil: L(u)_0 = D * (u_{-1} - 2*u_0 + u_1) / dx²
    double D = D_boundary[0];  // Or average: (D_boundary[0] + D_boundary[1]) / 2.0
    result[0] = D * (2.0*u[1] - 2.0*u[0] - 2.0*dx*g) / (dx * dx);
}
```

#### User Code Example

```c
// User provides diffusion coefficient explicitly
static void heat_diffusion_coeff(const double *x, double t, size_t n_points,
                                 double *D, void *user_data) {
    HeatEquationData *data = (HeatEquationData *)user_data;

    // Constant diffusion
    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        D[i] = data->diffusion_coeff;
    }
}

// With jump conditions (spatially varying)
static void heat_diffusion_coeff_jump(const double *x, double t, size_t n_points,
                                      double *D, void *user_data) {
    HeatEquationData *data = (HeatEquationData *)user_data;

    for (size_t i = 0; i < n_points; i++) {
        D[i] = (x[i] < data->jump_location) ?
               data->diffusion_left : data->diffusion_right;
    }
}

// Setup callbacks
PDECallbacks callbacks = {
    .initial_condition = heat_initial_condition,
    .left_boundary = heat_left_boundary,
    .right_boundary = heat_right_boundary,
    .spatial_operator = heat_spatial_operator,
    .diffusion_coeff = heat_diffusion_coeff,  // NEW
    .jump_condition = nullptr,
    .obstacle = nullptr,
    .temporal_event = nullptr,
    .n_temporal_events = 0,
    .temporal_event_times = nullptr,
    .user_data = &heat_data
};
```

#### Advantages
- ✅ Explicit and clear: Users know they need to provide D
- ✅ Numerically stable: No estimation from potentially noisy data
- ✅ Flexible: Supports constant, spatially-varying, and time-varying D
- ✅ Efficient: Vectorized callback minimizes overhead
- ✅ Enforces correctness: Compile-time requirement for Neumann BC

#### Disadvantages
- ❌ Breaking API change: Existing code needs updates
- ❌ Redundant information: User provides both L(u) and D
- ❌ Still assumes pure diffusion: Doesn't help with advection-diffusion

---

### Option 2: Operator Type Enumeration

Add metadata to declare what type of operator the user is providing.

#### API Changes

```c
// Spatial operator types
typedef enum {
    OPERATOR_PURE_DIFFUSION,         // L(u) = D·∂²u/∂x²
    OPERATOR_ADVECTION_DIFFUSION,    // L(u) = D·∂²u/∂x² + v·∂u/∂x
    OPERATOR_NONLINEAR_DIFFUSION,    // L(u) = ∇·(D(u)∇u)
    OPERATOR_CUSTOM                  // User-defined, no automatic boundary handling
} OperatorType;

// Add to PDECallbacks
struct PDECallbacks {
    // ... existing fields ...
    OperatorType operator_type;           // NEW: Declares operator type
    DiffusionCoeffFunc diffusion_coeff;   // Required for PURE_DIFFUSION
    // ... rest of fields ...
};
```

#### Implementation Changes

```c
void evaluate_spatial_operator(PDESolver *solver, double t,
                              const double * __restrict__ u,
                              double * __restrict__ result) {
    const size_t n = solver->grid.n_points;
    const double dx = solver->grid.dx;

    // Call user's spatial operator
    solver->callbacks.spatial_operator(solver->grid.x, t, u, n, result,
                                      solver->callbacks.user_data);

    // Apply boundary handling based on operator type
    switch (solver->callbacks.operator_type) {
        case OPERATOR_PURE_DIFFUSION:
            apply_neumann_bc_pure_diffusion(solver, t, u, result);
            break;
        case OPERATOR_ADVECTION_DIFFUSION:
            apply_neumann_bc_advection_diffusion(solver, t, u, result);
            break;
        case OPERATOR_CUSTOM:
            // User is responsible for boundary handling in spatial_operator
            break;
        default:
            fprintf(stderr, "ERROR: Unsupported operator type with Neumann BC\n");
            break;
    }
}
```

#### Advantages
- ✅ Documents operator assumptions clearly
- ✅ Enables operator-specific optimizations
- ✅ Validates compatibility at creation time
- ✅ Extensible for future operator types

#### Disadvantages
- ❌ Complex: Multiple code paths to maintain
- ❌ Incomplete: Still need diffusion_coeff for pure diffusion case
- ❌ Restrictive: Users must fit into predefined categories

---

### Option 3: Boundary Stencil Callback

Let users provide custom boundary stencil computation.

#### API Changes

```c
// Boundary stencil callback: Computes L(u) at boundary points
// Parameters: boundary_idx (0 = left, n-1 = right), g (boundary flux)
typedef double (*BoundaryStencilFunc)(size_t boundary_idx, const double *x,
                                     double t, const double *u,
                                     size_t n_points, double g,
                                     void *user_data);

struct PDECallbacks {
    // ... existing fields ...
    BoundaryStencilFunc boundary_stencil;  // Optional: for custom Neumann handling
    // ...
};
```

#### Implementation

```c
if (solver->bc_config.left_type == BC_NEUMANN) {
    double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);

    if (solver->callbacks.boundary_stencil != nullptr) {
        // User provides custom boundary stencil
        result[0] = solver->callbacks.boundary_stencil(0, solver->grid.x, t, u,
                                                       n, g, solver->callbacks.user_data);
    } else {
        // Fallback to estimation (with warning)
        // ... existing code ...
    }
}
```

#### Advantages
- ✅ Maximum flexibility: Users control exact boundary treatment
- ✅ Backward compatible: Optional callback
- ✅ Handles any operator type

#### Disadvantages
- ❌ Complex for users: Requires deep understanding of numerics
- ❌ Error-prone: Easy to get stencil wrong
- ❌ Code duplication: Users repeat boundary logic from spatial_operator

---

### Option 4: Runtime Validation (Diagnostic Only)

Add validation that checks if the operator behaves like pure diffusion.

#### Implementation

```c
// Validation function (called in pde_solver_create)
static bool validate_pure_diffusion_operator(PDESolver *solver) {
    if (solver->bc_config.left_type != BC_NEUMANN &&
        solver->bc_config.right_type != BC_NEUMANN) {
        return true;  // No Neumann BC, validation not needed
    }

    // Test if operator satisfies: L(u) ≈ D·∂²u/∂x²
    // Create test functions and check if operator output matches expected pattern

    // Test 1: Linear function u(x) = ax + b should give L(u) ≈ 0
    // Test 2: Quadratic u(x) = x² should give L(u) ≈ constant

    // ... validation logic ...

    return true;  // Passes validation
}
```

#### Advantages
- ✅ Catches user errors early
- ✅ No API changes needed
- ✅ Educational: Helps users understand assumptions

#### Disadvantages
- ❌ Doesn't solve the problem: Still estimates D incorrectly
- ❌ Complex: Hard to validate reliably for all cases
- ❌ Runtime cost: Adds overhead at solver creation

---

## Recommendation

**Implement Option 1 (Explicit Diffusion Coefficient) + Option 4 (Validation)**

### Rationale

1. **Option 1** solves the core problem:
   - Removes numerical instability from D estimation
   - Makes pure diffusion assumption explicit and enforced
   - Aligns with the library's use case (financial PDEs are typically pure diffusion)

2. **Option 4** adds safety:
   - Validates that spatial_operator is consistent with diffusion_coeff
   - Catches user errors (e.g., providing wrong D value)
   - Can be disabled in release builds for performance

### Implementation Plan

**Phase 1: Add callback (backward compatible)**
1. Add `diffusion_coeff` to `PDECallbacks` as **optional** (nullable)
2. Update `evaluate_spatial_operator` to use explicit D when provided
3. Fall back to estimation if callback is nullptr (with warning)
4. Update all examples to provide explicit D

**Phase 2: Make it mandatory (breaking change)**
1. Return error if Neumann BC is used without diffusion_coeff
2. Update CLAUDE.md to document requirement
3. Version bump to indicate breaking change

**Phase 3: Add validation**
1. Implement validation in `pde_solver_create`
2. Check consistency between spatial_operator and diffusion_coeff
3. Provide helpful error messages

### Migration Guide for Users

**Before:**
```c
PDECallbacks callbacks = {
    .spatial_operator = heat_spatial_operator,
    // ... other callbacks ...
};

BoundaryConfig bc = {
    .left_type = BC_NEUMANN,  // Used to work (with warnings)
};
```

**After:**
```c
// Add diffusion coefficient callback
static void diffusion_coeff(const double *x, double t, size_t n,
                            double *D, void *user_data) {
    double D_val = *(double*)user_data;
    for (size_t i = 0; i < n; i++) {
        D[i] = D_val;
    }
}

PDECallbacks callbacks = {
    .spatial_operator = heat_spatial_operator,
    .diffusion_coeff = diffusion_coeff,  // NEW: Required for Neumann BC
    // ... other callbacks ...
};

BoundaryConfig bc = {
    .left_type = BC_NEUMANN,  // Now requires diffusion_coeff
};
```

---

## Alternative: Document and Warn (Minimal Change)

If breaking API changes are not acceptable, a minimal approach is:

1. **Document the limitation clearly** in CLAUDE.md and pde_solver.h
2. **Add runtime warning** when Neumann BC is used:
   ```c
   fprintf(stderr, "WARNING: Neumann BC assumes pure diffusion operator L(u) = D·∂²u/∂x²\n");
   fprintf(stderr, "         For other operators, consider Dirichlet BC or custom implementation\n");
   ```
3. **Improve estimation robustness**:
   - Check for division by zero
   - Add fallback methods
   - Use multiple interior points for averaging

This approach maintains backward compatibility but doesn't solve the fundamental issue.

---

## References

- Ghost Point Method: LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations" (2007), Chapter 2.12
- TR-BDF2 Solver: Ascher, Ruuth, Wetton, "Implicit-Explicit Methods for Time-Dependent PDE's" (1995)
- Current Implementation: `src/pde_solver.c:176-195`
