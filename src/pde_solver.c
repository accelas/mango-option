#include "pde_solver.h"
#include "ivcalc_trace.h"
#include "tridiagonal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// GCC/Clang restrict keyword for optimization
#ifndef __restrict__
#define __restrict__ restrict
#endif

// SIMD alignment boundary (AVX-512 requires 64-byte alignment)
#define SIMD_ALIGNMENT 64

// Apply boundary conditions
static void apply_boundary_conditions(PDESolver *solver, double t, double *u) {
    const size_t n = solver->grid.n_points;
    const double dx = solver->grid.dx;

    // Left boundary
    if (solver->bc_config.left_type == BC_DIRICHLET) {
        u[0] = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
    } else if (solver->bc_config.left_type == BC_NEUMANN) {
        // du/dx = g => u[0] = u[1] - dx*g (first-order)
        double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
        u[0] = u[1] - dx * g;
    } else if (solver->bc_config.left_type == BC_ROBIN) {
        // a*u + b*du/dx = g
        double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
        double a = solver->bc_config.left_robin_a;
        double b = solver->bc_config.left_robin_b;
        u[0] = (g - b * (u[1] - u[0]) / dx) / a;
    }

    // Right boundary
    if (solver->bc_config.right_type == BC_DIRICHLET) {
        u[n - 1] = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
    } else if (solver->bc_config.right_type == BC_NEUMANN) {
        double g = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
        u[n - 1] = u[n - 2] + dx * g;
    } else if (solver->bc_config.right_type == BC_ROBIN) {
        double g = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
        double a = solver->bc_config.right_robin_a;
        double b = solver->bc_config.right_robin_b;
        u[n - 1] = (g - b * (u[n - 1] - u[n - 2]) / dx) / a;
    }

    // Apply obstacle condition if provided
    if (solver->callbacks.obstacle != nullptr) {
        // Reuse workspace array for psi (no allocation overhead)
        double *psi = solver->u_temp;
        solver->callbacks.obstacle(solver->grid.x, t, n, psi, solver->callbacks.user_data);

        for (size_t i = 0; i < n; i++) {
            if (u[i] < psi[i]) {
                u[i] = psi[i];
            }
        }
    }
}

// Evaluate spatial operator for all points
static void evaluate_spatial_operator(PDESolver *solver, double t,
                                      const double * __restrict__ u,
                                      double * __restrict__ result) {
    const size_t n = solver->grid.n_points;
    const double dx = solver->grid.dx;

    // Call vectorized spatial operator
    solver->callbacks.spatial_operator(solver->grid.x, t, u, n, result,
                                      solver->callbacks.user_data);

    // Apply ghost point method for Neumann boundaries
    // The user's callback may set Lu[0] = Lu[n-1] = 0, but for conservation
    // we need to compute L(u) properly at boundaries using ghost points

    if (solver->bc_config.left_type == BC_NEUMANN) {
        // Ghost point: u_{-1} = u_1 (for zero flux du/dx = 0)
        // More generally: u_{-1} = u_1 - 2*dx*g where du/dx = g
        double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
        double D = solver->callbacks.diffusion_coeff;

        // Check if explicit diffusion coefficient is provided
        if (!isnan(D)) {
            // Use explicit diffusion coefficient (pure diffusion operator L(u) = D·∂²u/∂x²)
            // Ghost point stencil: L(u)_0 = D * (u_{-1} - 2*u_0 + u_1) / dx²
            //                             = D * (u_1 - 2*dx*g - 2*u_0 + u_1) / dx²
            //                             = D * (2*u_1 - 2*u_0 - 2*dx*g) / dx²
            result[0] = D * (2.0*u[1] - 2.0*u[0] - 2.0*dx*g) / (dx * dx);
        } else {
            // Fall back to estimation for variable/non-constant diffusion
            // ASSUMPTION: This assumes pure diffusion operator L(u) = D·∂²u/∂x²
            // For advection-diffusion or nonlinear operators, this may be inaccurate
            if (n >= 3 && fabs(u[0] - 2.0*u[1] + u[2]) > 1e-12) {
                double D_estimate = result[1] * dx * dx / (u[0] - 2.0*u[1] + u[2]);
                result[0] = D_estimate * (2.0*u[1] - 2.0*u[0] - 2.0*dx*g) / (dx * dx);
            } else {
                // Fallback for edge cases
                result[0] = 0.0;
            }
        }
    }

    if (solver->bc_config.right_type == BC_NEUMANN) {
        double g = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
        double D = solver->callbacks.diffusion_coeff;

        // Check if explicit diffusion coefficient is provided
        if (!isnan(D)) {
            // Use explicit diffusion coefficient (pure diffusion operator L(u) = D·∂²u/∂x²)
            // Ghost point: u_n = u_{n-2} + 2*dx*g
            // L(u)_{n-1} = D * (u_{n-2} - 2*u_{n-1} + u_n) / dx²
            result[n-1] = D * (2.0*u[n-2] - 2.0*u[n-1] + 2.0*dx*g) / (dx * dx);
        } else {
            // Fall back to estimation for variable/non-constant diffusion
            // ASSUMPTION: Same as left boundary - assumes pure diffusion operator
            if (n >= 3 && fabs(u[n-3] - 2.0*u[n-2] + u[n-1]) > 1e-12) {
                double D_estimate = result[n-2] * dx * dx / (u[n-3] - 2.0*u[n-2] + u[n-1]);
                result[n-1] = D_estimate * (2.0*u[n-2] - 2.0*u[n-1] + 2.0*dx*g) / (dx * dx);
            } else {
                result[n-1] = 0.0;
            }
        }
    }
}

// Solve implicit system: (I - coeff*dt*L)*u_new = rhs
// Uses linearized Newton iteration with tridiagonal solver
static int solve_implicit_step(PDESolver *solver, double t, double coeff_dt,
                               const double *rhs, double *u_new, size_t step) {
    const size_t n = solver->grid.n_points;
    const size_t max_iter = solver->trbdf2_config.max_iter;
    const double tol = solver->trbdf2_config.tolerance;
    const double eps = 1e-7;  // Finite difference epsilon (balance between truncation and roundoff error)

    // Note: u_new is pre-initialized by caller with appropriate initial guess
    // (u_current for Stage 1, u_stage for Stage 2)

    // Initialize boundary values to satisfy constraints (needed for Jacobian computation)
    if (solver->bc_config.left_type == BC_DIRICHLET) {
        u_new[0] = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
    } else if (solver->bc_config.left_type == BC_NEUMANN) {
        double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
        u_new[0] = u_new[1] - solver->grid.dx * g;
    }

    if (solver->bc_config.right_type == BC_DIRICHLET) {
        u_new[n-1] = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
    } else if (solver->bc_config.right_type == BC_NEUMANN) {
        double g = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
        u_new[n-1] = u_new[n-2] + solver->grid.dx * g;
    }

    // Use pre-allocated workspace arrays
    double *u_old = solver->u_old;
    double *Lu = solver->Lu;
    double *Lu_pert = solver->u_temp;  // Reuse u_temp for perturbed L evaluation
    double *diag = solver->matrix_diag;
    double *upper = solver->matrix_upper;
    double *lower = solver->matrix_lower;
    double rel_error = 0.0;

    // Compute Jacobian once at the beginning (assumes nearly linear operator)
    // For truly nonlinear problems, this could be recomputed every few iterations
    evaluate_spatial_operator(solver, t, u_new, Lu);

    // Build Jacobian matrix using finite differences
    // For Neumann BC, we now apply the PDE at ALL points (including boundaries)
    // using ghost point method

    // Left boundary (i = 0)
    if (solver->bc_config.left_type == BC_DIRICHLET) {
        // Row 0: u[0] = g(t) (algebraic constraint)
        diag[0] = 1.0;
        upper[0] = 0.0;
    } else if (solver->bc_config.left_type == BC_NEUMANN) {
        // Row 0: PDE with ghost point stencil
        // Compute Jacobian entries via finite differences
        double u_save = u_new[0];
        u_new[0] = u_save + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dL0_du0 = (Lu_pert[0] - Lu[0]) / eps;
        u_new[0] = u_save;

        u_new[1] = u_new[1] + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dL0_du1 = (Lu_pert[0] - Lu[0]) / eps;
        u_new[1] = u_new[1] - eps;

        diag[0] = 1.0 - coeff_dt * dL0_du0;
        upper[0] = -coeff_dt * dL0_du1;
    }

    // Interior points (i = 1 to n-2)
    for (size_t i = 1; i < n - 1; i++) {
        // Diagonal element: ∂L_i/∂u_i
        double u_save = u_new[i];
        u_new[i] = u_save + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dLi_dui = (Lu_pert[i] - Lu[i]) / eps;
        u_new[i] = u_save;

        // Build system matrix: (I - coeff_dt * J)
        diag[i] = 1.0 - coeff_dt * dLi_dui;

        // Lower diagonal: ∂L_i/∂u_{i-1}
        u_new[i-1] = u_new[i-1] + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dLi_duim1 = (Lu_pert[i] - Lu[i]) / eps;
        u_new[i-1] = u_new[i-1] - eps;
        lower[i-1] = -coeff_dt * dLi_duim1;

        // Upper diagonal: ∂L_i/∂u_{i+1}
        u_new[i+1] = u_new[i+1] + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dLi_duip1 = (Lu_pert[i] - Lu[i]) / eps;
        u_new[i+1] = u_new[i+1] - eps;
        upper[i] = -coeff_dt * dLi_duip1;
    }

    // Right boundary (i = n-1)
    if (solver->bc_config.right_type == BC_DIRICHLET) {
        // Row n-1: u[n-1] = g(t) (algebraic constraint)
        diag[n-1] = 1.0;
        lower[n-2] = 0.0;
    } else if (solver->bc_config.right_type == BC_NEUMANN) {
        // Row n-1: PDE with ghost point stencil
        u_new[n-2] = u_new[n-2] + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dLn1_dun2 = (Lu_pert[n-1] - Lu[n-1]) / eps;
        u_new[n-2] = u_new[n-2] - eps;

        double u_save = u_new[n-1];
        u_new[n-1] = u_save + eps;
        evaluate_spatial_operator(solver, t, u_new, Lu_pert);
        double dLn1_dun1 = (Lu_pert[n-1] - Lu[n-1]) / eps;
        u_new[n-1] = u_save;

        lower[n-2] = -coeff_dt * dLn1_dun2;
        diag[n-1] = 1.0 - coeff_dt * dLn1_dun1;
    }

    for (size_t iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u_new, n * sizeof(double));

        // Evaluate L(u_old)
        evaluate_spatial_operator(solver, t, u_old, Lu);

        // Compute residual: r = rhs - (u_old - coeff_dt * L(u_old))
        //                      = rhs - u_old + coeff_dt * L(u_old)
        double *residual = Lu_pert;  // Reuse Lu_pert as residual storage

        // Boundary residuals - use PDE for Neumann, constraint for Dirichlet
        // Left boundary
        if (solver->bc_config.left_type == BC_DIRICHLET) {
            double g = solver->callbacks.left_boundary(t, solver->callbacks.user_data);
            residual[0] = g - u_old[0];
        } else if (solver->bc_config.left_type == BC_NEUMANN) {
            // Use PDE at boundary (same as interior)
            residual[0] = rhs[0] - u_old[0] + coeff_dt * Lu[0];
        }

        // Interior points
        #pragma omp simd
        for (size_t i = 1; i < n - 1; i++) {
            residual[i] = fma(coeff_dt, Lu[i], rhs[i] - u_old[i]);
        }

        // Right boundary
        if (solver->bc_config.right_type == BC_DIRICHLET) {
            double g = solver->callbacks.right_boundary(t, solver->callbacks.user_data);
            residual[n-1] = g - u_old[n-1];
        } else if (solver->bc_config.right_type == BC_NEUMANN) {
            // Use PDE at boundary (same as interior)
            residual[n-1] = rhs[n-1] - u_old[n-1] + coeff_dt * Lu[n-1];
        }

        // Solve for correction: (I - coeff_dt * J) * δu = residual
        double *delta_u = u_new;  // Use u_new to store δu temporarily
        solve_tridiagonal(n, lower, diag, upper, residual, delta_u, solver->tridiag_workspace);

        // Update: u_new = u_old + δu
        #pragma omp simd
        for (size_t i = 0; i < n; i++) {
            u_new[i] = u_old[i] + delta_u[i];
        }

        // Apply obstacle condition if provided (BCs already enforced in linear system)
        if (solver->callbacks.obstacle != nullptr) {
            double *psi = Lu;  // Reuse Lu workspace for psi
            solver->callbacks.obstacle(solver->grid.x, t, n, psi, solver->callbacks.user_data);
            for (size_t i = 0; i < n; i++) {
                if (u_new[i] < psi[i]) {
                    u_new[i] = psi[i];
                }
            }
        }

        // Check convergence
        double error = 0.0;
        double norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = u_new[i] - u_old[i];
            error += diff * diff;
            norm += u_new[i] * u_new[i];
        }
        error = sqrt(error / n);
        norm = sqrt(norm / n);

        rel_error = (norm > 1e-12) ? error / (norm + 1e-12) : error;

        // Trace iteration progress
        MANGO_TRACE_PDE_IMPLICIT_ITER(step, iter, rel_error, tol);

        if (rel_error < tol || error < tol) {
            MANGO_TRACE_PDE_IMPLICIT_CONVERGED(step, iter, rel_error);
            return 0;
        }
    }

    MANGO_TRACE_PDE_IMPLICIT_FAILED(step, max_iter, rel_error);
    return -1;
}

// Utility functions

SpatialGrid pde_create_grid(double x_min, double x_max, size_t n_points) {
    SpatialGrid grid;
    grid.x_min = x_min;
    grid.x_max = x_max;
    grid.n_points = n_points;
    grid.dx = (x_max - x_min) / (n_points - 1);
    grid.x = malloc(n_points * sizeof(double));

    #pragma omp simd
    for (size_t i = 0; i < n_points; i++) {
        grid.x[i] = x_min + i * grid.dx;
    }

    return grid;
}

void pde_free_grid(SpatialGrid *grid) {
    if (grid->x != nullptr) {
        free(grid->x);
        grid->x = nullptr;
    }
}

TRBDF2Config pde_default_trbdf2_config(void) {
    TRBDF2Config config;
    config.gamma = 2.0 - sqrt(2.0);  // ≈ 0.5858
    config.max_iter = 20;  // Newton iteration converges quickly for linear/near-linear problems
    config.tolerance = 1e-8;  // Tighter tolerance for better accuracy
    return config;
}

BoundaryConfig pde_default_boundary_config(void) {
    BoundaryConfig config;
    config.left_type = BC_DIRICHLET;
    config.right_type = BC_DIRICHLET;
    config.left_robin_a = 1.0;
    config.left_robin_b = 0.0;
    config.right_robin_a = 1.0;
    config.right_robin_b = 0.0;
    return config;
}

// Core API

PDESolver* pde_solver_create(SpatialGrid *grid,
                              const TimeDomain *time,
                              const BoundaryConfig *bc_config,
                              const TRBDF2Config *trbdf2_config,
                              const PDECallbacks *callbacks) {
    PDESolver *solver = malloc(sizeof(PDESolver));

    // Take ownership of grid (shallow copy, transfer ownership)
    solver->grid = *grid;
    grid->x = nullptr;  // Prevent double-free, ownership transferred to solver

    solver->time = *time;
    solver->bc_config = *bc_config;
    solver->trbdf2_config = *trbdf2_config;
    solver->callbacks = *callbacks;

    // Validate Robin boundary condition coefficients
    if (bc_config->left_type == BC_ROBIN && fabs(bc_config->left_robin_a) < 1e-15) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_PDE_SOLVER, 1, bc_config->left_robin_a, 1e-15);
        free(solver);
        return nullptr;
    }
    if (bc_config->right_type == BC_ROBIN && fabs(bc_config->right_robin_a) < 1e-15) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_PDE_SOLVER, 2, bc_config->right_robin_a, 1e-15);
        free(solver);
        return nullptr;
    }

    // Allocate single workspace buffer for all arrays (better cache locality)
    // Need arrays totaling 12n doubles:
    //   - Solution: u_current, u_next, u_stage, rhs (4n)
    //   - Matrix: matrix_diag, matrix_upper, matrix_lower (3n)
    //   - Temps: u_old, Lu, u_temp (3n)
    //   - Tridiagonal workspace: c_prime, d_prime (2n)
    const size_t n = grid->n_points;
    const size_t workspace_size = 12 * n;

    // Use aligned allocation for SIMD vectorization (64-byte alignment for AVX-512)
    // Each array starts at aligned boundary by padding n to alignment
    const size_t alignment = SIMD_ALIGNMENT;
    const size_t n_aligned = ((n * sizeof(double) + alignment - 1) / alignment) * alignment / sizeof(double);
    const size_t workspace_aligned_size = 12 * n_aligned;

    solver->workspace = aligned_alloc(alignment, workspace_aligned_size * sizeof(double));
    if (solver->workspace == nullptr) {
        // Fallback to regular malloc if aligned_alloc fails
        solver->workspace = malloc(workspace_size * sizeof(double));
        if (solver->workspace == nullptr) {
            // Both allocations failed
            MANGO_TRACE_VALIDATION_ERROR(MODULE_PDE_SOLVER, 0, workspace_size, 0.0);
            free(solver);
            return nullptr;
        }
    }

    // Slice workspace into individual arrays (each aligned)
    size_t offset = 0;
    solver->u_current = solver->workspace + offset; offset += n_aligned;
    solver->u_next = solver->workspace + offset; offset += n_aligned;
    solver->u_stage = solver->workspace + offset; offset += n_aligned;
    solver->rhs = solver->workspace + offset; offset += n_aligned;
    solver->matrix_diag = solver->workspace + offset; offset += n_aligned;
    solver->matrix_upper = solver->workspace + offset; offset += n_aligned;
    solver->matrix_lower = solver->workspace + offset; offset += n_aligned;
    solver->u_old = solver->workspace + offset; offset += n_aligned;
    solver->Lu = solver->workspace + offset; offset += n_aligned;
    solver->u_temp = solver->workspace + offset; offset += n_aligned;
    solver->tridiag_workspace = solver->workspace + offset; offset += 2 * n_aligned;  // 2n for c_prime + d_prime

    return solver;
}

void pde_solver_destroy(PDESolver *solver) {
    if (solver == nullptr) return;

    pde_free_grid(&solver->grid);
    free(solver->workspace);  // Single free for all arrays
    free(solver);
}

void pde_solver_initialize(PDESolver *solver) {
    const size_t n = solver->grid.n_points;

    // Call vectorized initial condition
    solver->callbacks.initial_condition(solver->grid.x, n, solver->u_current,
                                       solver->callbacks.user_data);

    apply_boundary_conditions(solver, solver->time.t_start, solver->u_current);
}

// Internal version with step tracking for tracing
static int pde_solver_step_internal(PDESolver *solver, double t_current, size_t step) {
    const double dt = solver->time.dt;
    const double gamma = solver->trbdf2_config.gamma;
    const size_t n = solver->grid.n_points;

    // Hint to compiler that arrays are 64-byte aligned for SIMD
    double *u_current = __builtin_assume_aligned(solver->u_current, SIMD_ALIGNMENT);
    double *u_stage = __builtin_assume_aligned(solver->u_stage, SIMD_ALIGNMENT);
    double *u_next = __builtin_assume_aligned(solver->u_next, SIMD_ALIGNMENT);
    double *rhs = __builtin_assume_aligned(solver->rhs, SIMD_ALIGNMENT);

    // TR-BDF2 scheme
    // Stage 1: Trapezoidal rule from t_n to t_n + γ·dt
    // u* = u^n + (γ·dt/2) · [L(u^n) + L(u*)]
    // Rearranging: u* - (γ·dt/2)·L(u*) = u^n + (γ·dt/2)·L(u^n)

    // Reuse workspace Lu array (no allocation overhead)
    double *Lu_n = __builtin_assume_aligned(solver->Lu, SIMD_ALIGNMENT);
    evaluate_spatial_operator(solver, t_current, u_current, Lu_n);

    // RHS for stage 1
    const double gamma_dt_half = gamma * dt * 0.5;
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        rhs[i] = fma(gamma_dt_half, Lu_n[i], u_current[i]);
    }

    // Initialize u_stage with u_current as initial guess
    memcpy(u_stage, u_current, n * sizeof(double));

    // Solve implicit equation for stage 1
    int status = solve_implicit_step(solver, t_current + gamma * dt,
                                     gamma * dt / 2.0, solver->rhs, solver->u_stage, step);
    if (status != 0) {
        return status;
    }

    // Stage 2: BDF2 from t_n to t_n+1
    // Standard TR-BDF2 formulation (Ascher, Ruuth, Wetton 1995):
    // u^{n+1} - [(1-γ)Δt/(2-γ)]L(u^{n+1}) = [1/(γ(2-γ))]u^* - [(1-γ)²/(γ(2-γ))]u^n

    const double one_minus_gamma = 1.0 - gamma;
    const double two_minus_gamma = 2.0 - gamma;
    const double denom = gamma * two_minus_gamma;
    const double coeff = one_minus_gamma * dt / two_minus_gamma;

    // RHS for stage 2
    const double inv_denom = 1.0 / denom;
    const double neg_coeff = -(one_minus_gamma * one_minus_gamma * inv_denom);
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        rhs[i] = fma(neg_coeff, u_current[i], u_stage[i] * inv_denom);
    }

    // Initialize u_next with u_stage as initial guess
    memcpy(u_next, u_stage, n * sizeof(double));

    // Solve implicit equation for stage 2
    status = solve_implicit_step(solver, t_current + dt, coeff,
                                solver->rhs, solver->u_next, step);

    if (status == 0) {
        // Update current solution
        memcpy(u_current, u_next, n * sizeof(double));
    }

    return status;
}

int pde_solver_step(PDESolver *solver, double t_current) {
    // Public API: call internal version with step=0 (not tracked)
    return pde_solver_step_internal(solver, t_current, 0);
}

int pde_solver_solve(PDESolver *solver) {
    double t = solver->time.t_start;
    size_t next_event_idx = 0; // Track next event to check

    // Trace solver start
    MANGO_TRACE_PDE_START(solver->time.t_start, solver->time.t_end,
                           solver->time.dt, solver->time.n_steps);

    for (size_t step = 0; step < solver->time.n_steps; step++) {
        double t_prev = t;

        // Use internal version with step tracking for better tracing
        int status = pde_solver_step_internal(solver, t, step);
        if (status != 0) {
            // Note: convergence failure already traced in solve_implicit_step
            return status;
        }

        t += solver->time.dt;

        // Handle temporal events if callback is provided and events are registered
        if (solver->callbacks.temporal_event != nullptr &&
            solver->callbacks.n_temporal_events > 0 &&
            solver->callbacks.temporal_event_times != nullptr) {

            // Collect all events that occurred in (t_prev, t]
            size_t events_triggered[16]; // Static array for up to 16 events per step
            size_t n_triggered = 0;

            while (next_event_idx < solver->callbacks.n_temporal_events &&
                   n_triggered < 16) {
                double event_time = solver->callbacks.temporal_event_times[next_event_idx];

                // Check if event occurred in this time step
                if (event_time > t_prev && event_time <= t) {
                    events_triggered[n_triggered++] = next_event_idx;
                    next_event_idx++;
                } else if (event_time > t) {
                    // Event is in the future, stop checking
                    break;
                } else {
                    // Event is in the past (shouldn't happen if sorted), skip it
                    next_event_idx++;
                }
            }

            // Call callback if any events were triggered
            if (n_triggered > 0) {
                solver->callbacks.temporal_event(t, solver->grid.x,
                                                solver->grid.n_points,
                                                solver->u_current,
                                                events_triggered,
                                                n_triggered,
                                                solver->callbacks.user_data,
                                                solver->u_temp);  // Reuse u_temp as workspace
            }
        }

        // Trace progress periodically (every 10%)
        if (step % (solver->time.n_steps / 10 + 1) == 0) {
            MANGO_TRACE_PDE_PROGRESS(step, solver->time.n_steps, t);
        }
    }

    // Trace successful completion
    MANGO_TRACE_PDE_COMPLETE(solver->time.n_steps, t);
    return 0;
}

const double* pde_solver_get_solution(const PDESolver *solver) {
    return solver->u_current;
}

const double* pde_solver_get_grid(const PDESolver *solver) {
    return solver->grid.x;
}

// Cubic spline interpolation implementation

double pde_solver_interpolate(const PDESolver *solver, double x_eval) {
    CubicSpline *spline = pde_spline_create(solver->grid.x,
                                            solver->u_current,
                                            solver->grid.n_points);
    if (spline == nullptr) {
        return 0.0;
    }

    double result = pde_spline_eval(spline, x_eval);
    pde_spline_destroy(spline);

    return result;
}
