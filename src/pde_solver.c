#include "pde_solver.h"
#include "ivcalc_trace.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
static void evaluate_spatial_operator(PDESolver *solver, double t, const double *u,
                                      double *result) {
    const size_t n = solver->grid.n_points;

    // Call vectorized spatial operator
    solver->callbacks.spatial_operator(solver->grid.x, t, u, n, result,
                                      solver->callbacks.user_data);
}

// Solve implicit system: (I - coeff*dt*L)*u_new = rhs
// Uses fixed-point iteration for nonlinear cases
static int solve_implicit_step(PDESolver *solver, double t, double coeff_dt,
                               const double *rhs, double *u_new, size_t step) {
    const size_t n = solver->grid.n_points;
    const size_t max_iter = solver->trbdf2_config.max_iter;
    const double tol = solver->trbdf2_config.tolerance;

    // Initialize with rhs (better initial guess)
    memcpy(u_new, rhs, n * sizeof(double));
    apply_boundary_conditions(solver, t, u_new);

    // Use pre-allocated workspace arrays (no malloc overhead)
    double *u_old = solver->u_old;
    double *Lu = solver->Lu;
    double *u_temp = solver->u_temp;
    const double omega = 0.7;  // Relaxation parameter (under-relaxation)
    double rel_error = 0.0;  // Track relative error for convergence monitoring

    for (size_t iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u_new, n * sizeof(double));

        // Evaluate L(u_old)
        evaluate_spatial_operator(solver, t, u_old, Lu);

        // u_temp = rhs + coeff_dt * L(u_old)
        #pragma omp simd
        for (size_t i = 0; i < n; i++) {
            u_temp[i] = rhs[i] + coeff_dt * Lu[i];
        }

        // Apply under-relaxation: u_new = omega * u_temp + (1-omega) * u_old
        #pragma omp simd
        for (size_t i = 0; i < n; i++) {
            u_new[i] = omega * u_temp[i] + (1.0 - omega) * u_old[i];
        }

        apply_boundary_conditions(solver, t, u_new);

        // Check convergence using relative error
        double error = 0.0;
        double norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = u_new[i] - u_old[i];
            error += diff * diff;
            norm += u_new[i] * u_new[i];
        }
        error = sqrt(error / n);
        norm = sqrt(norm / n);

        // Use relative tolerance if norm is significant, otherwise absolute
        rel_error = (norm > 1e-12) ? error / (norm + 1e-12) : error;

        // Trace iteration progress
        IVCALC_TRACE_PDE_IMPLICIT_ITER(step, iter, rel_error, tol);

        if (rel_error < tol || error < tol) {
            // Trace successful convergence
            IVCALC_TRACE_PDE_IMPLICIT_CONVERGED(step, iter, rel_error);
            return 0; // Success
        }
    }

    // Trace convergence failure
    IVCALC_TRACE_PDE_IMPLICIT_FAILED(step, max_iter, rel_error);
    return -1; // Failed to converge
}

// Utility functions

SpatialGrid pde_create_grid(double x_min, double x_max, size_t n_points) {
    SpatialGrid grid;
    grid.x_min = x_min;
    grid.x_max = x_max;
    grid.n_points = n_points;
    grid.dx = (x_max - x_min) / (n_points - 1);
    grid.x = malloc(n_points * sizeof(double));

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
    config.max_iter = 100;
    config.tolerance = 1e-6;  // Reasonable tolerance for most applications
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

    // Allocate single workspace buffer for all arrays (better cache locality)
    // Need 10 arrays of size n:
    //   - Solution: u_current, u_next, u_stage, rhs
    //   - Matrix: matrix_diag, matrix_upper, matrix_lower
    //   - Temps: u_old, Lu, u_temp
    const size_t n = grid->n_points;
    const size_t workspace_size = 10 * n;

    // Use aligned allocation for SIMD vectorization (64-byte alignment for AVX-512)
    // Each array starts at aligned boundary by padding n to alignment
    const size_t alignment = 64;  // Cache line and AVX-512 alignment
    const size_t n_aligned = ((n * sizeof(double) + alignment - 1) / alignment) * alignment / sizeof(double);
    const size_t workspace_aligned_size = 10 * n_aligned;

    solver->workspace = aligned_alloc(alignment, workspace_aligned_size * sizeof(double));
    if (solver->workspace == nullptr) {
        // Fallback to regular malloc if aligned_alloc fails
        solver->workspace = malloc(workspace_size * sizeof(double));
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

    // TR-BDF2 scheme
    // Stage 1: Trapezoidal rule from t_n to t_n + γ·dt
    // u* = u^n + (γ·dt/2) · [L(u^n) + L(u*)]
    // Rearranging: u* - (γ·dt/2)·L(u*) = u^n + (γ·dt/2)·L(u^n)

    // Reuse workspace Lu array (no allocation overhead)
    double *Lu_n = solver->Lu;
    evaluate_spatial_operator(solver, t_current, solver->u_current, Lu_n);

    // RHS for stage 1
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        solver->rhs[i] = solver->u_current[i] + (gamma * dt / 2.0) * Lu_n[i];
    }

    // Solve implicit equation for stage 1
    int status = solve_implicit_step(solver, t_current + gamma * dt,
                                     gamma * dt / 2.0, solver->rhs, solver->u_stage, step);
    if (status != 0) {
        return status;
    }

    // Stage 2: BDF2 from t_n to t_n+1
    // (1+2·α)·u^{n+1} - (1+α)·u* + α·u^n = (1-α)·dt·L(u^{n+1})
    // where α = 1 - γ

    const double alpha = 1.0 - gamma;
    const double coeff = (1.0 - alpha) * dt / (1.0 + 2.0 * alpha);

    // RHS for stage 2
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        solver->rhs[i] = ((1.0 + alpha) * solver->u_stage[i] - alpha * solver->u_current[i]) /
                        (1.0 + 2.0 * alpha);
    }

    // Solve implicit equation for stage 2
    status = solve_implicit_step(solver, t_current + dt, coeff,
                                solver->rhs, solver->u_next, step);

    if (status == 0) {
        // Update current solution
        memcpy(solver->u_current, solver->u_next, n * sizeof(double));
    }

    return status;
}

int pde_solver_step(PDESolver *solver, double t_current) {
    // Public API: call internal version with step=0 (not tracked)
    return pde_solver_step_internal(solver, t_current, 0);
}

int pde_solver_solve(PDESolver *solver) {
    double t = solver->time.t_start;

    // Trace solver start
    IVCALC_TRACE_PDE_START(solver->time.t_start, solver->time.t_end,
                           solver->time.dt, solver->time.n_steps);

    for (size_t step = 0; step < solver->time.n_steps; step++) {
        // Use internal version with step tracking for better tracing
        int status = pde_solver_step_internal(solver, t, step);
        if (status != 0) {
            // Note: convergence failure already traced in solve_implicit_step
            return status;
        }

        t += solver->time.dt;

        // Trace progress periodically (every 10%)
        if (step % (solver->time.n_steps / 10 + 1) == 0) {
            IVCALC_TRACE_PDE_PROGRESS(step, solver->time.n_steps, t);
        }
    }

    // Trace successful completion
    IVCALC_TRACE_PDE_COMPLETE(solver->time.n_steps, t);
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
