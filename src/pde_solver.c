#include "pde_solver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Tridiagonal matrix solver (Thomas algorithm)
static void solve_tridiagonal(size_t n, const double *lower, const double *diag,
                              const double *upper, const double *rhs, double *solution) {
    double *c_prime = malloc(n * sizeof(double));
    double *d_prime = malloc(n * sizeof(double));

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (size_t i = 1; i < n; i++) {
        double m = 1.0 / (diag[i] - lower[i] * c_prime[i - 1]);
        c_prime[i] = (i < n - 1) ? upper[i] * m : 0.0;
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) * m;
    }

    // Back substitution
    solution[n - 1] = d_prime[n - 1];
    for (int i = (int)n - 2; i >= 0; i--) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    free(c_prime);
    free(d_prime);
}

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
                               const double *rhs, double *u_new) {
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
        double rel_error = (norm > 1e-12) ? error / (norm + 1e-12) : error;

        if (rel_error < tol || error < tol) {
            return 0; // Success
        }
    }

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

PDESolver* pde_solver_create(const SpatialGrid *grid,
                              const TimeDomain *time,
                              const BoundaryConfig *bc_config,
                              const TRBDF2Config *trbdf2_config,
                              const PDECallbacks *callbacks) {
    PDESolver *solver = malloc(sizeof(PDESolver));

    // Copy grid (deep copy)
    solver->grid = *grid;
    solver->grid.x = malloc(grid->n_points * sizeof(double));
    memcpy(solver->grid.x, grid->x, grid->n_points * sizeof(double));

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

int pde_solver_step(PDESolver *solver, double t_current) {
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
                                     gamma * dt / 2.0, solver->rhs, solver->u_stage);
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
                                solver->rhs, solver->u_next);

    if (status == 0) {
        // Update current solution
        memcpy(solver->u_current, solver->u_next, n * sizeof(double));
    }

    return status;
}

int pde_solver_solve(PDESolver *solver) {
    double t = solver->time.t_start;

    printf("Starting PDE solve from t=%.6f to t=%.6f with dt=%.6f\n",
           solver->time.t_start, solver->time.t_end, solver->time.dt);

    for (size_t step = 0; step < solver->time.n_steps; step++) {
        int status = pde_solver_step(solver, t);
        if (status != 0) {
            fprintf(stderr, "Error: Failed to converge at step %zu, t=%.6f\n", step, t);
            return status;
        }

        t += solver->time.dt;

        if (step % (solver->time.n_steps / 10 + 1) == 0) {
            printf("Progress: step %zu/%zu, t=%.6f\n", step, solver->time.n_steps, t);
        }
    }

    printf("PDE solve completed successfully.\n");
    return 0;
}

const double* pde_solver_get_solution(const PDESolver *solver) {
    return solver->u_current;
}

const double* pde_solver_get_grid(const PDESolver *solver) {
    return solver->grid.x;
}

// Cubic spline interpolation implementation

CubicSpline* pde_spline_create(const double *x, const double *y, size_t n_points) {
    if (n_points < 2) {
        fprintf(stderr, "Error: Need at least 2 points for spline interpolation\n");
        return nullptr;
    }

    CubicSpline *spline = malloc(sizeof(CubicSpline));
    spline->n_points = n_points;
    spline->x = x;  // Store pointer (not owned)
    spline->y = y;  // Store pointer (not owned)

    // Allocate coefficient arrays
    // For n points, we have n-1 intervals
    // Each interval has coefficients: a, b, c, d
    const size_t n = n_points;
    spline->coeffs_a = malloc(n * sizeof(double));
    spline->coeffs_b = malloc(n * sizeof(double));
    spline->coeffs_c = malloc(n * sizeof(double));
    spline->coeffs_d = malloc(n * sizeof(double));

    // Compute spline coefficients using natural cubic spline
    // S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    // for x in [x_i, x_{i+1}]

    // a_i = y_i
    for (size_t i = 0; i < n; i++) {
        spline->coeffs_a[i] = y[i];
    }

    // Set up tridiagonal system for c coefficients (second derivatives)
    double *h = malloc((n - 1) * sizeof(double));  // Interval widths
    double *alpha = malloc((n - 1) * sizeof(double));

    for (size_t i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];
    }

    for (size_t i = 1; i < n - 1; i++) {
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) -
                   (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Set up tridiagonal system: A*c = rhs
    // For natural spline: c[0] = 0, c[n-1] = 0
    // Interior equations: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = alpha[i]

    double *lower = malloc(n * sizeof(double));
    double *diag = malloc(n * sizeof(double));
    double *upper = malloc(n * sizeof(double));
    double *rhs = malloc(n * sizeof(double));

    // Boundary conditions for natural spline
    diag[0] = 1.0;
    upper[0] = 0.0;
    rhs[0] = 0.0;

    for (size_t i = 1; i < n - 1; i++) {
        lower[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        rhs[i] = alpha[i];
    }

    lower[n - 1] = 0.0;
    diag[n - 1] = 1.0;
    rhs[n - 1] = 0.0;

    // Solve tridiagonal system using shared solver
    solve_tridiagonal(n, lower, diag, upper, rhs, spline->coeffs_c);

    // Compute b and d coefficients from c
    for (size_t j = 0; j < n - 1; j++) {
        spline->coeffs_b[j] = (y[j + 1] - y[j]) / h[j] -
                             h[j] * (spline->coeffs_c[j + 1] + 2.0 * spline->coeffs_c[j]) / 3.0;
        spline->coeffs_d[j] = (spline->coeffs_c[j + 1] - spline->coeffs_c[j]) / (3.0 * h[j]);
    }

    free(h);
    free(alpha);
    free(lower);
    free(diag);
    free(upper);
    free(rhs);

    return spline;
}

void pde_spline_destroy(CubicSpline *spline) {
    if (spline == nullptr) return;

    free(spline->coeffs_a);
    free(spline->coeffs_b);
    free(spline->coeffs_c);
    free(spline->coeffs_d);
    free(spline);
}

// Binary search to find interval containing x_eval
static size_t find_interval(const double *x, size_t n, double x_eval) {
    // Handle boundary cases
    if (x_eval <= x[0]) return 0;
    if (x_eval >= x[n - 1]) return n - 2;

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        size_t mid = (left + right) / 2;
        if (x_eval < x[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    return left;
}

double pde_spline_eval(const CubicSpline *spline, double x_eval) {
    // Find the interval containing x_eval
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    // Evaluate spline in interval i
    // S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    double dx = x_eval - spline->x[i];
    double result = spline->coeffs_a[i] +
                   spline->coeffs_b[i] * dx +
                   spline->coeffs_c[i] * dx * dx +
                   spline->coeffs_d[i] * dx * dx * dx;

    return result;
}

double pde_spline_eval_derivative(const CubicSpline *spline, double x_eval) {
    // Find the interval containing x_eval
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    // Evaluate derivative: S'_i(x) = b_i + 2*c_i*(x - x_i) + 3*d_i*(x - x_i)^2
    double dx = x_eval - spline->x[i];
    double result = spline->coeffs_b[i] +
                   2.0 * spline->coeffs_c[i] * dx +
                   3.0 * spline->coeffs_d[i] * dx * dx;

    return result;
}

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
