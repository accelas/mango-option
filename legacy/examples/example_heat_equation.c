#include "src/pde_solver.h"
#include <stdio.h>
#include <math.h>

// Example 1: Simple heat equation du/dt = D * d²u/dx²
// with initial Gaussian profile and Dirichlet boundary conditions

typedef struct {
    double diffusion_coeff;
    double jump_location;     // For jump condition example
    double diffusion_left;
    double diffusion_right;
} HeatEquationData;

// Initial condition: Gaussian profile (vectorized)
static void heat_initial_condition(const double *x, size_t n_points,
                                   double *u0, [[maybe_unused]] void *user_data) {
    const double x0 = 0.5;
    const double sigma = 0.1;

    for (size_t i = 0; i < n_points; i++) {
        u0[i] = exp(-pow(x[i] - x0, 2) / (2 * sigma * sigma));
    }
}

// Left boundary: u(0, t) = 0
static double heat_left_boundary([[maybe_unused]] double t, [[maybe_unused]] double x_boundary,
                                 [[maybe_unused]] BoundaryType bc_type,
                                 [[maybe_unused]] void *user_data) {
    return 0.0;
}

// Right boundary: u(1, t) = 0
static double heat_right_boundary([[maybe_unused]] double t, [[maybe_unused]] double x_boundary,
                                  [[maybe_unused]] BoundaryType bc_type,
                                  [[maybe_unused]] void *user_data) {
    return 0.0;
}

// Spatial operator for heat equation: D * d²u/dx² (vectorized)
// Using central finite differences: (u[i-1] - 2*u[i] + u[i+1]) / dx²
static void heat_spatial_operator(const double *x, [[maybe_unused]] double t, const double *u,
                                  size_t n_points, double *Lu, void *user_data) {
    HeatEquationData *data = (HeatEquationData *)user_data;
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double D = data->diffusion_coeff;
    const double dx2_inv = 1.0 / (dx * dx);

    // Boundaries (will be overwritten by BC)
    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    // Interior points - vectorized computation with SIMD
    #pragma omp simd
    for (size_t i = 1; i < n_points - 1; i++) {
        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = D * d2u_dx2;
    }
}

// Example 2: Jump condition - different diffusion coefficients (vectorized)
static void heat_spatial_operator_with_jump(const double *x, [[maybe_unused]] double t, const double *u,
                                            size_t n_points, double *Lu, void *user_data) {
    HeatEquationData *data = (HeatEquationData *)user_data;
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    // Boundaries
    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    // Interior points with spatially varying diffusion coefficient
    for (size_t i = 1; i < n_points - 1; i++) {
        // Use different diffusion coefficients based on position
        double D_center = (x[i] < data->jump_location) ?
                         data->diffusion_left : data->diffusion_right;

        // At the jump, use harmonic mean for the interface
        if (fabs(x[i] - data->jump_location) < dx) {
            D_center = 2.0 * data->diffusion_left * data->diffusion_right /
                      (data->diffusion_left + data->diffusion_right);
        }

        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = D_center * d2u_dx2;
    }
}

// Jump condition callback
static bool heat_jump_condition(double x, double *jump_value, void *user_data) {
    HeatEquationData *data = (HeatEquationData *)user_data;
    if (fabs(x - data->jump_location) < 1e-10) {
        *jump_value = 0.0; // Continuous solution, but flux jumps
        return true;
    }
    return false;
}

// Example 3: Obstacle condition (American option pricing) (vectorized)
static void obstacle_condition(const double *x, [[maybe_unused]] double t, size_t n_points,
                              double *psi, [[maybe_unused]] void *user_data) {
    // Example: minimum value is the payoff function max(x - 0.5, 0)
    for (size_t i = 0; i < n_points; i++) {
        psi[i] = fmax(x[i] - 0.5, 0.0);
    }
}

// Utility: Print solution to file
static void print_solution(const PDESolver *solver, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == nullptr) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    const double *x = pde_solver_get_grid(solver);
    const double *u = pde_solver_get_solution(solver);
    const size_t n = solver->grid.n_points;

    fprintf(fp, "# x u(x,t_final)\n");
    for (size_t i = 0; i < n; i++) {
        fprintf(fp, "%.10e %.10e\n", x[i], u[i]);
    }

    fclose(fp);
    printf("Solution written to %s\n", filename);
}

int main(void) {
    printf("=== PDE Solver Examples with TR-BDF2 ===\n\n");

    // Example 1: Basic heat equation
    printf("Example 1: Heat equation with Dirichlet BC\n");
    printf("------------------------------------------\n");

    HeatEquationData heat_data = {
        .diffusion_coeff = 0.1,
        .jump_location = 0.5,
        .diffusion_left = 0.1,
        .diffusion_right = 0.01
    };

    // Create spatial grid
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);

    // Create time domain
    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 1.0,
        .dt = 0.001,
        .n_steps = 1000
    };

    // Setup callbacks
    PDECallbacks callbacks = {
        .initial_condition = heat_initial_condition,
        .left_boundary = heat_left_boundary,
        .right_boundary = heat_right_boundary,
        .spatial_operator = heat_spatial_operator,
        .diffusion_coeff = heat_data.diffusion_coeff,  // Constant diffusion coefficient
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .temporal_event = nullptr,
        .n_temporal_events = 0,
        .temporal_event_times = nullptr,
        .user_data = &heat_data
    };

    // Create solver configuration
    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();

    // Create and run solver
    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);

    if (status == 0) {
        print_solution(solver, "heat_equation_solution.dat");

        // Demonstrate cubic spline interpolation
        printf("\nDemonstrating cubic spline interpolation:\n");
        printf("Grid spacing: dx = %.6f\n", grid.dx);

        // Evaluate solution at off-grid points
        double test_points[] = {0.123, 0.456, 0.789};
        for (size_t i = 0; i < sizeof(test_points) / sizeof(test_points[0]); i++) {
            double x_eval = test_points[i];
            double u_interp = pde_solver_interpolate(solver, x_eval);
            printf("  u(%.3f) = %.10e (interpolated)\n", x_eval, u_interp);
        }
    }

    pde_solver_destroy(solver);

    // Example 2: Heat equation with jump condition
    printf("\nExample 2: Heat equation with jump in diffusion coefficient\n");
    printf("-----------------------------------------------------------\n");

    // Create new grid (ownership was transferred to previous solver)
    grid = pde_create_grid(0.0, 1.0, 101);

    callbacks.spatial_operator = heat_spatial_operator_with_jump;
    callbacks.diffusion_coeff = NAN;  // Variable diffusion, use estimation
    callbacks.jump_condition = heat_jump_condition;

    solver = pde_solver_create(&grid, &time, &bc_config, &trbdf2_config, &callbacks);
    pde_solver_initialize(solver);
    status = pde_solver_solve(solver);

    if (status == 0) {
        print_solution(solver, "heat_equation_jump_solution.dat");
    }

    pde_solver_destroy(solver);

    // Example 3: Heat equation with obstacle condition
    printf("\nExample 3: Heat equation with obstacle condition\n");
    printf("------------------------------------------------\n");

    // Create new grid (ownership was transferred to previous solver)
    grid = pde_create_grid(0.0, 1.0, 101);

    callbacks.spatial_operator = heat_spatial_operator;
    callbacks.diffusion_coeff = heat_data.diffusion_coeff;  // Restore constant diffusion
    callbacks.jump_condition = nullptr;
    callbacks.obstacle = obstacle_condition;

    solver = pde_solver_create(&grid, &time, &bc_config, &trbdf2_config, &callbacks);
    pde_solver_initialize(solver);
    status = pde_solver_solve(solver);

    if (status == 0) {
        print_solution(solver, "heat_equation_obstacle_solution.dat");
    }

    pde_solver_destroy(solver);
    // Note: grid ownership was transferred to first solver, no need to free

    printf("\nAll examples completed!\n");
    return 0;
}
