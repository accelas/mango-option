#include <gtest/gtest.h>
#include <cmath>

extern "C" {
#include "../src/pde_solver.h"
}

// Test fixture for PDE solver tests
class PDESolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }

    void TearDown() override {
        // Common cleanup code
    }
};

// Test grid creation
TEST_F(PDESolverTest, GridCreation) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 11);

    EXPECT_EQ(grid.n_points, 11);
    EXPECT_DOUBLE_EQ(grid.x_min, 0.0);
    EXPECT_DOUBLE_EQ(grid.x_max, 1.0);
    EXPECT_DOUBLE_EQ(grid.dx, 0.1);
    EXPECT_NE(grid.x, nullptr);

    // Check grid points
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_NEAR(grid.x[i], i * 0.1, 1e-14);
    }

    pde_free_grid(&grid);
}

// Test default configurations
TEST_F(PDESolverTest, DefaultConfigurations) {
    TRBDF2Config trbdf2 = pde_default_trbdf2_config();
    EXPECT_NEAR(trbdf2.gamma, 2.0 - std::sqrt(2.0), 1e-14);
    EXPECT_GT(trbdf2.max_iter, 0);
    EXPECT_GT(trbdf2.tolerance, 0.0);

    BoundaryConfig bc = pde_default_boundary_config();
    EXPECT_EQ(bc.left_type, BC_DIRICHLET);
    EXPECT_EQ(bc.right_type, BC_DIRICHLET);
}

// Test data for steady-state problem
struct SteadyStateData {
    double source_term;
};

// Callbacks for steady-state test: du/dt = d²u/dx² - u + 1
// Steady state: d²u/dx² = u - 1, with u(0)=u(1)=0
// Analytical solution: u(x) = 1 - sinh(x)/sinh(1)

static double steady_initial(double x, void *user_data) {
    return 0.0; // Start from zero
}

static double steady_left_bc(double t, void *user_data) {
    return 0.0;
}

static double steady_right_bc(double t, void *user_data) {
    return 0.0;
}

static double steady_spatial_op(const double *x, double t, const double *u,
                               size_t idx, size_t n_points, void *user_data) {
    if (idx == 0 || idx == n_points - 1) {
        return 0.0;
    }

    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    double d2u_dx2 = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / (dx * dx);

    // du/dt = d²u/dx² - u + 1
    return d2u_dx2 - u[idx] + 1.0;
}

// Test steady-state convergence
TEST_F(PDESolverTest, SteadyStateConvergence) {
    SteadyStateData data = {1.0};

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 5.0,  // Run long enough to reach steady state
        .dt = 0.01,
        .n_steps = 500
    };

    PDECallbacks callbacks = {
        .initial_condition = steady_initial,
        .left_boundary = steady_left_bc,
        .right_boundary = steady_right_bc,
        .spatial_operator = steady_spatial_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();

    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);
    EXPECT_EQ(status, 0);

    // Check against analytical solution: u(x) = 1 - cosh(x - 0.5)/cosh(0.5)
    const double *u = pde_solver_get_solution(solver);
    const double *x = pde_solver_get_grid(solver);

    double max_error = 0.0;
    for (size_t i = 1; i < grid.n_points - 1; i++) {
        double analytical = 1.0 - std::cosh(x[i] - 0.5) / std::cosh(0.5);
        double error = std::abs(u[i] - analytical);
        max_error = std::max(max_error, error);
    }

    // Should be close to analytical solution (within discretization error)
    EXPECT_LT(max_error, 0.01);

    pde_solver_destroy(solver);
    pde_free_grid(&grid);
}

// Test data for heat equation
struct HeatData {
    double diffusion;
};

static double heat_initial(double x, void *user_data) {
    // Gaussian initial condition
    return std::exp(-std::pow(x - 0.5, 2) / 0.02);
}

static double heat_zero_bc(double t, void *user_data) {
    return 0.0;
}

static double heat_diffusion_op(const double *x, double t, const double *u,
                                size_t idx, size_t n_points, void *user_data) {
    if (idx == 0 || idx == n_points - 1) {
        return 0.0;
    }

    HeatData *data = static_cast<HeatData*>(user_data);
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    double d2u_dx2 = (u[idx - 1] - 2.0 * u[idx] + u[idx + 1]) / (dx * dx);

    return data->diffusion * d2u_dx2;
}

// Test heat equation properties
TEST_F(PDESolverTest, HeatEquationProperties) {
    HeatData data = {0.1};

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 0.5,
        .dt = 0.001,
        .n_steps = 500
    };

    PDECallbacks callbacks = {
        .initial_condition = heat_initial,
        .left_boundary = heat_zero_bc,
        .right_boundary = heat_zero_bc,
        .spatial_operator = heat_diffusion_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();

    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);

    // Check initial condition
    const double *u0 = pde_solver_get_solution(solver);
    double initial_peak = u0[50];
    EXPECT_NEAR(initial_peak, std::exp(0.0), 0.01); // Peak at x=0.5

    int status = pde_solver_solve(solver);
    EXPECT_EQ(status, 0);

    // Check properties:
    // 1. Solution should be non-negative (maximum principle)
    // 2. Peak should decrease over time (diffusion)
    // 3. Boundary conditions should be satisfied
    const double *u = pde_solver_get_solution(solver);

    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_GE(u[i], -1e-10); // Non-negative (with small tolerance)
    }

    // Peak should have diffused
    EXPECT_LT(u[50], initial_peak);

    // Boundary conditions
    EXPECT_NEAR(u[0], 0.0, 1e-10);
    EXPECT_NEAR(u[grid.n_points - 1], 0.0, 1e-10);

    pde_solver_destroy(solver);
    pde_free_grid(&grid);
}

// Test obstacle condition
static double obstacle_func(double x, double t, void *user_data) {
    return 0.2; // Minimum value
}

TEST_F(PDESolverTest, ObstacleCondition) {
    HeatData data = {0.1};

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 2.0,  // Long time to see obstacle effect
        .dt = 0.01,
        .n_steps = 200
    };

    PDECallbacks callbacks = {
        .initial_condition = heat_initial,
        .left_boundary = heat_zero_bc,
        .right_boundary = heat_zero_bc,
        .spatial_operator = heat_diffusion_op,
        .jump_condition = nullptr,
        .obstacle = obstacle_func,
        .user_data = &data
    };

    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();

    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);
    EXPECT_EQ(status, 0);

    // Check that solution respects obstacle
    const double *u = pde_solver_get_solution(solver);
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_GE(u[i], 0.2 - 1e-10);
    }

    pde_solver_destroy(solver);
    pde_free_grid(&grid);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
