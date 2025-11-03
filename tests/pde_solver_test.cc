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

static void steady_initial([[maybe_unused]] const double *x, size_t n_points,
                          double *u0, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        u0[i] = 0.0; // Start from zero
    }
}

static double steady_left_bc([[maybe_unused]] double t, [[maybe_unused]] void *user_data) {
    return 0.0;
}

static double steady_right_bc([[maybe_unused]] double t, [[maybe_unused]] void *user_data) {
    return 0.0;
}

static void steady_spatial_op(const double *x, [[maybe_unused]] double t, const double *u,
                              size_t n_points, double *Lu, [[maybe_unused]] void *user_data) {
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    // du/dt = d²u/dx² - u + 1
    for (size_t i = 1; i < n_points - 1; i++) {
        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = d2u_dx2 - u[i] + 1.0;
    }
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
    // Note: grid ownership transferred to solver
}

// Test data for heat equation
struct HeatData {
    double diffusion;
};

static void heat_initial(const double *x, size_t n_points,
                        double *u0, [[maybe_unused]] void *user_data) {
    // Gaussian initial condition
    for (size_t i = 0; i < n_points; i++) {
        u0[i] = std::exp(-std::pow(x[i] - 0.5, 2) / 0.02);
    }
}

static double heat_zero_bc([[maybe_unused]] double t, [[maybe_unused]] void *user_data) {
    return 0.0;
}

static void heat_diffusion_op(const double *x, [[maybe_unused]] double t, const double *u,
                              size_t n_points, double *Lu, void *user_data) {
    HeatData *data = static_cast<HeatData*>(user_data);
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    for (size_t i = 1; i < n_points - 1; i++) {
        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = data->diffusion * d2u_dx2;
    }
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
    // Note: grid ownership transferred to solver
}

// Test obstacle condition
static void obstacle_func([[maybe_unused]] const double *x, [[maybe_unused]] double t, size_t n_points,
                         double *psi, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        psi[i] = 0.2; // Minimum value
    }
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
    // Note: grid ownership transferred to solver
}

// Robin BC helper data for tests
struct RobinBCData {
    double left_g;
    double right_g;
};

static void robin_initial_condition([[maybe_unused]] const double *x, size_t n_points,
                                    double *u0, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        u0[i] = 1.0 + 0.5 * static_cast<double>(i);
    }
}

static double robin_left_bc([[maybe_unused]] double t, void *user_data) {
    RobinBCData *data = static_cast<RobinBCData*>(user_data);
    return data->left_g;
}

static double robin_right_bc([[maybe_unused]] double t, void *user_data) {
    RobinBCData *data = static_cast<RobinBCData*>(user_data);
    return data->right_g;
}

static void robin_zero_spatial_op([[maybe_unused]] const double *x, [[maybe_unused]] double t,
                                  [[maybe_unused]] const double *u, size_t n_points,
                                  double *Lu, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        Lu[i] = 0.0;
    }
}

TEST_F(PDESolverTest, RobinBoundaryConditionUsesOutwardNormal) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 5);
    TimeDomain time = {.t_start = 0.0, .t_end = 0.1, .dt = 0.1, .n_steps = 1};

    BoundaryConfig bc = pde_default_boundary_config();
    bc.left_type = BC_ROBIN;
    bc.right_type = BC_ROBIN;
    bc.left_robin_a = 2.0;
    bc.left_robin_b = 1.0;
    bc.right_robin_a = 1.5;
    bc.right_robin_b = -1.2;

    RobinBCData data = {3.0, 0.25};

    PDECallbacks callbacks = {
        .initial_condition = robin_initial_condition,
        .left_boundary = robin_left_bc,
        .right_boundary = robin_right_bc,
        .spatial_operator = robin_zero_spatial_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    TRBDF2Config trbdf2 = pde_default_trbdf2_config();

    PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);

    const double *u = pde_solver_get_solution(solver);
    ASSERT_NE(u, nullptr);

    const size_t n = solver->grid.n_points;
    const double dx = solver->grid.dx;

    double expected_left = (data.left_g + bc.left_robin_b * u[1] / dx) /
                           (bc.left_robin_a + bc.left_robin_b / dx);
    double expected_right = (data.right_g + bc.right_robin_b * u[n - 2] / dx) /
                            (bc.right_robin_a + bc.right_robin_b / dx);

    EXPECT_NEAR(u[0], expected_left, 1e-12);
    EXPECT_NEAR(u[n - 1], expected_right, 1e-12);

    pde_solver_destroy(solver);
    // Grid ownership transferred to solver; no need to free grid.x separately
}

// Negative test: Invalid Robin BC coefficient (division by zero)
TEST_F(PDESolverTest, InvalidRobinCoefficient) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 11);
    TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = 0.01, .n_steps = 100};

    BoundaryConfig bc = pde_default_boundary_config();
    bc.left_type = BC_ROBIN;
    bc.left_robin_a = 0.0;  // Invalid - would cause division by zero
    bc.left_robin_b = 1.0;

    TRBDF2Config trbdf2 = pde_default_trbdf2_config();

    SteadyStateData data = {1.0};
    PDECallbacks callbacks = {
        .initial_condition = steady_initial,
        .left_boundary = steady_left_bc,
        .right_boundary = steady_right_bc,
        .spatial_operator = steady_spatial_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    // Should return nullptr due to invalid Robin coefficient
    PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
    EXPECT_EQ(solver, nullptr);

    // Grid not transferred if creation failed, need to free it
    if (grid.x != nullptr) {
        pde_free_grid(&grid);
    }
}

// Negative test: Very small grid (n < 3)
TEST_F(PDESolverTest, TooSmallGrid) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 2);
    TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = 0.01, .n_steps = 100};

    BoundaryConfig bc = pde_default_boundary_config();
    TRBDF2Config trbdf2 = pde_default_trbdf2_config();

    SteadyStateData data = {1.0};
    PDECallbacks callbacks = {
        .initial_condition = steady_initial,
        .left_boundary = steady_left_bc,
        .right_boundary = steady_right_bc,
        .spatial_operator = steady_spatial_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    // Should handle gracefully (may return nullptr or work with degraded accuracy)
    PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);

    if (solver != nullptr) {
        // If solver created, it should at least not crash
        pde_solver_initialize(solver);
        [[maybe_unused]] int status = pde_solver_solve(solver);
        // May converge or not, but shouldn't crash
        pde_solver_destroy(solver);
    } else {
        // Grid not transferred if creation failed
        if (grid.x != nullptr) {
            pde_free_grid(&grid);
        }
    }
}

// Negative test: Negative time step
TEST_F(PDESolverTest, NegativeTimeStep) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 11);
    TimeDomain time = {.t_start = 0.0, .t_end = 1.0, .dt = -0.01, .n_steps = 100};  // Negative dt

    BoundaryConfig bc = pde_default_boundary_config();
    TRBDF2Config trbdf2 = pde_default_trbdf2_config();

    SteadyStateData data = {1.0};
    PDECallbacks callbacks = {
        .initial_condition = steady_initial,
        .left_boundary = steady_left_bc,
        .right_boundary = steady_right_bc,
        .spatial_operator = steady_spatial_op,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);

    if (solver != nullptr) {
        pde_solver_initialize(solver);
        // Should handle gracefully (may fail to solve or time-reverse)
        [[maybe_unused]] int status = pde_solver_solve(solver);
        // Just verify it doesn't crash
        pde_solver_destroy(solver);
    } else {
        if (grid.x != nullptr) {
            pde_free_grid(&grid);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
