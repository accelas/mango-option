#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

extern "C" {
#include "../src/pde_solver.h"
}

// Test fixture for numerical stability tests
class StabilityTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test data structures
struct StiffData {
    double diffusion;
    double reaction_rate;
};

struct AdvectionData {
    double velocity;
    double diffusion;
};

// Stiff reaction-diffusion: du/dt = D*d²u/dx² - k*u
static void stiff_initial(const double *x, size_t n_points,
                         double *u0, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        u0[i] = std::sin(M_PI * x[i]);
    }
}

static double zero_bc([[maybe_unused]] double t, [[maybe_unused]] void *user_data) {
    return 0.0;
}

static void stiff_operator(const double *x, [[maybe_unused]] double t, const double *u,
                          size_t n_points, double *Lu, void *user_data) {
    StiffData *data = static_cast<StiffData*>(user_data);
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    // Diffusion + reaction terms
    for (size_t i = 1; i < n_points - 1; i++) {
        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = data->diffusion * d2u_dx2 - data->reaction_rate * u[i];
    }
}

// Test 1: Stiff equation stability
TEST_F(StabilityTest, StiffEquationStability) {
    StiffData data = {0.01, 10.0};  // Large reaction rate (stiff)

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 0.5,
        .dt = 0.01,  // Relatively large time step
        .n_steps = 50
    };

    PDECallbacks callbacks = {
        .initial_condition = stiff_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
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

    // TR-BDF2 should handle stiff equations without instability
    EXPECT_EQ(status, 0);

    // Check solution remains bounded
    const double *u = pde_solver_get_solution(solver);
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_FALSE(std::isnan(u[i]));
        EXPECT_FALSE(std::isinf(u[i]));
        EXPECT_LT(std::abs(u[i]), 10.0);  // Should decay, not grow
    }

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 2: Fine grid stability
TEST_F(StabilityTest, FineGridStability) {
    StiffData data = {0.1, 0.0};  // Pure diffusion

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 201);  // Very fine grid

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 0.1,
        .dt = 0.0001,  // Small time step for fine grid
        .n_steps = 1000
    };

    PDECallbacks callbacks = {
        .initial_condition = stiff_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
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

    // Solution should remain stable
    const double *u = pde_solver_get_solution(solver);
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_FALSE(std::isnan(u[i]));
        EXPECT_FALSE(std::isinf(u[i]));
    }

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 3: Maximum principle for heat equation
TEST_F(StabilityTest, MaximumPrinciple) {
    StiffData data = {0.1, 0.0};

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 1.0,
        .dt = 0.01,
        .n_steps = 100
    };

    PDECallbacks callbacks = {
        .initial_condition = stiff_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
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

    // Find initial maximum
    const double *u0 = pde_solver_get_solution(solver);
    double max_initial = 0.0;
    for (size_t i = 0; i < grid.n_points; i++) {
        max_initial = std::max(max_initial, std::abs(u0[i]));
    }

    int status = pde_solver_solve(solver);
    EXPECT_EQ(status, 0);

    // Maximum principle: max at final time <= max at initial time
    const double *u = pde_solver_get_solution(solver);
    double max_final = 0.0;
    for (size_t i = 0; i < grid.n_points; i++) {
        max_final = std::max(max_final, std::abs(u[i]));
    }

    EXPECT_LE(max_final, max_initial + 1e-10);

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 4: Mass conservation for closed system
static void conservation_initial(const double *x, size_t n_points,
                                double *u0, [[maybe_unused]] void *user_data) {
    for (size_t i = 0; i < n_points; i++) {
        u0[i] = std::exp(-50.0 * std::pow(x[i] - 0.5, 2));
    }
}

static double neumann_zero([[maybe_unused]] double t, [[maybe_unused]] void *user_data) {
    return 0.0;  // Zero flux
}

static void diffusion_only(const double *x, [[maybe_unused]] double t, const double *u,
                           size_t n_points, double *Lu, [[maybe_unused]] void *user_data) {
    const double dx = (x[n_points - 1] - x[0]) / (n_points - 1);
    const double dx2_inv = 1.0 / (dx * dx);

    Lu[0] = 0.0;
    Lu[n_points - 1] = 0.0;

    for (size_t i = 1; i < n_points - 1; i++) {
        double d2u_dx2 = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * dx2_inv;
        Lu[i] = 0.1 * d2u_dx2;
    }
}

TEST_F(StabilityTest, MassConservation) {
    SpatialGrid grid = pde_create_grid(0.0, 1.0, 101);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 1.0,
        .dt = 0.01,
        .n_steps = 100
    };

    PDECallbacks callbacks = {
        .initial_condition = conservation_initial,
        .left_boundary = neumann_zero,
        .right_boundary = neumann_zero,
        .spatial_operator = diffusion_only,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = nullptr
    };

    BoundaryConfig bc_config = {
        .left_type = BC_NEUMANN,
        .right_type = BC_NEUMANN,
        .left_robin_a = 0.0,
        .left_robin_b = 0.0,
        .right_robin_a = 0.0,
        .right_robin_b = 0.0
    };

    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();

    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);

    // Compute initial mass using trapezoidal rule
    // For ghost point method, mass is conserved with trapezoidal quadrature
    const double *u0 = pde_solver_get_solution(solver);
    double mass_initial = 0.0;
    mass_initial += 0.5 * u0[0] * grid.dx;  // Half weight at left boundary
    for (size_t i = 1; i < grid.n_points - 1; i++) {
        mass_initial += u0[i] * grid.dx;
    }
    mass_initial += 0.5 * u0[grid.n_points - 1] * grid.dx;  // Half weight at right boundary

    int status = pde_solver_solve(solver);
    EXPECT_EQ(status, 0);

    // Compute final mass using trapezoidal rule
    const double *u = pde_solver_get_solution(solver);
    double mass_final = 0.0;
    mass_final += 0.5 * u[0] * grid.dx;
    for (size_t i = 1; i < grid.n_points - 1; i++) {
        mass_final += u[i] * grid.dx;
    }
    mass_final += 0.5 * u[grid.n_points - 1] * grid.dx;

    // Mass should be conserved (within numerical error)
    EXPECT_NEAR(mass_final / mass_initial, 1.0, 0.01);

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 5: Long-time stability
TEST_F(StabilityTest, LongTimeStability) {
    StiffData data = {0.05, 0.1};  // Small decay

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 10.0,  // Long time integration
        .dt = 0.02,
        .n_steps = 500
    };

    PDECallbacks callbacks = {
        .initial_condition = stiff_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
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

    // Solution should remain stable over long time
    const double *u = pde_solver_get_solution(solver);
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_FALSE(std::isnan(u[i]));
        EXPECT_FALSE(std::isinf(u[i]));
        EXPECT_LT(std::abs(u[i]), 1.0);
    }

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 6: Non-negative preservation for suitable problems
TEST_F(StabilityTest, NonNegativityPreservation) {
    StiffData data = {0.1, 0.5};

    // Start with non-negative initial condition
    auto nonneg_initial = [](const double *x, size_t n_points,
                            double *u0, [[maybe_unused]] void *user_data) -> void {
        for (size_t i = 0; i < n_points; i++) {
            u0[i] = std::exp(-10.0 * std::pow(x[i] - 0.5, 2));
        }
    };

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 1.0,
        .dt = 0.01,
        .n_steps = 100
    };

    PDECallbacks callbacks = {
        .initial_condition = nonneg_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
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

    // Solution should remain non-negative (within small tolerance)
    const double *u = pde_solver_get_solution(solver);
    for (size_t i = 0; i < grid.n_points; i++) {
        EXPECT_GE(u[i], -1e-10);
    }

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

// Test 7: Convergence iteration test
TEST_F(StabilityTest, ConvergenceIterations) {
    StiffData data = {0.1, 1.0};

    SpatialGrid grid = pde_create_grid(0.0, 1.0, 51);

    TimeDomain time = {
        .t_start = 0.0,
        .t_end = 0.1,
        .dt = 0.01,
        .n_steps = 10
    };

    PDECallbacks callbacks = {
        .initial_condition = stiff_initial,
        .left_boundary = zero_bc,
        .right_boundary = zero_bc,
        .spatial_operator = stiff_operator,
        .jump_condition = nullptr,
        .obstacle = nullptr,
        .user_data = &data
    };

    BoundaryConfig bc_config = pde_default_boundary_config();
    TRBDF2Config trbdf2_config = pde_default_trbdf2_config();
    trbdf2_config.tolerance = 1e-12;  // Tight tolerance

    PDESolver *solver = pde_solver_create(&grid, &time, &bc_config,
                                          &trbdf2_config, &callbacks);
    ASSERT_NE(solver, nullptr);

    pde_solver_initialize(solver);
    int status = pde_solver_solve(solver);

    // Should converge even with tight tolerance
    EXPECT_EQ(status, 0);

    pde_solver_destroy(solver);
    // Note: grid ownership transferred to solver
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
