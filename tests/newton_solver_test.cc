#include "src/cpp/newton_solver.hpp"
#include "src/cpp/root_finding.hpp"
#include "src/cpp/workspace.hpp"
#include "src/cpp/boundary_conditions.hpp"
#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Test fixture for Newton solver
class NewtonSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        n = 101;
        grid_data.resize(n);
        for (size_t i = 0; i < n; ++i) {
            grid_data[i] = static_cast<double>(i) / (n - 1);
        }
    }

    size_t n;
    std::vector<double> grid_data;
};

TEST_F(NewtonSolverTest, ConvergesForLinearProblem) {
    // Setup: Solve u = rhs + coeff_dt·∂²u/∂x² with Dirichlet BCs
    // This is a linear problem, Newton should converge in 1-2 iterations

    mango::RootFindingConfig config{.max_iter = 10, .tolerance = 1e-8};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    // Dirichlet boundaries: u(0) = 0, u(1) = 0
    mango::DirichletBC left_bc{[](double, double) { return 0.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 0.0; }};

    // Spatial operator: L(u) = ∂²u/∂x²
    mango::LaplacianOperator spatial_op{1.0};  // Diffusion coefficient D = 1.0

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    // Initial guess: u = sin(πx)
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(M_PI * grid_data[i]);
    }

    // RHS: rhs = u (i.e., solve u = u + 0·L(u), trivial fixed point)
    std::vector<double> rhs(u);

    double t = 0.0;
    double coeff_dt = 0.01;

    auto result = solver.solve(t, coeff_dt, std::span{u}, std::span{rhs});

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 5);  // Should converge quickly
    EXPECT_LT(result.final_error, config.tolerance);
    EXPECT_FALSE(result.failure_reason.has_value());
}

TEST_F(NewtonSolverTest, RespectsDirichletBoundaries) {
    mango::RootFindingConfig config{.max_iter = 20, .tolerance = 1e-6};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::DirichletBC left_bc{[](double, double) { return 1.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 2.0; }};

    mango::LaplacianOperator spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    std::vector<double> u(n, 1.5);  // Initial guess
    std::vector<double> rhs(n, 1.5);

    auto result = solver.solve(0.0, 0.01, std::span{u}, std::span{rhs});

    EXPECT_TRUE(result.converged);
    EXPECT_DOUBLE_EQ(u[0], 1.0);  // Left BC
    EXPECT_DOUBLE_EQ(u[n-1], 2.0);  // Right BC
}

TEST_F(NewtonSolverTest, ReportsConvergenceFailure) {
    mango::RootFindingConfig config{.max_iter = 2, .tolerance = 1e-12};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::DirichletBC left_bc{[](double, double) { return 0.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 0.0; }};
    mango::LaplacianOperator spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    std::vector<double> u(n, 1.0);
    std::vector<double> rhs(n, 0.0);

    auto result = solver.solve(0.0, 0.01, std::span{u}, std::span{rhs});

    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_TRUE(result.failure_reason.has_value());
    EXPECT_EQ(result.failure_reason.value(), "Max iterations reached");
}

TEST_F(NewtonSolverTest, ReuseAcrossMultipleSolves) {
    mango::RootFindingConfig config{.max_iter = 20, .tolerance = 1e-6};
    mango::WorkspaceStorage workspace(n, std::span{grid_data});

    mango::DirichletBC left_bc{[](double, double) { return 0.0; }};
    mango::DirichletBC right_bc{[](double, double) { return 0.0; }};
    mango::LaplacianOperator spatial_op{1.0};

    mango::NewtonSolver solver(n, config, workspace, left_bc, right_bc,
                              spatial_op, std::span{grid_data});

    // Solve twice with different RHS
    std::vector<double> u1(n, 1.0), rhs1(n, 1.0);
    auto result1 = solver.solve(0.0, 0.01, std::span{u1}, std::span{rhs1});

    std::vector<double> u2(n, 2.0), rhs2(n, 2.0);
    auto result2 = solver.solve(0.1, 0.01, std::span{u2}, std::span{rhs2});

    EXPECT_TRUE(result1.converged);
    EXPECT_TRUE(result2.converged);
}
