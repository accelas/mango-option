#include <gtest/gtest.h>
#include "kokkos/src/pde/core/pde_solver.hpp"
#include "kokkos/src/pde/core/grid.hpp"
#include "kokkos/src/pde/core/workspace.hpp"
#include <cmath>

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class PDESolverTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(PDESolverTest, HeatEquationConverges) {
    // Heat equation: u_t = u_xx
    // Initial: u(x,0) = sin(pi*x)
    // Boundary: u(0,t) = u(1,t) = 0
    // Exact: u(x,t) = exp(-pi^2*t) * sin(pi*x)

    constexpr size_t n = 51;
    constexpr double T = 0.1;
    constexpr size_t n_steps = 100;

    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, n).value();
    auto workspace = PDEWorkspace<HostMemSpace>::create(n).value();

    // Initialize
    auto u = grid.u_current();
    auto x = grid.x();
    for (size_t i = 0; i < n; ++i) {
        u(i) = std::sin(M_PI * x(i));
    }

    // Solve
    HeatEquationSolver<HostMemSpace> solver(grid, workspace);
    solver.solve(T, n_steps);

    // Check against exact solution
    double exact_decay = std::exp(-M_PI * M_PI * T);
    for (size_t i = 1; i < n - 1; ++i) {
        double exact = exact_decay * std::sin(M_PI * x(i));
        EXPECT_NEAR(u(i), exact, 0.01) << "at i=" << i;
    }
}

TEST_F(PDESolverTest, HeatEquationDecays) {
    // Verify energy decreases over time
    constexpr size_t n = 31;
    constexpr double T = 0.5;
    constexpr size_t n_steps = 500;

    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, n).value();
    auto workspace = PDEWorkspace<HostMemSpace>::create(n).value();

    // Initialize with sin(pi*x)
    auto u = grid.u_current();
    auto x = grid.x();
    double initial_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        u(i) = std::sin(M_PI * x(i));
        initial_energy += u(i) * u(i);
    }

    // Solve
    HeatEquationSolver<HostMemSpace> solver(grid, workspace);
    solver.solve(T, n_steps);

    // Check energy has decreased
    double final_energy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        final_energy += u(i) * u(i);
    }

    EXPECT_LT(final_energy, initial_energy);

    // With T=0.5 and sin(pi*x) initial condition:
    // exact decay = exp(-pi^2 * 0.5) ≈ 0.007
    // So final_energy / initial_energy ≈ 0.007^2 ≈ 5e-5
    EXPECT_LT(final_energy / initial_energy, 0.01);
}

TEST_F(PDESolverTest, BoundaryConditionsRespected) {
    // Verify boundaries stay at zero
    constexpr size_t n = 21;
    constexpr double T = 0.1;
    constexpr size_t n_steps = 100;

    auto grid = Grid<HostMemSpace>::uniform(0.0, 1.0, n).value();
    auto workspace = PDEWorkspace<HostMemSpace>::create(n).value();

    // Initialize with bump in the middle
    auto u = grid.u_current();
    for (size_t i = 0; i < n; ++i) {
        double xi = static_cast<double>(i) / (n - 1);
        u(i) = std::sin(M_PI * xi);
    }

    HeatEquationSolver<HostMemSpace> solver(grid, workspace);
    solver.solve(T, n_steps);

    // Boundaries should be exactly zero
    EXPECT_DOUBLE_EQ(u(0), 0.0);
    EXPECT_DOUBLE_EQ(u(n - 1), 0.0);
}

}  // namespace mango::kokkos::test
