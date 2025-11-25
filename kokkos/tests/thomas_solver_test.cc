#include <gtest/gtest.h>
#include "kokkos/src/math/thomas_solver.hpp"
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

class ThomasSolverTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(ThomasSolverTest, SolvesSimpleSystem) {
    // System: 2x1 - x2 = 1
    //        -x1 + 2x2 - x3 = 0
    //        -x2 + 2x3 = 1
    // Solution: x = [1, 1, 1]

    constexpr size_t n = 3;

    Kokkos::View<double*, HostMemSpace> lower("lower", n - 1);
    Kokkos::View<double*, HostMemSpace> diag("diag", n);
    Kokkos::View<double*, HostMemSpace> upper("upper", n - 1);
    Kokkos::View<double*, HostMemSpace> rhs("rhs", n);
    Kokkos::View<double*, HostMemSpace> solution("solution", n);

    lower(0) = -1.0; lower(1) = -1.0;
    diag(0) = 2.0; diag(1) = 2.0; diag(2) = 2.0;
    upper(0) = -1.0; upper(1) = -1.0;
    rhs(0) = 1.0; rhs(1) = 0.0; rhs(2) = 1.0;

    ThomasSolver<HostMemSpace> solver;
    auto result = solver.solve(lower, diag, upper, rhs, solution);

    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(solution(0), 1.0, 1e-10);
    EXPECT_NEAR(solution(1), 1.0, 1e-10);
    EXPECT_NEAR(solution(2), 1.0, 1e-10);
}

TEST_F(ThomasSolverTest, SolvesDiffusionSystem) {
    // Discretized diffusion: -u_{i-1} + 2*u_i - u_{i+1} = h^2 * f_i
    // With u(0) = 0, u(1) = 0, f = 1
    // Analytical: u(x) = x(1-x)/2

    constexpr size_t n = 11;
    double h = 1.0 / static_cast<double>(n + 1);

    Kokkos::View<double*, HostMemSpace> lower("lower", n - 1);
    Kokkos::View<double*, HostMemSpace> diag("diag", n);
    Kokkos::View<double*, HostMemSpace> upper("upper", n - 1);
    Kokkos::View<double*, HostMemSpace> rhs("rhs", n);
    Kokkos::View<double*, HostMemSpace> solution("solution", n);

    for (size_t i = 0; i < n; ++i) {
        diag(i) = 2.0;
        rhs(i) = h * h;  // f = 1
    }
    for (size_t i = 0; i < n - 1; ++i) {
        lower(i) = -1.0;
        upper(i) = -1.0;
    }

    ThomasSolver<HostMemSpace> solver;
    auto result = solver.solve(lower, diag, upper, rhs, solution);

    ASSERT_TRUE(result.has_value());

    // Check against analytical solution at midpoint
    double x_mid = 0.5;
    double u_analytical = x_mid * (1.0 - x_mid) / 2.0;
    EXPECT_NEAR(solution(n / 2), u_analytical, 1e-3);
}

TEST_F(ThomasSolverTest, DetectsSingularMatrix) {
    constexpr size_t n = 3;

    Kokkos::View<double*, HostMemSpace> lower("lower", n - 1);
    Kokkos::View<double*, HostMemSpace> diag("diag", n);
    Kokkos::View<double*, HostMemSpace> upper("upper", n - 1);
    Kokkos::View<double*, HostMemSpace> rhs("rhs", n);
    Kokkos::View<double*, HostMemSpace> solution("solution", n);

    // Singular: diagonal is zero
    lower(0) = 1.0; lower(1) = 1.0;
    diag(0) = 0.0; diag(1) = 0.0; diag(2) = 0.0;
    upper(0) = 1.0; upper(1) = 1.0;
    rhs(0) = 1.0; rhs(1) = 1.0; rhs(2) = 1.0;

    ThomasSolver<HostMemSpace> solver;
    auto result = solver.solve(lower, diag, upper, rhs, solution);

    EXPECT_FALSE(result.has_value());
}

}  // namespace mango::kokkos::test
