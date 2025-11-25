#include <gtest/gtest.h>
#include "kokkos/src/pde/operators/black_scholes.hpp"
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

class BlackScholesTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(BlackScholesTest, ApplyOperatorConstantFunction) {
    // Black-Scholes operator: L(u) = 0.5*sigma^2*u_xx + drift*u_x - r*u
    // For constant u=1: u_xx=0, u_x=0, so L(1) = -r

    constexpr size_t n = 11;
    double sigma = 0.2;
    double r = 0.05;
    double q = 0.02;

    Kokkos::View<double*, HostMemSpace> x("x", n);
    Kokkos::View<double*, HostMemSpace> u("u", n);
    Kokkos::View<double*, HostMemSpace> Lu("Lu", n);

    // Uniform grid centered at 0
    double dx = 0.1;
    for (size_t i = 0; i < n; ++i) {
        x(i) = -0.5 + static_cast<double>(i) * dx;
        u(i) = 1.0;  // Constant function
    }

    BlackScholesOperator<HostMemSpace> op(sigma, r, q);
    op.apply(x, u, Lu, dx);

    // Interior points should have L(1) = -r
    for (size_t i = 1; i < n - 1; ++i) {
        EXPECT_NEAR(Lu(i), -r, 1e-10);
    }
}

TEST_F(BlackScholesTest, ApplyOperatorQuadratic) {
    // For u = x^2, u_xx = 2, u_x = 2x
    // L(x^2) = 0.5*sigma^2*2 + drift*2x - r*x^2

    constexpr size_t n = 11;
    double sigma = 0.2;
    double r = 0.05;
    double q = 0.02;

    Kokkos::View<double*, HostMemSpace> x("x", n);
    Kokkos::View<double*, HostMemSpace> u("u", n);
    Kokkos::View<double*, HostMemSpace> Lu("Lu", n);

    double dx = 0.1;
    for (size_t i = 0; i < n; ++i) {
        x(i) = -0.5 + static_cast<double>(i) * dx;
        u(i) = x(i) * x(i);
    }

    BlackScholesOperator<HostMemSpace> op(sigma, r, q);
    op.apply(x, u, Lu, dx);

    // Check at x = 0 (index 5)
    // L(x^2)|_{x=0} = 0.5*sigma^2*2 + drift*0 - r*0 = sigma^2
    double expected_at_zero = sigma * sigma;
    EXPECT_NEAR(Lu(5), expected_at_zero, 1e-3);

    // Check at other interior points
    double drift = r - q - 0.5 * sigma * sigma;
    for (size_t i = 2; i < n - 2; ++i) {  // Avoid boundary effects
        double xi = x(i);
        double expected = 0.5 * sigma * sigma * 2.0 + drift * 2.0 * xi - r * xi * xi;
        EXPECT_NEAR(Lu(i), expected, 1e-2);  // Discretization error
    }
}

TEST_F(BlackScholesTest, JacobianAssembly) {
    // Test that Jacobian is assembled correctly
    // J = I - dt*L where L is tridiagonal

    constexpr size_t n = 5;
    double sigma = 0.2;
    double r = 0.05;
    double q = 0.02;
    double dt = 0.01;
    double dx = 0.1;

    Kokkos::View<double*, HostMemSpace> lower("lower", n - 1);
    Kokkos::View<double*, HostMemSpace> diag("diag", n);
    Kokkos::View<double*, HostMemSpace> upper("upper", n - 1);

    BlackScholesOperator<HostMemSpace> op(sigma, r, q);
    op.assemble_jacobian(dt, dx, lower, diag, upper);

    // Compute expected coefficients
    double half_sigma_sq = 0.5 * sigma * sigma;
    double drift = r - q - half_sigma_sq;
    double dx_sq = dx * dx;
    double two_dx = 2.0 * dx;

    double L_lower = half_sigma_sq / dx_sq - drift / two_dx;
    double L_diag = -2.0 * half_sigma_sq / dx_sq - r;
    double L_upper = half_sigma_sq / dx_sq + drift / two_dx;

    // J = I - dt*L
    double expected_diag = 1.0 - dt * L_diag;
    double expected_lower = -dt * L_lower;
    double expected_upper = -dt * L_upper;

    // Check diagonal
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(diag(i), expected_diag, 1e-12);
    }

    // Check off-diagonals
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_NEAR(lower(i), expected_lower, 1e-12);
        EXPECT_NEAR(upper(i), expected_upper, 1e-12);
    }
}

TEST_F(BlackScholesTest, CoefficientAccessors) {
    double sigma = 0.2;
    double r = 0.05;
    double q = 0.02;

    BlackScholesOperator<HostMemSpace> op(sigma, r, q);

    EXPECT_DOUBLE_EQ(op.half_sigma_sq(), 0.5 * sigma * sigma);
    EXPECT_DOUBLE_EQ(op.drift(), r - q - 0.5 * sigma * sigma);
    EXPECT_DOUBLE_EQ(op.discount_rate(), r);
}

}  // namespace mango::kokkos::test
