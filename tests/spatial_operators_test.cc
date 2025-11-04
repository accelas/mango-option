#include "src/cpp/spatial_operators.hpp"
#include "src/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(BlackScholesOperatorTest, EquityOperatorBasic) {
    // Create operator for equity option
    // Parameters: r=0.05, sigma=0.2, no dividends
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    // Create simple grid
    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();

    // Test input: linear function u(S) = S (delta = 1, gamma = 0)
    std::vector<double> u(41);
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    std::vector<double> Lu(41);

    // Apply operator
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S, the Black-Scholes operator gives:
    // L(u) = r*S*du/dS - r*u = r*S*1 - r*S = 0
    // (Middle points should be approximately zero)
    EXPECT_NEAR(Lu[20], 0.0, 0.01);  // S=100
}

TEST(BlackScholesOperatorTest, EquityOperatorParabolic) {
    // Test with u(S) = S^2 (delta = 2S, gamma = 2)
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();

    std::vector<double> u(21);
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i] * grid[i];
    }

    std::vector<double> Lu(21);
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S^2:
    // du/dS = 2S, d2u/dS2 = 2
    // L(u) = 0.5*sigma^2*S^2*2 + r*S*2S - r*S^2
    //      = sigma^2*S^2 + 2*r*S^2 - r*S^2
    //      = sigma^2*S^2 + r*S^2
    double S = grid[10];  // Middle point
    double expected = 0.2*0.2*S*S + 0.05*S*S;
    EXPECT_NEAR(Lu[10], expected, 0.1);
}
