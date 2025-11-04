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

TEST(BlackScholesOperatorTest, IndexOperatorWithDividend) {
    // Create operator with dividend yield q=0.03
    mango::IndexBlackScholesOperator op(0.05, 0.2, 0.03);

    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();

    // Test with u(S) = S (delta = 1, gamma = 0)
    std::vector<double> u(41);
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    std::vector<double> Lu(41);
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S with dividend:
    // du/dS = 1, d2u/dS2 = 0
    // L(u) = 0 + (r - q)*S*1 - r*S = (r - q - r)*S = -q*S
    double S = grid[20];  // S = 100
    double expected = -0.03 * S;  // -q*S
    EXPECT_NEAR(Lu[20], expected, 0.01);
}

TEST(BlackScholesOperatorTest, IndexVsEquityDifference) {
    // Verify dividend yield affects operator output
    double r = 0.05, sigma = 0.2, q = 0.03;

    mango::EquityBlackScholesOperator equity_op(r, sigma);
    mango::IndexBlackScholesOperator index_op(r, sigma, q);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();

    std::vector<double> u(21);
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i];  // u(S) = S
    }

    std::vector<double> Lu_equity(21), Lu_index(21);
    equity_op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu_equity));
    index_op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu_index));

    // For u(S) = S:
    // Equity: L(u) = r*S - r*S = 0
    // Index:  L(u) = (r-q)*S - r*S = -q*S
    // Difference should be -q*S
    double S = grid[10];
    EXPECT_NEAR(Lu_equity[10], 0.0, 0.01);
    EXPECT_NEAR(Lu_index[10], -q * S, 0.01);
    EXPECT_NEAR(Lu_index[10] - Lu_equity[10], -q * S, 0.01);
}
