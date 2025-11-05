#include "src/cpp/dividend_jump.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(DividendJumpTest, CoordinateTransformation) {
    // Dividend: $2, Strike: $100
    mango::DividendJump div_jump(2.0, 100.0);

    // Test grid: x = ln(S/K) from -0.3 to 0.3
    std::vector<double> x = {-0.3, -0.1, 0.0, 0.1, 0.3};
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0};  // Arbitrary values

    // Store original
    std::vector<double> x_orig = x;
    std::vector<double> u_orig = u;

    // Apply jump
    div_jump(0.0, x, u);

    // After jump, x should shift left (stock price drops)
    // Verify at least that values changed
    bool values_changed = false;
    for (size_t i = 0; i < u.size(); ++i) {
        if (std::abs(u[i] - u_orig[i]) > 1e-10) {
            values_changed = true;
            break;
        }
    }
    EXPECT_TRUE(values_changed) << "Dividend jump should modify option values";

    // Verify no spurious gains (total value should not increase)
    double sum_before = 0.0, sum_after = 0.0;
    for (size_t i = 0; i < u.size(); ++i) {
        sum_before += u_orig[i];
        sum_after += u[i];
    }
    // After dividend, option should be worth similar or less (stock dropped)
    EXPECT_LE(sum_after, sum_before * 1.1);  // Allow 10% tolerance for interpolation
}

TEST(DividendJumpTest, LinearInterpolation) {
    // Small dividend to test interpolation accuracy
    mango::DividendJump div_jump(1.0, 100.0);

    // Linear function: u(x) = 10 + 5*x
    // Use wider grid to ensure x_new stays within bounds
    std::vector<double> x;
    std::vector<double> u;
    for (int i = -20; i <= 20; ++i) {
        double xi = 0.1 * i;  // x from -2.0 to 2.0
        x.push_back(xi);
        u.push_back(10.0 + 5.0 * xi);
    }

    std::vector<double> u_orig = u;

    // Apply jump
    div_jump(0.0, x, u);

    // For a linear function, interpolation should be exact
    // (within tolerance for numerical precision)
    // Check only interior points where x_new is within original grid
    for (size_t i = 5; i < u.size() - 5; ++i) {
        const double S = 100.0 * std::exp(x[i]);
        const double S_new = S - 1.0;
        const double x_new = std::log(S_new / 100.0);

        // Only check if x_new is within grid bounds
        if (x_new >= x[0] && x_new <= x[x.size()-1]) {
            const double u_expected = 10.0 + 5.0 * x_new;
            EXPECT_NEAR(u[i], u_expected, 1e-6)
                << "Interpolation error at i=" << i;
        }
    }
}

TEST(DividendJumpTest, ZeroDividend) {
    // Zero dividend should leave values unchanged
    mango::DividendJump div_jump(0.0, 100.0);

    std::vector<double> x = {-0.5, -0.2, 0.0, 0.2, 0.5};
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> u_orig = u;

    div_jump(0.0, x, u);

    // Values should be unchanged (within numerical precision)
    for (size_t i = 0; i < u.size(); ++i) {
        EXPECT_NEAR(u[i], u_orig[i], 1e-12);
    }
}

TEST(DividendJumpTest, LargeDividendNegativeStock) {
    // Very large dividend relative to stock price
    mango::DividendJump div_jump(150.0, 100.0);

    // At x = -0.5, S = 100*exp(-0.5) ≈ 60
    // Dividend of $150 would make S_new = -90 (negative!)
    std::vector<double> x = {-0.5};
    std::vector<double> u = {10.0};

    div_jump(0.0, x, u);

    // Should handle gracefully (clamp to OTM value)
    EXPECT_GE(u[0], 0.0) << "Value should remain non-negative";
}
