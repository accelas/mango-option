// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/discrete_dividend_event.hpp"
#include <vector>
#include <cmath>

using namespace mango;

TEST(DiscreteDividendEventTest, BasicPutShift) {
    std::vector<double> x = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<double> u(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
    }

    double original_atm = u[2];

    auto callback = make_dividend_event(5.0, 100.0, OptionType::PUT);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    EXPECT_GT(u[2], original_atm)
        << "Put value at ATM should increase after dividend (spot drops)";

    // With a coarse 5-point grid, cubic spline interpolation may produce
    // small negative overshoots in OTM regions. Use a relaxed tolerance.
    for (size_t i = 0; i < u.size(); ++i) {
        EXPECT_GE(u[i], -0.01) << "Solution must be approximately non-negative at index " << i;
    }
}

TEST(DiscreteDividendEventTest, NoShiftWhenSpotBelowDividendPut) {
    // Use a grid where the lowest points have S/K < D/K = 0.10,
    // so the dividend shift drives them to S - D <= 0 (put fallback = 1.0)
    std::vector<double> x = {-3.0, -2.5, -2.0, -1.5, -1.0};
    std::vector<double> u(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::max(1.0 - std::exp(x[i]), 0.0);
    }

    // D = 10, K = 100 → d = 0.10. exp(-3.0) ≈ 0.0498 < 0.10 → fallback
    auto callback = make_dividend_event(10.0, 100.0, OptionType::PUT);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    EXPECT_DOUBLE_EQ(u[0], 1.0);
}

TEST(DiscreteDividendEventTest, NoShiftWhenSpotBelowDividendCall) {
    // Same grid setup; exp(-3.0) ≈ 0.0498 < d = 0.10 → call fallback = 0.0
    std::vector<double> x = {-3.0, -2.5, -2.0, -1.5, -1.0};
    std::vector<double> u(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        u[i] = std::max(std::exp(x[i]) - 1.0, 0.0);  // call payoff
    }

    auto callback = make_dividend_event(10.0, 100.0, OptionType::CALL);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    EXPECT_NEAR(u[0], 0.0, 1e-10);
}

TEST(DiscreteDividendEventTest, ZeroDividendNoOp) {
    std::vector<double> x = {-1.0, 0.0, 1.0};
    std::vector<double> u = {0.5, 0.3, 0.1};
    std::vector<double> u_orig = u;

    auto callback = make_dividend_event(0.0, 100.0, OptionType::PUT);
    callback(0.5, std::span<const double>(x), std::span<double>(u));

    for (size_t i = 0; i < u.size(); ++i) {
        EXPECT_DOUBLE_EQ(u[i], u_orig[i]);
    }
}
