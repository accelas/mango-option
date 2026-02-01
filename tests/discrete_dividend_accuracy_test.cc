// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include <cmath>

using namespace mango;

TEST(DiscreteDividendAccuracyTest, PutPriceWithinReasonableBounds) {
    // ATM put, S=100, K=100, T=1, sigma=0.20, r=0.05
    // Discrete dividend: $3 at t=0.5
    PricingParams with_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                           {{0.5, 3.0}});

    auto result = solve_american_option_auto(with_div);
    ASSERT_TRUE(result.has_value());
    double price = result->value();

    // Lower bound: American put on S-PV(D) (spot adjusted for PV of dividend)
    // PV(D) = 3 * exp(-0.05 * 0.5) ≈ 2.926
    // S_adj ≈ 97.07
    PricingParams lower_bound(97.07, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20);
    auto lb_result = solve_american_option_auto(lower_bound);
    ASSERT_TRUE(lb_result.has_value());

    // Price should be in a reasonable range
    EXPECT_GT(price, 3.0) << "Put with dividend must be worth more than intrinsic bump";
    EXPECT_LT(price, 25.0) << "Put price should be reasonable";

    // Should be close to the spot-adjusted price (within ~15% relative)
    double ref = lb_result->value();
    EXPECT_NEAR(price, ref, ref * 0.15)
        << "Discrete dividend put should be close to spot-adjusted reference";
}

TEST(DiscreteDividendAccuracyTest, LargeDividendStressTest) {
    // Large dividend: $20 on S=100 (20% of spot)
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.30,
                         {{0.5, 20.0}});

    auto result = solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_TRUE(std::isfinite(result->value()));
}

TEST(DiscreteDividendAccuracyTest, DividendNearExpiry) {
    // Dividend very close to expiry (t=0.95, T=1.0)
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                         {{0.95, 2.0}});

    auto result = solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value(), 0.0);
    EXPECT_TRUE(std::isfinite(result->value()));
}

TEST(DiscreteDividendAccuracyTest, EventAlignsWithMandatoryTimePoint) {
    // Verify that the time grid actually contains the dividend date
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                         {{0.3, 2.0}});

    auto [grid_spec, td] = estimate_grid_for_option(params);
    auto pts = td.time_points();

    // tau = T - t_cal = 1.0 - 0.3 = 0.7
    double tau_div = 0.7;
    bool found = false;
    for (double p : pts) {
        if (std::abs(p - tau_div) < 1e-14) { found = true; break; }
    }
    EXPECT_TRUE(found) << "Time grid must land exactly on dividend tau=" << tau_div;
}

TEST(DiscreteDividendAccuracyTest, DividendAtBoundariesIgnored) {
    // Dividends at t=0 or t=T should be silently ignored (no crash)
    PricingParams params(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20,
                         {{0.0, 5.0}, {1.0, 5.0}});

    auto result = solve_american_option_auto(params);
    ASSERT_TRUE(result.has_value());

    // Should produce the same price as no dividends
    PricingParams no_div(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 0.20);
    auto result_no_div = solve_american_option_auto(no_div);
    ASSERT_TRUE(result_no_div.has_value());

    EXPECT_NEAR(result->value(), result_no_div->value(), 1e-10)
        << "Boundary dividends should be ignored";
}
