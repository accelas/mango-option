// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"

using namespace mango;

// Regression: raw segmented samples must retain direct PDE price accuracy.
// Bug: Fitted initial conditions contaminated subsequent temporal segments.
TEST(SegmentedPriceTableBuilderTest, LowVolatilityFixedExpiryMatchesDirectSolve) {
    std::vector<double> x;
    for (int i = -10; i <= 10; ++i) x.push_back(0.05 * i);
    SegmentedPriceTableBuilder::Config config{
        .K_ref = 100.0,
        .option_type = OptionType::PUT,
        .dividends = {.discrete_dividends = {{0.5, 3.0}}},
        .grid = {.moneyness = x, .vol = {0.1, 0.15, 0.2, 0.3},
                 .rate = {0.02, 0.03, 0.05, 0.07}},
        .maturity = 1.0,
        .tau_points_per_segment = 5,
        .pde_accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra),
    };
    auto surface = SegmentedPriceTableBuilder::build(config);
    ASSERT_TRUE(surface.has_value()) << surface.error();

    // At remaining maturity .75, elapsed time is .25, so the originally
    // anchored .5 dividend is now .25 away. K equals K_ref: no strike blend.
    PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
        .maturity = 0.75, .rate = 0.05, .option_type = OptionType::PUT}, 0.1);
    p.discrete_dividends = {{0.25, 3.0}};
    auto high = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
    auto ultra = AmericanOptionSolver::create(
        p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
    ASSERT_TRUE(high.has_value());
    ASSERT_TRUE(ultra.has_value());
    auto reference = high->solve();
    auto converged = ultra->solve();
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(converged.has_value());
    ASSERT_NEAR(reference->value(), converged->value(), 0.001);
    const double actual = surface->price(100.0, 100.0, 0.75, 0.1, 0.05);
    EXPECT_NEAR(actual, converged->value(), 0.003);
}

TEST(SegmentedPriceTableBuilderTest, RawCallAndPutPricesAndGreeksShareTheSameSurface) {
    std::vector<double> x;
    for (int i = -10; i <= 10; ++i) x.push_back(0.05 * i);
    for (auto type : {OptionType::PUT, OptionType::CALL}) {
        SCOPED_TRACE(static_cast<int>(type));
        SegmentedPriceTableBuilder::Config config{
            .K_ref = 100.0, .option_type = type,
            .dividends = {.discrete_dividends = {{0.5, 3.0}}},
            .grid = {.moneyness = x, .vol = {0.1, 0.15, 0.2, 0.3},
                     .rate = {0.02, 0.03, 0.05, 0.07}},
            .maturity = 1.0,
            .pde_accuracy = make_grid_accuracy(GridAccuracyProfile::High),
        };
        auto surface = SegmentedPriceTableBuilder::build(config);
        ASSERT_TRUE(surface.has_value()) << surface.error();
        for (double tau : {0.25, 0.75}) {
            PricingParams p(OptionSpec{.spot = 100.0, .strike = 100.0,
                .maturity = tau, .rate = 0.05, .option_type = type}, 0.2);
            if (tau > 0.5) p.discrete_dividends = {{tau - 0.5, 3.0}};
            auto fine = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::High)});
            auto finer = AmericanOptionSolver::create(
                p, PDEGridSpec{make_grid_accuracy(GridAccuracyProfile::Ultra)});
            ASSERT_TRUE(fine.has_value());
            ASSERT_TRUE(finer.has_value());
            auto reference = fine->solve();
            auto converged = finer->solve();
            ASSERT_TRUE(reference.has_value());
            ASSERT_TRUE(converged.has_value());
            ASSERT_NEAR(reference->value(), converged->value(), 0.001);
            const auto price = [&](double spot, double time, double sigma, double rate) {
                return surface->price(spot, 100.0, time, sigma, rate);
            };
            EXPECT_NEAR(price(100, tau, 0.2, 0.05), converged->value(), 0.003);

            // Public Greeks must differentiate the delivered raw-sample
            // surface. Finite differences use interior points in each axis.
            auto delta = surface->greek(Greek::Delta, p);
            auto gamma = surface->gamma(p);
            auto theta = surface->greek(Greek::Theta, p);
            auto rho = surface->greek(Greek::Rho, p);
            ASSERT_TRUE(delta.has_value());
            ASSERT_TRUE(gamma.has_value());
            ASSERT_TRUE(theta.has_value());
            ASSERT_TRUE(rho.has_value());
            constexpr double hs = 0.005, h = 1e-5;
            const double mid = price(100, tau, 0.2, 0.05);
            const double up = price(100 + hs, tau, 0.2, 0.05);
            const double down = price(100 - hs, tau, 0.2, 0.05);
            EXPECT_NEAR(*delta, (up - down) / (2 * hs), 1e-7);
            EXPECT_NEAR(*gamma, (up - 2 * mid + down) / (hs * hs), 1e-7);
            EXPECT_NEAR(surface->vega(100, 100, tau, 0.2, 0.05),
                (price(100, tau, 0.2 + h, 0.05) - price(100, tau, 0.2 - h, 0.05)) / (2 * h), 1e-6);
            EXPECT_NEAR(*theta,
                (price(100, tau - h, 0.2, 0.05) - price(100, tau + h, 0.2, 0.05)) / (2 * h), 1e-6);
            EXPECT_NEAR(*rho,
                (price(100, tau, 0.2, 0.05 + h) - price(100, tau, 0.2, 0.05 - h)) / (2 * h), 1e-6);
        }
    }
}
