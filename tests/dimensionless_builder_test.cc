// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/dimensionless/dimensionless_builder.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/european_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(DimensionlessBuilderTest, BuildsPutSurface) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    axes.tau_prime = {0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14};
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5};

    auto result = build_dimensionless_surface(axes, 100.0, OptionType::PUT);
    ASSERT_TRUE(result.has_value())
        << "Build failed: code=" << static_cast<int>(result.error().code);

    EXPECT_EQ(result->n_pde_solves, static_cast<int>(axes.ln_kappa.size()));
    EXPECT_NE(result->surface, nullptr);

    // EEP at ATM, moderate tau', kappa=1 should be non-negative and small
    double eep = result->surface->value({0.0, 0.04, 0.0});
    EXPECT_GE(eep, 0.0);
    EXPECT_LT(eep, 0.2);
}

TEST(DimensionlessBuilderTest, EEPMatchesPDE) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    axes.tau_prime = {0.01, 0.02, 0.03125, 0.04, 0.06, 0.08, 0.10, 0.14};
    axes.ln_kappa = {-1.5, -1.0, -0.5, 0.0, 0.247, 0.5, 1.0, 1.5};

    auto build_result = build_dimensionless_surface(axes, 100.0, OptionType::PUT);
    ASSERT_TRUE(build_result.has_value());

    // Physical parameters: sigma=0.25, rate=0.04, maturity=1.0
    const double sigma = 0.25, rate = 0.04, maturity = 1.0;
    const double tau_prime = sigma * sigma * maturity / 2.0;
    const double kappa = 2.0 * rate / (sigma * sigma);
    const double ln_kappa = std::log(kappa);
    const double K_ref = 100.0;

    // Solve American option in physical coordinates
    auto am = solve_american_option(PricingParams(
        OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = maturity,
            .rate = rate, .dividend_yield = 0.0,
            .option_type = OptionType::PUT}, sigma));
    ASSERT_TRUE(am.has_value());

    // Solve European option in physical coordinates
    auto eu = EuropeanOptionSolver(
        OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = maturity,
            .rate = rate, .dividend_yield = 0.0,
            .option_type = OptionType::PUT}, sigma).solve();
    ASSERT_TRUE(eu.has_value());

    // Reference EEP normalized by K
    double ref_eep = (am->value() - eu->value()) / K_ref;

    // Surface EEP at the same dimensionless coordinates
    double surface_eep = build_result->surface->value({0.0, tau_prime, ln_kappa});

    EXPECT_NEAR(surface_eep, ref_eep, 0.002);
}

TEST(DimensionlessBuilderTest, RejectsInsufficientGrid) {
    DimensionlessAxes axes;
    axes.log_moneyness = {-0.1, 0.0, 0.1};  // Only 3 -- too few
    axes.tau_prime = {0.01, 0.02, 0.04, 0.06};
    axes.ln_kappa = {-1.0, 0.0, 0.5, 1.0};

    auto result = build_dimensionless_surface(axes, 100.0, OptionType::PUT);
    ASSERT_FALSE(result.has_value());
}

}  // namespace
}  // namespace mango
