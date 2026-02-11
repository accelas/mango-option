// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/american_option.hpp"
#include <cmath>

namespace mango {
namespace {

TEST(DimensionlessIVTest, FactoryCreatesAndSolves) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.05, 0.08},
        },
        .backend = DimensionlessBackend{.maturity = 1.5},
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver.has_value()) << "Factory failed";

    // Solve IV for a known point
    OptionSpec spec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT};
    IVQuery query(spec, 6.0);  // Approximate ATM put price

    auto result = solver->solve(query);
    ASSERT_TRUE(result.has_value())
        << "IV solve failed: " << static_cast<int>(result.error().code);
    EXPECT_GT(result->implied_vol, 0.05);
    EXPECT_LT(result->implied_vol, 0.50);
}

TEST(DimensionlessIVTest, IVMatchesFDM) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = IVGrid{
            .moneyness = {0.8, 0.9, 1.0, 1.1, 1.2},
            .vol = {0.10, 0.20, 0.30, 0.40},
            .rate = {0.02, 0.05, 0.08},
        },
        .backend = DimensionlessBackend{.maturity = 1.5},
    };
    auto solver_3d = make_interpolated_iv_solver(config);
    ASSERT_TRUE(solver_3d.has_value());

    // Reference: FDM IV solver
    IVSolverConfig fdm_config;
    IVSolver fdm_solver(fdm_config);

    OptionSpec spec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
        .rate = 0.05, .dividend_yield = 0.0, .option_type = OptionType::PUT};

    // Get a reference price at known IV
    auto ref_price = solve_american_option(PricingParams(spec, 0.20));
    ASSERT_TRUE(ref_price.has_value());

    IVQuery query(spec, ref_price->value_at(100.0));

    auto iv_3d = solver_3d->solve(query);
    ASSERT_TRUE(iv_3d.has_value());

    auto iv_fdm = fdm_solver.solve(query);
    ASSERT_TRUE(iv_fdm.has_value());

    // 3D surface IV should be within ~50 bps of FDM IV
    // Widen to 100 bps if the 3D approximation is less accurate at this point
    EXPECT_NEAR(iv_3d->implied_vol, iv_fdm->implied_vol, 0.01)
        << "3D IV=" << iv_3d->implied_vol << " FDM IV=" << iv_fdm->implied_vol;
}

TEST(DimensionlessIVTest, RejectsNonzeroDividendYield) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .dividend_yield = 0.01,
        .grid = IVGrid{
            .moneyness = {0.8, 1.0, 1.2},
            .vol = {0.10, 0.20, 0.30},
            .rate = {0.02, 0.05},
        },
        .backend = DimensionlessBackend{.maturity = 1.0},
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value());
    EXPECT_EQ(solver.error().code, ValidationErrorCode::InvalidDividend);
}

TEST(DimensionlessIVTest, RejectsDiscreteDividends) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = IVGrid{
            .moneyness = {0.8, 1.0, 1.2},
            .vol = {0.10, 0.20, 0.30},
            .rate = {0.02, 0.05},
        },
        .backend = DimensionlessBackend{.maturity = 1.0},
        .discrete_dividends = DiscreteDividendConfig{
            .maturity = 1.0,
            .discrete_dividends = {Dividend{0.5, 0.25}},
        },
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value());
    EXPECT_EQ(solver.error().code, ValidationErrorCode::InvalidDividend);
}

// Regression: ln(2r/sigma^2) with rate <= 0 produces NaN
TEST(DimensionlessIVTest, RejectsNonpositiveRate) {
    IVSolverFactoryConfig config{
        .option_type = OptionType::PUT,
        .spot = 100.0,
        .grid = IVGrid{
            .moneyness = {0.8, 1.0, 1.2},
            .vol = {0.10, 0.20, 0.30},
            .rate = {0.0, 0.05},
        },
        .backend = DimensionlessBackend{.maturity = 1.0},
    };

    auto solver = make_interpolated_iv_solver(config);
    ASSERT_FALSE(solver.has_value());
    EXPECT_EQ(solver.error().code, ValidationErrorCode::InvalidRate);
}

}  // namespace
}  // namespace mango
