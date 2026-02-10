// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_config.hpp"
#include <cmath>

TEST(PriceTableBuilderResultTest, BuildReturnsDiagnostics) {
    // Create minimal valid axes (4 points per axis for B-spline)
    mango::PriceTableAxes axes;
    axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)};  // log-moneyness
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};  // maturity
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};  // volatility
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};  // rate
    axes.names = {"log_moneyness", "maturity", "volatility", "rate"};

    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.pde_grid = mango::PDEGridConfig{mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(), 100};

    mango::PriceTableBuilder builder(config);
    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();

    // Check diagnostics are populated
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->n_pde_solves, 0);
    EXPECT_GT(result->precompute_time_seconds, 0.0);
    // Fitting stats should be populated (exact values depend on data)
    EXPECT_GE(result->fitting_stats.max_residual_overall, 0.0);
}
