// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_config.hpp"

TEST(PriceTableBuilderResultTest, BuildReturnsDiagnostics) {
    // Create minimal valid axes (4 points per axis for B-spline)
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};  // moneyness
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};  // maturity
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};  // volatility
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};  // rate
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.pde_grid = mango::PDEGridConfig{mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(), 100};

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();

    // Check diagnostics are populated
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->n_pde_solves, 0);
    EXPECT_GT(result->precompute_time_seconds, 0.0);
    // Fitting stats should be populated (exact values depend on data)
    EXPECT_GE(result->fitting_stats.max_residual_overall, 0.0);
}
