// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_config.hpp"
#include <cmath>

TEST(PriceTableBuilderGridTest, RespectsUserGridBounds) {
    // Create axes with specific log-moneyness range
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};  // log-moneyness
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"log_moneyness", "maturity", "volatility", "rate"};

    // Configure grid with specific bounds that cover log(0.8) to log(1.2)
    // log(0.8) ≈ -0.223, log(1.2) ≈ 0.182
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.pde_grid = mango::PDEGridConfig{mango::GridSpec<double>::uniform(-0.5, 0.5, 51).value(), 100};

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should succeed because grid bounds cover moneyness range
    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();
    EXPECT_NE(result->surface, nullptr);
}

TEST(PriceTableBuilderGridTest, RejectsInsufficientGridBounds) {
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.5), std::log(0.75), std::log(1.0), std::log(1.5), std::log(2.0)};  // Wide log-moneyness range
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"log_moneyness", "maturity", "volatility", "rate"};

    // Configure grid with narrow bounds that don't cover the range
    // log(0.5) ≈ -0.693, log(2.0) ≈ 0.693, but grid only [-0.1, 0.1]
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.pde_grid = mango::PDEGridConfig{mango::GridSpec<double>::uniform(-0.1, 0.1, 51).value(), 100};

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should fail validation because grid bounds don't cover moneyness range
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::PriceTableErrorCode::InvalidConfig);
}
