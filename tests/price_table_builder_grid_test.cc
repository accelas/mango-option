#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_config.hpp"

TEST(PriceTableBuilderGridTest, RespectsUserGridBounds) {
    // Create axes with specific moneyness range
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1, 1.2};  // moneyness range [0.8, 1.2]
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Configure grid with specific bounds that cover log(0.8) to log(1.2)
    // log(0.8) ≈ -0.223, log(1.2) ≈ 0.182
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.grid_estimator = mango::GridSpec<double>::uniform(-0.5, 0.5, 51).value();
    config.n_time = 100;

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should succeed because grid bounds cover moneyness range
    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();
    EXPECT_NE(result->surface, nullptr);
}

TEST(PriceTableBuilderGridTest, RejectsInsufficientGridBounds) {
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {0.5, 0.75, 1.0, 1.5, 2.0};  // Wide moneyness range
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.03, 0.04, 0.05, 0.06};
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Configure grid with narrow bounds that don't cover the range
    // log(0.5) ≈ -0.693, log(2.0) ≈ 0.693, but grid only [-0.1, 0.1]
    mango::PriceTableConfig config;
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;
    config.grid_estimator = mango::GridSpec<double>::uniform(-0.1, 0.1, 51).value();
    config.n_time = 100;

    mango::PriceTableBuilder<4> builder(config);
    auto result = builder.build(axes);

    // Should fail validation because grid bounds don't cover moneyness range
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::PriceTableErrorCode::InvalidConfig);
}
