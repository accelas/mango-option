#include <gtest/gtest.h>
#include "src/option/price_table_builder.hpp"

namespace mango {
namespace {

TEST(PriceTableBuilderTest, ConstructFromConfig) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    // Just verify construction succeeds
    SUCCEED();
}

TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    // This will fail until we implement the pipeline
    // For now, just verify it returns an error
    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());  // Not implemented yet
}

TEST(PriceTableBuilderTest, MakeBatch4D) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 1.0}}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};      // moneyness: 2 points
    axes.grids[1] = {0.1, 0.5};      // maturity: 2 points
    axes.grids[2] = {0.20};          // volatility: 1 point
    axes.grids[3] = {0.05};          // rate: 1 point

    // Should create 2*2*1*1 = 4 option parameter sets
    auto batch = builder.make_batch_for_testing(axes, 100.0);

    EXPECT_EQ(batch.size(), 4);

    // Check first parameter set
    EXPECT_DOUBLE_EQ(batch[0].spot, 90.0);  // m=0.9, K_ref=100 => S=90
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);
    EXPECT_DOUBLE_EQ(batch[0].maturity, 0.1);
    EXPECT_DOUBLE_EQ(batch[0].volatility, 0.20);
    EXPECT_DOUBLE_EQ(batch[0].rate, 0.05);
    EXPECT_DOUBLE_EQ(batch[0].dividend_yield, 0.02);

    // Check discrete dividends were copied
    EXPECT_EQ(batch[0].discrete_dividends.size(), 1);
    EXPECT_DOUBLE_EQ(batch[0].discrete_dividends[0].first, 0.25);
}

} // namespace
} // namespace mango
