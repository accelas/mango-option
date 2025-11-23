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

// REGRESSION TEST: Verify build() returns "not yet implemented" for all dimensions
// Issue: build() is incomplete, test documents this limitation
TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    PriceTableConfig config;
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    // NOTE: build() is a skeleton implementation
    // Will be completed in Phases 8-10 of price table refactor
    // For now, verify it returns expected error
    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "PriceTableBuilder::build() not yet implemented");
}

// REGRESSION TEST: build() incomplete for 2D
TEST(PriceTableBuilderTest, Build2DNotImplemented) {
    PriceTableConfig config;
    PriceTableBuilder<2> builder(config);

    PriceTableAxes<2> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};

    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "PriceTableBuilder::build() not yet implemented");
}

// REGRESSION TEST: build() incomplete for 3D
TEST(PriceTableBuilderTest, Build3DNotImplemented) {
    PriceTableConfig config;
    PriceTableBuilder<3> builder(config);

    PriceTableAxes<3> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};

    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "PriceTableBuilder::build() not yet implemented");
}

// REGRESSION TEST: build() incomplete for 5D
TEST(PriceTableBuilderTest, Build5DNotImplemented) {
    PriceTableConfig config;
    PriceTableBuilder<5> builder(config);

    PriceTableAxes<5> axes;
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0, 1.5};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};
    axes.grids[4] = {0.0, 0.01, 0.02, 0.03};

    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "PriceTableBuilder::build() not yet implemented");
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

// REGRESSION TEST: make_batch returns empty for N=2 (not 4D)
// Issue: make_batch only supports 4D grids, returns empty for other dimensions
TEST(PriceTableBuilderTest, MakeBatch2DReturnsEmpty) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<2> builder(config);

    PriceTableAxes<2> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};
    axes.grids[1] = {0.1, 0.5, 1.0};

    // Should return empty batch (N != 4)
    auto batch = builder.make_batch_for_testing(axes, 100.0);
    EXPECT_TRUE(batch.empty());
}

// REGRESSION TEST: make_batch returns empty for N=3 (not 4D)
TEST(PriceTableBuilderTest, MakeBatch3DReturnsEmpty) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<3> builder(config);

    PriceTableAxes<3> axes;
    axes.grids[0] = {0.9, 1.0};
    axes.grids[1] = {0.1, 0.5};
    axes.grids[2] = {0.20, 0.25};

    // Should return empty batch (N != 4)
    auto batch = builder.make_batch_for_testing(axes, 100.0);
    EXPECT_TRUE(batch.empty());
}

// REGRESSION TEST: make_batch returns empty for N=5 (not 4D)
TEST(PriceTableBuilderTest, MakeBatch5DReturnsEmpty) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<5> builder(config);

    PriceTableAxes<5> axes;
    axes.grids[0] = {0.9, 1.0};
    axes.grids[1] = {0.1, 0.5};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};
    axes.grids[4] = {0.01};

    // Should return empty batch (N != 4)
    // This documents the 4D-only limitation
    auto batch = builder.make_batch_for_testing(axes, 100.0);
    EXPECT_TRUE(batch.empty());
}

} // namespace
} // namespace mango
