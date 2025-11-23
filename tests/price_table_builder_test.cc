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

TEST(PriceTableBuilderTest, MakeBatchIteratesVolatilityAndRateOnly) {
    // Design: make_batch should iterate axes[2] × axes[3] only (vol × rate)
    // NOT all grid points (would explode PDE count)

    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 1000,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};      // moneyness: 3 points
    axes.grids[1] = {0.1, 0.5, 1.0};      // maturity: 3 points
    axes.grids[2] = {0.15, 0.20, 0.25};   // volatility: 3 points
    axes.grids[3] = {0.02, 0.05};         // rate: 2 points

    // Should create 3 × 2 = 6 batch entries (vol × rate)
    // NOT 3 × 3 × 3 × 2 = 54 entries (all axes)
    auto batch = builder.make_batch_for_testing(axes);

    EXPECT_EQ(batch.size(), 6);  // Nσ × Nr

    // Verify all batch entries use normalized params (Spot = Strike = K_ref)
    for (const auto& params : batch) {
        EXPECT_DOUBLE_EQ(params.spot, 100.0);
        EXPECT_DOUBLE_EQ(params.strike, 100.0);
    }
}

TEST(PriceTableBuilderTest, MakeBatch4D) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .dividend_yield = 0.02,
        .discrete_dividends = {{0.25, 1.0}}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};      // moneyness: 2 points
    axes.grids[1] = {0.1, 0.5};      // maturity: 2 points
    axes.grids[2] = {0.20};          // volatility: 1 point
    axes.grids[3] = {0.05};          // rate: 1 point

    // Should create 1 × 1 = 1 option (vol × rate)
    // NOT 2 × 2 × 1 × 1 = 4 options
    auto batch = builder.make_batch_for_testing(axes);
    EXPECT_EQ(batch.size(), 1);  // 1 vol × 1 rate

    // Check parameter set - should be normalized (Spot = Strike = K_ref)
    EXPECT_DOUBLE_EQ(batch[0].spot, 100.0);     // Normalized
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);   // K_ref
    EXPECT_DOUBLE_EQ(batch[0].maturity, 0.5);   // Max maturity
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
    auto batch = builder.make_batch_for_testing(axes);
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
    auto batch = builder.make_batch_for_testing(axes);
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
    auto batch = builder.make_batch_for_testing(axes);
    EXPECT_TRUE(batch.empty());
}

TEST(PriceTableBuilderTest, SolveBatchRegistersMaturitySnapshots) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 101).value(),
        .n_time = 1000,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0};
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 maturity points
    axes.grids[2] = {0.20};           // 1 vol
    axes.grids[3] = {0.05};           // 1 rate

    auto batch_params = builder.make_batch_for_testing(axes);
    auto batch_result = builder.solve_batch_for_testing(batch_params, axes);

    // Verify snapshots were registered (should have 3 snapshots)
    ASSERT_EQ(batch_result.results.size(), 1);
    ASSERT_TRUE(batch_result.results[0].has_value());

    auto grid = batch_result.results[0]->grid();
    EXPECT_GE(grid->num_snapshots(), axes.grids[1].size());

    // Verify grid size matches config (should be 101 points)
    EXPECT_EQ(grid->n_space(), 101);

    // Verify snapshots match maturity grid values
    EXPECT_EQ(grid->num_snapshots(), 3);
}

} // namespace
} // namespace mango
