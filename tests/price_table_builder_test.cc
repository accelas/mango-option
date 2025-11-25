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

// Smoke test: Verify build() pipeline works with minimal grid
// Uses small grid (4×4×4×4 minimum for B-spline, auto-estimated spatial/time)
TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    // Use default grid estimator (auto-estimation) with reduced time steps
    PriceTableConfig config{
        .n_time = 100  // Reduce from default 5000 for faster test
    };
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    // Minimum 4 points per axis for cubic B-spline fitting
    axes.grids[0] = {0.8, 0.9, 1.0, 1.1};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.15, 0.20, 0.25, 0.30};
    axes.grids[3] = {0.02, 0.04, 0.06, 0.08};

    // Full pipeline should succeed (4×4=16 PDE solves, ~1s)
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();
    EXPECT_NE(result->surface, nullptr);
}

// Note: N≠4 tests removed - PriceTableBuilder uses static_assert(N == 4)
// which produces compile-time errors for unsupported dimensions.

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

TEST(PriceTableBuilderTest, SolveBatchRegistersMaturitySnapshots) {
    // Use small grid for fast test (21 spatial, 100 time steps)
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 21).value(),
        .n_time = 100,
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

    // Verify grid is reasonable (auto-estimated, so don't assert exact size)
    EXPECT_GE(grid->n_space(), 20);  // At least 20 spatial points
    EXPECT_LE(grid->n_space(), 1200);  // At most 1200 spatial points

    // Verify snapshots match maturity grid values
    EXPECT_EQ(grid->num_snapshots(), 3);
}

TEST(PriceTableBuilderTest, ExtractTensorInterpolatesSurfaces) {
    // Use small grid for fast test (21 spatial, 100 time steps)
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .grid_estimator = GridSpec<double>::uniform(-3.0, 3.0, 21).value(),
        .n_time = 100,
        .dividend_yield = 0.02
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1};      // 3 moneyness points
    axes.grids[1] = {0.1, 0.5, 1.0};      // 3 maturity points
    axes.grids[2] = {0.20};               // 1 vol
    axes.grids[3] = {0.05};               // 1 rate

    auto batch_params = builder.make_batch_for_testing(axes);
    auto batch_result = builder.solve_batch_for_testing(batch_params, axes);
    auto tensor_result = builder.extract_tensor_for_testing(batch_result, axes);

    ASSERT_TRUE(tensor_result.has_value());
    auto tensor = tensor_result.value();

    // Tensor should have full 4D shape: 3×3×1×1 = 9 points
    EXPECT_EQ(tensor.view.extent(0), 3);  // moneyness
    EXPECT_EQ(tensor.view.extent(1), 3);  // maturity
    EXPECT_EQ(tensor.view.extent(2), 1);  // volatility
    EXPECT_EQ(tensor.view.extent(3), 1);  // rate

    // Verify prices are populated (not NaN or zero)
    // Note: K_ref scaling should now be applied
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double price = tensor.view[i, j, 0, 0];
            EXPECT_TRUE(std::isfinite(price));
            EXPECT_GT(price, 0.0);
        }
    }
}

} // namespace
} // namespace mango
