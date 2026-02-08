// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/price_table_builder.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include "mango/option/table/price_table_metadata.hpp"
#include "mango/option/table/price_tensor.hpp"

namespace mango {
namespace {

using Access = testing::PriceTableBuilderAccess<4>;

// Smoke test: Verify build() pipeline works with minimal grid
// Uses small grid (4×4×4×4 minimum for B-spline, auto-estimated spatial/time)
TEST(PriceTableBuilderTest, BuildEmpty4DSurface) {
    // Use default grid estimator (auto-estimation) with reduced time steps
    PriceTableConfig config{
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100}  // Reduce from default 1000 for faster test
    };
    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    // Minimum 4 points per axis for cubic B-spline fitting
    axes.grids[0] = {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)};
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
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 1000},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0), std::log(1.1)};  // log-moneyness: 3 points
    axes.grids[1] = {0.1, 0.5, 1.0};      // maturity: 3 points
    axes.grids[2] = {0.15, 0.20, 0.25};   // volatility: 3 points
    axes.grids[3] = {0.02, 0.05};         // rate: 2 points

    // Should create 3 × 2 = 6 batch entries (vol × rate)
    // NOT 3 × 3 × 3 × 2 = 54 entries (all axes)
    auto batch = Access::make_batch(builder, axes);

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
        .dividends = {.dividend_yield = 0.02, .discrete_dividends = {{.calendar_time = 0.25, .amount = 1.0}}}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};  // log-moneyness: 2 points
    axes.grids[1] = {0.1, 0.5};      // maturity: 2 points
    axes.grids[2] = {0.20};          // volatility: 1 point
    axes.grids[3] = {0.05};          // rate: 1 point

    // Should create 1 × 1 = 1 option (vol × rate)
    // NOT 2 × 2 × 1 × 1 = 4 options
    auto batch = Access::make_batch(builder, axes);
    EXPECT_EQ(batch.size(), 1);  // 1 vol × 1 rate

    // Check parameter set - should be normalized (Spot = Strike = K_ref)
    EXPECT_DOUBLE_EQ(batch[0].spot, 100.0);     // Normalized
    EXPECT_DOUBLE_EQ(batch[0].strike, 100.0);   // K_ref
    EXPECT_DOUBLE_EQ(batch[0].maturity, 0.5);   // Max maturity
    EXPECT_DOUBLE_EQ(batch[0].volatility, 0.20);
    EXPECT_TRUE(std::holds_alternative<double>(batch[0].rate));
    EXPECT_DOUBLE_EQ(std::get<double>(batch[0].rate), 0.05);
    EXPECT_DOUBLE_EQ(batch[0].dividend_yield, 0.02);

    // Check discrete dividends were copied
    EXPECT_EQ(batch[0].discrete_dividends.size(), 1);
    EXPECT_DOUBLE_EQ(batch[0].discrete_dividends[0].calendar_time, 0.25);
}

TEST(PriceTableBuilderTest, SolveBatchRegistersMaturitySnapshots) {
    // Use small grid for fast test (21 spatial, 100 time steps)
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 21).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};
    axes.grids[1] = {0.1, 0.5, 1.0};  // 3 maturity points
    axes.grids[2] = {0.20};           // 1 vol
    axes.grids[3] = {0.05};           // 1 rate

    auto batch_params = Access::make_batch(builder, axes);
    auto batch_result = Access::solve_batch(builder, batch_params, axes);

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

TEST(PriceTableBuilderTest, DISABLED_ExtractTensorInterpolatesSurfaces) {
    // Use small grid for fast test (21 spatial, 100 time steps)
    // DISABLED: EEP → 0 at (m=0.9, T=0.1) — ITM put near expiry where
    //   American price ≈ intrinsic and early exercise premium vanishes (#352)
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 21).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0), std::log(1.1)};  // 3 log-moneyness points
    axes.grids[1] = {0.1, 0.5, 1.0};      // 3 maturity points
    axes.grids[2] = {0.20};               // 1 vol
    axes.grids[3] = {0.05};               // 1 rate

    auto batch_params = Access::make_batch(builder, axes);
    auto batch_result = Access::solve_batch(builder, batch_params, axes);
    auto tensor_result = Access::extract_tensor(builder, batch_result, axes);

    ASSERT_TRUE(tensor_result.has_value());
    auto& extraction = tensor_result.value();

    // Tensor should have full 4D shape: 3×3×1×1 = 9 points
    EXPECT_EQ(extraction.tensor.view.extent(0), 3);  // moneyness
    EXPECT_EQ(extraction.tensor.view.extent(1), 3);  // maturity
    EXPECT_EQ(extraction.tensor.view.extent(2), 1);  // volatility
    EXPECT_EQ(extraction.tensor.view.extent(3), 1);  // rate

    // Verify prices are populated (not NaN or zero)
    // Note: K_ref scaling should now be applied
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double price = extraction.tensor.view[i, j, 0, 0];
            EXPECT_TRUE(std::isfinite(price));
            EXPECT_GT(price, 0.0);
        }
    }
}

TEST(PriceTableConfigTest, MaxFailureRateDefault) {
    mango::PriceTableConfig config;
    EXPECT_DOUBLE_EQ(config.max_failure_rate, 0.0);
}

TEST(PriceTableConfigTest, ValidateConfigRejectsInvalidRate) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 1.5;  // Invalid
    auto err = mango::validate_config(config);
    EXPECT_TRUE(err.has_value());
    EXPECT_NE(err->find("max_failure_rate"), std::string::npos);
}

TEST(PriceTableConfigTest, ValidateConfigAcceptsValidRate) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 0.1;  // Valid
    auto err = mango::validate_config(config);
    EXPECT_FALSE(err.has_value());
}

TEST(PriceTableBuilderTest, BuildRejectsInvalidConfig) {
    mango::PriceTableConfig config;
    config.max_failure_rate = 2.0;  // Invalid
    config.option_type = mango::OptionType::PUT;
    config.K_ref = 100.0;

    mango::PriceTableBuilder<4> builder(config);

    // Create minimal valid axes (4 points per axis for B-spline)
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0), std::log(1.1), std::log(1.2)};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.2, 0.25, 0.3, 0.35};
    axes.grids[3] = {0.05, 0.06, 0.07, 0.08};

    auto result = builder.build(axes);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::PriceTableErrorCode::InvalidConfig);
}

TEST(PriceTableBuilderTest, FromVectorsRejectsInvalidMaxFailureRate) {
    auto result = mango::PriceTableBuilder<4>::from_vectors(
        {std::log(0.9), std::log(1.0), std::log(1.1)},  // log-moneyness
        {0.25, 0.5},      // maturity
        {0.2, 0.3},       // volatility
        {0.05},           // rate
        100.0,            // K_ref
        mango::PDEGridConfig{mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(), 500},
        mango::OptionType::PUT,
        0.0,              // dividend_yield
        1.5               // max_failure_rate - INVALID
    );
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::PriceTableErrorCode::InvalidConfig);
}

TEST(PriceTableBuilderTest, FindNearestValidNeighborFindsAdjacent) {
    // Test helper directly via builder's testing interface
    // Create 3x3 grid, mark center invalid, verify finds adjacent
    std::vector<bool> slice_valid(9, true);
    slice_valid[4] = false;  // Center (1,1) invalid

    mango::PriceTableConfig config;
    mango::PriceTableBuilder<4> builder(config);

    auto result = Access::find_nearest_valid_neighbor(builder, 1, 1, 3, 3, slice_valid);
    ASSERT_TRUE(result.has_value());
    // Should find one of (0,1), (1,0), (1,2), (2,1) at distance 1
    auto [nσ, nr] = result.value();
    size_t dist = std::abs(static_cast<int>(nσ) - 1) + std::abs(static_cast<int>(nr) - 1);
    EXPECT_EQ(dist, 1);
}

TEST(PriceTableBuilderTest, RepairFailedSlicesInterpolatesPartial) {
    // Create a minimal 4D tensor and inject a NaN at a known τ position
    // Then verify repair fills it via τ-interpolation

    // Use 2x3x2x2 tensor: Nm=2, Nt=3, Nσ=2, Nr=2
    auto tensor_result = mango::PriceTensor<4>::create({2, 3, 2, 2});
    ASSERT_TRUE(tensor_result.has_value());
    auto& tensor = tensor_result.value();

    // Initialize all values to m_idx + τ_idx (so we can verify interpolation)
    for (size_t m = 0; m < 2; ++m) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t s = 0; s < 2; ++s) {
                for (size_t r = 0; r < 2; ++r) {
                    tensor.view[m, t, s, r] = static_cast<double>(m + t);
                }
            }
        }
    }

    // Inject NaN at (σ=0, r=0, τ=1) - middle maturity
    tensor.view[0, 1, 0, 0] = std::numeric_limits<double>::quiet_NaN();
    tensor.view[1, 1, 0, 0] = std::numeric_limits<double>::quiet_NaN();

    // Create axes
    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};
    axes.grids[1] = {0.25, 0.5, 0.75};
    axes.grids[2] = {0.2, 0.3};
    axes.grids[3] = {0.05, 0.06};

    // Build config and builder
    mango::PriceTableConfig config;
    mango::PriceTableBuilder<4> builder(config);

    // Call repair with failed_spline indicating τ=1 failed
    std::vector<size_t> failed_pde;  // No PDE failures
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
    failed_spline.emplace_back(0, 0, 1);  // (σ=0, r=0, τ=1)

    auto result = Access::repair_failed_slices(builder, 
        tensor, failed_pde, failed_spline, axes);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repaired_partial_points, 1);
    EXPECT_EQ(result->repaired_full_slices, 0);

    // Verify interpolated values (τ=0 has val=m+0, τ=2 has val=m+2, so τ=1 should be m+1)
    double val1 = tensor.view[0, 1, 0, 0];
    double val2 = tensor.view[1, 1, 0, 0];
    EXPECT_NEAR(val1, 1.0, 1e-10);
    EXPECT_NEAR(val2, 2.0, 1e-10);
}

TEST(PriceTableBuilderTest, RepairFailedSlicesCopiesFromNeighbor) {
    // Create tensor with full slice NaN, verify neighbor copy
    auto tensor_result = mango::PriceTensor<4>::create({2, 2, 2, 2});
    ASSERT_TRUE(tensor_result.has_value());
    auto& tensor = tensor_result.value();

    // Initialize: slice (0,0) gets value 10, slice (0,1) gets value 20, etc.
    for (size_t m = 0; m < 2; ++m) {
        for (size_t t = 0; t < 2; ++t) {
            for (size_t s = 0; s < 2; ++s) {
                for (size_t r = 0; r < 2; ++r) {
                    tensor.view[m, t, s, r] = static_cast<double>(10 * (s * 2 + r + 1));
                }
            }
        }
    }

    // Mark slice (0,0) as NaN (simulating PDE failure)
    for (size_t m = 0; m < 2; ++m) {
        for (size_t t = 0; t < 2; ++t) {
            tensor.view[m, t, 0, 0] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};
    axes.grids[1] = {0.25, 0.5};
    axes.grids[2] = {0.2, 0.3};
    axes.grids[3] = {0.05, 0.06};

    mango::PriceTableConfig config;
    mango::PriceTableBuilder<4> builder(config);

    // PDE failed at flat index 0 (σ=0, r=0)
    std::vector<size_t> failed_pde = {0};
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;

    auto result = Access::repair_failed_slices(builder, 
        tensor, failed_pde, failed_spline, axes);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->repaired_full_slices, 1);

    // Slice (0,0) should now have values from nearest neighbor
    // Neighbors are (0,1), (1,0), (1,1) - (0,1) or (1,0) should be picked first
    // Just verify it's no longer NaN and is one of the valid slice values
    double val1 = tensor.view[0, 0, 0, 0];
    double val2 = tensor.view[1, 1, 0, 0];
    EXPECT_FALSE(std::isnan(val1));
    EXPECT_FALSE(std::isnan(val2));
}

TEST(PriceTableBuilderTest, RepairFailedSlicesFailsWhenNoValidDonor) {
    // All slices invalid, verify returns error
    auto tensor_result = mango::PriceTensor<4>::create({2, 2, 1, 1});
    ASSERT_TRUE(tensor_result.has_value());
    auto& tensor = tensor_result.value();

    // Only one (σ,r) slice exists, and it failed
    for (size_t m = 0; m < 2; ++m) {
        for (size_t t = 0; t < 2; ++t) {
            tensor.view[m, t, 0, 0] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    mango::PriceTableAxes<4> axes;
    axes.grids[0] = {std::log(0.9), std::log(1.0)};
    axes.grids[1] = {0.25, 0.5};
    axes.grids[2] = {0.2};
    axes.grids[3] = {0.05};

    mango::PriceTableConfig config;
    mango::PriceTableBuilder<4> builder(config);

    std::vector<size_t> failed_pde = {0};  // Only slice failed
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;

    auto result = Access::repair_failed_slices(builder, 
        tensor, failed_pde, failed_spline, axes);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, mango::PriceTableErrorCode::RepairFailed);
}

TEST(PriceTableBuilderTest, BuildPopulatesTotalSlicesAndPoints) {
    auto result = mango::PriceTableBuilder<4>::from_vectors(
        {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)}, {0.25, 0.5, 0.75, 1.0}, {0.15, 0.2, 0.25, 0.3}, {0.02, 0.04, 0.06, 0.08},
        100.0,
        mango::PDEGridConfig{mango::GridSpec<double>::uniform(-3.0, 3.0, 51).value(), 500});
    ASSERT_TRUE(result.has_value());
    auto& [builder, axes] = result.value();

    auto build_result = builder.build(axes);
    ASSERT_TRUE(build_result.has_value()) << "Error: " << build_result.error();

    EXPECT_EQ(build_result->total_slices, 4 * 4);  // Nσ × Nr = 4 × 4
    EXPECT_EQ(build_result->total_points, 4 * 4 * 4);  // Nσ × Nr × Nt = 4 × 4 × 4
}

// Default mode always produces NormalizedPrice metadata
TEST(PriceTableBuilderTest, DefaultModeProducesNormalizedPriceMetadata) {
    auto setup = PriceTableBuilder<4>::from_vectors(
        {std::log(0.8), std::log(0.9), std::log(1.0), std::log(1.1)},
        {0.25, 0.5, 0.75, 1.0},
        {0.15, 0.20, 0.25, 0.30},
        {0.02, 0.04, 0.06, 0.08},
        100.0,
        PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        OptionType::PUT);
    ASSERT_TRUE(setup.has_value());
    auto& [builder, axes] = setup.value();

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value()) << "Build failed: " << result.error();
    EXPECT_EQ(result->surface->metadata().content,
              SurfaceContent::NormalizedPrice);
}

} // namespace
} // namespace mango
