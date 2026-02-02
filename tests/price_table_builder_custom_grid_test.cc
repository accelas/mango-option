// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/table/price_table_builder.hpp"
#include "tests/price_table_builder_test_access.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {
namespace {

using Access = testing::PriceTableBuilderAccess<4>;

// Test to investigate the claimed failure when using custom_grid
// The claim is: "custom_grid causes all PDE solves to fail when options have spot==strike (normalized case)"
TEST(PriceTableBuilderCustomGridTest, CustomGridWithNormalizedCase) {
    // Setup: minimal 4D grid with small spatial grid for fast test
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100},
        .dividends = {.dividend_yield = 0.02}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1, 1.2};     // moneyness: 4 points
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};   // maturity: 4 points
    axes.grids[2] = {0.20, 0.25};             // volatility: 2 points
    axes.grids[3] = {0.05, 0.06};             // rate: 2 points

    // Generate batch (should be 2×2=4 entries with spot==strike==K_ref)
    auto batch = Access::make_batch(builder, axes);
    ASSERT_EQ(batch.size(), 4);  // 2 vols × 2 rates

    // Verify all batch entries are normalized (spot == strike == K_ref)
    for (const auto& params : batch) {
        EXPECT_DOUBLE_EQ(params.spot, 100.0);
        EXPECT_DOUBLE_EQ(params.strike, 100.0);
    }

    // Now test solve_batch WITHOUT custom_grid (baseline - should work)
    auto batch_result_baseline = Access::solve_batch(builder, batch, axes);

    std::cout << "=== Baseline (no custom_grid) ===" << std::endl;
    std::cout << "Total results: " << batch_result_baseline.results.size() << std::endl;
    std::cout << "Failed count: " << batch_result_baseline.failed_count << std::endl;

    for (size_t i = 0; i < batch_result_baseline.results.size(); ++i) {
        if (batch_result_baseline.results[i].has_value()) {
            std::cout << "  Result " << i << ": SUCCESS" << std::endl;
        } else {
            const auto& error = batch_result_baseline.results[i].error();
            std::cout << "  Result " << i << ": FAILED - code="
                      << static_cast<int>(error.code) << std::endl;
        }
    }

    // Baseline should succeed
    EXPECT_EQ(batch_result_baseline.failed_count, 0);

    // Now test the CLAIMED FAILURE: using custom_grid with normalized case
    // Build custom_grid as the plan specified
    std::cout << "\n=== Testing custom_grid (claimed to fail) ===" << std::endl;

    // Create custom grid/time domain as plan specified
    const auto& cg_grid = std::get<PDEGridConfig>(config.pde_grid);
    GridSpec<double> user_grid = cg_grid.grid_spec;
    auto time_domain = TimeDomain::from_n_steps(0.0, axes.grids[1].back(), cg_grid.n_time);
    std::optional<PDEGridSpec> custom_grid =
        PDEGridConfig{user_grid, time_domain.n_steps(), {}};

    // Create a BatchAmericanOptionSolver to test with custom_grid
    BatchAmericanOptionSolver solver;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = std::min(cg_grid.grid_spec.n_points(), size_t(100));
    accuracy.max_spatial_points = std::max(cg_grid.grid_spec.n_points(), size_t(1200));
    accuracy.max_time_steps = cg_grid.n_time;
    if (cg_grid.grid_spec.type() == GridSpec<double>::Type::SinhSpaced) {
        accuracy.alpha = cg_grid.grid_spec.concentration();
    }
    solver.set_grid_accuracy(accuracy);
    solver.set_snapshot_times(axes.grids[1]);

    // Call solve_batch with custom_grid parameter
    auto batch_result_custom = solver.solve_batch(batch, true, nullptr, custom_grid);

    std::cout << "Total results: " << batch_result_custom.results.size() << std::endl;
    std::cout << "Failed count: " << batch_result_custom.failed_count << std::endl;

    for (size_t i = 0; i < batch_result_custom.results.size(); ++i) {
        if (batch_result_custom.results[i].has_value()) {
            std::cout << "  Result " << i << ": SUCCESS" << std::endl;
        } else {
            const auto& error = batch_result_custom.results[i].error();
            std::cout << "  Result " << i << ": FAILED - code="
                      << static_cast<int>(error.code)
                      << " (iterations=" << error.iterations << ")" << std::endl;
        }
    }

    // Check if the claim is true: does custom_grid cause failures?
    if (batch_result_custom.failed_count > 0) {
        std::cout << "\n*** CLAIM VERIFIED: custom_grid causes "
                  << batch_result_custom.failed_count << " failures ***" << std::endl;
    } else {
        std::cout << "\n*** CLAIM REFUTED: custom_grid works fine ***" << std::endl;
    }

    // For investigation: print details about the grid
    std::cout << "\nGrid details:" << std::endl;
    std::cout << "  x_min: " << user_grid.x_min() << std::endl;
    std::cout << "  x_max: " << user_grid.x_max() << std::endl;
    std::cout << "  n_points: " << user_grid.n_points() << std::endl;
    std::cout << "  time domain: [" << time_domain.t_start() << ", " << time_domain.t_end() << "]" << std::endl;
    std::cout << "  n_steps: " << time_domain.n_steps() << std::endl;
}

// Additional test: verify the exact spot==strike condition
TEST(PriceTableBuilderCustomGridTest, VerifyNormalizedBatchConditions) {
    PriceTableConfig config{
        .option_type = OptionType::PUT,
        .K_ref = 100.0,
        .pde_grid = PDEGridConfig{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 100}
    };

    PriceTableBuilder<4> builder(config);

    PriceTableAxes<4> axes;
    axes.grids[0] = {0.9, 1.0, 1.1, 1.2};
    axes.grids[1] = {0.25, 0.5, 0.75, 1.0};
    axes.grids[2] = {0.20};
    axes.grids[3] = {0.05};

    auto batch = Access::make_batch(builder, axes);

    ASSERT_EQ(batch.size(), 1);

    // Verify the normalized condition: spot == strike
    const auto& params = batch[0];
    EXPECT_DOUBLE_EQ(params.spot, params.strike);
    EXPECT_DOUBLE_EQ(params.spot, config.K_ref);

    // This is the "normalized case" mentioned in the claim
    double moneyness = params.spot / params.strike;
    EXPECT_DOUBLE_EQ(moneyness, 1.0);

    std::cout << "Normalized batch parameters:" << std::endl;
    std::cout << "  spot: " << params.spot << std::endl;
    std::cout << "  strike: " << params.strike << std::endl;
    std::cout << "  moneyness: " << moneyness << std::endl;
    std::cout << "  maturity: " << params.maturity << std::endl;
    std::cout << "  volatility: " << params.volatility << std::endl;
    std::cout << "  rate: ";
    if (std::holds_alternative<double>(params.rate)) {
        std::cout << std::get<double>(params.rate);
    } else {
        std::cout << "<YieldCurve>";
    }
    std::cout << std::endl;
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: Auto-estimated PDE grid must cover wide moneyness axes
// Bug: compute_global_grid_for_batch() used spot=strike=K_ref (x0=0) and
// sized the domain solely from n_sigma * σ√T.  Wide moneyness axes could
// fall outside the PDE grid, causing extrapolation failures in extract_tensor.
TEST(PriceTableBuilderCustomGridTest, AutoGridCoversWideMoneyness) {
    // Wide moneyness with LOW vol and SHORT maturity to trigger the bug.
    // m ∈ [0.5, 2.0] → log(m) ∈ [-0.69, 0.69]
    // All σ ≤ 0.15, T ≤ 0.25 → max(σ√T) = 0.15 * √0.25 = 0.075
    // Without fix: n_sigma=5 → half-width = 0.375, does NOT cover |-0.69|.
    // With fix: n_sigma bumped to ≥ (0.69/0.075)*1.1 ≈ 10.1
    std::vector<double> moneyness = {0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0};
    std::vector<double> maturity = {0.04, 0.08, 0.17, 0.25};
    std::vector<double> volatility = {0.10, 0.12, 0.14, 0.15};  // All small
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    GridAccuracyParams accuracy;
    accuracy.tol = 1e-2;  // Fast mode for test speed

    auto setup = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate,
        100.0,                    // K_ref
        accuracy,                 // auto-estimated PDE grid
        OptionType::PUT,
        0.02,                     // dividend_yield
        0.0                       // strict: no failures allowed
    );

    ASSERT_TRUE(setup.has_value()) << "from_vectors failed";
    auto& [builder, axes] = setup.value();

    // Build the full table — this exercises estimate_pde_grid → solve_batch → extract_tensor
    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value())
        << "build() failed — PDE grid likely did not cover moneyness axis";

    // Verify no PDE or spline failures
    EXPECT_EQ(result->failed_pde_slices, 0)
        << "PDE failures indicate grid coverage issue";
    EXPECT_EQ(result->failed_spline_points, 0)
        << "Spline failures indicate extrapolation outside PDE domain";

    // Verify surface returns sensible prices at extreme moneyness
    auto& surface = result->surface;
    ASSERT_NE(surface, nullptr);

    // Deep ITM put (m=0.5 → spot/strike=0.5 → strike >> spot): high price
    double deep_itm = surface->value({0.5, 0.17, 0.14, 0.05});
    EXPECT_GT(deep_itm, 0.0) << "Deep ITM put should have positive price";
    EXPECT_FALSE(std::isnan(deep_itm)) << "Deep ITM should not be NaN";

    // Deep OTM put (m=2.0 → spot/strike=2.0 → spot >> strike): near zero
    double deep_otm = surface->value({2.0, 0.17, 0.14, 0.05});
    EXPECT_GE(deep_otm, 0.0) << "Put price should be non-negative";
    EXPECT_FALSE(std::isnan(deep_otm)) << "Deep OTM should not be NaN";
}

// Verify GridAccuracyParams in PriceTableConfig actually drives grid choice
TEST(PriceTableBuilderCustomGridTest, AutoGridAccuracyParamsDriveGridChoice) {
    std::vector<double> moneyness = {0.9, 0.95, 1.0, 1.05, 1.1};
    std::vector<double> maturity = {0.25, 0.5, 0.75, 1.0};
    std::vector<double> volatility = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    // Build with Coarse accuracy (few spatial points)
    GridAccuracyParams coarse;
    coarse.tol = 1e-1;
    coarse.min_spatial_points = 51;
    coarse.max_spatial_points = 51;

    auto setup_coarse = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate,
        100.0, coarse, OptionType::PUT, 0.02, 0.0);

    ASSERT_TRUE(setup_coarse.has_value());
    auto& [builder_coarse, axes_coarse] = setup_coarse.value();
    auto result_coarse = builder_coarse.build(axes_coarse);
    ASSERT_TRUE(result_coarse.has_value());

    // Build with Fine accuracy (many spatial points)
    GridAccuracyParams fine;
    fine.tol = 1e-4;
    fine.min_spatial_points = 401;
    fine.max_spatial_points = 401;

    auto setup_fine = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate,
        100.0, fine, OptionType::PUT, 0.02, 0.0);

    ASSERT_TRUE(setup_fine.has_value());
    auto& [builder_fine, axes_fine] = setup_fine.value();
    auto result_fine = builder_fine.build(axes_fine);
    ASSERT_TRUE(result_fine.has_value());

    // Both should succeed with no failures
    EXPECT_EQ(result_coarse->failed_pde_slices, 0);
    EXPECT_EQ(result_fine->failed_pde_slices, 0);

    // ATM put prices from both
    double price_coarse = result_coarse->surface->value({1.0, 0.5, 0.20, 0.05});
    double price_fine   = result_fine->surface->value({1.0, 0.5, 0.20, 0.05});

    // Both should be reasonable ATM put prices
    EXPECT_GT(price_coarse, 0.0);
    EXPECT_GT(price_fine, 0.0);
    EXPECT_LT(price_coarse, 20.0);
    EXPECT_LT(price_fine, 20.0);

    // Different grid resolutions should produce measurably different results
    // (51 vs 401 points is a large enough gap to cause observable difference)
    double diff = std::abs(price_coarse - price_fine);
    EXPECT_GT(diff, 1e-6) << "Accuracy params should influence the result";
    EXPECT_LT(diff, 2.0) << "Prices should still be in the same ballpark";
}

// Regression: Explicit-grid fallback path must also cover wide moneyness
// Bug: When an explicit grid violates solver constraints, the fallback
// auto-estimation path ignored moneyness axis bounds (same root cause as above).
TEST(PriceTableBuilderCustomGridTest, ExplicitGridFallbackCoversWideMoneyness) {
    // Use a coarse explicit grid to trigger the fallback path.
    // Uniform(-1.0, 1.0, 21): dx = 2.0/20 = 0.1 > MAX_DX (0.05) → fallback
    auto coarse_grid = GridSpec<double>::uniform(-1.0, 1.0, 21).value();
    PDEGridConfig explicit_pde{coarse_grid, 200};

    // Wide moneyness + low vol/short maturity (same scenario as AutoGridCoversWideMoneyness)
    std::vector<double> moneyness = {0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0};
    std::vector<double> maturity = {0.04, 0.08, 0.17, 0.25};
    std::vector<double> volatility = {0.10, 0.12, 0.14, 0.15};
    std::vector<double> rate = {0.03, 0.04, 0.05, 0.06};

    auto setup = PriceTableBuilder<4>::from_vectors(
        moneyness, maturity, volatility, rate,
        100.0,
        explicit_pde,             // triggers fallback due to coarse spacing
        OptionType::PUT,
        0.02,
        0.0                       // strict
    );

    // from_vectors may reject if explicit grid doesn't cover log(m) range
    // (the coverage check at build() line 75-84 only applies to explicit grids
    // that pass constraints — for fallback, we need to get past from_vectors first)
    // Note: from_vectors doesn't do the coverage check, build() does.
    ASSERT_TRUE(setup.has_value()) << "from_vectors failed";
    auto& [builder, axes] = setup.value();

    auto result = builder.build(axes);

    // The explicit grid [-1, 1] doesn't cover log(0.5)=-0.69... wait, it does.
    // But the explicit grid coverage check (lines 73-84) checks explicit grids:
    // x_min_requested = log(0.5) = -0.69, x_min = -1.0 → covered.
    // However the grid SPACING violates max_dx, so it hits the fallback path.
    // The fallback must then produce a grid that covers the moneyness axis.
    ASSERT_TRUE(result.has_value())
        << "build() failed — fallback path did not cover moneyness axis";

    EXPECT_EQ(result->failed_pde_slices, 0)
        << "PDE failures in fallback path";
    EXPECT_EQ(result->failed_spline_points, 0)
        << "Spline failures indicate fallback grid under-coverage";

    auto& surface = result->surface;
    ASSERT_NE(surface, nullptr);

    double deep_itm = surface->value({0.5, 0.17, 0.14, 0.05});
    EXPECT_GT(deep_itm, 0.0);
    EXPECT_FALSE(std::isnan(deep_itm));

    double deep_otm = surface->value({2.0, 0.17, 0.14, 0.05});
    EXPECT_GE(deep_otm, 0.0);
    EXPECT_FALSE(std::isnan(deep_otm));
}

} // namespace
} // namespace mango
