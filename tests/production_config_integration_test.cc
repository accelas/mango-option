// SPDX-License-Identifier: MIT
/**
 * @file production_config_integration_test.cc
 * @brief Integration tests using production-like configurations
 *
 * These tests mirror actual production usage patterns to catch bugs that
 * unit tests with default parameters might miss. Created in response to
 * issue #272 where the batch solver failed 100% for production configs
 * but passed all unit tests.
 *
 * Key differences from unit tests:
 * - Explicit small grid sizes (51 pts vs default 101)
 * - Realistic market data grids (SPY-like parameters)
 * - Full PriceTableBuilderND workflow
 */

#include <gtest/gtest.h>
#include <cmath>
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/eep/eep_decomposer.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/standard_surface.hpp"

using namespace mango;

// ============================================================================
// Production-like Market Grid (mirrors benchmark setup)
// ============================================================================

namespace {

struct MarketGrid {
    std::vector<double> log_moneyness;
    std::vector<double> maturities;
    std::vector<double> volatilities;
    std::vector<double> rates;
    double K_ref;
    double spot;
    double dividend;
};

/// Generate realistic market grid (based on SPY options)
MarketGrid generate_market_grid() {
    MarketGrid grid;

    grid.spot = 450.0;
    grid.K_ref = 450.0;
    grid.dividend = 0.015;

    // Log-moneyness grid: ln(0.85) to ln(1.15)
    grid.log_moneyness = {
        std::log(0.85), std::log(0.90), std::log(0.93), std::log(0.95), std::log(0.97), std::log(0.99),
        std::log(1.00),
        std::log(1.01), std::log(1.03), std::log(1.05), std::log(1.07), std::log(1.10), std::log(1.15)
    };

    // Maturities: weekly to 2 years
    grid.maturities = {
        7.0/365, 14.0/365, 30.0/365, 60.0/365,
        90.0/365, 180.0/365, 1.0, 2.0
    };

    // Volatility grid: 10% to 50%
    grid.volatilities = {
        0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50
    };

    // Rate grid: 2% to 5%
    grid.rates = {0.02, 0.03, 0.04, 0.05};

    return grid;
}

/// Smaller grid for faster tests (min 4 points per axis for B-spline)
MarketGrid generate_small_market_grid() {
    MarketGrid grid;

    grid.spot = 100.0;
    grid.K_ref = 100.0;
    grid.dividend = 0.02;

    grid.log_moneyness = {std::log(0.90), std::log(0.95), std::log(1.00), std::log(1.05), std::log(1.10)};
    grid.maturities = {0.25, 0.5, 1.0, 2.0};  // Need 4+ for B-spline
    grid.volatilities = {0.15, 0.20, 0.25, 0.30};
    grid.rates = {0.02, 0.03, 0.04, 0.05};  // Need 4+ for B-spline

    return grid;
}

}  // namespace

// ============================================================================
// PriceTableBuilderND Integration Tests
// ============================================================================

// This test directly mirrors the benchmark that exposed issue #272
TEST(ProductionConfig, PriceTableBuilder_SmallGrid_51Points) {
    auto grid = generate_small_market_grid();

    // Production uses small grid (51 points) for speed
    // This was the configuration that failed before issue #272 fix
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec, 500},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value())
        << "PriceTableBuilderND::from_vectors failed";

    auto [builder, axes] = std::move(builder_result.value());

    // This was failing 100% before issue #272 fix
    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value())
        << "PriceTableBuilderND::build failed with error code "
        << static_cast<int>(result.error().code)
        << ", axis_index=" << result.error().axis_index
        << ", count=" << result.error().count;

    // Verify surface is usable
    EXPECT_NE(result->surface, nullptr);
}

TEST(ProductionConfig, PriceTableBuilder_VerySmallGrid_31Points) {
    auto grid = generate_small_market_grid();

    // Even smaller grid - edge case
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 31, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec_result.value(), 300},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value());
    auto [builder, axes] = std::move(builder_result.value());

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
}

TEST(ProductionConfig, PriceTableBuilder_LargeGrid_201Points) {
    auto grid = generate_small_market_grid();

    // Larger grid for higher accuracy
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 201, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec_result.value(), 1000},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value());
    auto [builder, axes] = std::move(builder_result.value());

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value());
}

TEST(ProductionConfig, PriceTableBuilder_FullMarketGrid) {
    auto grid = generate_market_grid();

    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec_result.value(), 500},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value());
    auto [builder, axes] = std::move(builder_result.value());

    auto result = builder.build(axes);
    ASSERT_TRUE(result.has_value())
        << "Full market grid build failed";

    // Verify surface dimensions match input
    EXPECT_NE(result->surface, nullptr);
}

// ============================================================================
// Batch Solver with Explicit Grid Sizes (Parameterized)
// ============================================================================

class BatchSolverGridSizeTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BatchSolverGridSizeTest, SharedGrid_ExplicitSize) {
    size_t n_points = GetParam();

    // Create grid spec with explicit size
    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, n_points, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    // Create batch of options
    std::vector<PricingParams> params;
    for (double K : {90.0, 95.0, 100.0, 105.0, 110.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    // Create custom grid config
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, 500);
    PDEGridSpec custom_grid = PDEGridConfig{grid_spec, time_domain.n_steps(), {}};

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true, nullptr, custom_grid);

    EXPECT_EQ(results.failed_count, 0)
        << "Batch solver failed with grid size " << n_points;
    EXPECT_EQ(results.results.size(), 5);

    for (size_t i = 0; i < results.results.size(); ++i) {
        EXPECT_TRUE(results.results[i].has_value())
            << "Option " << i << " failed with grid size " << n_points;
        if (results.results[i].has_value()) {
            EXPECT_GT(results.results[i]->value(), 0.0);
        }
    }
}

TEST_P(BatchSolverGridSizeTest, PerOptionGrid_ExplicitSize) {
    size_t n_points = GetParam();

    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, n_points, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    std::vector<PricingParams> params;
    for (double K : {90.0, 100.0, 110.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20));
    }

    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, 500);
    PDEGridSpec custom_grid = PDEGridConfig{grid_spec, time_domain.n_steps(), {}};

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/false, nullptr, custom_grid);

    EXPECT_EQ(results.failed_count, 0);
    for (const auto& r : results.results) {
        EXPECT_TRUE(r.has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(
    GridSizes,
    BatchSolverGridSizeTest,
    ::testing::Values(31, 51, 75, 101, 151, 201),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return "GridSize_" + std::to_string(info.param);
    });

// ============================================================================
// Batch Solver Configuration Matrix
// ============================================================================

struct BatchConfig {
    size_t batch_size;
    bool use_shared_grid;
    OptionType option_type;
    std::string name;
};

class BatchSolverConfigTest : public ::testing::TestWithParam<BatchConfig> {};

TEST_P(BatchSolverConfigTest, ConfigurationMatrix) {
    const auto& config = GetParam();

    std::vector<PricingParams> params;
    double spot = 100.0;
    double base_strike = 90.0;
    double strike_step = 20.0 / config.batch_size;

    for (size_t i = 0; i < config.batch_size; ++i) {
        double strike = base_strike + i * strike_step;
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = strike, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = config.option_type}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, config.use_shared_grid);

    EXPECT_EQ(results.results.size(), config.batch_size);
    EXPECT_EQ(results.failed_count, 0)
        << "Failed for config: " << config.name;

    for (size_t i = 0; i < results.results.size(); ++i) {
        ASSERT_TRUE(results.results[i].has_value())
            << "Option " << i << " failed for config: " << config.name;
        EXPECT_GT(results.results[i]->value(), 0.0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Configurations,
    BatchSolverConfigTest,
    ::testing::Values(
        // Batch sizes
        BatchConfig{1, true, OptionType::PUT, "single_shared_put"},
        BatchConfig{5, true, OptionType::PUT, "small_shared_put"},
        BatchConfig{20, true, OptionType::PUT, "medium_shared_put"},
        BatchConfig{50, true, OptionType::PUT, "large_shared_put"},

        // Per-option grids
        BatchConfig{1, false, OptionType::PUT, "single_per_option_put"},
        BatchConfig{5, false, OptionType::PUT, "small_per_option_put"},
        BatchConfig{20, false, OptionType::PUT, "medium_per_option_put"},

        // Call options
        BatchConfig{5, true, OptionType::CALL, "small_shared_call"},
        BatchConfig{5, false, OptionType::CALL, "small_per_option_call"},
        BatchConfig{20, true, OptionType::CALL, "medium_shared_call"}
    ),
    [](const ::testing::TestParamInfo<BatchConfig>& info) {
        return info.param.name;
    });

// ============================================================================
// Moneyness Range Tests
// ============================================================================

TEST(ProductionConfig, BatchSolver_DeepITM_Puts) {
    // Deep ITM puts (high strikes)
    std::vector<PricingParams> params;
    for (double K : {120.0, 130.0, 140.0, 150.0, 160.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 0.5, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.25));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0);
    for (const auto& r : results.results) {
        ASSERT_TRUE(r.has_value());
        // Deep ITM put should have high value (close to K - S)
        EXPECT_GT(r->value(), 15.0);
    }
}

TEST(ProductionConfig, BatchSolver_DeepOTM_Puts) {
    // Deep OTM puts (low strikes)
    std::vector<PricingParams> params;
    for (double K : {50.0, 60.0, 70.0, 80.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 0.5, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.25));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0);
    for (const auto& r : results.results) {
        ASSERT_TRUE(r.has_value());
        // Deep OTM put should have low value
        EXPECT_LT(r->value(), 5.0);
    }
}

TEST(ProductionConfig, BatchSolver_WideStrikeRange) {
    // Very wide strike range in single batch
    std::vector<PricingParams> params;
    for (double K : {50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 150.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.25));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0)
        << "Wide strike range batch failed";
    EXPECT_EQ(results.results.size(), 7);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ProductionConfig, BatchSolver_ShortMaturity) {
    // Very short maturities (weekly options)
    std::vector<PricingParams> params;
    for (double K : {95.0, 100.0, 105.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 7.0/365.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.30));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0);
}

TEST(ProductionConfig, BatchSolver_HighVolatility) {
    // High volatility (meme stock scenario)
    std::vector<PricingParams> params;
    for (double K : {80.0, 100.0, 120.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 0.5, .rate = 0.05, .option_type = OptionType::PUT}, 0.80));  // 80% vol
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0);
}

TEST(ProductionConfig, BatchSolver_NegativeRate) {
    // Negative interest rate (European scenario)
    std::vector<PricingParams> params;
    for (double K : {95.0, 100.0, 105.0}) {
        params.push_back(PricingParams(OptionSpec{.spot = 100.0, .strike = K, .maturity = 1.0, .rate = -0.01, .option_type = OptionType::PUT}, 0.20));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, true);

    EXPECT_EQ(results.failed_count, 0);
}

// ============================================================================
// Benchmark-as-Test: Full E2E Workflow
// ============================================================================
// These tests mirror the benchmark setup to catch regressions that benchmarks
// would expose but aren't run in CI.

TEST(BenchmarkAsTest, MarketIVE2E_BuildPriceTable) {
    // Mirrors BM_API_BuildPriceTable from market_iv_e2e_benchmark.cc
    auto grid = generate_market_grid();

    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec_result.value(), 500},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value())
        << "PriceTableBuilderND::from_vectors failed";

    auto [builder, axes] = std::move(builder_result.value());
    auto result = builder.build(axes);

    ASSERT_TRUE(result.has_value())
        << "PriceTableBuilderND::build failed with error code "
        << static_cast<int>(result.error().code);

    // Verify result structure
    EXPECT_NE(result->surface, nullptr);
    EXPECT_GT(result->n_pde_solves, 0);
    EXPECT_EQ(result->failed_pde_slices, 0)
        << "PDE slices failed - indicates solver configuration issue";
}

TEST(BenchmarkAsTest, MarketIVE2E_IVSolverCreation) {
    // Build price table first
    auto grid = generate_market_grid();

    auto grid_spec_result = GridSpec<double>::sinh_spaced(-3.0, 3.0, 51, 2.0);
    ASSERT_TRUE(grid_spec_result.has_value());

    auto builder_result = PriceTableBuilder::from_vectors(
        grid.log_moneyness,
        grid.maturities,
        grid.volatilities,
        grid.rates,
        grid.K_ref,
        PDEGridConfig{grid_spec_result.value(), 500},
        OptionType::PUT,
        grid.dividend,
        0.0);    // max_failure_rate

    ASSERT_TRUE(builder_result.has_value());
    auto [builder, axes] = std::move(builder_result.value());
    EEPDecomposer decomposer{OptionType::PUT, grid.K_ref, grid.dividend};
    auto table_result = builder.build(axes, SurfaceContent::EarlyExercisePremium,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            decomposer.decompose(tensor, a);
        });
    ASSERT_TRUE(table_result.has_value());

    // Create IV solver from surface via StandardSurfaceWrapper
    InterpolatedIVSolverConfig solver_config;
    solver_config.max_iter = 50;
    solver_config.tolerance = 1e-6;

    auto wrapper = make_standard_wrapper(table_result->surface, OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value()) << wrapper.error();
    auto iv_solver_result = DefaultInterpolatedIVSolver::create(
        std::move(wrapper).value(), solver_config);

    ASSERT_TRUE(iv_solver_result.has_value())
        << "DefaultInterpolatedIVSolver::create failed";

    // Test IV solve at a sample point
    const auto& iv_solver = iv_solver_result.value();
    double spot = grid.spot;
    double strike = spot / 1.0;  // ATM
    double maturity = 0.5;
    double rate = 0.04;
    double vol = 0.20;

    // Get reconstructed American price from StandardSurfaceWrapper
    auto wrapper_for_price = make_standard_wrapper(table_result->surface, OptionType::PUT);
    ASSERT_TRUE(wrapper_for_price.has_value()) << wrapper_for_price.error();
    double price = wrapper_for_price->price(spot, strike, maturity, vol, rate);
    EXPECT_GT(price, 0.0);

    // Solve for IV
    IVQuery query(OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                      .rate = rate, .dividend_yield = grid.dividend, .option_type = OptionType::PUT}, price);
    auto iv_result = iv_solver.solve(query);

    ASSERT_TRUE(iv_result.has_value())
        << "IV solve failed for ATM option";

    // Should recover approximately the input volatility
    EXPECT_NEAR(iv_result->implied_vol, vol, 0.01)
        << "IV solve did not recover input volatility";
}
