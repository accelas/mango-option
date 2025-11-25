#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "kokkos/src/option/pricing_pipeline.hpp"

namespace mango::kokkos {

class PricingPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            int argc = 0;
            char** argv = nullptr;
            Kokkos::initialize(argc, argv);
        }
    }

    static void TearDownTestSuite() {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }

    // Default configuration for tests
    PricingPipelineConfig default_config() {
        PricingPipelineConfig config;
        config.moneyness = {0.8, 1.2, 9};
        config.maturity = {0.25, 1.0, 4};
        config.volatility = {0.15, 0.35, 5};
        config.rate = {0.02, 0.06, 3};
        // n_space and n_time default to 0 = auto-estimate
        config.K_ref = 100.0;
        config.dividend_yield = 0.01;
        config.is_put = true;
        return config;
    }
};

// Test 1: Pipeline construction
TEST_F(PricingPipelineTest, Construction) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    EXPECT_FALSE(pipeline.is_price_table_built());
    EXPECT_EQ(pipeline.config().K_ref, 100.0);
    // n_space and n_time default to 0 = auto-estimate
    EXPECT_EQ(pipeline.config().n_space, 0);
    EXPECT_EQ(pipeline.config().n_time, 0);
}

// Test 2: Price table building
TEST_F(PricingPipelineTest, BuildPriceTable) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Build price table
    auto result = pipeline.build_price_table();
    ASSERT_TRUE(result.has_value()) << "Price table build should succeed";

    // Check that table is now built
    EXPECT_TRUE(pipeline.is_price_table_built());

    // Verify table exists and has correct shape
    auto table = pipeline.get_price_table();
    ASSERT_TRUE(table.has_value());

    const auto& shape = table->get().shape;
    EXPECT_EQ(shape[0], 9);   // moneyness
    EXPECT_EQ(shape[1], 4);   // maturity
    EXPECT_EQ(shape[2], 5);   // volatility
    EXPECT_EQ(shape[3], 3);   // rate
}

// Test 3: Batch option pricing
TEST_F(PricingPipelineTest, BatchOptionPricing) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Create batch of options (3 puts)
    const size_t n_options = 3;
    Kokkos::View<double*, Kokkos::HostSpace> strikes("strikes", n_options);
    Kokkos::View<double*, Kokkos::HostSpace> spots("spots", n_options);

    strikes(0) = 100.0; spots(0) = 95.0;   // ITM
    strikes(1) = 100.0; spots(1) = 100.0;  // ATM
    strikes(2) = 100.0; spots(2) = 105.0;  // OTM

    // Price options
    BatchPricingParams params{
        .maturity = 0.5,
        .volatility = 0.25,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .is_put = true
    };

    auto result = pipeline.price_options(params, strikes, spots);
    ASSERT_TRUE(result.has_value()) << "Batch pricing should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, result.value());

    // Verify prices
    EXPECT_GT(results_h(0).price, 0.0);
    EXPECT_GT(results_h(1).price, 0.0);
    EXPECT_GT(results_h(2).price, 0.0);

    // ITM put should be most expensive
    EXPECT_GT(results_h(0).price, results_h(1).price);
    EXPECT_GT(results_h(1).price, results_h(2).price);

    // Put delta should be negative
    EXPECT_LT(results_h(0).delta, 0.0);
    EXPECT_LT(results_h(1).delta, 0.0);
    EXPECT_LT(results_h(2).delta, 0.0);
}

// Test 4: IV solve via interpolation
TEST_F(PricingPipelineTest, IVSolveInterpolated) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Build price table first
    auto build_result = pipeline.build_price_table();
    ASSERT_TRUE(build_result.has_value()) << "Price table build should succeed";

    // Create IV query
    const size_t n_queries = 1;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> queries("queries", n_queries);

    queries(0) = IVQuery{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 0.5,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .type = OptionType::Put,
        .market_price = 5.0
    };

    // Solve IV
    auto result = pipeline.solve_iv_interpolated(queries);
    ASSERT_TRUE(result.has_value()) << "IV solve should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, result.value());

    // Verify result
    EXPECT_TRUE(results_h(0).converged);
    EXPECT_GT(results_h(0).implied_vol, 0.0);
    EXPECT_LT(results_h(0).implied_vol, 1.0);  // Reasonable volatility
    EXPECT_LT(results_h(0).final_error, 1e-4);  // Converged
}

// Test 5: IV solve via FDM
TEST_F(PricingPipelineTest, IVSolveFDM) {
    auto config = default_config();
    // Uses auto-estimation by default
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Create IV query
    const size_t n_queries = 1;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> queries("queries", n_queries);

    queries(0) = IVQuery{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 0.5,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .type = OptionType::Put,
        .market_price = 5.0
    };

    // Solve IV using FDM
    auto result = pipeline.solve_iv_fdm(queries);
    ASSERT_TRUE(result.has_value()) << "FDM IV solve should succeed";

    auto results_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, result.value());

    // Verify result
    EXPECT_TRUE(results_h(0).converged);
    EXPECT_GT(results_h(0).implied_vol, 0.0);
    EXPECT_LT(results_h(0).implied_vol, 1.0);
    EXPECT_EQ(results_h(0).code, IVResultCode::Success);
}

// Test 6: Comparison - Interpolated vs FDM IV
TEST_F(PricingPipelineTest, InterpolatedVsFDMComparison) {
    auto config = default_config();
    // Uses auto-estimation by default
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Build price table
    auto build_result = pipeline.build_price_table();
    ASSERT_TRUE(build_result.has_value());

    // Create IV query
    const size_t n_queries = 1;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> queries("queries", n_queries);

    queries(0) = IVQuery{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 0.5,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .type = OptionType::Put,
        .market_price = 5.0
    };

    // Solve with both methods
    auto interp_result = pipeline.solve_iv_interpolated(queries);
    auto fdm_result = pipeline.solve_iv_fdm(queries);

    ASSERT_TRUE(interp_result.has_value());
    ASSERT_TRUE(fdm_result.has_value());

    auto interp_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, interp_result.value());
    auto fdm_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fdm_result.value());

    // Both should converge
    EXPECT_TRUE(interp_h(0).converged);
    EXPECT_TRUE(fdm_h(0).converged);

    // Results should be close (within 5% relative error)
    double iv_interp = interp_h(0).implied_vol;
    double iv_fdm = fdm_h(0).implied_vol;
    double relative_error = std::abs(iv_interp - iv_fdm) / iv_fdm;

    EXPECT_LT(relative_error, 0.05) << "Interpolated IV: " << iv_interp
                                     << ", FDM IV: " << iv_fdm;
}

// Test 7: End-to-end workflow test
TEST_F(PricingPipelineTest, EndToEndWorkflow) {
    // Step 1: Configure pipeline
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Step 2: Build price table (one-time startup cost)
    auto build_result = pipeline.build_price_table();
    ASSERT_TRUE(build_result.has_value()) << "Price table build failed";

    // Step 3: Price some options
    const size_t n_options = 5;
    Kokkos::View<double*, Kokkos::HostSpace> strikes("strikes", n_options);
    Kokkos::View<double*, Kokkos::HostSpace> spots("spots", n_options);

    for (size_t i = 0; i < n_options; ++i) {
        strikes(i) = 100.0;
        spots(i) = 90.0 + static_cast<double>(i) * 5.0;  // 90, 95, 100, 105, 110
    }

    BatchPricingParams batch_params{
        .maturity = 0.5,
        .volatility = 0.25,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .is_put = true
    };

    auto price_result = pipeline.price_options(batch_params, strikes, spots);
    ASSERT_TRUE(price_result.has_value());

    auto prices_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, price_result.value());

    // Step 4: Solve IV for some market prices using interpolation
    const size_t n_iv = 3;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> iv_queries("iv_queries", n_iv);

    for (size_t i = 0; i < n_iv; ++i) {
        iv_queries(i) = IVQuery{
            .strike = 100.0,
            .spot = 95.0 + static_cast<double>(i) * 5.0,
            .maturity = 0.5,
            .rate = 0.04,
            .dividend_yield = 0.01,
            .type = OptionType::Put,
            .market_price = prices_h(i).price  // Use computed prices
        };
    }

    auto iv_result = pipeline.solve_iv_interpolated(iv_queries);
    ASSERT_TRUE(iv_result.has_value());

    auto iv_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, iv_result.value());

    // Step 5: Verify IV results
    for (size_t i = 0; i < n_iv; ++i) {
        EXPECT_TRUE(iv_h(i).converged) << "Query " << i << " did not converge";
        EXPECT_GT(iv_h(i).implied_vol, 0.0);
        // Should recover approximately the input volatility (0.25)
        // Allow wider tolerance due to interpolation error and price table coarseness
        EXPECT_NEAR(iv_h(i).implied_vol, 0.25, 0.15);
    }

    // Step 6: Validate with FDM for one query
    Kokkos::View<IVQuery*, Kokkos::HostSpace> single_query("single", 1);
    single_query(0) = iv_queries(1);  // Middle query

    auto fdm_result = pipeline.solve_iv_fdm(single_query);
    ASSERT_TRUE(fdm_result.has_value());

    auto fdm_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fdm_result.value());

    EXPECT_TRUE(fdm_h(0).converged);
    // FDM and interpolated should agree
    EXPECT_NEAR(fdm_h(0).implied_vol, iv_h(1).implied_vol, 0.02);
}

// Test 8: Attempt IV solve before building table
TEST_F(PricingPipelineTest, IVSolveBeforeBuild) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    // Try to solve IV without building table first
    const size_t n_queries = 1;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> queries("queries", n_queries);

    queries(0) = IVQuery{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 0.5,
        .rate = 0.04,
        .dividend_yield = 0.01,
        .type = OptionType::Put,
        .market_price = 5.0
    };

    auto result = pipeline.solve_iv_interpolated(queries);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), PipelineError::PriceTableNotBuilt);
}

// Test 9: Batch IV queries
TEST_F(PricingPipelineTest, BatchIVQueries) {
    auto config = default_config();
    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    auto build_result = pipeline.build_price_table();
    ASSERT_TRUE(build_result.has_value());

    // Create batch of IV queries
    const size_t n_queries = 10;
    Kokkos::View<IVQuery*, Kokkos::HostSpace> queries("queries", n_queries);

    for (size_t i = 0; i < n_queries; ++i) {
        double spot = 90.0 + static_cast<double>(i) * 2.0;  // 90 to 108
        queries(i) = IVQuery{
            .strike = 100.0,
            .spot = spot,
            .maturity = 0.5,
            .rate = 0.04,
            .dividend_yield = 0.01,
            .type = OptionType::Put,
            .market_price = std::max(100.0 - spot, 0.0) + 3.0  // Rough put price
        };
    }

    auto result = pipeline.solve_iv_interpolated(queries);
    ASSERT_TRUE(result.has_value());

    auto results_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, result.value());

    // Check all queries
    for (size_t i = 0; i < n_queries; ++i) {
        EXPECT_TRUE(results_h(i).converged) << "Query " << i;
        EXPECT_GT(results_h(i).implied_vol, 0.0) << "Query " << i;
        EXPECT_LT(results_h(i).implied_vol, 1.0) << "Query " << i;
    }
}

// Test 10: Configuration access
TEST_F(PricingPipelineTest, ConfigurationAccess) {
    auto config = default_config();
    config.moneyness.min = 0.5;
    config.moneyness.max = 1.5;
    config.K_ref = 150.0;

    PricingPipeline<Kokkos::HostSpace> pipeline(config);

    const auto& retrieved_config = pipeline.config();
    EXPECT_EQ(retrieved_config.moneyness.min, 0.5);
    EXPECT_EQ(retrieved_config.moneyness.max, 1.5);
    EXPECT_EQ(retrieved_config.K_ref, 150.0);
}

}  // namespace mango::kokkos
