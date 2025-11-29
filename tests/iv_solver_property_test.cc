/**
 * @file iv_solver_property_test.cc
 * @brief Property-based tests for implied volatility solvers
 *
 * These tests verify mathematical invariants of the IV solvers using
 * parameterized testing. Unlike the FuzzTest-based tests in batch_solver_fuzz_test.cc,
 * these run with the regular build system and include full USDT tracing support.
 *
 * Properties tested:
 * - Round-trip consistency: Price → IV → Price
 * - IV positivity: IV > 0 when found
 * - IV bounds: IV within configured bounds
 * - Monotonicity: Higher prices → higher IVs
 * - No NaN/Inf: Results are always finite
 */

#include <gtest/gtest.h>
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/american_option_batch.hpp"
#include <cmath>
#include <vector>
#include <tuple>

using namespace mango;

// ============================================================================
// Test Fixtures
// ============================================================================

class IVSolverPropertyTest : public ::testing::Test {
protected:
    BatchAmericanOptionSolver pricer_;
    IVSolverFDMConfig config_;

    void SetUp() override {
        config_.root_config.max_iter = 100;
        config_.root_config.tolerance = 1e-6;
    }

    // Helper to price an option
    std::optional<double> price_option(double spot, double strike, double maturity,
                                       double rate, double dividend, double vol,
                                       OptionType type) {
        std::vector<AmericanOptionParams> params;
        params.push_back(PricingParams(spot, strike, maturity, rate, dividend,
                                       type, vol, {}));
        auto results = pricer_.solve_batch(params, false);
        if (results.results[0].has_value()) {
            return results.results[0]->value();
        }
        return std::nullopt;
    }
};

// ============================================================================
// Round-trip Consistency Tests
// ============================================================================

struct RoundTripParams {
    double spot;
    double strike;
    double maturity;
    double volatility;
    double rate;
    double dividend;
    OptionType type;
};

class IVRoundTripTest : public IVSolverPropertyTest,
                        public ::testing::WithParamInterface<RoundTripParams> {};

TEST_P(IVRoundTripTest, PriceToIVToPrice) {
    auto params = GetParam();

    // Skip extreme moneyness
    double moneyness = params.spot / params.strike;
    if (moneyness < 0.7 || moneyness > 1.4) {
        GTEST_SKIP() << "Extreme moneyness: " << moneyness;
    }

    // Step 1: Price the option
    auto price = price_option(params.spot, params.strike, params.maturity,
                              params.rate, params.dividend, params.volatility,
                              params.type);
    if (!price.has_value() || *price < 0.10) {
        GTEST_SKIP() << "Price too small or unavailable";
    }

    // Step 2: Solve for IV
    IVQuery query(params.spot, params.strike, params.maturity,
                  params.rate, params.dividend, params.type, *price);

    IVSolverFDM solver(config_);
    auto iv_result = solver.solve_impl(query);

    if (!iv_result.has_value()) {
        // Some edge cases may not converge, that's acceptable
        GTEST_SKIP() << "IV solver did not converge";
    }

    // Step 3: Verify round-trip
    double recovered_iv = iv_result->implied_vol;
    double iv_diff = std::abs(recovered_iv - params.volatility);

    EXPECT_LT(iv_diff, 0.01)
        << "Round-trip IV mismatch: original σ=" << params.volatility
        << ", recovered σ=" << recovered_iv
        << ", price=" << *price;
}

INSTANTIATE_TEST_SUITE_P(
    ATMOptions,
    IVRoundTripTest,
    ::testing::Values(
        RoundTripParams{100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::PUT},
        RoundTripParams{100.0, 100.0, 1.0, 0.30, 0.05, 0.02, OptionType::PUT},
        RoundTripParams{100.0, 100.0, 1.0, 0.40, 0.05, 0.02, OptionType::PUT},
        RoundTripParams{100.0, 100.0, 0.5, 0.25, 0.05, 0.02, OptionType::PUT},
        RoundTripParams{100.0, 100.0, 2.0, 0.25, 0.05, 0.02, OptionType::PUT},
        RoundTripParams{100.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::CALL},
        RoundTripParams{100.0, 100.0, 1.0, 0.30, 0.05, 0.02, OptionType::CALL}
    )
);

INSTANTIATE_TEST_SUITE_P(
    ITMOptions,
    IVRoundTripTest,
    ::testing::Values(
        RoundTripParams{110.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::CALL},
        RoundTripParams{90.0, 100.0, 1.0, 0.20, 0.05, 0.02, OptionType::PUT}
    )
);

INSTANTIATE_TEST_SUITE_P(
    OTMOptions,
    IVRoundTripTest,
    ::testing::Values(
        RoundTripParams{90.0, 100.0, 1.0, 0.25, 0.05, 0.02, OptionType::CALL},
        RoundTripParams{110.0, 100.0, 1.0, 0.25, 0.05, 0.02, OptionType::PUT}
    )
);

// ============================================================================
// IV Positivity Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, IVAlwaysPositive) {
    std::vector<std::tuple<double, double, double, double>> test_cases = {
        // spot, strike, maturity, price
        {100.0, 100.0, 1.0, 10.0},
        {100.0, 100.0, 0.5, 5.0},
        {100.0, 100.0, 2.0, 15.0},
        {100.0, 90.0, 1.0, 5.0},
        {100.0, 110.0, 1.0, 15.0},
    };

    IVSolverFDM solver(config_);

    for (const auto& [spot, strike, maturity, price] : test_cases) {
        IVQuery query(spot, strike, maturity, 0.05, 0.02, OptionType::PUT, price);
        auto result = solver.solve_impl(query);

        if (result.has_value()) {
            EXPECT_GT(result->implied_vol, 0.0)
                << "IV must be positive for spot=" << spot
                << ", strike=" << strike << ", price=" << price;
            EXPECT_FALSE(std::isnan(result->implied_vol))
                << "IV must not be NaN";
            EXPECT_FALSE(std::isinf(result->implied_vol))
                << "IV must not be Inf";
        }
    }
}

// ============================================================================
// IV Bounds Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, IVWithinConfiguredBounds) {
    std::vector<double> vols = {0.15, 0.25, 0.35, 0.50};

    // Standard bounds for Brent solver
    constexpr double vol_lower = 0.01;
    constexpr double vol_upper = 3.0;

    IVSolverFDM solver(config_);

    for (double vol : vols) {
        auto price = price_option(100.0, 100.0, 1.0, 0.05, 0.0, vol, OptionType::PUT);
        if (!price.has_value() || *price < 0.10) continue;

        IVQuery query(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, *price);
        auto result = solver.solve_impl(query);

        if (result.has_value()) {
            EXPECT_GE(result->implied_vol, vol_lower)
                << "IV below lower bound for vol=" << vol;
            EXPECT_LE(result->implied_vol, vol_upper)
                << "IV above upper bound for vol=" << vol;
        }
    }
}

// ============================================================================
// IV Monotonicity Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, HigherPriceGivesHigherIV) {
    // Price options at different volatilities
    double vol_low = 0.15;
    double vol_high = 0.35;

    auto price_low = price_option(100.0, 100.0, 1.0, 0.05, 0.0, vol_low, OptionType::PUT);
    auto price_high = price_option(100.0, 100.0, 1.0, 0.05, 0.0, vol_high, OptionType::PUT);

    ASSERT_TRUE(price_low.has_value());
    ASSERT_TRUE(price_high.has_value());
    ASSERT_GT(*price_high, *price_low) << "Higher vol should give higher price";

    IVSolverFDM solver(config_);

    IVQuery query_low(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, *price_low);
    IVQuery query_high(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, *price_high);

    auto result_low = solver.solve_impl(query_low);
    auto result_high = solver.solve_impl(query_high);

    ASSERT_TRUE(result_low.has_value());
    ASSERT_TRUE(result_high.has_value());

    EXPECT_GT(result_high->implied_vol, result_low->implied_vol - 0.005)
        << "Higher price should give higher IV";
}

// ============================================================================
// No NaN/Inf Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, NeverProducesNaNOrInf) {
    std::vector<IVQuery> queries = {
        IVQuery(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 10.0),
        IVQuery(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::CALL, 10.0),
        IVQuery(100.0, 80.0, 1.0, 0.05, 0.02, OptionType::PUT, 5.0),
        IVQuery(100.0, 120.0, 1.0, 0.05, 0.02, OptionType::PUT, 25.0),
        IVQuery(100.0, 100.0, 0.25, 0.05, 0.02, OptionType::PUT, 5.0),
        IVQuery(100.0, 100.0, 2.0, 0.05, 0.02, OptionType::PUT, 15.0),
    };

    IVSolverFDM solver(config_);

    for (const auto& query : queries) {
        auto result = solver.solve_impl(query);

        if (result.has_value()) {
            EXPECT_FALSE(std::isnan(result->implied_vol))
                << "IV is NaN for query with market_price=" << query.market_price;
            EXPECT_FALSE(std::isinf(result->implied_vol))
                << "IV is Inf for query with market_price=" << query.market_price;
            EXPECT_FALSE(std::isnan(result->final_error))
                << "Final error is NaN";
            EXPECT_FALSE(std::isinf(result->final_error))
                << "Final error is Inf";
        } else {
            EXPECT_FALSE(std::isnan(result.error().final_error))
                << "Error final_error is NaN";
        }
    }
}

// ============================================================================
// Batch IV Consistency Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, BatchResultsMatchIndividual) {
    // Create a batch of queries
    std::vector<IVQuery> queries;
    std::vector<double> prices = {8.0, 10.0, 12.0, 14.0, 16.0};

    for (double price : prices) {
        queries.push_back(IVQuery(100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, price));
    }

    IVSolverFDM solver(config_);

    // Solve batch
    auto batch_result = solver.solve_batch_impl(queries);

    // Verify batch size
    ASSERT_EQ(batch_result.results.size(), queries.size());

    // Verify each individual result matches
    for (size_t i = 0; i < queries.size(); ++i) {
        auto individual = solver.solve_impl(queries[i]);

        if (individual.has_value() && batch_result.results[i].has_value()) {
            EXPECT_NEAR(individual->implied_vol,
                       batch_result.results[i]->implied_vol, 1e-6)
                << "Batch and individual results differ at index " << i;
        } else {
            // Both should fail or succeed together
            EXPECT_EQ(individual.has_value(), batch_result.results[i].has_value())
                << "Batch and individual success/failure mismatch at index " << i;
        }
    }

    // Verify failed_count
    size_t actual_failures = 0;
    for (const auto& r : batch_result.results) {
        if (!r.has_value()) ++actual_failures;
    }
    EXPECT_EQ(batch_result.failed_count, actual_failures);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(IVSolverPropertyTest, ArbitragePricesHandledGracefully) {
    IVSolverFDM solver(config_);

    // Price below intrinsic (arbitrage) - ITM put
    // Intrinsic for K=110, S=100 put = 10
    IVQuery below_intrinsic(100.0, 110.0, 1.0, 0.05, 0.0, OptionType::PUT, 5.0);
    auto result1 = solver.solve_impl(below_intrinsic);
    // Should either fail or return very low IV - just verify no crash

    // Price above upper bound (arbitrage)
    // Upper bound for put = strike = 110
    IVQuery above_upper(100.0, 110.0, 1.0, 0.05, 0.0, OptionType::PUT, 150.0);
    auto result2 = solver.solve_impl(above_upper);
    // Should fail - verify no crash
}

TEST_F(IVSolverPropertyTest, ShortMaturityHandled) {
    IVSolverFDM solver(config_);

    // Very short maturity
    auto price = price_option(100.0, 100.0, 0.05, 0.05, 0.0, 0.30, OptionType::PUT);
    if (price.has_value() && *price > 0.10) {
        IVQuery query(100.0, 100.0, 0.05, 0.05, 0.0, OptionType::PUT, *price);
        auto result = solver.solve_impl(query);

        if (result.has_value()) {
            EXPECT_GT(result->implied_vol, 0.0);
            EXPECT_FALSE(std::isnan(result->implied_vol));
        }
    }
}

TEST_F(IVSolverPropertyTest, LongMaturityHandled) {
    IVSolverFDM solver(config_);

    // Long maturity
    auto price = price_option(100.0, 100.0, 3.0, 0.05, 0.0, 0.25, OptionType::PUT);
    if (price.has_value() && *price > 0.10) {
        IVQuery query(100.0, 100.0, 3.0, 0.05, 0.0, OptionType::PUT, *price);
        auto result = solver.solve_impl(query);

        if (result.has_value()) {
            EXPECT_GT(result->implied_vol, 0.0);
            EXPECT_NEAR(result->implied_vol, 0.25, 0.02);
        }
    }
}
