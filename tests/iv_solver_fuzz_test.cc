// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_fuzz_test.cc
 * @brief Fuzz tests for FDM implied volatility solver using FuzzTest
 *
 * These tests use property-based fuzzing to verify invariants of the
 * probe-based FDM IV solver. Each solve takes ~19ms, so domains are
 * kept tight to avoid timeouts.
 *
 * Requires Earthly container with Clang + libc++ due to C++23 ABI issues.
 *
 * Run via:
 *   earthly +fuzz-test
 *
 * Or manually in a libc++ environment:
 *   bazel test --config=fuzz //tests:iv_solver_fuzz_test
 */

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include <cmath>

using namespace mango;

// ============================================================================
// Property: IV round-trip (price -> IV -> price)
// ============================================================================
// Generate a known option with a known vol, price it, then solve for IV
// from that price. The recovered IV should match the input vol.

void IVRoundTrip(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity < 0.05 || volatility < 0.05) return;

    OptionSpec spec{.spot = spot, .strike = strike, .maturity = maturity,
                    .rate = rate,
                    .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    // Price with known vol
    PricingParams params(spec, volatility);
    auto price_result = solve_american_option(params);
    if (!price_result.has_value()) return;

    double market_price = price_result->value();
    if (market_price < 0.01) return;  // Skip near-zero prices (IV unstable)

    // Solve for IV from that price
    IVQuery query(spec, market_price);
    IVSolver solver(IVSolverConfig{});
    auto iv_result = solver.solve(query);

    if (!iv_result.has_value()) return;  // Solver may fail for edge cases

    EXPECT_NEAR(iv_result->implied_vol, volatility, 0.01)
        << "IV round-trip failed: input vol=" << volatility
        << ", recovered=" << iv_result->implied_vol
        << " for S=" << spot << " K=" << strike << " T=" << maturity;
}

FUZZ_TEST(IVSolverFuzz, IVRoundTrip)
    .WithDomains(
        fuzztest::InRange(90.0, 110.0),         // spot
        fuzztest::InRange(90.0, 110.0),         // strike
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.10, 0.50),          // volatility
        fuzztest::InRange(0.0, 0.10),           // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: IV is positive and bounded
// ============================================================================

void IVPositiveAndBounded(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity < 0.05 || volatility < 0.05) return;

    OptionSpec spec{.spot = spot, .strike = strike, .maturity = maturity,
                    .rate = rate,
                    .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    PricingParams params(spec, volatility);
    auto price_result = solve_american_option(params);
    if (!price_result.has_value()) return;

    double market_price = price_result->value();
    if (market_price < 0.01) return;

    IVQuery query(spec, market_price);
    IVSolver solver(IVSolverConfig{});
    auto iv_result = solver.solve(query);

    if (!iv_result.has_value()) return;

    EXPECT_GE(iv_result->implied_vol, 0.01)
        << "IV too small: " << iv_result->implied_vol;
    EXPECT_LE(iv_result->implied_vol, 3.0)
        << "IV too large: " << iv_result->implied_vol;
}

FUZZ_TEST(IVSolverFuzz, IVPositiveAndBounded)
    .WithDomains(
        fuzztest::InRange(80.0, 120.0),         // spot
        fuzztest::InRange(80.0, 120.0),         // strike
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.05, 0.80),          // volatility
        fuzztest::InRange(0.0, 0.10),           // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: IV is monotonic in market price
// ============================================================================
// For the same option spec, higher market price should give higher IV.

void IVMonotonicInPrice(
    double spot,
    double strike,
    double maturity,
    double vol1,
    double vol2,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity < 0.05) return;
    if (vol1 < 0.05 || vol2 < 0.05) return;
    if (vol1 >= vol2) std::swap(vol1, vol2);
    if (vol2 - vol1 < 0.05) return;

    OptionSpec spec{.spot = spot, .strike = strike, .maturity = maturity,
                    .rate = rate,
                    .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    // Price at two different vols
    PricingParams params1(spec, vol1);
    PricingParams params2(spec, vol2);
    auto price_result1 = solve_american_option(params1);
    auto price_result2 = solve_american_option(params2);
    if (!price_result1.has_value() || !price_result2.has_value()) return;

    double price1 = price_result1->value();
    double price2 = price_result2->value();
    if (price1 < 0.01 || price2 < 0.01) return;

    // Solve IV for each price
    IVSolver solver(IVSolverConfig{});
    auto iv1 = solver.solve(IVQuery(spec, price1));
    auto iv2 = solver.solve(IVQuery(spec, price2));

    if (!iv1.has_value() || !iv2.has_value()) return;

    // Higher price -> higher IV
    EXPECT_LE(iv1->implied_vol, iv2->implied_vol + 0.005)
        << "IV monotonicity violated: price1=" << price1 << " IV1=" << iv1->implied_vol
        << ", price2=" << price2 << " IV2=" << iv2->implied_vol;
}

FUZZ_TEST(IVSolverFuzz, IVMonotonicInPrice)
    .WithDomains(
        fuzztest::InRange(90.0, 110.0),         // spot
        fuzztest::InRange(90.0, 110.0),         // strike
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.05, 0.60),          // vol1
        fuzztest::InRange(0.05, 0.60),          // vol2
        fuzztest::InRange(0.0, 0.10),           // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: IV scale invariance
// ============================================================================
// IV depends only on moneyness, not on the absolute level of S and K.
// IV(S, K, P) should equal IV(λS, λK, λP).

void IVScaleInvariance(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double scale_factor,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity < 0.05 || volatility < 0.05) return;
    if (scale_factor < 0.5 || scale_factor > 2.0) return;

    OptionSpec spec1{.spot = spot, .strike = strike, .maturity = maturity,
                     .rate = rate,
                     .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    // Price original
    PricingParams params1(spec1, volatility);
    auto price_result1 = solve_american_option(params1);
    if (!price_result1.has_value()) return;

    double price1 = price_result1->value();
    if (price1 < 0.01) return;

    // Scaled option: λS, λK, λP
    double scaled_spot = spot * scale_factor;
    double scaled_strike = strike * scale_factor;
    double scaled_price = price1 * scale_factor;

    OptionSpec spec2{.spot = scaled_spot, .strike = scaled_strike,
                     .maturity = maturity, .rate = rate,
                     .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    // Solve IV for both
    IVSolver solver(IVSolverConfig{});
    auto iv1 = solver.solve(IVQuery(spec1, price1));
    auto iv2 = solver.solve(IVQuery(spec2, scaled_price));

    if (!iv1.has_value() || !iv2.has_value()) return;

    EXPECT_NEAR(iv1->implied_vol, iv2->implied_vol, 0.005)
        << "Scale invariance violated: IV(" << spot << "," << strike << ")="
        << iv1->implied_vol << ", IV(" << scaled_spot << "," << scaled_strike
        << ")=" << iv2->implied_vol;
}

FUZZ_TEST(IVSolverFuzz, IVScaleInvariance)
    .WithDomains(
        fuzztest::InRange(90.0, 110.0),         // spot
        fuzztest::InRange(90.0, 110.0),         // strike
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.10, 0.50),          // volatility
        fuzztest::InRange(0.0, 0.10),           // rate
        fuzztest::InRange(0.5, 2.0),            // scale_factor
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: No NaN or Inf in results
// ============================================================================

void IVNoNaNInf(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike <= 0 || maturity < 0.05 || volatility < 0.05) return;

    OptionSpec spec{.spot = spot, .strike = strike, .maturity = maturity,
                    .rate = rate,
                    .option_type = is_call ? OptionType::CALL : OptionType::PUT};

    PricingParams params(spec, volatility);
    auto price_result = solve_american_option(params);
    if (!price_result.has_value()) return;

    double market_price = price_result->value();
    if (market_price < 0.001) return;

    IVQuery query(spec, market_price);
    IVSolver solver(IVSolverConfig{});
    auto iv_result = solver.solve(query);

    if (iv_result.has_value()) {
        EXPECT_FALSE(std::isnan(iv_result->implied_vol))
            << "NaN implied_vol";
        EXPECT_FALSE(std::isinf(iv_result->implied_vol))
            << "Inf implied_vol";
        EXPECT_FALSE(std::isnan(iv_result->final_error))
            << "NaN final_error";
        EXPECT_FALSE(std::isinf(iv_result->final_error))
            << "Inf final_error";
        if (iv_result->vega.has_value()) {
            EXPECT_FALSE(std::isnan(*iv_result->vega))
                << "NaN vega";
            EXPECT_FALSE(std::isinf(*iv_result->vega))
                << "Inf vega";
        }
    } else {
        // Error results should also have finite diagnostics
        auto& err = iv_result.error();
        EXPECT_FALSE(std::isnan(err.final_error))
            << "NaN final_error in error result";
        EXPECT_FALSE(std::isinf(err.final_error))
            << "Inf final_error in error result";
        if (err.last_vol.has_value()) {
            EXPECT_FALSE(std::isnan(*err.last_vol))
                << "NaN last_vol in error result";
            EXPECT_FALSE(std::isinf(*err.last_vol))
                << "Inf last_vol in error result";
        }
    }
}

FUZZ_TEST(IVSolverFuzz, IVNoNaNInf)
    .WithDomains(
        fuzztest::InRange(50.0, 200.0),         // spot
        fuzztest::InRange(50.0, 200.0),         // strike
        fuzztest::InRange(0.05, 3.0),           // maturity
        fuzztest::InRange(0.05, 1.0),           // volatility
        fuzztest::InRange(-0.05, 0.20),         // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Property: Batch vs single consistency
// ============================================================================
// solve_batch([q1, q2, ...]) should match [solve(q1), solve(q2), ...].

void IVBatchVsSingleConsistency(
    double spot,
    double strike_base,
    double maturity,
    double volatility,
    double rate,
    bool is_call)
{
    if (spot <= 0 || strike_base <= 0 || maturity < 0.05 || volatility < 0.05) return;

    // Create 3 queries with different strikes
    std::vector<IVQuery> queries;
    for (int i = 0; i < 3; ++i) {
        double K = strike_base + i * 5.0;
        if (K <= 0) continue;

        OptionSpec spec{.spot = spot, .strike = K, .maturity = maturity,
                        .rate = rate,
                        .option_type = is_call ? OptionType::CALL : OptionType::PUT};

        PricingParams params(spec, volatility);
        auto price_result = solve_american_option(params);
        if (!price_result.has_value()) continue;

        double market_price = price_result->value();
        if (market_price < 0.01) continue;

        queries.push_back(IVQuery(spec, market_price));
    }

    if (queries.size() < 2) return;

    IVSolver solver(IVSolverConfig{});

    // Batch solve
    auto batch_result = solver.solve_batch(queries);

    // Single solves
    ASSERT_EQ(batch_result.results.size(), queries.size());

    for (size_t i = 0; i < queries.size(); ++i) {
        auto single_result = solver.solve(queries[i]);

        bool batch_ok = batch_result.results[i].has_value();
        bool single_ok = single_result.has_value();

        // Both should succeed or both should fail
        EXPECT_EQ(batch_ok, single_ok)
            << "Batch/single disagree on success at index " << i;

        if (batch_ok && single_ok) {
            EXPECT_NEAR(batch_result.results[i]->implied_vol,
                        single_result->implied_vol, 1e-6)
                << "Batch/single IV mismatch at index " << i;
        }
    }
}

FUZZ_TEST(IVSolverFuzz, IVBatchVsSingleConsistency)
    .WithDomains(
        fuzztest::InRange(90.0, 110.0),         // spot
        fuzztest::InRange(85.0, 105.0),         // strike_base
        fuzztest::InRange(0.1, 2.0),            // maturity
        fuzztest::InRange(0.10, 0.50),          // volatility
        fuzztest::InRange(0.0, 0.10),           // rate
        fuzztest::Arbitrary<bool>()             // is_call
    );

// ============================================================================
// Standard unit test to verify fuzz test setup works
// ============================================================================

TEST(IVSolverFuzz, SmokeTest) {
    IVRoundTrip(100.0, 100.0, 1.0, 0.20, 0.05, false);
    IVPositiveAndBounded(100.0, 100.0, 1.0, 0.20, 0.05, false);
    IVMonotonicInPrice(100.0, 100.0, 1.0, 0.15, 0.30, 0.05, false);
    IVScaleInvariance(100.0, 100.0, 1.0, 0.20, 0.05, 1.5, false);
    IVNoNaNInf(100.0, 100.0, 1.0, 0.20, 0.05, true);
    IVBatchVsSingleConsistency(100.0, 90.0, 1.0, 0.20, 0.05, false);
}
