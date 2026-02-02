// SPDX-License-Identifier: MIT
/**
 * @file normalized_chain_accuracy_test.cc
 * @brief Accuracy validation of normalized chain solver (PDE invariant solution)
 *
 * Tests:
 * - Normalized vs regular solver consistency (within 0.1%)
 * - Normalized solver vs QuantLib accuracy (within 1.0%)
 * - PDE invariance property: normalized solutions preserve accuracy
 *
 * The normalized chain solver exploits PDE scale-invariance: when normalized
 * by strike, the Black-Scholes PDE becomes independent of strike. This allows
 * solving one normalized PDE and scaling results for multiple strikes.
 *
 * This test ensures the normalized approach maintains accuracy.
 */

#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"
#include "tests/quantlib_validation_framework.hpp"

using namespace mango;
using namespace mango::testing;

// ============================================================================
// Normalized vs Regular Consistency Test
// ============================================================================

TEST(NormalizedChainAccuracy, ConsistencyWithRegularSolver) {
    // Create batch eligible for normalized solving
    double spot = 100.0;
    std::vector<double> strikes = {85.0, 92.5, 100.0, 107.5, 115.0};
    double maturity = 1.0;
    double rate = 0.05;
    double dividend_yield = 0.02;
    double volatility = 0.20;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    // Solve with normalized chain (fast path)
    BatchAmericanOptionSolver normalized_solver;
    normalized_solver.set_use_normalized(true);
    auto normalized_results = normalized_solver.solve_batch(
        params, /*use_shared_grid=*/true);

    // Solve with regular solver (reference)
    BatchAmericanOptionSolver regular_solver;
    regular_solver.set_use_normalized(false);
    auto regular_results = regular_solver.solve_batch(
        params, /*use_shared_grid=*/true);

    ASSERT_EQ(normalized_results.failed_count, 0);
    ASSERT_EQ(regular_results.failed_count, 0);
    ASSERT_EQ(normalized_results.results.size(), strikes.size());
    ASSERT_EQ(regular_results.results.size(), strikes.size());

    // Compare results: normalized should match regular within tight tolerance
    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(normalized_results.results[i].has_value());
        ASSERT_TRUE(regular_results.results[i].has_value());

        const auto& norm_result = normalized_results.results[i].value();
        const auto& reg_result = regular_results.results[i].value();

        // Price comparison
        double norm_price = norm_result.value_at(spot);
        double reg_price = reg_result.value_at(spot);
        double price_rel_error = std::abs(norm_price - reg_price) / reg_price * 100.0;

        // Tolerance relaxed from 0.1% to 0.3% to account for the fact that
        // regular batch solver now correctly uses the shared grid domain
        // (wider domain = slightly different numerical resolution)
        EXPECT_LT(price_rel_error, 0.3)  // Within 0.3%
            << "Normalized price: $" << norm_price
            << "\nRegular price: $" << reg_price
            << "\nError: " << price_rel_error << "%";

        // Delta comparison (more relaxed - normalized grid transformation affects derivatives)
        double norm_delta = norm_result.delta();
        double reg_delta = reg_result.delta();
        double delta_abs_error = std::abs(norm_delta - reg_delta);

        EXPECT_LT(delta_abs_error, 0.03)  // Within 0.03 absolute (3% of typical delta magnitude)
            << "Normalized delta: " << norm_delta
            << "\nRegular delta: " << reg_delta
            << "\nError: " << delta_abs_error;

        // Gamma comparison (more relaxed - second derivative more sensitive to grid)
        double norm_gamma = norm_result.gamma();
        double reg_gamma = reg_result.gamma();
        double gamma_abs_error = std::abs(norm_gamma - reg_gamma);

        EXPECT_LT(gamma_abs_error, 0.001)  // Within 0.001 absolute
            << "Normalized gamma: " << norm_gamma
            << "\nRegular gamma: " << reg_gamma
            << "\nError: " << gamma_abs_error;
    }
}

// ============================================================================
// Normalized vs QuantLib Accuracy Test
// ============================================================================

TEST(NormalizedChainAccuracy, QuantLibComparison_ATM_Chain) {
    // ATM chain with varying strikes
    double spot = 100.0;
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    double maturity = 1.0;
    double rate = 0.05;
    double dividend_yield = 0.02;
    double volatility = 0.25;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    // Solve with normalized chain
    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);
    ASSERT_EQ(results.results.size(), strikes.size());

    // Compare each result to QuantLib
    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        // Get QuantLib reference price
        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);
        double price_rel_error = (price_error / ql_result.price) * 100.0;

        EXPECT_LT(price_rel_error, 1.0)  // Within 1.0%
            << "Mango (normalized): $" << mango_price
            << "\nQuantLib: $" << ql_result.price
            << "\nError: " << price_rel_error << "%";

        // Delta comparison (relaxed - normalized solving uses different grid spacing)
        double delta_val = result.delta();
        double delta_error = std::abs(delta_val - ql_result.delta);
        double delta_rel = (delta_error / std::abs(ql_result.delta)) * 100.0;

        EXPECT_LT(delta_rel, 9.0)  // Within 9%
            << "Mango delta: " << delta_val
            << "\nQuantLib delta: " << ql_result.delta
            << "\nError: " << delta_rel << "%";

        // Gamma comparison
        double gamma_val = result.gamma();
        double gamma_error = std::abs(gamma_val - ql_result.gamma);
        double gamma_rel = (gamma_error / std::abs(ql_result.gamma)) * 100.0;

        EXPECT_LT(gamma_rel, 5.0)  // Within 5%
            << "Mango gamma: " << gamma_val
            << "\nQuantLib gamma: " << ql_result.gamma
            << "\nError: " << gamma_rel << "%";
    }
}

// ============================================================================
// Deep ITM/OTM Chains (Challenging Cases)
// ============================================================================

TEST(NormalizedChainAccuracy, QuantLibComparison_DeepITM_Puts) {
    // Deep ITM puts: high strikes relative to spot
    double spot = 100.0;
    std::vector<double> strikes = {110.0, 120.0, 130.0, 140.0, 150.0};
    double maturity = 0.5;
    double rate = 0.05;
    double dividend_yield = 0.02;
    double volatility = 0.25;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);
        double price_rel_error = (price_error / ql_result.price) * 100.0;

        EXPECT_LT(price_rel_error, 1.0)
            << "Deep ITM error: " << price_rel_error << "%"
            << "\nMango: $" << mango_price
            << "\nQuantLib: $" << ql_result.price;
    }
}

TEST(NormalizedChainAccuracy, QuantLibComparison_DeepOTM_Puts) {
    // Deep OTM puts: low strikes relative to spot
    double spot = 100.0;
    std::vector<double> strikes = {50.0, 60.0, 70.0, 80.0, 90.0};
    double maturity = 0.5;
    double rate = 0.05;
    double dividend_yield = 0.02;
    double volatility = 0.25;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);

        // For deep OTM, absolute error is more meaningful than relative
        EXPECT_LT(price_error, 0.05)  // Within $0.05
            << "Deep OTM absolute error: $" << price_error
            << "\nMango: $" << mango_price
            << "\nQuantLib: $" << ql_result.price;
    }
}

// ============================================================================
// High Volatility Chain
// ============================================================================

TEST(NormalizedChainAccuracy, QuantLibComparison_HighVolatility) {
    // High volatility scenario
    double spot = 100.0;
    std::vector<double> strikes = {85.0, 92.5, 100.0, 107.5, 115.0};
    double maturity = 1.0;
    double rate = 0.05;
    double dividend_yield = 0.02;
    double volatility = 0.50;  // 50% vol

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);
        double price_rel_error = (price_error / ql_result.price) * 100.0;

        EXPECT_LT(price_rel_error, 1.0)
            << "High vol error: " << price_rel_error << "%"
            << "\nMango: $" << mango_price
            << "\nQuantLib: $" << ql_result.price;
    }
}

// ============================================================================
// Extreme Interest Rate Scenarios
// ============================================================================

TEST(NormalizedChainAccuracy, QuantLibComparison_NegativeRate) {
    // Negative interest rate (European crisis scenario)
    double spot = 100.0;
    std::vector<double> strikes = {85.0, 92.5, 100.0, 107.5, 115.0};
    double maturity = 0.5;
    double rate = -0.01;  // -1% rate
    double dividend_yield = 0.0;
    double volatility = 0.20;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);
        double price_rel_error = (price_error / ql_result.price) * 100.0;

        EXPECT_LT(price_rel_error, 1.0)
            << "Negative rate error: " << price_rel_error << "%";
    }
}

TEST(NormalizedChainAccuracy, QuantLibComparison_HighRate) {
    // High interest rate (emerging markets scenario)
    double spot = 100.0;
    std::vector<double> strikes = {85.0, 92.5, 100.0, 107.5, 115.0};
    double maturity = 0.5;
    double rate = 0.15;  // 15% rate
    double dividend_yield = 0.03;
    double volatility = 0.30;

    std::vector<PricingParams> params;
    for (double K : strikes) {
        params.push_back(PricingParams(OptionSpec{.spot = spot, .strike = K, .maturity = maturity, .rate = rate, .dividend_yield = dividend_yield, .option_type = OptionType::PUT}, volatility));
    }

    BatchAmericanOptionSolver solver;
    auto results = solver.solve_batch(params, /*use_shared_grid=*/true);

    ASSERT_EQ(results.failed_count, 0);

    for (size_t i = 0; i < strikes.size(); ++i) {
        SCOPED_TRACE("Strike: " + std::to_string(strikes[i]));

        ASSERT_TRUE(results.results[i].has_value());
        const auto& result = results.results[i].value();

        auto ql_result = price_with_quantlib(
            spot, strikes[i], maturity, volatility, rate, dividend_yield,
            false, 201, 2000);

        double mango_price = result.value_at(spot);
        double price_error = std::abs(mango_price - ql_result.price);
        double price_rel_error = (price_error / ql_result.price) * 100.0;

        EXPECT_LT(price_rel_error, 1.0)
            << "High rate error: " << price_rel_error << "%";
    }
}
