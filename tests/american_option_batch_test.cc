#include <gtest/gtest.h>
#include "src/option/american_option_batch.hpp"

using namespace mango;

TEST(BatchAmericanOptionSolver, NormalizedEligibility) {
    // Test eligible batch: varying strikes with same maturity
    std::vector<AmericanOptionParams> eligible_params;
    double spot = 100.0;
    std::vector<double> strikes = {90, 95, 100, 105, 110};

    for (double K : strikes) {
        eligible_params.push_back(PricingParams(
            spot,                  // spot
            K,                     // strike (varying)
            1.0,                   // maturity (same)
            0.05,                  // rate
            0.02,                  // dividend_yield
            OptionType::PUT,       // type
            0.20,                  // volatility
            {}                     // discrete_dividends
        ));
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(eligible_params, /*use_shared_grid=*/true);

    // Should use normalized path: 1 PDE solve for 5 options
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);

    // All results should have converged
    for (const auto& r : result.results) {
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(r->converged);
        EXPECT_GT(r->value(), 0.0);
    }
}

TEST(BatchAmericanOptionSolver, NormalizedIneligibleDividends) {
    // Test ineligible batch (discrete dividends)
    std::vector<AmericanOptionParams> ineligible_params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        ineligible_params.push_back(PricingParams(
            spot,                          // spot
            90.0 + i * 5.0,                // strike
            1.0,                           // maturity
            0.05,                          // rate
            0.02,                          // dividend_yield
            OptionType::PUT,               // type
            0.20,                          // volatility
            {{0.5, 2.0}}                   // discrete_dividends (has discrete dividend)
        ));
    }

    BatchAmericanOptionSolver solver;
    auto result = solver.solve_batch(ineligible_params, /*use_shared_grid=*/true);

    // Should fall back to regular path
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}

TEST(BatchAmericanOptionSolver, DisableNormalizedOptimization) {
    // Test forcing regular path
    std::vector<AmericanOptionParams> params;
    double spot = 100.0;

    for (int i = 0; i < 5; ++i) {
        params.push_back(PricingParams(
            spot,                  // spot
            90.0 + i * 5.0,        // strike
            1.0,                   // maturity
            0.05,                  // rate
            0.02,                  // dividend_yield
            OptionType::PUT,       // type
            0.20,                  // volatility
            {}                     // discrete_dividends
        ));
    }

    BatchAmericanOptionSolver solver;
    solver.set_use_normalized(false);  // Force regular path

    auto result = solver.solve_batch(params, /*use_shared_grid=*/true);
    EXPECT_EQ(result.failed_count, 0);
    EXPECT_EQ(result.results.size(), 5);
}
