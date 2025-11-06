#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>

extern "C" {
#include "../src/implied_volatility.h"
#include "../src/american_option.h"
#include "../src/price_table.h"
}

// Test fixture for American option implied volatility tests
class ImpliedVolatilityTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid;

    void SetUp() override {
        default_grid.x_min = -0.7;
        default_grid.x_max = 0.7;
        default_grid.n_points = 141;
        default_grid.dt = 0.001;
        default_grid.n_steps = 1000;
    }
};

// Test ATM American put IV
TEST_F(ImpliedVolatilityTest, ATMPutIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 6.08,  // Typical American put price
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.15);  // Reasonable range
    EXPECT_LT(result.implied_vol, 0.35);
    EXPECT_LT(result.iterations, 20);      // Should converge quickly
    EXPECT_EQ(result.error, nullptr);
}

// Test OTM American call IV
TEST_F(ImpliedVolatilityTest, OTMCallIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.5,
        .risk_free_rate = 0.03,
        .dividend_yield = 0.0,
        .market_price = 3.0,
        .option_type = OPTION_CALL,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.10);
    EXPECT_LT(result.implied_vol, 0.50);
    EXPECT_EQ(result.error, nullptr);
}

// Test ITM American put IV
TEST_F(ImpliedVolatilityTest, ITMPutIV) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 0.25,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 11.0,  // ITM put
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.05);
    EXPECT_LT(result.implied_vol, 0.60);
    EXPECT_EQ(result.error, nullptr);
}

// Test convenience function
TEST_F(ImpliedVolatilityTest, SimpleCalculation) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 6.0,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv_simple(&params, nullptr);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 1.0);
}

// Test invalid inputs
TEST_F(ImpliedVolatilityTest, InvalidSpot) {
    IVParams params = {
        .spot_price = -100.0,  // Invalid
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Test arbitrage violation (price below intrinsic)
TEST_F(ImpliedVolatilityTest, BelowIntrinsicValue) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 110.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,  // Below intrinsic (10)
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_FALSE(result.converged);
    EXPECT_NE(result.error, nullptr);
}

// Test short maturity
TEST_F(ImpliedVolatilityTest, ShortMaturity) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 0.027,  // ~1 week
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 2.0,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
}

// Test consistency: reverse-engineer IV
TEST_F(ImpliedVolatilityTest, RoundTripConsistency) {
    // Start with known volatility, compute price, then recover IV
    double true_vol = 0.30;

    OptionData option = {
        .strike = 100.0,
        .volatility = true_vol,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = NULL,
        .dividend_amounts = NULL
    };

    // Compute American option price with known vol
    AmericanOptionResult price_result = american_option_price(&option, &default_grid);
    ASSERT_EQ(price_result.status, 0);

    double market_price = american_option_get_value_at_spot(
        price_result.solver, 100.0, 100.0);

    american_option_free_result(&price_result);

    // Now recover IV from that price
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = market_price,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult iv_result = calculate_iv(&params, &default_grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(iv_result.converged);
    // Should recover original volatility within tolerance
    EXPECT_NEAR(iv_result.implied_vol, true_vol, 0.01);  // Within 1%
}

// Test new API with NULL table (Task 1)
TEST_F(ImpliedVolatilityTest, AcceptsNullTable) {
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EUROPEAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    // Should work with NULL table (uses FDM fallback)
    IVResult result = calculate_iv(&params, &grid, nullptr, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.implied_vol, 0.0);
    EXPECT_LT(result.implied_vol, 1.0);
}

// Test in-bounds detection (Task 2)
TEST_F(ImpliedVolatilityTest, DetectsInBoundsPoint) {
    // Create simple 4D table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, EUROPEAN,
        COORD_RAW, LAYOUT_M_INNER);

    // In-bounds point
    IVParams params_in = {
        .spot_price = 100.0,
        .strike = 100.0,  // m = 1.0 (in bounds)
        .time_to_maturity = 1.0,  // tau = 1.0 (in bounds)
        .risk_free_rate = 0.05,  // r = 0.05 (in bounds)
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_CALL,
        .exercise_type = EUROPEAN
    };

    // This should use table interpolation (when implemented)
    // For now, just testing it doesn't crash
    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    IVResult result = calculate_iv(&params_in, &grid, table, 1e-6, 100);
    EXPECT_TRUE(result.converged);

    price_table_destroy(table);
}

// Test out-of-bounds detection (Task 2)
TEST_F(ImpliedVolatilityTest, DetectsOutOfBoundsPoint) {
    // Create simple 4D table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    // Out-of-bounds point (tau too large)
    IVParams params_out = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 5.0,  // tau = 5.0 (out of bounds)
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 10.0,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    // Should fallback to FDM
    IVResult result = calculate_iv(&params_out, &grid, table, 1e-6, 100);
    EXPECT_TRUE(result.converged);

    price_table_destroy(table);
}

// Test Newton's method with precomputed table (Task 3)
TEST_F(ImpliedVolatilityTest, NewtonMethodWithTable) {
    // Create table and precompute
    std::vector<double> m = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3};
    std::vector<double> tau = {0.25, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> sigma = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};
    std::vector<double> r = {0.0, 0.03, 0.05, 0.07, 0.10};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 101, .dt = 0.001, .n_steps = 1000
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Get a reference price at known volatility
    double test_vol = 0.25;
    double test_price = price_table_interpolate_4d(table, 1.0, 1.0, test_vol, 0.05);

    // Now solve for IV using that price
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = test_price,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.iterations, 10);  // Newton should converge quickly
    EXPECT_NEAR(result.implied_vol, test_vol, 0.01);  // Should recover the volatility

    price_table_destroy(table);
}

// Test fallback when table has no vegas (Task 4)
TEST_F(ImpliedVolatilityTest, FallbackWhenNoVegas) {
    // Create table but don't precompute (no vegas)
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25, 0.30};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_PUT, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    // Don't precompute - no vegas available
    EXPECT_EQ(table->vegas, nullptr);

    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,
        .option_type = OPTION_PUT,
        .exercise_type = AMERICAN
    };

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    // Should fallback to FDM
    EXPECT_TRUE(result.converged);

    price_table_destroy(table);
}

// Test fallback on option type mismatch (Task 4)
TEST_F(ImpliedVolatilityTest, FallbackOnOptionTypeMismatch) {
    // Create CALL table
    std::vector<double> m = {0.8, 0.9, 1.0, 1.1, 1.2};
    std::vector<double> tau = {0.5, 1.0, 1.5};
    std::vector<double> sigma = {0.15, 0.20, 0.25};
    std::vector<double> r = {0.03, 0.05, 0.07};

    OptionPriceTable *table = price_table_create_ex(
        m.data(), m.size(),
        tau.data(), tau.size(),
        sigma.data(), sigma.size(),
        r.data(), r.size(),
        nullptr, 0,
        OPTION_CALL, AMERICAN,
        COORD_RAW, LAYOUT_M_INNER);

    AmericanOptionGrid grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 51, .dt = 0.01, .n_steps = 100
    };

    price_table_precompute(table, &grid);
    price_table_build_interpolation(table);

    // Query for PUT (mismatch)
    IVParams params = {
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .market_price = 5.0,
        .option_type = OPTION_PUT,  // Mismatch!
        .exercise_type = AMERICAN
    };

    IVResult result = calculate_iv(&params, &grid, table, 1e-6, 100);

    // Should fallback to FDM
    EXPECT_TRUE(result.converged);

    price_table_destroy(table);
}

// Comprehensive accuracy comparison: FDM vs Table-based IV
TEST_F(ImpliedVolatilityTest, AccuracyComparison) {
    printf("\n=== IV Accuracy Comparison ===\n\n");

    // Create tables with different resolutions
    struct TableConfig {
        const char *name;
        std::vector<double> m;
        std::vector<double> tau;
        std::vector<double> sigma;
        std::vector<double> r;
    };

    std::vector<TableConfig> configs = {
        {"Coarse (7×3×5×3 = 315 points)",
         {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15},
         {0.5, 1.0, 2.0},
         {0.15, 0.20, 0.25, 0.30, 0.35},
         {0.03, 0.05, 0.07}},

        {"Medium (10×5×7×5 = 1750 points)",
         {0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20},
         {0.25, 0.5, 1.0, 1.5, 2.0},
         {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40},
         {0.0, 0.03, 0.05, 0.07, 0.10}}
    };

    // Test cases (ground truth: calculate IV using FDM)
    struct TestCase {
        double spot;
        double strike;
        double maturity;
        double rate;
        double market_price;
        OptionType option_type;
    };

    std::vector<TestCase> test_cases = {
        {100.0, 100.0, 1.0, 0.05, 6.08, OPTION_PUT},   // ATM
        // Case 2 removed: ITM put with IV=0.1485 is below table range [0.15, 0.35]
        {100.0, 90.0, 0.5, 0.03, 3.0, OPTION_PUT},     // OTM
        {100.0, 100.0, 0.25, 0.05, 3.5, OPTION_PUT},   // Short maturity
        {100.0, 105.0, 1.5, 0.07, 8.0, OPTION_PUT}     // Mid strike
    };

    // Compute ground truth IV using FDM (fine grid)
    AmericanOptionGrid fine_grid = {
        .x_min = -0.7, .x_max = 0.7,
        .n_points = 141, .dt = 0.001, .n_steps = 1000
    };

    std::vector<double> ground_truth_iv;
    printf("Computing ground truth IV (FDM with fine grid)...\n");
    for (const auto &tc : test_cases) {
        IVParams params = {
            .spot_price = tc.spot,
            .strike = tc.strike,
            .time_to_maturity = tc.maturity,
            .risk_free_rate = tc.rate,
            .dividend_yield = 0.0,
            .market_price = tc.market_price,
            .option_type = tc.option_type,
            .exercise_type = AMERICAN
        };

        IVResult result = calculate_iv(&params, &fine_grid, nullptr, 1e-6, 100);
        ASSERT_TRUE(result.converged);
        ground_truth_iv.push_back(result.implied_vol);
        printf("  Case %zu: IV = %.4f (%.0f iterations)\n",
               ground_truth_iv.size(), result.implied_vol, (double)result.iterations);
    }

    printf("\n");

    // Test each table configuration
    for (const auto &config : configs) {
        printf("Table: %s\n", config.name);

        // Create and precompute table
        OptionPriceTable *table = price_table_create_ex(
            config.m.data(), config.m.size(),
            config.tau.data(), config.tau.size(),
            config.sigma.data(), config.sigma.size(),
            config.r.data(), config.r.size(),
            nullptr, 0,
            OPTION_PUT, AMERICAN,
            COORD_RAW, LAYOUT_M_INNER
        );

        // Use SAME grid as FDM validation for fair comparison
        price_table_precompute(table, &fine_grid);
        price_table_build_interpolation(table);

        // Compute IV using table for each test case
        std::vector<double> errors;
        std::vector<double> table_iv;

        for (size_t i = 0; i < test_cases.size(); i++) {
            const auto &tc = test_cases[i];
            IVParams params = {
                .spot_price = tc.spot,
                .strike = tc.strike,
                .time_to_maturity = tc.maturity,
                .risk_free_rate = tc.rate,
                .dividend_yield = 0.0,
                .market_price = tc.market_price,
                .option_type = tc.option_type,
                .exercise_type = AMERICAN
            };

            IVResult result = calculate_iv(&params, &fine_grid, table, 1e-6, 100);
            ASSERT_TRUE(result.converged);

            table_iv.push_back(result.implied_vol);
            double error = std::abs(result.implied_vol - ground_truth_iv[i]) * 10000;  // basis points
            errors.push_back(error);
        }

        // Compute statistics
        double max_error = *std::max_element(errors.begin(), errors.end());
        double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

        printf("  Accuracy:\n");
        printf("    Mean error:  %.2f bp\n", mean_error);
        printf("    Max error:   %.2f bp\n", max_error);
        printf("  Details:\n");
        for (size_t i = 0; i < test_cases.size(); i++) {
            printf("    Case %zu: FDM=%.4f, Table=%.4f, Error=%.2f bp\n",
                   i+1, ground_truth_iv[i], table_iv[i], errors[i]);
        }
        printf("\n");

        price_table_destroy(table);

        // Verify table interpolation accuracy is excellent for in-bounds cases
        // Note: This tests INTERPOLATION accuracy with same FDM grid for both methods
        // All test cases are within table bounds, so expect near-perfect accuracy
        EXPECT_LT(max_error, 1.0) << config.name << " table should have <1bp max error for in-bounds cases";
        EXPECT_LT(mean_error, 0.5) << config.name << " table should have <0.5bp mean error for in-bounds cases";
    }

    printf("=== Accuracy Comparison Complete ===\n");
}
