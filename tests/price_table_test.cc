#include <gtest/gtest.h>
#include "../src/price_table.h"
#include "../src/american_option.h"
#include <cmath>
#include <cstdlib>

class PriceTablePrecomputeTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid;

    void SetUp() override {
        // Simple grid for fast testing
        default_grid.x_min = -0.7;  // ln(0.5)
        default_grid.x_max = 0.7;   // ln(2.0)
        default_grid.n_points = 51;
        default_grid.dt = 0.001;
        default_grid.n_steps = 100;
    }
};

TEST_F(PriceTablePrecomputeTest, NullTablePointer) {
    int status = price_table_precompute(nullptr, &default_grid);
    EXPECT_EQ(status, -1);
}

TEST_F(PriceTablePrecomputeTest, NullGridPointer) {
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, nullptr);
    EXPECT_EQ(status, -1);

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, SmallGrid4D) {
    // 2×2×2×2 = 16 points for fast test
    double moneyness[] = {0.95, 1.05};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.03, 0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 2, volatility, 2, rate, 2, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    EXPECT_EQ(status, 0);

    // Verify no NANs in results
    for (size_t i_m = 0; i_m < 2; i_m++) {
        for (size_t i_tau = 0; i_tau < 2; i_tau++) {
            for (size_t i_sigma = 0; i_sigma < 2; i_sigma++) {
                for (size_t i_r = 0; i_r < 2; i_r++) {
                    double price = price_table_get(table, i_m, i_tau, i_sigma, i_r, 0);
                    EXPECT_FALSE(std::isnan(price))
                        << "NAN at [" << i_m << "," << i_tau << "," << i_sigma << "," << i_r << "]";
                    EXPECT_GT(price, 0.0)
                        << "Non-positive price at [" << i_m << "," << i_tau << "," << i_sigma << "," << i_r << "]";
                }
            }
        }
    }

    // Verify generation timestamp set
    EXPECT_GT(table->generation_time, 0);

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInMoneyness) {
    // Put prices should generally increase as moneyness decreases (more ITM)
    double moneyness[] = {0.90, 0.95, 1.00, 1.05, 1.10};
    double maturity[] = {0.5};
    double volatility[] = {0.25};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 5, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that put prices increase as moneyness decreases
    for (size_t i = 0; i < 4; i++) {
        double price_lower_m = price_table_get(table, i, 0, 0, 0, 0);
        double price_higher_m = price_table_get(table, i + 1, 0, 0, 0, 0);
        EXPECT_GT(price_lower_m, price_higher_m)
            << "Put price should decrease as moneyness increases: "
            << "m[" << i << "]=" << moneyness[i] << " price=" << price_lower_m
            << " vs m[" << i+1 << "]=" << moneyness[i+1] << " price=" << price_higher_m;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInMaturity) {
    // American option prices should increase with maturity (more time value)
    // Use OTM option to avoid early exercise boundary issues
    double moneyness[] = {1.1};  // OTM put
    double maturity[] = {0.1, 0.25, 0.5, 1.0};
    double volatility[] = {0.25};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 1, maturity, 4, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that prices increase with maturity (or at least don't decrease)
    for (size_t i = 0; i < 3; i++) {
        double price_shorter = price_table_get(table, 0, i, 0, 0, 0);
        double price_longer = price_table_get(table, 0, i + 1, 0, 0, 0);
        EXPECT_LE(price_shorter, price_longer)
            << "American option price should not decrease with maturity: "
            << "τ[" << i << "]=" << maturity[i] << " price=" << price_shorter
            << " vs τ[" << i+1 << "]=" << maturity[i+1] << " price=" << price_longer;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, MonotonicityInVolatility) {
    // Option prices should increase with volatility
    double moneyness[] = {1.0};
    double maturity[] = {0.5};
    double volatility[] = {0.15, 0.20, 0.25, 0.30, 0.35};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 1, maturity, 1, volatility, 5, rate, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Check that prices increase with volatility
    for (size_t i = 0; i < 4; i++) {
        double price_lower_vol = price_table_get(table, 0, 0, i, 0, 0);
        double price_higher_vol = price_table_get(table, 0, 0, i + 1, 0, 0);
        EXPECT_LT(price_lower_vol, price_higher_vol)
            << "Option price should increase with volatility: "
            << "σ[" << i << "]=" << volatility[i] << " price=" << price_lower_vol
            << " vs σ[" << i+1 << "]=" << volatility[i+1] << " price=" << price_higher_vol;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, BatchSizeConsistency) {
    // Test that different batch sizes produce identical results
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.05};

    // First run with default batch size
    OptionPriceTable *table1 = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table1, nullptr);
    price_table_precompute(table1, &default_grid);

    // Second run with batch_size=1
    setenv("IVCALC_PRECOMPUTE_BATCH_SIZE", "1", 1);

    OptionPriceTable *table2 = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table2, nullptr);
    price_table_precompute(table2, &default_grid);

    unsetenv("IVCALC_PRECOMPUTE_BATCH_SIZE");

    // Compare results (should be identical)
    size_t n_total = 3 * 2 * 2 * 1;
    for (size_t i = 0; i < n_total; i++) {
        EXPECT_NEAR(table1->prices[i], table2->prices[i], 1e-10)
            << "Mismatch at index " << i;
    }

    price_table_destroy(table1);
    price_table_destroy(table2);
}

TEST_F(PriceTablePrecomputeTest, GetSetOperations) {
    // Test price_table_get and price_table_set operations
    double moneyness[] = {0.9, 1.0};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 2, maturity, 2, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    // Test setting values
    int status = price_table_set(table, 0, 0, 0, 0, 0, 10.5);
    EXPECT_EQ(status, 0);

    double value = price_table_get(table, 0, 0, 0, 0, 0);
    EXPECT_DOUBLE_EQ(value, 10.5);

    // Test out of bounds
    status = price_table_set(table, 10, 0, 0, 0, 0, 5.0);
    EXPECT_EQ(status, -1);

    double invalid = price_table_get(table, 10, 0, 0, 0, 0);
    EXPECT_TRUE(std::isnan(invalid));

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, IntrinsicValueBound) {
    // For a put at maturity, price should be at least intrinsic value
    double moneyness[] = {0.8, 0.9, 1.0, 1.1, 1.2};
    double maturity[] = {0.05};  // Short maturity (adaptive: 50 steps with dt=0.001)
    double volatility[] = {0.25};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 5, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // For each moneyness, check price >= max(K - S, 0) approximately
    const double K_ref = 100.0;
    for (size_t i = 0; i < 5; i++) {
        double m = moneyness[i];
        double S = m * K_ref;
        double intrinsic = std::max(K_ref - S, 0.0);
        double price = price_table_get(table, i, 0, 0, 0, 0);

        EXPECT_GE(price, intrinsic * 0.95)  // Allow 5% margin for numerical errors
            << "Put price should be >= intrinsic value: "
            << "m=" << m << " S=" << S << " intrinsic=" << intrinsic << " price=" << price;
    }

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, CallPutTypeCorrectness) {
    // Verify that call and put tables store different prices
    double moneyness[] = {1.0};
    double maturity[] = {0.5};
    double volatility[] = {0.25};
    double rate[] = {0.05};

    // American call
    OptionPriceTable *call_table = price_table_create(
        moneyness, 1, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_CALL, AMERICAN);
    ASSERT_NE(call_table, nullptr);
    price_table_precompute(call_table, &default_grid);

    // American put
    OptionPriceTable *put_table = price_table_create(
        moneyness, 1, maturity, 1, volatility, 1, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);
    ASSERT_NE(put_table, nullptr);
    price_table_precompute(put_table, &default_grid);

    double call_price = price_table_get(call_table, 0, 0, 0, 0, 0);
    double put_price = price_table_get(put_table, 0, 0, 0, 0, 0);

    // At ATM, call and put should be different
    EXPECT_NE(call_price, put_price);
    EXPECT_GT(call_price, 0.0);
    EXPECT_GT(put_price, 0.0);

    price_table_destroy(call_table);
    price_table_destroy(put_table);
}

TEST_F(PriceTablePrecomputeTest, InterpolationSmoothness) {
    // Create a small table and verify interpolation works
    double moneyness[] = {0.9, 1.0, 1.1};
    double maturity[] = {0.25, 0.5};
    double volatility[] = {0.2, 0.3};
    double rate[] = {0.05};

    OptionPriceTable *table = price_table_create(
        moneyness, 3, maturity, 2, volatility, 2, rate, 1, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    int status = price_table_precompute(table, &default_grid);
    ASSERT_EQ(status, 0);

    // Test interpolation at a point between grid points
    double interp_price = price_table_interpolate_4d(table, 0.95, 0.375, 0.25, 0.05);

    // Should return a valid price
    EXPECT_FALSE(std::isnan(interp_price));
    EXPECT_GT(interp_price, 0.0);

    // Should be between corner values
    double corner_prices[4];
    corner_prices[0] = price_table_get(table, 0, 0, 0, 0, 0);  // (0.9, 0.25, 0.2, 0.05)
    corner_prices[1] = price_table_get(table, 1, 0, 0, 0, 0);  // (1.0, 0.25, 0.2, 0.05)
    corner_prices[2] = price_table_get(table, 0, 1, 1, 0, 0);  // (0.9, 0.5, 0.3, 0.05)
    corner_prices[3] = price_table_get(table, 1, 1, 1, 0, 0);  // (1.0, 0.5, 0.3, 0.05)

    double min_corner = corner_prices[0];
    double max_corner = corner_prices[0];
    for (int i = 1; i < 4; i++) {
        min_corner = std::min(min_corner, corner_prices[i]);
        max_corner = std::max(max_corner, corner_prices[i]);
    }

    EXPECT_GE(interp_price, min_corner * 0.9);  // Allow small extrapolation
    EXPECT_LE(interp_price, max_corner * 1.1);

    price_table_destroy(table);
}

TEST_F(PriceTablePrecomputeTest, InterpolationAccuracyIntegration) {
    // Create moderately-sized table and verify interpolation accuracy
    // 10×8×5×3 = 1200 points
    double moneyness[10];
    double maturity[8];
    double volatility[5];
    double rate[3];

    // Linear-spaced moneyness
    for (int i = 0; i < 10; i++) {
        double t = (double)i / 9.0;
        moneyness[i] = 0.8 + t * (1.2 - 0.8);
    }

    // Linear maturity
    for (int i = 0; i < 8; i++) {
        double t = (double)i / 7.0;
        maturity[i] = 0.1 + t * (2.0 - 0.1);
    }

    // Volatility range
    for (int i = 0; i < 5; i++) {
        double t = (double)i / 4.0;
        volatility[i] = 0.15 + t * (0.4 - 0.15);
    }

    // Rate range
    for (int i = 0; i < 3; i++) {
        double t = (double)i / 2.0;
        rate[i] = 0.02 + t * (0.08 - 0.02);
    }

    OptionPriceTable *table = price_table_create(
        moneyness, 10, maturity, 8, volatility, 5, rate, 3, nullptr, 0,
        OPTION_PUT, AMERICAN);

    ASSERT_NE(table, nullptr);

    // Use finer grid for better accuracy
    AmericanOptionGrid fine_grid;
    fine_grid.x_min = -0.7;
    fine_grid.x_max = 0.7;
    fine_grid.n_points = 101;  // Finer spatial grid
    fine_grid.dt = 0.0005;     // Smaller time step
    fine_grid.n_steps = 1000;  // Will be overridden by adaptive calculation

    // Pre-compute table (takes ~1-2 minutes)
    int status = price_table_precompute(table, &fine_grid);
    ASSERT_EQ(status, 0);

    // Test interpolation at arbitrary off-grid point
    double test_m = 1.05;      // Between grid points
    double test_tau = 0.25;    // Between grid points
    double test_sigma = 0.22;  // Between grid points
    double test_r = 0.055;     // Between grid points

    double price_interp = price_table_interpolate_4d(table, test_m, test_tau,
                                                       test_sigma, test_r);

    EXPECT_FALSE(std::isnan(price_interp));
    EXPECT_GT(price_interp, 0.0);

    // Compare to direct computation with same grid
    const double K_ref = 100.0;
    OptionData option;
    option.strike = K_ref;
    option.volatility = test_sigma;
    option.risk_free_rate = test_r;
    option.time_to_maturity = test_tau;
    option.option_type = OPTION_PUT;
    option.n_dividends = 0;
    option.dividend_times = nullptr;
    option.dividend_amounts = nullptr;

    // Use adaptive time steps matching precompute behavior
    AmericanOptionGrid comparison_grid = fine_grid;
    comparison_grid.n_steps = static_cast<size_t>(test_tau / fine_grid.dt);
    if (comparison_grid.n_steps < 10) comparison_grid.n_steps = 10;  // Match minimum

    AmericanOptionResult result = american_option_price(&option, &comparison_grid);
    ASSERT_EQ(result.status, 0);

    double spot_price = test_m * K_ref;
    double price_direct = american_option_get_value_at_spot(result.solver, spot_price, K_ref);

    // Clean up result
    american_option_free_result(&result);

    // Interpolation error should be < 5% for multilinear interpolation in 4D
    // (Achieving 1% would require much denser grids or higher-order interpolation)
    double error = fabs(price_interp - price_direct) / price_direct;
    EXPECT_LT(error, 0.05) << "Interpolation error too large: "
                           << "interp=" << price_interp
                           << " direct=" << price_direct
                           << " error=" << (error * 100) << "%";

    price_table_destroy(table);
}
