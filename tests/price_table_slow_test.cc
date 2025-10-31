#include <gtest/gtest.h>
#include "../src/price_table.h"
#include "../src/american_option.h"
#include <cmath>

class PriceTablePrecomputeSlowTest : public ::testing::Test {
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

TEST_F(PriceTablePrecomputeSlowTest, InterpolationAccuracyIntegration) {
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
