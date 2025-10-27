#include <gtest/gtest.h>
#include <cmath>
#include <vector>

extern "C" {
#include "../src/american_option.h"
#include "../src/implied_volatility.h"
}

// Test fixture for American option tests
class AmericanOptionTest : public ::testing::Test {
protected:
    AmericanOptionGrid default_grid = {
        .x_min = -0.7,
        .x_max = 0.7,
        .n_points = 101,
        .dt = 0.001,
        .n_steps = 500
    };

    void TearDown() override {
        // Cleanup if needed
    }
};

// Test basic call option pricing
TEST_F(AmericanOptionTest, BasicCallOption) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);
    ASSERT_NE(result.solver, nullptr);

    // Get ATM value
    double value = american_option_get_value_at_spot(result.solver, 100.0, option.strike);

    // American call without dividends should equal European call
    double european = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, true);

    EXPECT_NEAR(value, european, 0.1);  // Within 10 cents
    EXPECT_GT(value, 0.0);

    pde_solver_destroy(result.solver);
}

// Test basic put option pricing
TEST_F(AmericanOptionTest, BasicPutOption) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);
    ASSERT_NE(result.solver, nullptr);

    double value = american_option_get_value_at_spot(result.solver, 100.0, option.strike);

    // American put should be more valuable than European put (early exercise premium)
    double european = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, false);

    EXPECT_GT(value, european - 0.01);  // At least as valuable as European
    EXPECT_GT(value, 0.0);

    pde_solver_destroy(result.solver);
}

// Test put-call relationship (American options don't satisfy exact parity)
TEST_F(AmericanOptionTest, PutCallRelationship) {
    OptionData call_option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.5,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    OptionData put_option = call_option;
    put_option.option_type = OPTION_PUT;

    AmericanOptionResult call_result = american_option_price(&call_option, &default_grid);
    AmericanOptionResult put_result = american_option_price(&put_option, &default_grid);

    double call_value = american_option_get_value_at_spot(call_result.solver, 100.0, 100.0);
    double put_value = american_option_get_value_at_spot(put_result.solver, 100.0, 100.0);

    // For American options: C - P <= S - K*e^(-rT) <= C - P + PV(dividends)
    double parity_bound = 100.0 - 100.0 * std::exp(-0.05 * 0.5);
    double diff = call_value - put_value;

    EXPECT_LE(diff, parity_bound + 0.1);
    EXPECT_GT(diff, 0.0);  // Call should be worth more than put for ATM with positive rate

    pde_solver_destroy(call_result.solver);
    pde_solver_destroy(put_result.solver);
}

// Test early exercise premium for puts
TEST_F(AmericanOptionTest, EarlyExercisePremium) {
    OptionData put_option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&put_option, &default_grid);

    double american_value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    double european_value = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, false);

    double early_exercise_premium = american_value - european_value;

    // American put should have positive early exercise premium
    EXPECT_GT(early_exercise_premium, 0.0);
    EXPECT_LT(early_exercise_premium, 5.0);  // But not excessive

    pde_solver_destroy(result.solver);
}

// Test intrinsic value bound
TEST_F(AmericanOptionTest, IntrinsicValueBound) {
    OptionData put_option = {
        .strike = 100.0,
        .volatility = 0.3,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.5,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&put_option, &default_grid);

    // Test at various spot prices
    std::vector<double> spot_prices = {80.0, 90.0, 100.0, 110.0, 120.0};

    for (double spot : spot_prices) {
        double value = american_option_get_value_at_spot(result.solver, spot, 100.0);
        double intrinsic = std::max(100.0 - spot, 0.0);

        // American option value must be at least intrinsic value
        EXPECT_GE(value, intrinsic - 0.01);
    }

    pde_solver_destroy(result.solver);
}

// Test monotonicity in volatility
TEST_F(AmericanOptionTest, MonotonicityInVolatility) {
    std::vector<double> volatilities = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> values;

    for (double vol : volatilities) {
        OptionData option = {
            .strike = 100.0,
            .volatility = vol,
            .risk_free_rate = 0.05,
            .time_to_maturity = 1.0,
            .option_type = OPTION_CALL,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };

        AmericanOptionResult result = american_option_price(&option, &default_grid);
        double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
        values.push_back(value);

        pde_solver_destroy(result.solver);
    }

    // Values should increase with volatility
    for (size_t i = 1; i < values.size(); i++) {
        EXPECT_GT(values[i], values[i-1]);
    }
}

// Test monotonicity in time to maturity
TEST_F(AmericanOptionTest, MonotonicityInMaturity) {
    std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0};
    std::vector<double> values;

    for (double T : maturities) {
        OptionData option = {
            .strike = 100.0,
            .volatility = 0.25,
            .risk_free_rate = 0.05,
            .time_to_maturity = T,
            .option_type = OPTION_PUT,
            .n_dividends = 0,
            .dividend_times = nullptr,
            .dividend_amounts = nullptr
        };

        AmericanOptionGrid grid = default_grid;
        grid.n_steps = static_cast<size_t>(T * 1000);  // Scale steps with maturity

        AmericanOptionResult result = american_option_price(&option, &grid);
        double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
        values.push_back(value);

        pde_solver_destroy(result.solver);
    }

    // American option values should increase with maturity
    for (size_t i = 1; i < values.size(); i++) {
        EXPECT_GT(values[i], values[i-1]);
    }
}

// Test OTM call option
TEST_F(AmericanOptionTest, OTMCallOption) {
    OptionData option = {
        .strike = 110.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 110.0);

    // OTM call should have positive time value
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 20.0);  // But not excessive

    // Should be close to European value
    double european = black_scholes_price(100.0, 110.0, 1.0, 0.05, 0.2, true);
    EXPECT_NEAR(value, european, 0.2);

    pde_solver_destroy(result.solver);
}

// Test ITM put option
TEST_F(AmericanOptionTest, ITMPutOption) {
    OptionData option = {
        .strike = 110.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.5,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 110.0);
    double intrinsic = 110.0 - 100.0;

    // ITM put should be worth at least intrinsic
    EXPECT_GE(value, intrinsic);

    // But also have some time value
    EXPECT_GT(value, intrinsic);

    pde_solver_destroy(result.solver);
}

// Test deep OTM option
TEST_F(AmericanOptionTest, DeepOTMOption) {
    OptionData option = {
        .strike = 150.0,
        .volatility = 0.3,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.25,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 150.0);

    // Deep OTM should have small but positive value
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 1.0);

    pde_solver_destroy(result.solver);
}

// Test deep ITM option
TEST_F(AmericanOptionTest, DeepITMOption) {
    OptionData option = {
        .strike = 150.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 150.0);
    double intrinsic = 150.0 - 100.0;

    // Deep ITM put might be close to intrinsic (optimal to exercise early)
    EXPECT_NEAR(value, intrinsic, 5.0);
    EXPECT_GE(value, intrinsic - 0.01);

    pde_solver_destroy(result.solver);
}

// Test short maturity
TEST_F(AmericanOptionTest, ShortMaturity) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.05,  // ~18 days
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = default_grid;
    grid.n_steps = 100;  // Fewer steps for short maturity

    AmericanOptionResult result = american_option_price(&option, &grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 10.0);

    pde_solver_destroy(result.solver);
}

// Test long maturity
TEST_F(AmericanOptionTest, LongMaturity) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.05,
        .time_to_maturity = 3.0,  // 3 years
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = default_grid;
    grid.n_steps = 2000;  // More steps for long maturity

    AmericanOptionResult result = american_option_price(&option, &grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    EXPECT_GT(value, 10.0);
    EXPECT_LT(value, 100.0);

    pde_solver_destroy(result.solver);
}

// Test high volatility
TEST_F(AmericanOptionTest, HighVolatility) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.8,  // 80% volatility
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);

    // High vol means high option value
    EXPECT_GT(value, 20.0);

    pde_solver_destroy(result.solver);
}

// Test low volatility
TEST_F(AmericanOptionTest, LowVolatility) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.05,  // 5% volatility
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);

    // Low vol means low option value
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 5.0);

    pde_solver_destroy(result.solver);
}

// Test zero interest rate
TEST_F(AmericanOptionTest, ZeroInterestRate) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = 0.0,
        .time_to_maturity = 1.0,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    EXPECT_GT(value, 0.0);

    pde_solver_destroy(result.solver);
}

// Test negative interest rate
TEST_F(AmericanOptionTest, NegativeInterestRate) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.2,
        .risk_free_rate = -0.02,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    EXPECT_GT(value, 0.0);

    pde_solver_destroy(result.solver);
}

// Test grid resolution impact
TEST_F(AmericanOptionTest, GridResolution) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Coarse grid
    AmericanOptionGrid coarse_grid = {-0.7, 0.7, 51, 0.002, 500};
    AmericanOptionResult coarse_result = american_option_price(&option, &coarse_grid);
    double coarse_value = american_option_get_value_at_spot(coarse_result.solver, 100.0, 100.0);

    // Fine grid
    AmericanOptionGrid fine_grid = {-0.7, 0.7, 201, 0.001, 1000};
    AmericanOptionResult fine_result = american_option_price(&option, &fine_grid);
    double fine_value = american_option_get_value_at_spot(fine_result.solver, 100.0, 100.0);

    // Values should be close (within 5%)
    EXPECT_NEAR(coarse_value / fine_value, 1.0, 0.05);

    pde_solver_destroy(coarse_result.solver);
    pde_solver_destroy(fine_result.solver);
}

// Stress test: Very short maturity
TEST_F(AmericanOptionTest, VeryShortMaturity) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.3,
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.01,  // ~4 days
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = default_grid;
    grid.n_steps = 50;

    AmericanOptionResult result = american_option_price(&option, &grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);

    // Should converge to intrinsic value
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 5.0);

    pde_solver_destroy(result.solver);
}

// Stress test: Extreme volatility
TEST_F(AmericanOptionTest, ExtremeVolatility) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 2.0,  // 200% volatility
        .risk_free_rate = 0.05,
        .time_to_maturity = 0.5,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionGrid grid = default_grid;
    grid.x_min = -2.0;
    grid.x_max = 2.0;

    AmericanOptionResult result = american_option_price(&option, &grid);

    EXPECT_EQ(result.status, 0);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);

    // Should still produce reasonable value
    EXPECT_GT(value, 0.0);
    EXPECT_LT(value, 100.0);
    EXPECT_FALSE(std::isnan(value));
    EXPECT_FALSE(std::isinf(value));

    pde_solver_destroy(result.solver);
}

// Stress test: Extreme moneyness
TEST_F(AmericanOptionTest, ExtremeMoneyness) {
    OptionData option = {
        .strike = 50.0,  // Deep ITM call (spot = 100)
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_CALL,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    AmericanOptionResult result = american_option_price(&option, &default_grid);

    double value = american_option_get_value_at_spot(result.solver, 100.0, 50.0);
    double intrinsic = 100.0 - 50.0;

    // Deep ITM call should be close to intrinsic (or spot - PV(strike))
    EXPECT_NEAR(value, intrinsic, 5.0);
    EXPECT_GE(value, intrinsic - 0.1);

    pde_solver_destroy(result.solver);
}

// Test consistency: Multiple runs should give same result
TEST_F(AmericanOptionTest, ConsistencyTest) {
    OptionData option = {
        .strike = 100.0,
        .volatility = 0.25,
        .risk_free_rate = 0.05,
        .time_to_maturity = 1.0,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    std::vector<double> values;

    for (int i = 0; i < 3; i++) {
        AmericanOptionResult result = american_option_price(&option, &default_grid);
        double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
        values.push_back(value);
        pde_solver_destroy(result.solver);
    }

    // All values should be identical
    for (size_t i = 1; i < values.size(); i++) {
        EXPECT_DOUBLE_EQ(values[0], values[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
