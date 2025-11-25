/**
 * @file american_option_accuracy_test.cc
 * @brief Accuracy validation tests for American option pricing
 *
 * Tests against known reference values and validates:
 * - ATM/ITM/OTM options
 * - Various maturities
 * - Different volatility regimes
 * - Greeks accuracy
 */

#include <gtest/gtest.h>
#include "kokkos/src/option/american_option.hpp"
#include <cmath>

namespace mango::kokkos::test {

// Global Kokkos environment
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class AmericanOptionAccuracyTest : public ::testing::Test {};

// Helper to test scenarios against expected ranges
void test_pricing_scenario(
    const std::string& name,
    double spot, double strike, double maturity,
    double volatility, double rate, double dividend_yield,
    OptionType type,
    double expected_min, double expected_max,
    double delta_min = -2.0, double delta_max = 2.0)
{
    SCOPED_TRACE(name);

    PricingParams params{
        .strike = strike,
        .spot = spot,
        .maturity = maturity,
        .volatility = volatility,
        .rate = rate,
        .dividend_yield = dividend_yield,
        .type = type
    };

    AmericanOptionSolver<Kokkos::HostSpace> solver(params);
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value()) << "Solver failed for " << name;

    EXPECT_GT(result->price, expected_min)
        << name << ": price " << result->price << " below min " << expected_min;
    EXPECT_LT(result->price, expected_max)
        << name << ": price " << result->price << " above max " << expected_max;

    if (delta_min > -2.0 || delta_max < 2.0) {
        EXPECT_GT(result->delta, delta_min)
            << name << ": delta " << result->delta << " below min " << delta_min;
        EXPECT_LT(result->delta, delta_max)
            << name << ": delta " << result->delta << " above max " << delta_max;
    }
}

// ============================================================================
// Core Accuracy Tests - Compare to known approximate values
// ============================================================================

TEST_F(AmericanOptionAccuracyTest, ATM_Put_1Y) {
    // ATM put K=S=100, T=1Y, σ=20%, r=5%, q=2%
    // Expected: ~6.5-7.0 based on standard pricing
    test_pricing_scenario("ATM Put 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02,
        OptionType::Put, 6.0, 8.0, -0.6, -0.3);
}

TEST_F(AmericanOptionAccuracyTest, OTM_Put_3M) {
    // OTM put S=110 > K=100, T=3M
    test_pricing_scenario("OTM Put 3M",
        110.0, 100.0, 0.25, 0.30, 0.05, 0.02,
        OptionType::Put, 1.0, 5.0, -0.4, -0.05);
}

TEST_F(AmericanOptionAccuracyTest, ITM_Put_2Y) {
    // ITM put S=90 < K=100, T=2Y
    test_pricing_scenario("ITM Put 2Y",
        90.0, 100.0, 2.0, 0.25, 0.05, 0.02,
        OptionType::Put, 12.0, 20.0, -0.8, -0.4);
}

TEST_F(AmericanOptionAccuracyTest, DeepITM_Put_6M) {
    // Deep ITM put S=80 << K=100
    // Should be close to intrinsic (20) with early exercise premium
    test_pricing_scenario("Deep ITM Put 6M",
        80.0, 100.0, 0.5, 0.25, 0.05, 0.02,
        OptionType::Put, 19.0, 25.0, -0.99, -0.7);
}

TEST_F(AmericanOptionAccuracyTest, ATM_Call_1Y) {
    // ATM call (no dividend: American ~ European)
    test_pricing_scenario("ATM Call 1Y",
        100.0, 100.0, 1.0, 0.20, 0.05, 0.0,
        OptionType::Call, 9.0, 12.0, 0.5, 0.8);
}

TEST_F(AmericanOptionAccuracyTest, HighVol_Put_1Y) {
    // High volatility increases option value
    test_pricing_scenario("High Vol Put 1Y",
        100.0, 100.0, 1.0, 0.50, 0.05, 0.02,
        OptionType::Put, 12.0, 22.0);
}

TEST_F(AmericanOptionAccuracyTest, LowVol_Put_1Y) {
    // Low volatility reduces option value
    test_pricing_scenario("Low Vol Put 1Y",
        100.0, 100.0, 1.0, 0.10, 0.05, 0.02,
        OptionType::Put, 2.0, 5.0);
}

TEST_F(AmericanOptionAccuracyTest, LongMaturity_Put_5Y) {
    // Long maturity increases time value
    test_pricing_scenario("Long Maturity Put 5Y",
        100.0, 100.0, 5.0, 0.20, 0.05, 0.02,
        OptionType::Put, 10.0, 18.0);
}

// ============================================================================
// Intrinsic Value Tests
// ============================================================================

TEST_F(AmericanOptionAccuracyTest, PutValueAboveIntrinsic) {
    // American put should always be >= intrinsic value
    std::vector<double> spots = {80.0, 90.0, 100.0, 110.0, 120.0};
    double strike = 100.0;

    for (double spot : spots) {
        PricingParams params{
            .strike = strike, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        double intrinsic = std::max(0.0, strike - spot);
        EXPECT_GE(result->price, intrinsic - 0.01)
            << "Put at spot=" << spot << " below intrinsic " << intrinsic;
    }
}

TEST_F(AmericanOptionAccuracyTest, CallValueAboveIntrinsic) {
    // American call should always be >= intrinsic value
    std::vector<double> spots = {80.0, 90.0, 100.0, 110.0, 120.0};
    double strike = 100.0;

    for (double spot : spots) {
        PricingParams params{
            .strike = strike, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Call
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        double intrinsic = std::max(0.0, spot - strike);
        EXPECT_GE(result->price, intrinsic - 0.01)
            << "Call at spot=" << spot << " below intrinsic " << intrinsic;
    }
}

// ============================================================================
// Monotonicity Tests
// ============================================================================

TEST_F(AmericanOptionAccuracyTest, PutDecreasingWithSpot) {
    // Put value should decrease as spot increases
    double prev_price = 1e10;
    for (double spot = 80.0; spot <= 120.0; spot += 10.0) {
        PricingParams params{
            .strike = 100.0, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_LT(result->price, prev_price)
            << "Put price not decreasing at spot=" << spot;
        prev_price = result->price;
    }
}

TEST_F(AmericanOptionAccuracyTest, CallIncreasingWithSpot) {
    // Call value should increase as spot increases
    double prev_price = 0.0;
    for (double spot = 80.0; spot <= 120.0; spot += 10.0) {
        PricingParams params{
            .strike = 100.0, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Call
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_GT(result->price, prev_price)
            << "Call price not increasing at spot=" << spot;
        prev_price = result->price;
    }
}

TEST_F(AmericanOptionAccuracyTest, OptionIncreasingWithVolatility) {
    // Option value should increase with volatility
    double prev_price = 0.0;
    for (double vol = 0.10; vol <= 0.50; vol += 0.10) {
        PricingParams params{
            .strike = 100.0, .spot = 100.0, .maturity = 1.0,
            .volatility = vol, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_GT(result->price, prev_price)
            << "Put price not increasing at vol=" << vol;
        prev_price = result->price;
    }
}

TEST_F(AmericanOptionAccuracyTest, OptionIncreasingWithMaturity) {
    // Option value should generally increase with maturity
    double prev_price = 0.0;
    for (double T = 0.25; T <= 2.0; T += 0.25) {
        PricingParams params{
            .strike = 100.0, .spot = 100.0, .maturity = T,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_GE(result->price, prev_price - 0.1)
            << "Put price decreasing significantly at T=" << T;
        prev_price = result->price;
    }
}

// ============================================================================
// Delta Tests
// ============================================================================

TEST_F(AmericanOptionAccuracyTest, PutDeltaNegative) {
    // Put delta should always be negative
    std::vector<double> spots = {80.0, 90.0, 100.0, 110.0, 120.0};

    for (double spot : spots) {
        PricingParams params{
            .strike = 100.0, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_LT(result->delta, 0.0)
            << "Put delta positive at spot=" << spot;
        EXPECT_GT(result->delta, -1.0)
            << "Put delta below -1 at spot=" << spot;
    }
}

TEST_F(AmericanOptionAccuracyTest, CallDeltaPositive) {
    // Call delta should always be positive
    std::vector<double> spots = {80.0, 90.0, 100.0, 110.0, 120.0};

    for (double spot : spots) {
        PricingParams params{
            .strike = 100.0, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Call
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        EXPECT_GT(result->delta, 0.0)
            << "Call delta negative at spot=" << spot;
        EXPECT_LT(result->delta, 1.0)
            << "Call delta above 1 at spot=" << spot;
    }
}

TEST_F(AmericanOptionAccuracyTest, DeltaMonotonicity) {
    // Delta magnitude should increase as option becomes more ITM
    double prev_delta = 0.0;
    for (double spot = 120.0; spot >= 80.0; spot -= 10.0) {
        PricingParams params{
            .strike = 100.0, .spot = spot, .maturity = 1.0,
            .volatility = 0.20, .rate = 0.05, .dividend_yield = 0.02,
            .type = OptionType::Put
        };

        AmericanOptionSolver<Kokkos::HostSpace> solver(params);
        auto result = solver.solve();
        ASSERT_TRUE(result.has_value());

        // As spot decreases, put becomes more ITM, delta more negative
        EXPECT_LT(result->delta, prev_delta + 0.01)
            << "Put delta not becoming more negative at spot=" << spot;
        prev_delta = result->delta;
    }
}

}  // namespace mango::kokkos::test
