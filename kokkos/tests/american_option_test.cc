#include <gtest/gtest.h>
#include "kokkos/src/option/american_option.hpp"
#include <cmath>

namespace mango::kokkos::test {

// Global setup/teardown for Kokkos - once per test program
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the global environment
[[maybe_unused]] static ::testing::Environment* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

class AmericanOptionTest : public ::testing::Test {
    // No per-test setup/teardown needed for Kokkos
};

TEST_F(AmericanOptionTest, ATMPutPrice) {
    // ATM American put, compare to known value
    PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    // Debug: print grid parameters
    auto [grid_params, time_params] = estimate_grid_for_option(params);
    std::cout << "Grid: x=[" << grid_params.x_min << ", " << grid_params.x_max
              << "], n=" << grid_params.n_points
              << ", alpha=" << grid_params.alpha << "\n";
    std::cout << "Time: T=" << time_params.T << ", steps=" << time_params.n_steps << "\n";

    AmericanOptionSolver<HostMemSpace> solver(params);
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    std::cout << "Price: " << result->price << " (expected: 6.5-7.0)\n";
    // Expected price around 6.5-7.0 for these parameters
    EXPECT_GT(result->price, 6.0);
    EXPECT_LT(result->price, 8.0);
}

TEST_F(AmericanOptionTest, DeepITMPutIntrinsicValue) {
    // Deep ITM put should be worth at least intrinsic
    PricingParams params{
        .strike = 100.0,
        .spot = 80.0,  // Deep ITM put
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.0,
        .type = OptionType::Put
    };

    AmericanOptionSolver<HostMemSpace> solver(params);
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    double intrinsic = params.strike - params.spot;  // 20
    EXPECT_GE(result->price, intrinsic - 0.5);  // Allow small tolerance
}

TEST_F(AmericanOptionTest, CallWithNoDividend) {
    // American call with no dividend ~ European call
    // Should not exercise early (no dividend benefit)
    PricingParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.0,
        .type = OptionType::Call
    };

    AmericanOptionSolver<HostMemSpace> solver(params);
    auto result = solver.solve();

    ASSERT_TRUE(result.has_value());
    // Should be close to European Black-Scholes
    // BS ATM call with r=0.05, sigma=0.20, T=1.0 ~ 10.45
    EXPECT_NEAR(result->price, 10.45, 1.0);
}

TEST_F(AmericanOptionTest, PutPriceDecreaseWithSpot) {
    // Put value should decrease as spot increases
    PricingParams params1{
        .strike = 100.0,
        .spot = 90.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    PricingParams params2 = params1;
    params2.spot = 110.0;

    AmericanOptionSolver<HostMemSpace> solver1(params1);
    AmericanOptionSolver<HostMemSpace> solver2(params2);

    auto result1 = solver1.solve();
    auto result2 = solver2.solve();

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // ITM put (spot=90) should be worth more than OTM put (spot=110)
    EXPECT_GT(result1->price, result2->price);
}

TEST_F(AmericanOptionTest, DeltaSign) {
    // Put delta should be negative
    // Call delta should be positive
    PricingParams put_params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .dividend_yield = 0.02,
        .type = OptionType::Put
    };

    PricingParams call_params = put_params;
    call_params.type = OptionType::Call;

    AmericanOptionSolver<HostMemSpace> put_solver(put_params);
    AmericanOptionSolver<HostMemSpace> call_solver(call_params);

    auto put_result = put_solver.solve();
    auto call_result = call_solver.solve();

    ASSERT_TRUE(put_result.has_value());
    ASSERT_TRUE(call_result.has_value());

    EXPECT_LT(put_result->delta, 0.0);  // Put delta < 0
    EXPECT_GT(call_result->delta, 0.0); // Call delta > 0
}

}  // namespace mango::kokkos::test
