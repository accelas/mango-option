// SPDX-License-Identifier: MIT
/**
 * @file american_option_result_test.cc
 * @brief Tests for AmericanOptionResult wrapper class
 */

#include "src/option/american_option_result.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;

namespace {

// Test fixture with common setup
class AmericanOptionResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple uniform grid in log-moneyness space
        auto grid_spec_result = GridSpec<double>::uniform(-1.0, 1.0, 21);
        ASSERT_TRUE(grid_spec_result.has_value());
        auto grid_spec = grid_spec_result.value();

        auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);

        // Create grid with solution storage
        auto grid_result = Grid<double>::create(grid_spec, time_domain);
        ASSERT_TRUE(grid_result.has_value());
        grid = grid_result.value();

        // Setup pricing params (ATM put)
        params = PricingParams(OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0, .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

        // Fill grid with known payoff: max(K - S, 0) for put
        // In log-moneyness: x = ln(S/K), so S = K * exp(x)
        // Store normalized by K: V/K
        auto x_span = grid->x();
        auto solution = grid->solution();
        for (size_t i = 0; i < x_span.size(); ++i) {
            double S = params.strike * std::exp(x_span[i]);
            double payoff = std::max(params.strike - S, 0.0);
            solution[i] = payoff / params.strike;  // Normalize by K
        }
    }

    std::shared_ptr<Grid<double>> grid;
    PricingParams params;
};

// Test 1: Construction and basic accessors
TEST_F(AmericanOptionResultTest, ConstructionAndAccessors) {
    AmericanOptionResult result(grid, params);

    // Should have access to pricing params
    EXPECT_DOUBLE_EQ(result.spot(), 100.0);
    EXPECT_DOUBLE_EQ(result.strike(), 100.0);
    EXPECT_DOUBLE_EQ(result.maturity(), 1.0);
    EXPECT_EQ(result.option_type(), OptionType::PUT);
}

// Test 2: value() returns value at current spot
TEST_F(AmericanOptionResultTest, ValueAtCurrentSpot) {
    AmericanOptionResult result(grid, params);

    // For ATM put with x = ln(S/K), S = K means x = 0
    // Should interpolate to the value at x = 0
    double value = result.value();

    // Should be non-negative (put option)
    EXPECT_GE(value, 0.0);

    // For ATM, should be somewhere between 0 and strike
    EXPECT_LE(value, params.strike);
}

// Test 3: value_at() with known payoff
TEST_F(AmericanOptionResultTest, ValueAtInterpolation) {
    AmericanOptionResult result(grid, params);

    // Test at-the-money (S = K = 100)
    double value_atm = result.value_at(100.0);
    EXPECT_NEAR(value_atm, 0.0, 1e-10);  // ATM put has zero intrinsic value

    // Test in-the-money (S = 90, payoff = 10)
    double value_itm = result.value_at(90.0);
    EXPECT_NEAR(value_itm, 10.0, 0.5);  // Should be close to intrinsic value

    // Test out-of-the-money (S = 110, payoff = 0)
    double value_otm = result.value_at(110.0);
    EXPECT_NEAR(value_otm, 0.0, 1e-10);
}

// Test 4: value() matches value_at(spot)
TEST_F(AmericanOptionResultTest, ValueConsistency) {
    AmericanOptionResult result(grid, params);

    double value1 = result.value();
    double value2 = result.value_at(params.spot);

    EXPECT_DOUBLE_EQ(value1, value2);
}

// Test 5: Delta computation (sign and range)
TEST_F(AmericanOptionResultTest, DeltaComputation) {
    AmericanOptionResult result(grid, params);

    double delta = result.delta();

    // Put delta should be negative or zero
    EXPECT_LE(delta, 0.0);

    // Put delta should be >= -1
    EXPECT_GE(delta, -1.0);
}

// Test 6: Gamma computation (positive for both call/put)
TEST_F(AmericanOptionResultTest, GammaComputation) {
    AmericanOptionResult result(grid, params);

    double gamma = result.gamma();

    // Gamma should be non-negative (convexity)
    EXPECT_GE(gamma, 0.0);

    // Gamma should have reasonable magnitude
    EXPECT_LE(gamma, 1.0);  // Arbitrary upper bound for sanity check
}

// Test 7: Snapshot delegation
TEST_F(AmericanOptionResultTest, SnapshotDelegation) {
    // Create grid with snapshots
    auto grid_spec_result = GridSpec<double>::uniform(-1.0, 1.0, 21);
    ASSERT_TRUE(grid_spec_result.has_value());
    auto grid_spec = grid_spec_result.value();

    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);

    std::vector<double> snapshot_times = {0.25, 0.5, 0.75};
    auto grid_with_snaps = Grid<double>::create(
        grid_spec, time_domain, snapshot_times).value();

    AmericanOptionResult result(grid_with_snaps, params);

    EXPECT_TRUE(result.has_snapshots());
    EXPECT_EQ(result.num_snapshots(), 3);

    auto times = result.snapshot_times();
    EXPECT_EQ(times.size(), 3);
}

// Test 8: Grid access for advanced users
TEST_F(AmericanOptionResultTest, GridAccess) {
    AmericanOptionResult result(grid, params);

    auto grid_ptr = result.grid();
    EXPECT_NE(grid_ptr, nullptr);
    EXPECT_EQ(grid_ptr->n_space(), 21);
}

// Test 9: Call option (for symmetry testing)
TEST_F(AmericanOptionResultTest, CallOptionGreeks) {
    // Change to call option
    params.option_type = OptionType::CALL;

    // Fill grid with call payoff: max(S - K, 0)
    // Store normalized by K: V/K
    auto x_span = grid->x();
    auto solution = grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        double S = params.strike * std::exp(x_span[i]);
        double payoff = std::max(S - params.strike, 0.0);
        solution[i] = payoff / params.strike;  // Normalize by K
    }

    AmericanOptionResult result(grid, params);

    double delta = result.delta();

    // Call delta should be positive or zero
    EXPECT_GE(delta, 0.0);

    // Call delta should be <= 1
    EXPECT_LE(delta, 1.0);
}

// Test 10: Theta computation with known function
TEST_F(AmericanOptionResultTest, ThetaComputation) {
    // Fill both current and previous solutions with known values
    // V_current(x) = 1.0 (constant)
    // V_prev(x) = 1.5 (constant)
    // Expected theta = (V_prev - V_current) / dt = 0.5 / 0.01 = 50
    auto x_span = grid->x();
    auto solution = grid->solution();
    auto solution_prev = grid->solution_prev();

    for (size_t i = 0; i < x_span.size(); ++i) {
        solution[i] = 1.0;       // V(t=0) / K
        solution_prev[i] = 1.5;  // V(t=dt) / K
    }

    AmericanOptionResult result(grid, params);
    double theta = result.theta();

    // dt = (t_end - t_start) / n_steps = 1.0 / 100 = 0.01
    // theta_normalized = (1.5 - 1.0) / 0.01 = 50
    // theta = theta_normalized * K = 50 * 100 = 5000
    double dt = 1.0 / 100.0;
    double expected_theta = (1.5 - 1.0) / dt * params.strike;

    EXPECT_NEAR(theta, expected_theta, 1e-10)
        << "Theta should match analytical value for constant solution";
}

// Test 10b: Theta sign for time decay
TEST_F(AmericanOptionResultTest, ThetaTimeDecay) {
    // Simulate time decay: V_current > V_prev (option loses value as time passes)
    // This means theta = (V_prev - V_current) / dt < 0 (time decay)
    auto solution = grid->solution();
    auto solution_prev = grid->solution_prev();

    // Use simple linear values that decrease over time
    // At t=0 (current): V = 1.0 everywhere
    // At t=dt (prev): V = 0.9 everywhere (option worth less in future)
    for (size_t i = 0; i < grid->n_space(); ++i) {
        solution[i] = 1.0;       // Current value (t=0)
        solution_prev[i] = 0.9;  // Previous value (t=dt, option worth less)
    }

    AmericanOptionResult result(grid, params);
    double theta = result.theta();

    // Theta = (V_prev - V_current) / dt = (0.9 - 1.0) / dt < 0
    EXPECT_LT(theta, 0.0)
        << "Theta should be negative when option decays over time";

    // Check magnitude
    double dt = 1.0 / 100.0;  // n_steps = 100
    double expected = (0.9 - 1.0) / dt * params.strike;  // = -1000
    EXPECT_NEAR(theta, expected, 1e-10);
}

// Test 11: Gamma correction term verification
// Verify that gamma() uses the corrected formula with both first and second derivatives
TEST_F(AmericanOptionResultTest, GammaAccuracy) {
    // Create a fine grid for better finite difference accuracy
    auto fine_grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 201);
    ASSERT_TRUE(fine_grid_spec.has_value());
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto fine_grid = Grid<double>::create(fine_grid_spec.value(), time_domain).value();

    // Use a quadratic function: V(x) = 1 + 2x + 3x²
    // This has: dV/dx = 2 + 6x, d²V/dx² = 6
    auto x_span = fine_grid->x();
    auto solution = fine_grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        double x = x_span[i];
        solution[i] = 1.0 + 2.0 * x + 3.0 * x * x;
    }

    // Test at spot = 90 (ITM put)
    double spot = 90.0;
    PricingParams test_params(
        OptionSpec{.spot = spot, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    AmericanOptionResult result(fine_grid, test_params);
    double gamma_computed = result.gamma();

    // Analytical derivatives at x_spot = ln(90/100) ≈ -0.10536
    double x_spot = std::log(spot / test_params.strike);
    double dv_dx_exact = 2.0 + 6.0 * x_spot;
    double d2v_dx2_exact = 6.0;

    // Correct gamma formula: (K/S²) * [d²V/dx² - dV/dx]
    double K_over_S2 = test_params.strike / (spot * spot);
    double gamma_correct = K_over_S2 * (d2v_dx2_exact - dv_dx_exact);

    // With fine grid (201 points), finite differences should be accurate to ~1%
    double rel_error = std::abs(gamma_computed - gamma_correct) / std::abs(gamma_correct);
    EXPECT_LT(rel_error, 0.01)
        << "Gamma should match analytical formula within 1% on fine grid"
        << "\n  computed: " << gamma_computed
        << "\n  exact:    " << gamma_correct
        << "\n  error:    " << rel_error * 100 << "%";

    // Verify the correction term is significant
    double correction_term = -K_over_S2 * dv_dx_exact;
    double second_deriv_term = K_over_S2 * d2v_dx2_exact;
    double correction_fraction = std::abs(correction_term / second_deriv_term);

    EXPECT_GT(correction_fraction, 0.1)
        << "Correction term should be significant (> 10% of second derivative term)"
        << "\n  correction term: " << correction_term
        << "\n  second deriv term: " << second_deriv_term
        << "\n  fraction: " << correction_fraction * 100 << "%";
}

} // namespace
