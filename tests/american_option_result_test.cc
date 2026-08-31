// SPDX-License-Identifier: MIT
/**
 * @file american_option_result_test.cc
 * @brief Tests for AmericanOptionResult wrapper class
 */

#include "mango/option/american_option_result.hpp"
#include "mango/option/american_option.hpp"
#include <gtest/gtest.h>
#include <array>
#include <barrier>
#include <cmath>
#include <thread>

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

// ===========================================================================
// Regression: cubic spline interpolation vs linear (#331)
// Verify that cubic spline gives better accuracy at off-grid points
// ===========================================================================

// Verify cubic spline interpolation is accurate on a smooth function
// evaluated at off-grid points (where the improvement over linear matters)
TEST_F(AmericanOptionResultTest, CubicSplineOffGridAccuracy) {
    // Create a coarse grid to amplify interpolation error
    auto coarse_grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 21);
    ASSERT_TRUE(coarse_grid_spec.has_value());
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto coarse_grid = Grid<double>::create(coarse_grid_spec.value(), time_domain).value();

    // Fill with a smooth function: V(x) = exp(-x²) (Gaussian-like, always positive)
    // This approximates a smooth PDE solution without kinks
    auto x_span = coarse_grid->x();
    auto solution = coarse_grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        double x = x_span[i];
        solution[i] = std::exp(-x * x);
    }

    // Evaluate at a point between grid nodes
    // Grid spacing = 2.0/20 = 0.1, so midpoint x=0.05 is between nodes
    double spot_mid = params.strike * std::exp(0.05);  // S = K * exp(0.05)

    PricingParams mid_params(
        OptionSpec{.spot = spot_mid, .strike = params.strike, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    AmericanOptionResult result(coarse_grid, mid_params);
    double value = result.value_at(spot_mid);
    double value_normalized = value / params.strike;

    // Exact value at x=0.05: exp(-0.0025) ≈ 0.99750
    double exact = std::exp(-0.05 * 0.05);
    double error = std::abs(value_normalized - exact);

    // Cubic spline on a smooth function should be very accurate (< 1e-4)
    // Linear interpolation on this grid would give ~2.5e-4 error
    EXPECT_LT(error, 1e-4)
        << "Cubic spline should interpolate smooth functions accurately"
        << "\n  computed: " << value_normalized
        << "\n  exact:    " << exact
        << "\n  error:    " << error;
}

// Verify delta from spline derivative is more accurate than grid-snapping
TEST_F(AmericanOptionResultTest, CubicSplineDeltaAccuracy) {
    // Fine grid with smooth function: V(x) = exp(-x²)
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 51);
    ASSERT_TRUE(grid_spec.has_value());
    auto time_domain = TimeDomain::from_n_steps(0.0, 1.0, 100);
    auto test_grid = Grid<double>::create(grid_spec.value(), time_domain).value();

    auto x_span = test_grid->x();
    auto solution = test_grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        double x = x_span[i];
        solution[i] = std::exp(-x * x);
    }

    // Evaluate at off-grid spot: x = 0.03 (between nodes at dx = 0.04)
    double spot = params.strike * std::exp(0.03);
    PricingParams test_params(
        OptionSpec{.spot = spot, .strike = params.strike, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02, .option_type = OptionType::PUT}, 0.20);

    AmericanOptionResult result(test_grid, test_params);
    double delta = result.delta();

    // Analytical: V(x) = K·exp(-x²), dV/dS = (K/S)·dV_norm/dx = (K/S)·(-2x)·exp(-x²)
    double x_spot = 0.03;
    double dv_norm_dx = -2.0 * x_spot * std::exp(-x_spot * x_spot);
    double expected_delta = dv_norm_dx * (params.strike / spot);

    // Spline derivative should be accurate within 0.5%
    double rel_error = std::abs(delta - expected_delta) / std::abs(expected_delta);
    EXPECT_LT(rel_error, 0.005)
        << "Cubic spline derivative should be accurate at off-grid points"
        << "\n  computed delta: " << delta
        << "\n  exact delta:    " << expected_delta
        << "\n  rel error:      " << rel_error * 100 << "%";
}

// ===========================================================================
// Regression tests for bugs found during code review
// ===========================================================================

// Regression: const accessors raced on lazy spline/operator initialization
// Bug (#436): ensure_spline()/ensure_operator() mutated shared state under a
// documented "const methods are thread-safe" promise; concurrent FIRST calls
// were a data race (both threads see built==false and build concurrently).
TEST(AmericanOptionResultConcurrencyTest, ConcurrentFirstAccessMatchesSerial) {
    mango::PricingParams params(
        mango::OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
                          .rate = 0.05, .dividend_yield = 0.02,
                          .option_type = mango::OptionType::PUT},
        0.20);
    // Expected values from a separate, serially-used result.
    auto expected = mango::solve_american_option(params);
    ASSERT_TRUE(expected.has_value());
    const double e_value = expected->value_at(102.0);
    const double e_delta = expected->delta();
    const double e_gamma = expected->gamma();
    const double e_theta = expected->theta();

    constexpr int kThreads = 8;

    // Phase 1: spline race — value_at is every thread's FIRST accessor call.
    auto r1 = mango::solve_american_option(params);
    ASSERT_TRUE(r1.has_value());
    {
        std::barrier sync(kThreads);
        std::array<std::array<double, 3>, kThreads> got{};
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                sync.arrive_and_wait();
                got[t] = {r1->value_at(102.0), r1->delta(), r1->theta()};
            });
        }
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) {
            EXPECT_EQ(got[t][0], e_value);
            EXPECT_EQ(got[t][1], e_delta);
            EXPECT_EQ(got[t][2], e_theta);
        }
    }

    // Phase 2: operator race — fresh result, gamma() is every thread's FIRST
    // call, so the lazy operator build races independently of phase 1.
    auto r2 = mango::solve_american_option(params);
    ASSERT_TRUE(r2.has_value());
    {
        std::barrier sync(kThreads);
        std::array<double, kThreads> got{};
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                sync.arrive_and_wait();
                got[t] = r2->gamma();
            });
        }
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) EXPECT_EQ(got[t], e_gamma);
    }
}

// ===========================================================================
// Regression tests for bugs found during code review (issue #438)
// ===========================================================================

// Regression: gamma() never computed interior node n-2
// Bug: CenteredDifference ranges are exclusive-end ([start, end)); gamma()
// passed end = n-2, so d2v_dx2[n-2]/dv_dx[n-2] stayed 0.0 and any query
// interpolating against node n-2 blended with a spurious zero.
TEST_F(AmericanOptionResultTest, GammaAtLastInteriorNodeUsesComputedStencil) {
    // Synthetic smooth solution V/K = x^2: d2V/dx2 = 2 (exact under central
    // differences), dV/dx = 2x (exact on a uniform grid).
    auto x_span = grid->x();
    auto solution = grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        solution[i] = x_span[i] * x_span[i];
    }

    const size_t n = x_span.size();
    const double x_target = x_span[n - 2];  // last interior node
    PricingParams p = params;
    p.spot = p.strike * std::exp(x_target);  // spot exactly on node n-2

    AmericanOptionResult result(grid, p);
    const double gamma = result.gamma();

    // gamma = (K/S^2) * (d2V/dx2 - dV/dx) = (K/S^2) * (2 - 2*x_target)
    const double S = p.spot;
    const double expected = p.strike / (S * S) * (2.0 - 2.0 * x_target);
    // Under the bug both stencil values at n-2 are the unwritten 0.0, so
    // gamma is exactly 0.
    EXPECT_NEAR(gamma, expected, std::abs(expected) * 0.05)
        << "gamma at node n-2 must use computed stencil values";
}

// Regression: gamma interpolation in (x[n-3], x[n-2]) blended a zero
// Bug: with i_right = n-2 unwritten, the linear blend pulled gamma toward 0
// by alpha * 100%.
TEST_F(AmericanOptionResultTest, GammaNearRightEdgeMatchesAnalyticReference) {
    auto x_span = grid->x();
    auto solution = grid->solution();
    for (size_t i = 0; i < x_span.size(); ++i) {
        solution[i] = x_span[i] * x_span[i];
    }

    const size_t n = x_span.size();
    const double x_mid = 0.5 * (x_span[n - 3] + x_span[n - 2]);
    PricingParams p = params;
    p.spot = p.strike * std::exp(x_mid);

    AmericanOptionResult result(grid, p);

    // For V/K = x^2 the stencil values are exact (central differences of a
    // quadratic), and the linear blend of dV/dx = 2x is exact at x_mid, so
    // gamma must equal the analytic (K/S^2) * (2 - 2*x_mid).
    // Under the bug, i_right = n-2 held unwritten zeros and the blend gave
    // gamma ~33% low.
    const double S = p.spot;
    const double expected = p.strike / (S * S) * (2.0 - 2.0 * x_mid);
    EXPECT_NEAR(result.gamma(), expected, std::abs(expected) * 0.01)
        << "gamma mid-interval blend must not include unwritten stencil zeros";
}

// Regression: theta() divided by the average dt, not the actual final step
// Bug: Grid::dt() forwards TimeDomain::dt(), which is only an average for
// non-uniform time grids (the discrete-dividend case). solution_prev is one
// *actual* final step away, so theta was scaled by dt_avg / dt_last.
TEST(AmericanOptionResultNonUniformTimeTest, ThetaUsesActualFinalStep) {
    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 21);
    ASSERT_TRUE(grid_spec.has_value());

    // Segments: [0, 0.95] -> 10 steps of 0.095; [0.95, 1.0] -> 1 step of 0.05.
    auto time_domain =
        TimeDomain::with_mandatory_points(0.0, 1.0, 0.1, {0.95});
    const size_t n_steps = time_domain.n_steps();
    const double dt_last = time_domain.dt_at(n_steps - 1);
    // Precondition: grid actually non-uniform (guard against vacuous pass)
    ASSERT_GT(std::abs(dt_last - time_domain.dt()), 1e-3);

    auto grid_result = Grid<double>::create(*grid_spec, time_domain);
    ASSERT_TRUE(grid_result.has_value());
    auto grid = grid_result.value();

    auto solution = grid->solution();
    auto solution_prev = grid->solution_prev();
    for (size_t i = 0; i < grid->n_space(); ++i) {
        solution[i] = 1.0;
        solution_prev[i] = 1.5;
    }

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);
    AmericanOptionResult result(grid, params);

    // theta = (1.5 - 1.0) / dt_last * K. Under the bug the divisor is the
    // average dt() ≈ 0.0909, giving ~550 instead of 1000.
    EXPECT_NEAR(result.theta(), 0.5 / dt_last * params.strike, 1e-3);
}

// Regression: discrete-dividend theta was off by dt_avg/dt_last vs a
// calendar-time finite difference
// Bug: same divisor bug, observed end-to-end. The FD reference advances the
// valuation date: maturity T-h AND every dividend calendar_time reduced by h
// (dividends anchor to the valuation date; the solver places the jump at
// tau = maturity - calendar_time, so this keeps the event's tau fixed).
TEST(AmericanOptionResultNonUniformTimeTest,
     ThetaMatchesCalendarBumpWithDiscreteDividend) {
    auto grid_spec = GridSpec<double>::uniform(-2.0, 2.0, 201);
    ASSERT_TRUE(grid_spec.has_value());

    // Dividend at calendar 0.043 -> tau event at 0.957. n_time = 11 makes
    // the final segment [0.957, 1.0] one step of 0.043 vs average ~0.083.
    const double div_time = 0.043;
    const double tau_div = 1.0 - div_time;
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.0,
                   .option_type = OptionType::PUT},
        0.20,
        {Dividend{.calendar_time = div_time, .amount = 1.50}});
    PDEGridConfig grid_config{.grid_spec = *grid_spec, .n_time = 11,
                              .mandatory_times = {tau_div}};

    auto solver = AmericanOptionSolver::create(params, PDEGridSpec{grid_config});
    ASSERT_TRUE(solver.has_value());
    auto result = solver->solve();
    ASSERT_TRUE(result.has_value());

    // Precondition: final step differs measurably from the average
    const auto& time = result->grid()->time();
    ASSERT_GT(std::abs(time.dt_at(time.n_steps() - 1) - time.dt()), 1e-2);

    // Calendar bump: shorten maturity AND shift the dividend
    const double h = 0.02;
    PricingParams bumped = params;
    bumped.maturity -= h;
    bumped.discrete_dividends[0].calendar_time -= h;
    PDEGridConfig bumped_config{.grid_spec = *grid_spec, .n_time = 11,
                                .mandatory_times = {tau_div}};
    auto bumped_solver =
        AmericanOptionSolver::create(bumped, PDEGridSpec{bumped_config});
    ASSERT_TRUE(bumped_solver.has_value());
    auto bumped_result = bumped_solver->solve();
    ASSERT_TRUE(bumped_result.has_value());

    const double theta_fd =
        (bumped_result->value() - result->value()) / h;

    // Bug error is ~50% (dt_avg/dt_last ≈ 1.9); 20% tolerance separates it
    // while absorbing FD truncation + coarse-grid noise.
    EXPECT_NEAR(result->theta(), theta_fd, std::abs(theta_fd) * 0.20)
        << "theta must match the calendar-time FD reference";
}

// Regression: value_at(NaN) returned 0.0 (issue #466 family)
// Bug: std::max(0.0, spline eval) masked the NaN from log(NaN/K)
TEST(AmericanOptionResultNaNTest, ValueAtNaNSpotReturnsNaN) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
                   .rate = 0.05, .dividend_yield = 0.02,
                   .option_type = OptionType::PUT},
        0.20);
    auto result = solve_american_option(params);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(result->value_at(std::nan(""))));
}

} // namespace
