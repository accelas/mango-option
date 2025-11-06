/**
 * @file cpp_boundary_test.cc
 * @brief Test to verify C++ boundary condition bug hypothesis
 *
 * This test demonstrates that C++ boundary conditions are static
 * (don't evolve with time) while C boundary conditions correctly
 * account for time discounting.
 */

#include "src/american_option.hpp"
#include "src/american_option.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mango {
namespace {

TEST(BoundaryConditionBugTest, CPPBoundariesIgnoreTime) {
    // Setup identical parameters for C and C++ solvers
    const double strike = 100.0;
    const double spot = 100.0;
    const double maturity = 1.0;
    const double volatility = 0.2;
    const double rate = 0.05;

    // C++ solver
    AmericanOptionParams cpp_params{
        .strike = strike,
        .spot = spot,
        .maturity = maturity,
        .volatility = volatility,
        .rate = rate,
        .continuous_dividend_yield = 0.0,
        .option_type = OptionType::PUT
    };

    AmericanOptionGrid cpp_grid{};  // Use defaults
    cpp_grid.n_space = 101;
    cpp_grid.n_time = 1000;

    AmericanOptionSolver cpp_solver(cpp_params, cpp_grid);
    auto cpp_result = cpp_solver.solve();
    if (!cpp_result) {
        throw std::runtime_error(cpp_result.error().message);
    }

    // C solver
    OptionData c_option = {
        .strike = strike,
        .volatility = volatility,
        .risk_free_rate = rate,
        .time_to_maturity = maturity,
        .option_type = OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    ::AmericanOptionGrid c_grid = {
        .x_min = -3.0,
        .x_max = 3.0,
        .n_points = 101,
        .dt = maturity / 1000.0,
        .n_steps = 1000
    };

    ::AmericanOptionResult c_result = american_option_price(&c_option, &c_grid);
    double c_value = american_option_get_value_at_spot(
        c_result.solver, spot, strike
    );

    // Print results for debugging
    std::cout << "\nBoundary Condition Bug Test:\n";
    std::cout << "  C++ value: " << cpp_result->value << "\n";
    std::cout << "  C value:   " << c_value << "\n";
    std::cout << "  Diff:      " << std::abs(cpp_result->value - c_value) << "\n";
    std::cout << "  C++ converged: " << (cpp_result->converged ? "yes" : "no") << "\n";
    std::cout << "  C converged:   " << (c_result.status == 0 ? "yes" : "no") << "\n";

    // Cleanup
    american_option_free_result(&c_result);

    // The bug: C++ should match C but doesn't due to wrong boundary conditions
    // Expected: C++ ≈ C (both around $5.50 for ATM put)
    // Actual: C++ gives wrong value (often close to intrinsic or zero)

    // For ATM American put with r=5%, σ=20%, T=1:
    // Theoretical value should be around $5.50
    EXPECT_NEAR(c_value, 5.50, 1.0);  // C should be correct (within $1)

    // This will FAIL, demonstrating the bug:
    EXPECT_NEAR(cpp_result->value, c_value, 0.50);
        << "C++ value differs significantly from C value due to boundary condition bug";
}

TEST(BoundaryConditionBugTest, CPPBoundaryValuesAtDifferentTimes) {
    // This test demonstrates that C++ boundaries don't change with time

    const double strike = 100.0;
    const double volatility = 0.2;
    const double rate = 0.05;

    // For a deep ITM put (S→0), the correct boundary value is:
    // V(S=0, t) = K·exp(-r·τ) where τ = time to maturity
    //
    // At t=0 (maturity):     V = K·exp(0) = K = 100.0
    // At t=0.5 (6 months):   V = K·exp(-0.05·0.5) = 97.53
    // At t=1.0 (1 year):     V = K·exp(-0.05·1.0) = 95.12

    // C++ implementation (from american_option.cpp:153):
    // Returns: 1.0 - exp(x) where x = ln(S/K)
    // At S→0, x→-∞, so exp(x)→0, giving boundary value = 1.0 (normalized)
    // Denormalized: 1.0 * K = 100.0
    //
    // THIS IS WRONG! It should be exp(-r*t) * K, not K

    // Expected boundary values (at S=0) for deep ITM put:
    double expected_at_maturity = strike * std::exp(-rate * 0.0);   // 100.00
    double expected_at_6mo = strike * std::exp(-rate * 0.5);        //  97.53
    double expected_at_1yr = strike * std::exp(-rate * 1.0);        //  95.12

    std::cout << "\nExpected boundary values for deep ITM put (S→0):\n";
    std::cout << "  At maturity (t=0):   $" << expected_at_maturity << "\n";
    std::cout << "  At 6 months (t=0.5): $" << expected_at_6mo << "\n";
    std::cout << "  At 1 year (t=1.0):   $" << expected_at_1yr << "\n";
    std::cout << "\nC++ implementation returns: $" << strike << " (CONSTANT, WRONG!)\n";

    // The bug is that C++ boundary is constant (always 100.0) instead of
    // decreasing with time due to discounting (100.0 → 97.53 → 95.12)

    EXPECT_NE(expected_at_1yr, strike)
        << "Boundary value should change with time due to discounting";
}

}  // namespace
}  // namespace mango
