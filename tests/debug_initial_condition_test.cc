#include "src/option/american_option.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mango {

TEST(InitialConditionDebug, CheckT0Values) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    ASSERT_TRUE(grid_spec.has_value());
    auto workspace = AmericanSolverWorkspace::create(grid_spec.value(), 100, &pool);
    ASSERT_TRUE(workspace.has_value());

    AmericanOptionParams params(
        0.25,   // spot
        100.0,  // strike
        0.75,   // maturity
        0.05,   // rate
        0.0,    // dividend yield
        OptionType::PUT,
        0.2     // volatility
    );

    auto solver_result = AmericanOptionSolver::create(params, workspace.value());
    ASSERT_TRUE(solver_result.has_value());

    auto solve_result = solver_result.value().solve();
    ASSERT_TRUE(solve_result.has_value());

    const auto& result = solve_result.value();
    ASSERT_TRUE(result.converged);

    std::cout << "\n=== Initial Condition at t=0 (first time step) ===" << std::endl;
    std::cout << "Showing first 10 spatial points:\n" << std::endl;
    std::cout << std::setw(5) << "i"
              << std::setw(15) << "x"
              << std::setw(15) << "S"
              << std::setw(20) << "Payoff (K-S)"
              << std::setw(20) << "Solution V/K (t=0)"
              << std::setw(20) << "V (t=0)"
              << std::endl;

    auto t0_surface = result.at_time(0);  // First time step
    for (size_t i = 0; i < std::min(size_t(10), result.n_space); ++i) {
        double x_i = result.x_grid[i];
        double S_i = params.strike * std::exp(x_i);
        double payoff = std::max(params.strike - S_i, 0.0);
        double value_t0_normalized = t0_surface.empty() ? 0.0 : t0_surface[i];
        double value_t0 = value_t0_normalized * params.strike;

        std::cout << std::setw(5) << i
                  << std::setw(15) << x_i
                  << std::setw(15) << S_i
                  << std::setw(20) << payoff
                  << std::setw(20) << value_t0_normalized
                  << std::setw(20) << value_t0
                  << std::endl;
    }

    std::cout << "\n=== Final Condition at t=T (last time step) ===" << std::endl;
    auto tT_surface = result.at_time(result.n_time - 1);
    for (size_t i = 0; i < std::min(size_t(10), result.n_space); ++i) {
        double x_i = result.x_grid[i];
        double S_i = params.strike * std::exp(x_i);
        double payoff = std::max(params.strike - S_i, 0.0);
        double value_tT_normalized = tT_surface.empty() ? 0.0 : tT_surface[i];
        double value_tT = value_tT_normalized * params.strike;

        std::cout << std::setw(5) << i
                  << std::setw(15) << x_i
                  << std::setw(15) << S_i
                  << std::setw(20) << payoff
                  << std::setw(20) << value_tT_normalized
                  << std::setw(20) << value_tT
                  << std::endl;
    }

    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "If solution at t=0 equals payoff → solver starts at maturity (correct)" << std::endl;
    std::cout << "If solution at t=T equals payoff → solver ends at maturity (WRONG for backward solve)" << std::endl;
}

}  // namespace mango
