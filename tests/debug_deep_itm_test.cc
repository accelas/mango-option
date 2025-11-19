#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mango {

TEST(DeepITMDebug, InvestigateBoundaryValues) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    ASSERT_TRUE(grid_spec.has_value());
    auto workspace = AmericanSolverWorkspace::create(grid_spec.value(), 1500, &pool);
    ASSERT_TRUE(workspace.has_value());

    AmericanOptionParams params(
        0.25,   // spot deep ITM
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

    // Current moneyness
    double x_current = std::log(params.spot / params.strike);
    std::cout << "\n=== Deep ITM Put Analysis ===" << std::endl;
    std::cout << "Spot S = " << params.spot << ", Strike K = " << params.strike << std::endl;
    std::cout << "Current moneyness x = ln(S/K) = " << x_current << std::endl;
    std::cout << "Intrinsic value = K - S = " << (params.strike - params.spot) << std::endl;
    std::cout << "Computed value = " << result.value_at(params.spot) << std::endl;
    std::cout << "Difference = " << (result.value_at(params.spot) - (params.strike - params.spot)) << std::endl;

    // Examine left boundary region (first 10 points)
    std::cout << "\n=== Left Boundary Region (first 10 grid points) ===" << std::endl;
    std::cout << std::setw(5) << "i"
              << std::setw(15) << "x (moneyness)"
              << std::setw(15) << "S (spot)"
              << std::setw(15) << "Intrinsic"
              << std::setw(15) << "Solution V/K"
              << std::setw(15) << "Value V"
              << std::setw(15) << "V - Intrinsic"
              << std::endl;

    for (size_t i = 0; i < std::min(size_t(10), result.n_space); ++i) {
        double x_i = result.x_grid[i];
        double S_i = params.strike * std::exp(x_i);
        double intrinsic_i = std::max(params.strike - S_i, 0.0);
        double solution_normalized = result.solution[i];
        double value_i = solution_normalized * params.strike;
        double excess = value_i - intrinsic_i;

        std::cout << std::setw(5) << i
                  << std::setw(15) << x_i
                  << std::setw(15) << S_i
                  << std::setw(15) << intrinsic_i
                  << std::setw(15) << solution_normalized
                  << std::setw(15) << value_i
                  << std::setw(15) << excess
                  << std::endl;
    }

    // Find grid point closest to current spot
    size_t closest_idx = 0;
    double min_dist = std::abs(result.x_grid[0] - x_current);
    for (size_t i = 1; i < result.n_space; ++i) {
        double dist = std::abs(result.x_grid[i] - x_current);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    std::cout << "\n=== Closest Grid Point to Current Spot ===" << std::endl;
    std::cout << "Index: " << closest_idx << std::endl;
    std::cout << "Grid moneyness: " << result.x_grid[closest_idx] << std::endl;
    std::cout << "Target moneyness: " << x_current << std::endl;
    std::cout << "Distance: " << min_dist << std::endl;

    double S_closest = params.strike * std::exp(result.x_grid[closest_idx]);
    double intrinsic_closest = std::max(params.strike - S_closest, 0.0);
    double value_closest = result.solution[closest_idx] * params.strike;

    std::cout << "Spot at closest point: " << S_closest << std::endl;
    std::cout << "Intrinsic at closest: " << intrinsic_closest << std::endl;
    std::cout << "Solution value: " << value_closest << std::endl;
    std::cout << "Excess over intrinsic: " << (value_closest - intrinsic_closest) << std::endl;
}

}  // namespace mango
