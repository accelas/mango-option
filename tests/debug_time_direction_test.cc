#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mango {

TEST(TimeDirectionDebug, CheckSolutionEvolution) {
    std::pmr::synchronized_pool_resource pool;
    auto grid_spec = GridSpec<double>::sinh_spaced(-7.0, 2.0, 301, 2.0);
    ASSERT_TRUE(grid_spec.has_value());
    auto workspace = AmericanSolverWorkspace::create(grid_spec.value(), 10, &pool);  // Only 10 time steps
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

    // Allocate buffer for full surface
    size_t n_space = workspace.value()->n_space();
    size_t n_time = workspace.value()->n_time();
    std::vector<double> surface_buffer((n_time + 1) * n_space);

    auto solver_result = AmericanOptionSolver::create(params, workspace.value());
    ASSERT_TRUE(solver_result.has_value());

    auto solve_result = solver_result.value().solve();
    ASSERT_TRUE(solve_result.has_value());

    const auto& result = solve_result.value();
    ASSERT_TRUE(result.converged);

    std::cout << "\n=== Time Direction Analysis ===" << std::endl;
    std::cout << "Maturity T = " << params.maturity << " years" << std::endl;
    std::cout << "Number of time steps = " << n_time << std::endl;
    std::cout << "dt = " << (params.maturity / n_time) << " years per step" << std::endl;

    // Look at a specific spatial point (middle of domain)
    size_t mid_idx = n_space / 2;
    double x_mid = result.x_grid[mid_idx];
    double S_mid = params.strike * std::exp(x_mid);

    std::cout << "\n--- Value at mid-point x=" << x_mid << ", S=" << S_mid << " ---" << std::endl;

    // Print first and last few time steps only
    std::cout << std::setw(10) << "Time step"
              << std::setw(15) << "Time t"
              << std::setw(15) << "Time-to-mat τ"
              << std::setw(20) << "Value V/K"
              << std::endl;

    // First 3 steps
    for (size_t time_idx = 0; time_idx < std::min(size_t(3), n_time); ++time_idx) {
        double t = time_idx * (params.maturity / n_time);
        double tau = params.maturity - t;
        auto surface_slice = result.at_time(time_idx);
        if (!surface_slice.empty()) {
            double value_normalized = surface_slice[mid_idx];
            std::cout << std::setw(10) << time_idx
                      << std::setw(15) << t
                      << std::setw(15) << tau
                      << std::setw(20) << value_normalized
                      << std::endl;
        }
    }

    std::cout << "..." << std::endl;

    // Last 3 steps
    for (size_t time_idx = std::max(size_t(3), n_time) - 3; time_idx < n_time; ++time_idx) {
        double t = time_idx * (params.maturity / n_time);
        double tau = params.maturity - t;
        auto surface_slice = result.at_time(time_idx);
        if (!surface_slice.empty()) {
            double value_normalized = surface_slice[mid_idx];
            std::cout << std::setw(10) << time_idx
                      << std::setw(15) << t
                      << std::setw(15) << tau
                      << std::setw(20) << value_normalized
                      << std::endl;
        }
    }

    std::cout << "\n=== Interpretation ===" << std::endl;
    std::cout << "If value INCREASES as time_idx increases (t increases, τ decreases):" << std::endl;
    std::cout << "  → Solver is going FORWARD in calendar time (t=0 to t=T)" << std::endl;
    std::cout << "  → Initial condition (time_idx=0) is at t=0 (present)" << std::endl;
    std::cout << "  → Final condition (time_idx=n_time-1) is at t=T (maturity)" << std::endl;
    std::cout << "\nIf value DECREASES as time_idx increases:" << std::endl;
    std::cout << "  → Solver is going BACKWARD in time-to-maturity (τ=T to τ=0)" << std::endl;
    std::cout << "  → Initial condition is at maturity (payoff)" << std::endl;
    std::cout << "  → Final condition is at present (option value)" << std::endl;
}

}  // namespace mango
