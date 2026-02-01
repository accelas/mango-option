// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/american_option.hpp"

#include <cmath>
#include <memory_resource>
#include <vector>

using namespace mango;

TEST(AmericanOptionSolverTest, CustomInitialCondition) {
    // Use a custom IC that's slightly different from standard payoff
    // Verify the solver runs and produces a result
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20);
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate workspace buffer
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                     std::pmr::get_default_resource());
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    // Custom IC: payoff + small constant (simulates chained terminal condition)
    auto custom_ic = [](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = std::max(1.0 - std::exp(x[i]), 0.0) + 0.01;
        }
    };

    auto solver = AmericanOptionSolver::create(params, workspace).value();
    solver.set_initial_condition(custom_ic);
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value_at(params.spot), 0.0);
}

TEST(AmericanOptionSolverTest, DefaultPayoffStillWorks) {
    // Regression: ensure default behavior (no custom IC) is unchanged
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 0.5,
            .rate = 0.05, .dividend_yield = 0.02, .type = OptionType::PUT}, 0.20);
    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n),
                                     std::pmr::get_default_resource());
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    auto solver = AmericanOptionSolver::create(params, workspace).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value_at(params.spot), 0.0);
}
