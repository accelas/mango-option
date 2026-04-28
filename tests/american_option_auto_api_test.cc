// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "mango/option/american_option.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <memory_resource>
#include <vector>

namespace {

mango::PricingParams make_params() {
    return mango::PricingParams(
        mango::OptionSpec{
            .spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = mango::OptionType::PUT},
        0.20);
}

TEST(AmericanOptionAutoAPITest, MatchesExplicitFormToMachineEpsilon) {
    auto params = make_params();

    // Explicit form (existing API)
    auto [grid_spec, time_domain] = mango::estimate_pde_grid(params);
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(
        mango::PDEWorkspace::required_size(n),
        std::pmr::get_default_resource());
    auto ws = mango::PDEWorkspace::from_buffer(buffer, n).value();
    auto explicit_solver = mango::AmericanOptionSolver::create(params, ws).value();
    auto explicit_result = explicit_solver.solve();
    ASSERT_TRUE(explicit_result.has_value());

    // New auto form (no workspace param)
    auto auto_solver = mango::AmericanOptionSolver::create(params).value();
    auto auto_result = auto_solver.solve();
    ASSERT_TRUE(auto_result.has_value());

    // Pricing parity to machine epsilon — covers spot, delta, and an
    // off-spot value so the spatial-operator pointer-aliasing concern
    // (variant init must not move solver objects) is exercised.
    EXPECT_DOUBLE_EQ(
        explicit_result->value_at(params.spot),
        auto_result->value_at(params.spot));
    EXPECT_DOUBLE_EQ(
        explicit_result->delta(),
        auto_result->delta());
    EXPECT_DOUBLE_EQ(
        explicit_result->value_at(params.spot * 1.1),
        auto_result->value_at(params.spot * 1.1));
}

}  // namespace
