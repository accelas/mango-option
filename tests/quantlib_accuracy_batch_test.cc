// SPDX-License-Identifier: MIT
/**
 * @file quantlib_accuracy_batch_test.cc
 * @brief Batch validation of mango-option pricing and IV solvers against QuantLib
 *
 * Tests:
 * - FDM pricing accuracy (auto-estimation mode)
 * - FDM-based IV accuracy (ground truth)
 * - Interpolated IV accuracy (B-spline-based, fast)
 *
 * Uses unified testing framework from quantlib_validation_framework.hpp
 */

#include "tests/quantlib_validation_framework.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/bspline/bspline_tensor_accessor.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mango;
using namespace mango::testing;

// ============================================================================
// FDM Pricing + IV Batch Tests
// ============================================================================

TEST(QuantLibBatchTest, StandardScenarios_Pricing_And_IV_FDM) {
    auto scenarios = get_standard_test_scenarios();

    auto summary = validate_batch(scenarios,
        /*test_pricing=*/true,
        /*test_iv_fdm=*/true);

    summary.print_summary();
    EXPECT_TRUE(summary.all_passed());
}

// ============================================================================
// Interpolated IV Batch Test
// ============================================================================

TEST(QuantLibBatchTest, StandardScenarios_IV_Interpolated) {
    auto scenarios = get_standard_test_scenarios();

    // Build price table for interpolated IV (auto grid estimation)
    // Compute log-moneyness bounds from scenarios
    double lm_min = std::numeric_limits<double>::max();
    double lm_max = std::numeric_limits<double>::lowest();
    double tau_min = std::numeric_limits<double>::max();
    double tau_max = std::numeric_limits<double>::lowest();
    double sigma_min = std::numeric_limits<double>::max();
    double sigma_max = std::numeric_limits<double>::lowest();
    double r_min = std::numeric_limits<double>::max();
    double r_max = std::numeric_limits<double>::lowest();

    for (const auto& scenario : scenarios) {
        const double lm = std::log(scenario.spot / scenario.strike);
        lm_min = std::min(lm_min, lm);
        lm_max = std::max(lm_max, lm);
        tau_min = std::min(tau_min, scenario.maturity);
        tau_max = std::max(tau_max, scenario.maturity);
        sigma_min = std::min(sigma_min, scenario.volatility);
        sigma_max = std::max(sigma_max, scenario.volatility);
        r_min = std::min(r_min, scenario.rate);
        r_max = std::max(r_max, scenario.rate);
    }

    // Pad bounds to avoid extrapolation at edges
    // Log-moneyness: pad symmetrically
    lm_min -= 0.02;
    lm_max += 0.02;
    tau_min *= 0.9;
    tau_max *= 1.1;
    sigma_min = std::max(0.01, sigma_min * 0.9);
    sigma_max *= 1.1;
    r_min -= 0.01;
    r_max += 0.01;

    auto grid_params = make_price_table_grid_accuracy(PriceTableGridProfile::High);

    auto grid_estimate = estimate_grid_for_price_table(
        lm_min, lm_max, tau_min, tau_max, sigma_min, sigma_max, r_min, r_max, grid_params);

    std::vector<double> moneyness_grid = std::move(grid_estimate.moneyness_grid());
    std::vector<double> maturity_grid = std::move(grid_estimate.maturity_grid());
    std::vector<double> vol_grid = std::move(grid_estimate.volatility_grid());
    std::vector<double> rate_grid = std::move(grid_estimate.rate_grid());

    // Build price table (one-time cost)
    std::vector<PricingParams> pde_params;
    pde_params.reserve(scenarios.size());
    for (const auto& scenario : scenarios) {
        pde_params.emplace_back(PricingParams(
            OptionSpec{.spot = scenario.spot, .strike = scenario.strike,
                .maturity = scenario.maturity, .rate = scenario.rate,
                .dividend_yield = scenario.dividend_yield,
                .option_type = scenario.is_call ? OptionType::CALL : OptionType::PUT},
            scenario.volatility));
    }

    auto pde_accuracy = make_grid_accuracy(GridAccuracyProfile::High);

    auto [grid_spec, time_domain] = estimate_batch_pde_grid(pde_params, pde_accuracy);

    const double dividend_yield = scenarios.front().dividend_yield;
    for (const auto& scenario : scenarios) {
        ASSERT_DOUBLE_EQ(scenario.dividend_yield, dividend_yield)
            << "Interpolated IV test expects a single dividend yield across scenarios";
    }

    auto builder_axes_result = PriceTableBuilder::from_vectors(
        moneyness_grid,
        maturity_grid,
        vol_grid,
        rate_grid,
        100.0,  // K_ref
        PDEGridConfig{grid_spec, time_domain.n_steps()},
        OptionType::PUT,
        dividend_yield,
        0.0);    // max_failure_rate
    ASSERT_TRUE(builder_axes_result.has_value()) << "Failed to create builder: " << builder_axes_result.error();
    auto [builder, axes] = std::move(builder_axes_result.value());

    // Pre-compute prices for PUT options with EEP decomposition
    auto precompute_result = builder.build(axes,
        [&](PriceTensor& tensor, const PriceTableAxes& a) {
            BSplineTensorAccessor accessor(tensor, a, 100.0);
            eep_decompose(accessor, AnalyticalEEP(OptionType::PUT, dividend_yield));
        });
    ASSERT_TRUE(precompute_result.has_value())
        << "Price table precomputation failed: " << precompute_result.error();

    const auto& price_table_result = precompute_result.value();

    // Create interpolated IV solver from surface
    InterpolatedIVSolverConfig iv_config{
        .max_iter = 100,
        .tolerance = 1e-7,
        .sigma_min = 0.05,
        .sigma_max = 2.0
    };
    auto wrapper = make_bspline_surface(price_table_result.surface, OptionType::PUT);
    ASSERT_TRUE(wrapper.has_value()) << "Failed to create wrapper: " << wrapper.error();
    auto iv_solver_result = InterpolatedIVSolver<BSplinePriceTable>::create(std::move(*wrapper), iv_config);
    ASSERT_TRUE(iv_solver_result.has_value())
        << "Failed to create interpolated IV solver: " << iv_solver_result.error();

    const auto& iv_solver = iv_solver_result.value();

    // Test each scenario
    size_t passed = 0;
    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.name);

        // Skip call options (price table was built for puts only)
        if (scenario.is_call) {
            continue;
        }

        // Get market price from QuantLib
        auto ql_result = price_with_quantlib(
            scenario.spot, scenario.strike, scenario.maturity,
            scenario.volatility, scenario.rate, scenario.dividend_yield,
            scenario.is_call, 201, 2000);

        // Solve for IV using interpolated solver
        IVQuery query(
            OptionSpec{.spot = scenario.spot, .strike = scenario.strike,
                .maturity = scenario.maturity, .rate = scenario.rate,
                .dividend_yield = scenario.dividend_yield,
                .option_type = OptionType::PUT},
            ql_result.price);

        auto iv_result = iv_solver.solve(query);

        ASSERT_TRUE(iv_result.has_value())
            << "IV solver failed with code: " << static_cast<int>(iv_result.error().code);

        // Validate accuracy
        const auto& iv_success = iv_result.value();
        double abs_error = std::abs(iv_success.implied_vol - scenario.volatility);
        double rel_error_pct = (abs_error / scenario.volatility) * 100.0;
        double iv_tolerance_pct = scenario.iv_tolerance_pct;
        if (!scenario.is_call && (scenario.spot / scenario.strike) <= 0.85) {
            // Deep ITM options have low vega; small price errors can amplify IV error.
            iv_tolerance_pct = std::max(iv_tolerance_pct, 3.0);
        }

        EXPECT_LT(rel_error_pct, iv_tolerance_pct)
            << "IV error: " << rel_error_pct << "%"
            << "\n  True vol: " << scenario.volatility
            << "\n  Recovered: " << iv_success.implied_vol
            << "\n  Iterations: " << iv_success.iterations;

        if (rel_error_pct < iv_tolerance_pct) {
            passed++;
        }
    }

    // Report summary (excluding call options)
    size_t put_count = 0;
    for (const auto& s : scenarios) {
        if (!s.is_call) put_count++;
    }

    EXPECT_EQ(passed, put_count)
        << "Interpolated IV: " << passed << "/" << put_count << " tests passed";
}

// ============================================================================
// Convergence Test
// ============================================================================

TEST(QuantLibBatchTest, GridConvergence) {
    // Reference: Very high resolution QuantLib result
    auto ql_reference = price_with_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false,
        1001, 10000);

    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    // Use automatic grid estimation
    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    double mango_price = result->value_at(params.spot);
    double error = std::abs(mango_price - ql_reference.price);
    double rel_error = (error / ql_reference.price) * 100.0;

    EXPECT_LT(rel_error, 1.0)
        << "Convergence test: " << rel_error << "%"
        << "\n  Mango: $" << mango_price
        << "\n  Reference: $" << ql_reference.price
        << "\n  Grid: " << grid_spec.n_points() << "x" << time_domain.n_steps();
}

// ============================================================================
// Greeks Accuracy Test
// ============================================================================

TEST(QuantLibBatchTest, Greeks_ATM) {
    PricingParams params(
        OptionSpec{.spot = 100.0, .strike = 100.0, .maturity = 1.0,
            .rate = 0.05, .dividend_yield = 0.02,
            .option_type = OptionType::PUT},
        0.20);

    auto [grid_spec, time_domain] = estimate_pde_grid(params);

    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    auto solver = AmericanOptionSolver::create(params, workspace_result.value()).value();
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    double delta_val = result->delta();
    double gamma_val = result->gamma();

    auto ql_result = price_with_quantlib(
        100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false, 201, 2000);

    // Delta within 2%
    double delta_error = std::abs(delta_val - ql_result.delta);
    double delta_rel = (delta_error / std::abs(ql_result.delta)) * 100.0;
    EXPECT_LT(delta_rel, 2.0)
        << "Delta: mango=" << delta_val << " ql=" << ql_result.delta;

    // Gamma within 5%
    double gamma_error = std::abs(gamma_val - ql_result.gamma);
    double gamma_rel = (gamma_error / std::abs(ql_result.gamma)) * 100.0;
    EXPECT_LT(gamma_rel, 5.0)
        << "Gamma: mango=" << gamma_val << " ql=" << ql_result.gamma;
}
