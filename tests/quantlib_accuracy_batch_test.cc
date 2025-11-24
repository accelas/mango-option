/**
 * @file quantlib_accuracy_batch_test.cc
 * @brief Batch validation of mango-iv pricing and IV solvers against QuantLib
 *
 * Tests:
 * - FDM pricing accuracy (auto-estimation mode)
 * - FDM-based IV accuracy (ground truth)
 * - Interpolated IV accuracy (B-spline-based, fast)
 *
 * Uses unified testing framework from quantlib_validation_framework.hpp
 */

#include "tests/quantlib_validation_framework.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include <gtest/gtest.h>

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

// TODO: Interpolated IV needs finer price table grid resolution or better Newton initialization
TEST(QuantLibBatchTest, DISABLED_StandardScenarios_IV_Interpolated) {
    auto scenarios = get_standard_test_scenarios();

    // Build price table for interpolated IV
    // Use a grid that covers all test scenarios
    std::vector<double> moneyness_grid;
    for (double m = 0.7; m <= 1.3; m += 0.05) {
        moneyness_grid.push_back(m);
    }

    std::vector<double> maturity_grid = {0.027, 0.25, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> vol_grid;
    for (double v = 0.05; v <= 0.60; v += 0.05) {
        vol_grid.push_back(v);
    }

    std::vector<double> rate_grid = {0.00, 0.02, 0.05, 0.10};

    // Build price table (one-time cost)
    auto builder_result = PriceTable4DBuilder::create(
        moneyness_grid,
        maturity_grid,
        vol_grid,
        rate_grid,
        100.0  // K_ref
    );
    ASSERT_TRUE(builder_result.has_value()) << "Failed to create builder: " << builder_result.error();
    auto builder = builder_result.value();

    // Pre-compute prices for PUT options
    auto precompute_result = builder.precompute(OptionType::PUT, 101, 1000);
    ASSERT_TRUE(precompute_result.has_value())
        << "Price table precomputation failed: " << precompute_result.error();

    const auto& price_table_result = precompute_result.value();

    // Create interpolated IV solver from surface
    auto iv_solver_result = IVSolverInterpolated::create(price_table_result.surface);
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
        IVQuery query{
            scenario.spot, scenario.strike, scenario.maturity,
            scenario.rate, scenario.dividend_yield,
            OptionType::PUT,
            ql_result.price
        };

        auto iv_result = iv_solver.solve_impl(query);

        ASSERT_TRUE(iv_result.has_value())
            << "IV solver failed with code: " << static_cast<int>(iv_result.error().code);

        // Validate accuracy
        const auto& iv_success = iv_result.value();
        double abs_error = std::abs(iv_success.implied_vol - scenario.volatility);
        double rel_error_pct = (abs_error / scenario.volatility) * 100.0;

        EXPECT_LT(rel_error_pct, scenario.iv_tolerance_pct)
            << "IV error: " << rel_error_pct << "%"
            << "\n  True vol: " << scenario.volatility
            << "\n  Recovered: " << iv_success.implied_vol
            << "\n  Iterations: " << iv_success.iterations;

        if (rel_error_pct < scenario.iv_tolerance_pct) {
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

    AmericanOptionParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    // Use automatic grid estimation
    auto [grid_spec, n_time] = estimate_grid_for_option(params);

    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    AmericanOptionSolver solver(params, workspace_result.value());
    auto result = solver.solve();
    ASSERT_TRUE(result.has_value());

    double mango_price = result->value_at(params.spot);
    double error = std::abs(mango_price - ql_reference.price);
    double rel_error = (error / ql_reference.price) * 100.0;

    EXPECT_LT(rel_error, 1.0)
        << "Convergence test: " << rel_error << "%"
        << "\n  Mango: $" << mango_price
        << "\n  Reference: $" << ql_reference.price
        << "\n  Grid: " << grid_spec.n_points() << "x" << n_time;
}

// ============================================================================
// Greeks Accuracy Test
// ============================================================================

TEST(QuantLibBatchTest, Greeks_ATM) {
    AmericanOptionParams params(
        100.0, 100.0, 1.0, 0.05, 0.02, OptionType::PUT, 0.20);

    auto [grid_spec, n_time] = estimate_grid_for_option(params);

    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    ASSERT_TRUE(workspace_result.has_value());

    AmericanOptionSolver solver(params, workspace_result.value());
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
