// SPDX-License-Identifier: MIT
/**
 * @file quantlib_validation_framework.hpp
 * @brief Unified testing framework for validating mango-option against QuantLib
 *
 * Provides generic testing utilities for:
 * - American option pricing accuracy
 * - Implied volatility accuracy
 * - Greeks accuracy
 * - Batch processing validation
 *
 * Supports multiple mango-option solvers:
 * - FDM-based pricing (auto-estimation)
 * - FDM-based IV (ground truth)
 * - Interpolated IV (B-spline, fast)
 */

#pragma once

#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include <gtest/gtest.h>
#include <ql/quantlib.hpp>
#include <vector>
#include <string>
#include <functional>
#include <memory_resource>

namespace mango::testing {

namespace ql = QuantLib;

// ============================================================================
// QuantLib Reference Implementation
// ============================================================================

struct GreeksResult {
    double price;
    double delta;
    double gamma;
    double theta;
};

inline GreeksResult price_with_quantlib(
    double spot,
    double strike,
    double maturity,
    double volatility,
    double rate,
    double dividend_yield,
    bool is_call,
    size_t grid_steps = 201,
    size_t time_steps = 2000)
{
    // Setup QuantLib environment
    ql::Date today = ql::Date::todaysDate();
    ql::Settings::instance().evaluationDate() = today;

    // Option parameters
    ql::Option::Type option_type = is_call ? ql::Option::Call : ql::Option::Put;
    ql::Date maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

    ql::ext::shared_ptr<ql::Exercise> exercise =
        ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    ql::ext::shared_ptr<ql::StrikedTypePayoff> payoff =
        ql::ext::make_shared<ql::PlainVanillaPayoff>(option_type, strike);

    ql::VanillaOption american_option(payoff, exercise);

    // Market data
    ql::Handle<ql::Quote> spot_handle(ql::ext::make_shared<ql::SimpleQuote>(spot));
    ql::Handle<ql::YieldTermStructure> rate_ts(
        ql::ext::make_shared<ql::FlatForward>(today, rate, ql::Actual365Fixed()));
    ql::Handle<ql::YieldTermStructure> div_ts(
        ql::ext::make_shared<ql::FlatForward>(today, dividend_yield, ql::Actual365Fixed()));
    ql::Handle<ql::BlackVolTermStructure> vol_ts(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), volatility, ql::Actual365Fixed()));

    ql::ext::shared_ptr<ql::BlackScholesMertonProcess> bs_process =
        ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_handle, div_ts, rate_ts, vol_ts);

    // Finite difference pricing engine
    american_option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(
            bs_process, time_steps, grid_steps));

    GreeksResult result;
    result.price = american_option.NPV();
    result.delta = american_option.delta();
    result.gamma = american_option.gamma();
    result.theta = american_option.theta();

    return result;
}

// ============================================================================
// Test Scenario Specification
// ============================================================================

struct OptionTestScenario {
    std::string name;
    double spot;
    double strike;
    double maturity;
    double volatility;
    double rate;
    double dividend_yield;
    bool is_call;

    // Tolerance settings
    double price_tolerance_pct = 1.0;
    double greeks_tolerance_pct = 2.0;
    double iv_tolerance_pct = 2.0;
};

// ============================================================================
// Mango-Option Pricing Validation
// ============================================================================

struct PricingValidationResult {
    bool passed;
    double mango_price;
    double ql_price;
    double abs_error;
    double rel_error_pct;
    std::string failure_message;
};

inline PricingValidationResult validate_pricing(
    const OptionTestScenario& scenario)
{
    SCOPED_TRACE("Pricing: " + scenario.name);

    PricingValidationResult validation;

    // Mango-Option pricing with auto-estimation
    PricingParams mango_params(
        OptionSpec{.spot = scenario.spot, .strike = scenario.strike,
            .maturity = scenario.maturity, .rate = scenario.rate,
            .dividend_yield = scenario.dividend_yield,
            .option_type = scenario.is_call ? OptionType::CALL : OptionType::PUT},
        scenario.volatility);

    auto [grid_spec, time_domain] = estimate_pde_grid(mango_params);

    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result.has_value()) {
        validation.passed = false;
        validation.failure_message = "Failed to create workspace: " + workspace_result.error();
        return validation;
    }

    auto solver = AmericanOptionSolver::create(mango_params, workspace_result.value()).value();
    auto mango_result = solver.solve();
    if (!mango_result.has_value()) {
        validation.passed = false;
        validation.failure_message = "Solver failed with error code " +
            std::to_string(static_cast<int>(mango_result.error().code));
        return validation;
    }

    // QuantLib reference
    auto ql_result = price_with_quantlib(
        scenario.spot, scenario.strike, scenario.maturity, scenario.volatility,
        scenario.rate, scenario.dividend_yield, scenario.is_call,
        201, 2000);

    // Compare
    validation.mango_price = mango_result->value_at(scenario.spot);
    validation.ql_price = ql_result.price;
    validation.abs_error = std::abs(validation.mango_price - validation.ql_price);
    validation.rel_error_pct = (validation.abs_error / validation.ql_price) * 100.0;

    validation.passed = (validation.rel_error_pct < scenario.price_tolerance_pct);
    if (!validation.passed) {
        validation.failure_message =
            "Price error " + std::to_string(validation.rel_error_pct) + "% > " +
            std::to_string(scenario.price_tolerance_pct) + "%";
    }

    return validation;
}

// ============================================================================
// Mango-Option Implied Volatility Validation (FDM)
// ============================================================================

struct IVValidationResult {
    bool passed;
    double true_vol;
    double recovered_vol;
    double abs_error;
    double rel_error_pct;
    int iterations;
    std::string failure_message;
};

inline IVValidationResult validate_iv_fdm(
    const OptionTestScenario& scenario)
{
    SCOPED_TRACE("IV (FDM): " + scenario.name);

    IVValidationResult validation;
    validation.true_vol = scenario.volatility;

    // Get market price from QuantLib
    auto ql_result = price_with_quantlib(
        scenario.spot, scenario.strike, scenario.maturity, scenario.volatility,
        scenario.rate, scenario.dividend_yield, scenario.is_call,
        201, 2000);

    // Solve for IV using FDM
    IVQuery query(
        OptionSpec{.spot = scenario.spot, .strike = scenario.strike,
            .maturity = scenario.maturity, .rate = scenario.rate,
            .dividend_yield = scenario.dividend_yield,
            .option_type = scenario.is_call ? OptionType::CALL : OptionType::PUT},
        ql_result.price);

    IVSolverConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    IVSolver solver(config);
    auto iv_result = solver.solve(query);

    if (!iv_result.has_value()) {
        validation.passed = false;
        validation.failure_message = "IV solver failed with error code " +
            std::to_string(static_cast<int>(iv_result.error().code));
        return validation;
    }

    validation.recovered_vol = iv_result->implied_vol;
    validation.iterations = iv_result->iterations;
    validation.abs_error = std::abs(validation.recovered_vol - validation.true_vol);
    validation.rel_error_pct = (validation.abs_error / validation.true_vol) * 100.0;

    validation.passed = (validation.rel_error_pct < scenario.iv_tolerance_pct);
    if (!validation.passed) {
        validation.failure_message =
            "IV error " + std::to_string(validation.rel_error_pct) + "% > " +
            std::to_string(scenario.iv_tolerance_pct) + "%";
    }

    return validation;
}

// ============================================================================
// Batch Testing Framework
// ============================================================================

struct BatchValidationSummary {
    size_t total_tests;
    size_t pricing_passed;
    size_t iv_fdm_passed;

    bool all_passed() const {
        return (pricing_passed == total_tests) && (iv_fdm_passed == total_tests);
    }

    void print_summary() const {
        std::cout << "\n=== Batch Validation Summary ===\n";
        std::cout << "Total scenarios: " << total_tests << "\n";
        std::cout << "Pricing tests passed: " << pricing_passed << "/" << total_tests << "\n";
        std::cout << "IV (FDM) tests passed: " << iv_fdm_passed << "/" << total_tests << "\n";
        if (all_passed()) {
            std::cout << "✓ All tests PASSED\n";
        } else {
            std::cout << "✗ Some tests FAILED\n";
        }
    }
};

inline BatchValidationSummary validate_batch(
    const std::vector<OptionTestScenario>& scenarios,
    bool test_pricing = true,
    bool test_iv_fdm = true)
{
    BatchValidationSummary summary{.total_tests = scenarios.size()};

    for (const auto& scenario : scenarios) {
        if (test_pricing) {
            auto pricing_result = validate_pricing(scenario);
            EXPECT_TRUE(pricing_result.passed)
                << scenario.name << ": " << pricing_result.failure_message;
            if (pricing_result.passed) {
                summary.pricing_passed++;
            }
        }

        if (test_iv_fdm) {
            auto iv_result = validate_iv_fdm(scenario);
            EXPECT_TRUE(iv_result.passed)
                << scenario.name << ": " << iv_result.failure_message;
            if (iv_result.passed) {
                summary.iv_fdm_passed++;
            }
        }
    }

    return summary;
}

// ============================================================================
// Standard Test Scenarios
// ============================================================================

inline std::vector<OptionTestScenario> get_standard_test_scenarios() {
    return {
        {"ATM Put 1Y", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, false},
        {"OTM Put 3M", 110.0, 100.0, 0.25, 0.30, 0.05, 0.02, false},
        {"ITM Put 2Y", 90.0, 100.0, 2.0, 0.25, 0.05, 0.02, false},
        {"ATM Call 1Y", 100.0, 100.0, 1.0, 0.20, 0.05, 0.02, true},
        {"Deep ITM Put 6M", 80.0, 100.0, 0.5, 0.25, 0.05, 0.02, false},
        {"High Vol Put 1Y", 100.0, 100.0, 1.0, 0.50, 0.05, 0.02, false},
        {"Low Vol Put 1Y", 100.0, 100.0, 1.0, 0.10, 0.05, 0.02, false},
        {"Long Maturity Put 5Y", 100.0, 100.0, 5.0, 0.20, 0.05, 0.02, false},
    };
}

}  // namespace mango::testing
