// SPDX-License-Identifier: MIT
#include "mango/option/iv_solver.hpp"
#include "mango/math/root_finding.hpp"
#include "mango/option/american_option.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/support/parallel.hpp"
#include "mango/support/ivcalc_trace.h"
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <variant>

namespace mango {

IVSolver::IVSolver(const IVSolverConfig& config)
    : config_(config) {
    // Constructor - just stores configuration
}

double IVSolver::estimate_upper_bound(const IVQuery& query) const {
    // For American options, use intrinsic value approximation
    // Upper bound based on the relationship: V_market ≈ Intrinsic + Time Value
    // For deep ITM options, time value is small, so high vol is unlikely

    double intrinsic_val = intrinsic_value(query.spot, query.strike, query.option_type);

    // Time value = Market Price - Intrinsic Value
    double time_value = query.market_price - intrinsic_val;

    // For ATM/OTM options (high time value), use higher upper bound
    // For ITM options (low time value), use lower upper bound
    if (time_value > query.market_price * 0.5) {
        // High time value suggests moderate to high volatility
        return 3.0;  // 300% volatility
    } else if (time_value > query.market_price * 0.2) {
        return 2.0;  // 200% volatility
    } else {
        return 1.5;  // 150% volatility for deep ITM
    }
}

double IVSolver::estimate_lower_bound() const {
    // Lower bound: typically 1% volatility
    // No asset has zero volatility, and very low vol is rare
    return 0.01;  // 1%
}

double IVSolver::objective_function(const IVQuery& query, double volatility) const {
    // Create American option parameters
    PricingParams option_params;
    option_params.strike = query.strike;
    option_params.spot = query.spot;
    option_params.maturity = query.maturity;
    option_params.volatility = volatility;
    option_params.rate = query.rate;
    option_params.dividend_yield = query.dividend_yield;
    option_params.option_type = query.option_type;
    option_params.discrete_dividends = query.discrete_dividends;

    auto solver = AmericanOptionSolver::create(option_params, config_.grid);
    if (!solver) {
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::InvalidConfiguration, .iterations = 0};
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto price_result = solver->solve();

    if (!price_result) {
        last_solver_error_ = price_result.error();
        return std::numeric_limits<double>::quiet_NaN();
    }

    last_solver_error_.reset();
    const AmericanOptionResult& result = price_result.value();

    // Return difference: V(σ) - V_market
    return result.value_at(query.spot) - query.market_price;
}

// Validate query using centralized validation
std::expected<std::monostate, IVError>
IVSolver::validate_query(const IVQuery& query) const {
    // Use centralized IV query validation (option spec + market price + arbitrage)
    auto validation = validate_iv_query(query);
    if (!validation.has_value()) {
        return std::unexpected(validation_error_to_iv_error(validation.error()));
    }

    if (const auto* accuracy = std::get_if<GridAccuracyParams>(&config_.grid)) {
        auto valid = validate_grid_accuracy(*accuracy);
        if (!valid) return std::unexpected(IVError{
            .code = IVErrorCode::InvalidGridConfig,
            .final_error = valid.error().value,
            .last_vol = std::nullopt});
    }
    auto rate_validation = validate_pde_rate(query.rate, query.maturity);
    if (!rate_validation) {
        return std::unexpected(validation_error_to_iv_error(rate_validation.error()));
    }

    return std::monostate{};
}

std::expected<IVSuccess, IVError>
IVSolver::solve_brent(const IVQuery& query) const {
    // Adaptive bounds logic
    double intrinsic = intrinsic_value(query.spot, query.strike, query.option_type);

    double time_value = query.market_price - intrinsic;
    double time_value_ratio = time_value / query.market_price;

    double vol_upper;
    if (time_value_ratio > 0.5) {
        vol_upper = 3.0;
    } else if (time_value_ratio > 0.2) {
        vol_upper = 2.0;
    } else {
        vol_upper = 1.5;
    }

    double vol_lower = 0.01;

    // Objective function for root-finding
    auto objective = [this, &query](double vol) -> double {
        return this->objective_function(query, vol);
    };

    // Run Brent
    auto brent_result = brent_find_root(objective, vol_lower, vol_upper, config_.root_config);

    // Transform result: map both success and error types
    if (!brent_result.has_value()) {
        // Map RootFindingError to IVError
        const auto& root_error = brent_result.error();
        IVErrorCode error_code;
        switch (root_error.code) {
            case RootFindingErrorCode::MaxIterationsExceeded:
                error_code = IVErrorCode::MaxIterationsExceeded;
                break;
            case RootFindingErrorCode::InvalidBracket:
                error_code = IVErrorCode::BracketingFailed;
                break;
            case RootFindingErrorCode::NumericalInstability:
                error_code = IVErrorCode::NumericalInstability;
                break;
            case RootFindingErrorCode::NoProgress:
                error_code = IVErrorCode::NumericalInstability;
                break;
            default:
                error_code = IVErrorCode::NumericalInstability;
                break;
        }

        return std::unexpected(IVError{
            .code = error_code,
            .iterations = root_error.iterations,
            .final_error = root_error.final_error,
            .last_vol = root_error.last_value
        });
    }

    // Transform RootFindingSuccess to IVSuccess
    return IVSuccess{
        .implied_vol = brent_result->root,
        .iterations = brent_result->iterations,
        .final_error = brent_result->final_error,
        .vega = std::nullopt
    };
}

std::expected<IVSuccess, IVError> IVSolver::solve(const IVQuery& query) const {
    // Validate first
    auto validation = validate_query(query);
    if (!validation.has_value()) {
        return std::unexpected(validation.error());
    }

    // Run Brent solver (validation already done above)
    return solve_brent(query);
}

BatchIVResult IVSolver::solve_batch(const std::vector<IVQuery>& queries) const {
    std::vector<std::expected<IVSuccess, IVError>> results(queries.size());
    size_t failed_count = 0;

    // Use configured parallelization threshold to balance overhead vs parallelism
    // IV solves are expensive (multiple PDE solves), but small batches pay parallel tax
    if (queries.size() < config_.batch_parallel_threshold) {
        // Serial path for batches below threshold (avoid parallel overhead)
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = solve(queries[i]);
            if (!results[i].has_value()) {
                ++failed_count;
            }
        }
    } else {
        // Parallel path: each IV solve is independent (different PDE workspaces)
        // Mirrors InterpolatedIVSolver::solve_batch pattern
        MANGO_PRAGMA_PARALLEL_FOR
        for (size_t i = 0; i < queries.size(); ++i) {
            results[i] = solve(queries[i]);
            if (!results[i].has_value()) {
                MANGO_PRAGMA_ATOMIC
                ++failed_count;
            }
        }
    }

    return BatchIVResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

} // namespace mango
