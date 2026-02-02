// SPDX-License-Identifier: MIT
#include "src/option/iv_solver_fdm.hpp"
#include "src/math/root_finding.hpp"
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/support/parallel.hpp"
#include "src/support/ivcalc_trace.h"
#include <cmath>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <variant>

namespace mango {

IVSolverFDM::IVSolverFDM(const IVSolverFDMConfig& config)
    : config_(config) {
    // Constructor - just stores configuration
}

double IVSolverFDM::estimate_upper_bound(const IVQuery& query) const {
    // For American options, use intrinsic value approximation
    // Upper bound based on the relationship: V_market ≈ Intrinsic + Time Value
    // For deep ITM options, time value is small, so high vol is unlikely

    double intrinsic_value;
    if (query.option_type == OptionType::CALL) {
        intrinsic_value = std::max(query.spot - query.strike, 0.0);
    } else {
        intrinsic_value = std::max(query.strike - query.spot, 0.0);
    }

    // Time value = Market Price - Intrinsic Value
    double time_value = query.market_price - intrinsic_value;

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

double IVSolverFDM::estimate_lower_bound() const {
    // Lower bound: typically 1% volatility
    // No asset has zero volatility, and very low vol is rare
    return 0.01;  // 1%
}

double IVSolverFDM::objective_function(const IVQuery& query, double volatility) const {
    // Create American option parameters
    PricingParams option_params;
    option_params.strike = query.strike;
    option_params.spot = query.spot;
    option_params.maturity = query.maturity;
    option_params.volatility = volatility;
    option_params.rate = query.rate;
    option_params.dividend_yield = query.dividend_yield;
    option_params.option_type = query.option_type;

    // Resolve grid from config
    GridSpec<double> grid_spec = GridSpec<double>::uniform(0.0, 1.0, 10).value();  // Will be replaced
    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, query.maturity, 100);

    std::visit([&](const auto& grid_variant) {
        using T = std::decay_t<decltype(grid_variant)>;
        if constexpr (std::is_same_v<T, GridAccuracyParams>) {
            auto [auto_grid, auto_td] = estimate_grid_for_option(option_params, grid_variant);
            grid_spec = auto_grid;
            time_domain = auto_td;
        } else if constexpr (std::is_same_v<T, PDEGridConfig>) {
            grid_spec = grid_variant.grid_spec;
            if (grid_variant.mandatory_times.empty()) {
                time_domain = TimeDomain::from_n_steps(0.0, query.maturity, grid_variant.n_time);
            } else {
                time_domain = TimeDomain::with_mandatory_points(0.0, query.maturity,
                    query.maturity / static_cast<double>(grid_variant.n_time),
                    grid_variant.mandatory_times);
            }
        }
    }, config_.grid);

    // Single grow-only buffer per thread — avoids unordered_map overhead.
    // Grid size may vary across Brent iterations (volatility changes grid estimation)
    // but a single buffer that grows to the max is simpler and faster.
    thread_local std::vector<double> workspace_buffer;

    size_t n = grid_spec.n_points();
    size_t required_size = PDEWorkspace::required_size(n);

    if (workspace_buffer.size() < required_size) {
        workspace_buffer.resize(required_size);
    }

    auto pde_workspace_result = PDEWorkspace::from_buffer(
        std::span<double>(workspace_buffer.data(), workspace_buffer.size()), n);
    if (!pde_workspace_result.has_value()) {
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            // error code set above + pde_workspace_result.error(),
            .iterations = 0
        };
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Create solver and solve — always pass grid config to ensure the solver
    // uses the same grid we computed (matching the workspace size)
    auto explicit_grid = PDEGridConfig{grid_spec, time_domain.n_steps(), {}};

    auto solver_result = AmericanOptionSolver::create(
        option_params, pde_workspace_result.value(), PDEGridSpec{explicit_grid});
    if (!solver_result) {
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .iterations = 0
        };
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto& solver = solver_result.value();

    // Surface always collected for value_at()
    auto price_result = solver.solve();

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
IVSolverFDM::validate_query(const IVQuery& query) const {
    // Use centralized IV query validation (option spec + market price + arbitrage)
    auto validation = validate_iv_query(query);
    if (!validation.has_value()) {
        return std::unexpected(validation_error_to_iv_error(validation.error()));
    }

    return std::monostate{};
}

std::expected<IVSuccess, IVError>
IVSolverFDM::solve_brent(const IVQuery& query) const {
    // Adaptive bounds logic
    double intrinsic = (query.option_type == OptionType::CALL)
        ? std::max(0.0, query.spot - query.strike)
        : std::max(0.0, query.strike - query.spot);

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

    // Objective function
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

std::expected<IVSuccess, IVError> IVSolverFDM::solve(const IVQuery& query) const {
    // C++23 monadic validation pipeline: validate → solve
    return validate_query(query)
        .and_then([this, &query](auto) { return solve_brent(query); });
}

BatchIVResult IVSolverFDM::solve_batch(const std::vector<IVQuery>& queries) const {
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
        // Mirrors IVSolverInterpolated::solve_batch pattern
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
