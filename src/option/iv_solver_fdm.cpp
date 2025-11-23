#include "src/option/iv_solver_fdm.hpp"
#include "src/math/root_finding.hpp"
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/support/parallel.hpp"
#include "common/ivcalc_trace.h"
#include <cmath>
#include <algorithm>
#include <memory>
#include <memory_resource>

namespace mango {

IVSolverFDM::IVSolverFDM(const IVSolverFDMConfig& config)
    : config_(config) {
    // Constructor - just stores configuration
}

std::expected<void, std::string> IVSolverFDM::validate_query(const IVQuery& query) const {
    // Use common validation for option spec, market price, and arbitrage checks
    auto common_validation = validate_iv_query(query);
    if (!common_validation) {
        // Trace validation error
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 3, 0.0, 0.0);
        return common_validation;
    }

    // FDM-specific validation: grid parameters (only when manual mode enabled)
    if (config_.use_manual_grid) {
        if (config_.grid_n_space == 0) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 6, config_.grid_n_space, 0.0);
            return std::unexpected(std::string("Manual grid: n_space must be positive"));
        }

        if (config_.grid_n_time == 0) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 7, config_.grid_n_time, 0.0);
            return std::unexpected(std::string("Manual grid: n_time must be positive"));
        }

        if (config_.grid_x_min >= config_.grid_x_max) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 9, config_.grid_x_min, config_.grid_x_max);
            return std::unexpected(std::string("Manual grid: x_min must be < x_max"));
        }

        if (config_.grid_alpha < 0.0) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 10, config_.grid_alpha, 0.0);
            return std::unexpected(std::string("Manual grid: alpha must be non-negative"));
        }
    }

    return {};
}

double IVSolverFDM::estimate_upper_bound(const IVQuery& query) const {
    // For American options, use intrinsic value approximation
    // Upper bound based on the relationship: V_market ≈ Intrinsic + Time Value
    // For deep ITM options, time value is small, so high vol is unlikely

    double intrinsic_value;
    if (query.type == OptionType::CALL) {
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
    AmericanOptionParams option_params;
    option_params.strike = query.strike;
    option_params.spot = query.spot;
    option_params.maturity = query.maturity;
    option_params.volatility = volatility;
    option_params.rate = query.rate;
    option_params.dividend_yield = query.dividend_yield;
    option_params.type = query.type;

    // Choose grid: manual override or automatic estimation
    auto dummy_grid = GridSpec<double>::uniform(0.0, 1.0, 10);
    GridSpec<double> grid_spec = dummy_grid.value();  // Dummy init, will be replaced

    if (config_.use_manual_grid) {
        // Advanced mode: Use manually specified grid (for benchmarks)
        auto grid_result = GridSpec<double>::sinh_spaced(
            config_.grid_x_min,
            config_.grid_x_max,
            config_.grid_n_space,
            config_.grid_alpha
        );
        if (!grid_result.has_value()) {
            last_solver_error_ = SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Invalid manual grid: " + grid_result.error(),
                .iterations = 0
            };
            return std::numeric_limits<double>::quiet_NaN();
        }
        grid_spec = grid_result.value();
        // Note: config_.grid_n_time is not used here as AmericanOptionSolver
        // determines time stepping internally based on grid and params
    } else {
        // Default mode: Use automatic grid estimation
        auto [auto_grid, auto_nt] = estimate_grid_for_option(option_params);
        grid_spec = auto_grid;
        // Note: auto_nt (n_time) is not used as AmericanOptionSolver
        // handles time stepping internally
        (void)auto_nt;  // Suppress unused warning
    }

    // Allocate workspace buffer (local, temporary)
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), std::pmr::get_default_resource());

    auto pde_workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!pde_workspace_result.has_value()) {
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Invalid PDEWorkspace configuration: " + pde_workspace_result.error(),
            .iterations = 0
        };
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Create solver and solve
    try {
        // Pass custom grid if manual mode is enabled
        std::optional<GridSpec<double>> custom_grid_opt = std::nullopt;
        std::optional<size_t> custom_n_time_opt = std::nullopt;

        if (config_.use_manual_grid) {
            custom_grid_opt = grid_spec;
            custom_n_time_opt = config_.grid_n_time;
        }

        AmericanOptionSolver solver(option_params, pde_workspace_result.value(),
                                    std::nullopt,  // snapshot_times
                                    custom_grid_opt,
                                    custom_n_time_opt);
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
    } catch (...) {
        // If solver throws an exception, capture the error and return NaN
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::Unknown,
            .message = "AmericanOptionSolver threw during objective evaluation",
            .iterations = 0
        };
        return std::numeric_limits<double>::quiet_NaN();
    }
}

// Atomic validators (uniform API - all take const IVQuery&)
std::expected<std::monostate, IVError>
IVSolverFDM::validate_spot_positive(const IVQuery& query) const {
    if (query.spot <= 0.0) {
        return std::unexpected(IVError{
            .code = IVErrorCode::NegativeSpot,
            .message = "Spot price must be positive",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_strike_positive(const IVQuery& query) const {
    if (query.strike <= 0.0) {
        return std::unexpected(IVError{
            .code = IVErrorCode::NegativeStrike,
            .message = "Strike price must be positive",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_maturity_positive(const IVQuery& query) const {
    if (query.maturity <= 0.0) {
        return std::unexpected(IVError{
            .code = IVErrorCode::NegativeMaturity,
            .message = "Time to maturity must be positive",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_price_positive(const IVQuery& query) const {
    if (query.market_price <= 0.0) {
        return std::unexpected(IVError{
            .code = IVErrorCode::NegativeMarketPrice,
            .message = "Market price must be positive",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_call_price_bound(const IVQuery& query) const {
    if (query.type == OptionType::CALL && query.market_price > query.spot) {
        return std::unexpected(IVError{
            .code = IVErrorCode::ArbitrageViolation,
            .message = "Call price cannot exceed spot price (arbitrage)",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_put_price_bound(const IVQuery& query) const {
    if (query.type == OptionType::PUT && query.market_price > query.strike) {
        return std::unexpected(IVError{
            .code = IVErrorCode::ArbitrageViolation,
            .message = "Put price cannot exceed strike price (arbitrage)",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_intrinsic_value(const IVQuery& query) const {
    double intrinsic = (query.type == OptionType::CALL)
        ? std::max(0.0, query.spot - query.strike)
        : std::max(0.0, query.strike - query.spot);

    if (query.market_price < intrinsic) {
        return std::unexpected(IVError{
            .code = IVErrorCode::ArbitrageViolation,
            .message = "Market price below intrinsic value (arbitrage)",
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }
    return std::monostate{};
}

// Composite validators (monadic chains)
std::expected<std::monostate, IVError>
IVSolverFDM::validate_positive_parameters(const IVQuery& query) const {
    return validate_spot_positive(query)
        .and_then([this, &query](auto) { return validate_strike_positive(query); })
        .and_then([this, &query](auto) { return validate_maturity_positive(query); })
        .and_then([this, &query](auto) { return validate_price_positive(query); });
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_arbitrage_bounds(const IVQuery& query) const {
    return validate_call_price_bound(query)
        .and_then([this, &query](auto) { return validate_put_price_bound(query); })
        .and_then([this, &query](auto) { return validate_intrinsic_value(query); });
}

std::expected<std::monostate, IVError>
IVSolverFDM::validate_query_monadic(const IVQuery& query) const {
    return validate_positive_parameters(query)
        .and_then([this, &query](auto) { return validate_arbitrage_bounds(query); });
}

std::expected<IVSuccess, IVError>
IVSolverFDM::solve_brent(const IVQuery& query) const {
    // Adaptive bounds logic
    double intrinsic = (query.type == OptionType::CALL)
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

    // Check convergence
    if (!brent_result.converged) {
        IVErrorCode error_code;
        if (brent_result.failure_reason.has_value()) {
            const std::string& reason = brent_result.failure_reason.value();
            if (reason.find("Max iterations") != std::string::npos) {
                error_code = IVErrorCode::MaxIterationsExceeded;
            } else if (reason.find("not bracketed") != std::string::npos) {
                error_code = IVErrorCode::BracketingFailed;
            } else {
                error_code = IVErrorCode::MaxIterationsExceeded;
            }
        } else {
            error_code = IVErrorCode::MaxIterationsExceeded;
        }

        return std::unexpected(IVError{
            .code = error_code,
            .message = brent_result.failure_reason.value_or("Brent solver failed"),
            .iterations = brent_result.iterations,
            .final_error = brent_result.final_error,
            .last_vol = brent_result.root
        });
    }

    // Success
    return IVSuccess{
        .implied_vol = brent_result.root.value(),
        .iterations = brent_result.iterations,
        .final_error = brent_result.final_error,
        .vega = std::nullopt
    };
}

std::expected<IVSuccess, IVError> IVSolverFDM::solve_impl(const IVQuery& query) {
    // C++23 monadic validation pipeline: validate → solve
    return validate_query_monadic(query)
        .and_then([this, &query](auto) { return solve_brent(query); });
}

void IVSolverFDM::solve_batch_impl(std::span<const IVQuery> queries,
                                    std::span<IVResult> results) {
    // NOTE: This implementation is temporarily disabled as solve_impl now returns
    // std::expected<IVSuccess, IVError> instead of IVResult.
    // This will be updated in Task 5.1 (Update batch solver).
    (void)queries;
    (void)results;

    // TODO(Task 5.1): Update this to convert std::expected results to IVResult
}

} // namespace mango
