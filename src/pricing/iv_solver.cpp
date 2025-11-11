#include "src/pricing/iv_solver.hpp"
#include "src/core/root_finding.hpp"
#include "src/pricing/american_option.hpp"
#include <cmath>
#include <algorithm>

namespace mango {

IVSolver::IVSolver(const IVParams& params, const IVConfig& config)
    : params_(params), config_(config) {
    // Constructor - just stores parameters
    // Validation happens in solve()
}

expected<void, std::string> IVSolver::validate_params() const {
    // Validate spot price
    if (params_.spot_price <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 1, params_.spot_price, 0.0);
        return unexpected(std::string("Spot price must be positive"));
    }

    // Validate strike price
    if (params_.strike <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 2, params_.strike, 0.0);
        return unexpected(std::string("Strike price must be positive"));
    }

    // Validate time to maturity
    if (params_.time_to_maturity <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 3, params_.time_to_maturity, 0.0);
        return unexpected(std::string("Time to maturity must be positive"));
    }

    // Validate market price
    if (params_.market_price <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 4, params_.market_price, 0.0);
        return unexpected(std::string("Market price must be positive"));
    }

    // Validate grid parameters
    if (config_.grid_n_space == 0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 6, config_.grid_n_space, 0.0);
        return unexpected(std::string("Grid n_space must be positive"));
    }

    if (config_.grid_n_time == 0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 7, config_.grid_n_time, 0.0);
        return unexpected(std::string("Grid n_time must be positive"));
    }

    if (config_.grid_s_max <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 8, config_.grid_s_max, 0.0);
        return unexpected(std::string("Grid s_max must be positive"));
    }

    // Check arbitrage bounds
    double intrinsic_value;
    if (params_.is_call) {
        intrinsic_value = std::max(params_.spot_price - params_.strike, 0.0);
        if (params_.market_price > params_.spot_price) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, params_.spot_price);
            return unexpected(std::string("Call price exceeds spot price (arbitrage)"));
        }
    } else {
        intrinsic_value = std::max(params_.strike - params_.spot_price, 0.0);
        if (params_.market_price > params_.strike) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, params_.strike);
            return unexpected(std::string("Put price exceeds strike (arbitrage)"));
        }
    }

    // Market price should be at least intrinsic value (with small tolerance)
    const double tolerance = 1e-6;
    if (params_.market_price < intrinsic_value - tolerance) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, intrinsic_value);
        return unexpected(std::string("Market price below intrinsic value (arbitrage)"));
    }

    return {};
}

double IVSolver::estimate_upper_bound() const {
    // For American options, use intrinsic value approximation
    // Upper bound based on the relationship: V_market ≈ Intrinsic + Time Value
    // For deep ITM options, time value is small, so high vol is unlikely

    double intrinsic_value;
    if (params_.is_call) {
        intrinsic_value = std::max(params_.spot_price - params_.strike, 0.0);
    } else {
        intrinsic_value = std::max(params_.strike - params_.spot_price, 0.0);
    }

    // Time value = Market Price - Intrinsic Value
    double time_value = params_.market_price - intrinsic_value;

    // For ATM/OTM options (high time value), use higher upper bound
    // For ITM options (low time value), use lower upper bound
    if (time_value > params_.market_price * 0.5) {
        // High time value suggests moderate to high volatility
        return 3.0;  // 300% volatility
    } else if (time_value > params_.market_price * 0.2) {
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

double IVSolver::objective_function(double volatility) const {
    // Create American option parameters
    AmericanOptionParams option_params;
    option_params.strike = params_.strike;
    option_params.spot = params_.spot_price;
    option_params.maturity = params_.time_to_maturity;
    option_params.volatility = volatility;
    option_params.rate = params_.risk_free_rate;
    option_params.continuous_dividend_yield = 0.0;  // No dividends for now
    option_params.option_type = params_.is_call ? OptionType::CALL : OptionType::PUT;

    // Compute adaptive grid bounds based on spot/strike and config.grid_s_max
    // The grid should:
    // 1. Always contain the spot price (critical for interpolation)
    // 2. Extend to at least grid_s_max (default 200.0)
    // 3. Use reasonable lower bound (0.5 * spot or smaller if needed)

    double moneyness = params_.spot_price / params_.strike;

    // Lower bound: ensure we capture deep ITM scenarios
    // Use smaller of: 0.5 * moneyness or 0.5 (whichever extends grid more)
    double min_moneyness = std::min(0.5, moneyness * 0.5);

    // Upper bound: ensure we capture deep OTM scenarios
    // Use larger of: 2.0 * moneyness or grid_s_max / strike (whichever extends grid more)
    double max_s = std::max(params_.strike * 2.0, config_.grid_s_max);
    double max_moneyness = std::max(2.0, max_s / params_.strike);

    // Ensure spot is within bounds (with margin for interpolation)
    min_moneyness = std::min(min_moneyness, moneyness * 0.9);
    max_moneyness = std::max(max_moneyness, moneyness * 1.1);

    // Create grid for PDE solver
    AmericanOptionGrid grid_params;
    grid_params.n_space = config_.grid_n_space;
    grid_params.n_time = config_.grid_n_time;
    grid_params.x_min = std::log(min_moneyness);  // Adaptive lower bound
    grid_params.x_max = std::log(max_moneyness);  // Adaptive upper bound

    // Create solver and solve
    try {
        AmericanOptionSolver solver(option_params, grid_params);
        auto price_result = solver.solve();

        if (!price_result) {
            last_solver_error_ = price_result.error();
            return std::numeric_limits<double>::quiet_NaN();
        }

        last_solver_error_.reset();
        const AmericanOptionResult& result = price_result.value();

        // Return difference: V(σ) - V_market
        return result.value - params_.market_price;
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

IVResult IVSolver::solve() {
    // Trace calculation start
    MANGO_TRACE_ALGO_START(MODULE_IMPLIED_VOL,
                          static_cast<double>(config_.root_config.max_iter),
                          config_.root_config.tolerance,
                          0.0);

    // Validate input parameters
    auto validation_result = validate_params();
    if (!validation_result) {
        return IVResult{
            .converged = false,
            .iterations = 0,
            .implied_vol = 0.0,
            .final_error = 0.0,
            .failure_reason = validation_result.error(),
            .vega = std::nullopt
        };
    }

    // Estimate adaptive bounds for volatility search
    double lower_bound = estimate_lower_bound();
    double upper_bound = estimate_upper_bound();

    // Create objective function lambda for Brent's method
    auto objective = [this](double vol) {
        return this->objective_function(vol);
    };

    // Reset last solver error before root-finding
    last_solver_error_.reset();

    // Use Brent's method to find the root
    RootFindingResult root_result = brent_find_root(
        objective,
        lower_bound,
        upper_bound,
        config_.root_config
    );

    // Emit completion trace
    if (root_result.converged) {
        MANGO_TRACE_ALGO_COMPLETE(MODULE_IMPLIED_VOL, root_result.iterations, 1);
    } else {
        MANGO_TRACE_CONVERGENCE_FAILED(MODULE_IMPLIED_VOL, 0, root_result.iterations, root_result.final_error);
    }

    // Convert RootFindingResult to IVResult
    return IVResult{
        .converged = root_result.converged,
        .iterations = root_result.iterations,
        .implied_vol = root_result.converged ? root_result.root.value() : 0.0,
        .final_error = root_result.final_error,
        .failure_reason = root_result.converged
            ? std::nullopt
            : (root_result.failure_reason
               ? root_result.failure_reason
               : (last_solver_error_
                  ? std::optional<std::string>(last_solver_error_->message)
                  : std::nullopt)),
        .vega = std::nullopt  // Could be computed but not required for basic IV
    };
}

}  // namespace mango
