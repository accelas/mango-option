#include "iv_solver.hpp"
#include "brent.hpp"
#include <cmath>
#include <algorithm>

extern "C" {
#include "src/american_option.h"
}

namespace mango {

IVSolver::IVSolver(const IVParams& params, const IVConfig& config)
    : params_(params), config_(config) {
    // Constructor - just stores parameters
    // Validation happens in solve()
}

std::optional<std::string> IVSolver::validate_params() const {
    // Validate spot price
    if (params_.spot_price <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 1, params_.spot_price, 0.0);
        return "Spot price must be positive";
    }

    // Validate strike price
    if (params_.strike <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 2, params_.strike, 0.0);
        return "Strike price must be positive";
    }

    // Validate time to maturity
    if (params_.time_to_maturity <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 3, params_.time_to_maturity, 0.0);
        return "Time to maturity must be positive";
    }

    // Validate market price
    if (params_.market_price <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 4, params_.market_price, 0.0);
        return "Market price must be positive";
    }

    // Check arbitrage bounds
    double intrinsic_value;
    if (params_.is_call) {
        intrinsic_value = std::max(params_.spot_price - params_.strike, 0.0);
        if (params_.market_price > params_.spot_price) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, params_.spot_price);
            return "Call price exceeds spot price (arbitrage)";
        }
    } else {
        intrinsic_value = std::max(params_.strike - params_.spot_price, 0.0);
        if (params_.market_price > params_.strike) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, params_.strike);
            return "Put price exceeds strike (arbitrage)";
        }
    }

    // Market price should be at least intrinsic value (with small tolerance)
    const double tolerance = 1e-6;
    if (params_.market_price < intrinsic_value - tolerance) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 5, params_.market_price, intrinsic_value);
        return "Market price below intrinsic value (arbitrage)";
    }

    return std::nullopt;  // All validations passed
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
    // Create OptionData structure for American option solver
    OptionData option_data = {
        .strike = params_.strike,
        .volatility = volatility,
        .risk_free_rate = params_.risk_free_rate,
        .time_to_maturity = params_.time_to_maturity,
        .option_type = params_.is_call ? OPTION_CALL : OPTION_PUT,
        .n_dividends = 0,
        .dividend_times = nullptr,
        .dividend_amounts = nullptr
    };

    // Create grid for PDE solver
    size_t n_grid;
    AmericanOptionGrid grid_params = {
        .x_min = std::log(0.5),  // ln(0.5)
        .x_max = std::log(2.0),  // ln(2.0)
        .n_points = config_.grid_n_space,
        .dt = params_.time_to_maturity / config_.grid_n_time,
        .n_steps = config_.grid_n_time
    };

    double* m_grid = american_option_create_grid(&grid_params, &n_grid);
    if (!m_grid) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Solve American option PDE
    AmericanOptionResult result = american_option_solve(
        &option_data,
        m_grid,
        n_grid,
        grid_params.dt,
        grid_params.n_steps
    );

    free(m_grid);

    if (result.status != 0 || !result.solver) {
        american_option_free_result(&result);
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Get option value at current spot price
    double theoretical_price = american_option_get_value_at_spot(
        result.solver,
        params_.spot_price,
        params_.strike
    );

    // Clean up
    american_option_free_result(&result);

    // Return difference: V(σ) - V_market
    return theoretical_price - params_.market_price;
}

IVResult IVSolver::solve() {
    // Trace calculation start
    MANGO_TRACE_ALGO_START(MODULE_IMPLIED_VOL,
                          static_cast<double>(config_.root_config.max_iter),
                          config_.root_config.tolerance,
                          0.0);

    // Validate input parameters
    auto validation_error = validate_params();
    if (validation_error.has_value()) {
        return IVResult{
            .converged = false,
            .iterations = 0,
            .implied_vol = 0.0,
            .final_error = 0.0,
            .failure_reason = validation_error,
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
        .failure_reason = root_result.failure_reason,
        .vega = std::nullopt  // Could be computed but not required for basic IV
    };
}

}  // namespace mango
