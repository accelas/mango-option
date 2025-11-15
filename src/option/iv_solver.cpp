#include "src/option/iv_solver.hpp"
#include "src/math/root_finding.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

namespace mango {

IVSolver::IVSolver(const IVQuery& query, const IVConfig& config)
    : query_(query), config_(config) {
    // Constructor - just stores parameters
    // Validation happens in solve()
}

std::expected<void, std::string> validate_params_internal(
    const IVQuery& query,
    const IVConfig& config) {
    // Use unified validation from option_spec
    auto spec_validation = validate_iv_query(query);
    if (!spec_validation) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 0, 0.0, 0.0);
        return spec_validation;
    }

    // Validate grid parameters (config-specific)
    if (config.grid_n_space == 0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 6, config.grid_n_space, 0.0);
        return std::unexpected(std::string("Grid n_space must be positive"));
    }

    if (config.grid_n_time == 0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 7, config.grid_n_time, 0.0);
        return std::unexpected(std::string("Grid n_time must be positive"));
    }

    if (config.grid_s_max <= 0.0) {
        MANGO_TRACE_VALIDATION_ERROR(MODULE_IMPLIED_VOL, 8, config.grid_s_max, 0.0);
        return std::unexpected(std::string("Grid s_max must be positive"));
    }

    return {};
}

double IVSolver::estimate_upper_bound() const {
    // For American options, use intrinsic value approximation
    // Upper bound based on the relationship: V_market ≈ Intrinsic + Time Value
    // For deep ITM options, time value is small, so high vol is unlikely

    double intrinsic_value;
    if (query_.option.type == OptionType::CALL) {
        intrinsic_value = std::max(query_.option.spot - query_.option.strike, 0.0);
    } else {
        intrinsic_value = std::max(query_.option.strike - query_.option.spot, 0.0);
    }

    // Time value = Market Price - Intrinsic Value
    double time_value = query_.market_price - intrinsic_value;

    // For ATM/OTM options (high time value), use higher upper bound
    // For ITM options (low time value), use lower upper bound
    if (time_value > query_.market_price * 0.5) {
        // High time value suggests moderate to high volatility
        return 3.0;  // 300% volatility
    } else if (time_value > query_.market_price * 0.2) {
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
    option_params.strike = query_.option.strike;
    option_params.spot = query_.option.spot;
    option_params.maturity = query_.option.maturity;
    option_params.volatility = volatility;
    option_params.rate = query_.option.rate;
    option_params.continuous_dividend_yield = query_.option.dividend_yield;
    option_params.option_type = query_.option.type;

    // Compute adaptive grid bounds based on spot/strike and config.grid_s_max
    // The grid should:
    // 1. Always contain the spot price (critical for interpolation)
    // 2. Extend to at least grid_s_max (default 200.0)
    // 3. Use reasonable lower bound (0.5 * spot or smaller if needed)

    double moneyness = query_.option.spot / query_.option.strike;

    // Lower bound: ensure we capture deep ITM scenarios
    // Use smaller of: 0.5 * moneyness or 0.5 (whichever extends grid more)
    double min_moneyness = std::min(0.5, moneyness * 0.5);

    // Upper bound: ensure we capture deep OTM scenarios
    // Use larger of: 2.0 * moneyness or grid_s_max / strike (whichever extends grid more)
    double max_s = std::max(query_.option.strike * 2.0, config_.grid_s_max);
    double max_moneyness = std::max(2.0, max_s / query_.option.strike);

    // Ensure spot is within bounds (with margin for interpolation)
    min_moneyness = std::min(min_moneyness, moneyness * 0.9);
    max_moneyness = std::max(max_moneyness, moneyness * 1.1);

    // Create workspace for PDE solver
    double x_min = std::log(min_moneyness);  // Adaptive lower bound
    double x_max = std::log(max_moneyness);  // Adaptive upper bound
    auto workspace_result = AmericanSolverWorkspace::create(
        x_min, x_max, config_.grid_n_space, config_.grid_n_time);

    if (!workspace_result) {
        last_solver_error_ = SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Invalid workspace configuration: " + workspace_result.error(),
            .iterations = 0
        };
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Create solver and solve
    try {
        auto workspace = workspace_result.value();
        AmericanOptionSolver solver(option_params, workspace);
        auto price_result = solver.solve();

        if (!price_result) {
            last_solver_error_ = price_result.error();
            return std::numeric_limits<double>::quiet_NaN();
        }

        last_solver_error_.reset();
        const AmericanOptionResult& result = price_result.value();

        // Return difference: V(σ) - V_market
        return result.value - query_.market_price;
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
    auto validation_result = validate_params_internal(query_, config_);
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
