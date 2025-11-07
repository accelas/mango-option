/**
 * @file iv_solver_interpolated.cpp
 * @brief Implementation of interpolation-based IV solver
 */

#include "iv_solver_interpolated.hpp"
#include <algorithm>
#include <cmath>

namespace mango {

std::optional<std::string> IVSolverInterpolated::validate_query(const IVQuery& query) const {
    if (query.market_price <= 0.0) {
        return "Market price must be positive";
    }
    if (query.spot <= 0.0) {
        return "Spot price must be positive";
    }
    if (query.strike <= 0.0) {
        return "Strike must be positive";
    }
    if (query.maturity <= 0.0) {
        return "Maturity must be positive";
    }

    // Check for arbitrage violations (option-type specific)
    double intrinsic;
    double upper_bound;

    if (query.option_type == OptionType::CALL) {
        intrinsic = std::max(query.spot - query.strike, 0.0);  // Call intrinsic
        upper_bound = query.spot;  // Call price cannot exceed spot
    } else {  // PUT
        intrinsic = std::max(query.strike - query.spot, 0.0);  // Put intrinsic
        upper_bound = query.strike;  // Put price cannot exceed strike
    }

    if (query.market_price < intrinsic) {
        return "Market price below intrinsic value (arbitrage)";
    }
    if (query.market_price > upper_bound) {
        const char* opt_type = (query.option_type == OptionType::CALL) ? "Call" : "Put";
        const char* bound_type = (query.option_type == OptionType::CALL) ? "spot" : "strike";
        return std::string(opt_type) + " price above " + bound_type + " (arbitrage)";
    }

    return std::nullopt;
}

std::pair<double, double> IVSolverInterpolated::adaptive_bounds(const IVQuery& query) const {
    // Compute intrinsic value based on option type
    double intrinsic;
    if (query.option_type == OptionType::CALL) {
        intrinsic = std::max(query.spot - query.strike, 0.0);
    } else {  // PUT
        intrinsic = std::max(query.strike - query.spot, 0.0);
    }

    // Analyze time value to set adaptive bounds
    const double time_value = query.market_price - intrinsic;
    const double time_value_pct = time_value / query.market_price;

    double sigma_upper;
    if (time_value_pct > 0.5) {
        // High time value suggests high volatility
        sigma_upper = 3.0;  // 300%
    } else if (time_value_pct > 0.2) {
        // Moderate time value
        sigma_upper = 2.0;  // 200%
    } else {
        // Low time value (deep ITM)
        sigma_upper = 1.5;  // 150%
    }

    return {config_.sigma_min, std::min(sigma_upper, config_.sigma_max)};
}

IVResult IVSolverInterpolated::solve(const IVQuery& query) const {
    // Validate input
    auto error = validate_query(query);
    if (error.has_value()) {
        return IVResult{
            .implied_vol = 0.0,
            .converged = false,
            .iterations = 0,
            .final_error = 0.0,
            .error_message = *error
        };
    }

    // Compute moneyness
    const double moneyness = query.spot / query.strike;

    // Get adaptive bounds
    auto [sigma_min, sigma_max] = adaptive_bounds(query);

    // Check if query is within surface bounds (before we start iterating)
    if (!is_in_bounds(query, sigma_min) || !is_in_bounds(query, sigma_max)) {
        return IVResult{
            .implied_vol = 0.0,
            .converged = false,
            .iterations = 0,
            .final_error = 0.0,
            .error_message = "Query parameters out of surface bounds. "
                            "Moneyness: [" + std::to_string(m_range_.first) + ", " + std::to_string(m_range_.second) + "], "
                            "Maturity: [" + std::to_string(tau_range_.first) + ", " + std::to_string(tau_range_.second) + "], "
                            "Volatility: [" + std::to_string(sigma_range_.first) + ", " + std::to_string(sigma_range_.second) + "], "
                            "Rate: [" + std::to_string(r_range_.first) + ", " + std::to_string(r_range_.second) + "]. "
                            "Use PDE-based IV solver for out-of-grid queries."
        };
    }

    // Initial guess: midpoint of bounds
    double sigma = (sigma_min + sigma_max) / 2.0;

    // Newton-Raphson iterations
    int iter = 0;
    double error_abs = 0.0;

    for (; iter < config_.max_iterations; ++iter) {
        // Check if current sigma is within surface bounds
        if (!is_in_bounds(query, sigma)) {
            return IVResult{
                .implied_vol = sigma,
                .converged = false,
                .iterations = iter + 1,
                .final_error = error_abs,
                .error_message = "Newton iteration moved outside surface bounds"
            };
        }

        // Evaluate price at current volatility (with strike scaling)
        const double price = eval_price(moneyness, query.maturity, sigma, query.rate, query.strike);

        // Compute error
        error_abs = std::abs(price - query.market_price);

        // Check convergence
        if (error_abs < config_.tolerance) {
            return IVResult{
                .implied_vol = sigma,
                .converged = true,
                .iterations = iter + 1,
                .final_error = error_abs,
                .error_message = std::nullopt
            };
        }

        // Compute vega (∂Price/∂σ) with strike scaling
        const double vega = compute_vega(moneyness, query.maturity, sigma, query.rate, query.strike);

        // Check for numerical issues
        if (std::abs(vega) < 1e-10) {
            return IVResult{
                .implied_vol = sigma,
                .converged = false,
                .iterations = iter + 1,
                .final_error = error_abs,
                .error_message = "Vega too small (flat price surface)"
            };
        }

        // Newton step: σ_{n+1} = σ_n - f(σ_n)/f'(σ_n)
        const double f = price - query.market_price;
        const double sigma_new = sigma - f / vega;

        // Enforce bounds
        sigma = std::clamp(sigma_new, sigma_min, sigma_max);

        // Check if bounds are hit (may indicate convergence issues)
        if (sigma_new < sigma_min || sigma_new > sigma_max) {
            // Try to refine bounds
            if (iter > 10) {
                return IVResult{
                    .implied_vol = sigma,
                    .converged = false,
                    .iterations = iter + 1,
                    .final_error = error_abs,
                    .error_message = "Hit volatility bounds without convergence"
                };
            }
        }
    }

    // Max iterations reached
    return IVResult{
        .implied_vol = sigma,
        .converged = false,
        .iterations = iter,
        .final_error = error_abs,
        .error_message = "Maximum iterations reached without convergence"
    };
}

}  // namespace mango
