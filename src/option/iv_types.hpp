/**
 * @file iv_types.hpp
 * @brief Shared data structures for implied volatility solvers.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mango {

/**
 * @brief Legacy IV result type (deprecated - use std::expected<IVSuccess, IVError>)
 * @deprecated Use std::expected<IVSuccess, IVError> instead for type-safe error handling
 */
struct [[deprecated("Use std::expected<IVSuccess, IVError> instead of IVResult")]] IVResult {
    bool converged = false;                         ///< Convergence status
    std::size_t iterations = 0;                     ///< Number of iterations performed
    double implied_vol = 0.0;                       ///< Solved implied volatility (if converged)
    double final_error = 0.0;                       ///< Residual |Price(σ) - Market_Price|
    std::optional<std::string> failure_reason;      ///< Diagnostic string when not converged
    std::optional<double> vega;                     ///< Vega at the solution (when available)
};

}  // namespace mango
