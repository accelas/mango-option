/**
 * @file iv_types.hpp
 * @brief Shared data structures for implied volatility solvers.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mango {

/// Unified result type for implied volatility solvers (PDE-based and interpolation-based).
struct IVResult {
    bool converged = false;                         ///< Convergence status
    std::size_t iterations = 0;                     ///< Number of iterations performed
    double implied_vol = 0.0;                       ///< Solved implied volatility (if converged)
    double final_error = 0.0;                       ///< Residual |Price(σ) - Market_Price|
    std::optional<std::string> failure_reason;      ///< Diagnostic string when not converged
    std::optional<double> vega;                     ///< Vega at the solution (when available)
};

}  // namespace mango
