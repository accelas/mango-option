/**
 * @file iv_result.hpp
 * @brief IV solver result types for std::expected API
 */

#pragma once

#include <cstddef>
#include <optional>

namespace mango {

/// Success result from IV solver
struct IVSuccess {
    double implied_vol;              ///< Solved implied volatility
    size_t iterations;               ///< Number of iterations taken
    double final_error;              ///< |Price(σ) - Market_Price|
    std::optional<double> vega;      ///< Vega at solution (optional)
};

}  // namespace mango
