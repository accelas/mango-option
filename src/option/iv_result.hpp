/**
 * @file iv_result.hpp
 * @brief IV solver result types for std::expected API
 */

#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <expected>
#include "src/support/error_types.hpp"

namespace mango {

/// Success result from IV solver
struct IVSuccess {
    double implied_vol;              ///< Solved implied volatility
    size_t iterations;               ///< Number of iterations taken
    double final_error;              ///< |Price(σ) - Market_Price|
    std::optional<double> vega;      ///< Vega at solution (optional)
};

/// Batch IV solver result
struct BatchIVResult {
    std::vector<std::expected<IVSuccess, IVError>> results;  ///< Individual results
    size_t failed_count;                                      ///< Number of failures

    /// Check if all results succeeded
    bool all_succeeded() const {
        return failed_count == 0;
    }
};

}  // namespace mango
