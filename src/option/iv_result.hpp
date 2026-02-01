// SPDX-License-Identifier: MIT
/**
 * @file iv_result.hpp
 * @brief IV solver result types for std::expected API
 */

#pragma once

#include <cstddef>
#include <optional>
#include <vector>
#include <expected>
#include <cmath>
#include "src/support/error_types.hpp"

namespace mango {

/// Success result from IV solver
struct IVSuccess {
    double implied_vol;              ///< Solved implied volatility
    size_t iterations;               ///< Number of iterations taken
    double final_error;              ///< |Price(σ) - Market_Price|
    std::optional<double> vega;      ///< Vega at solution (optional)

    /// True if yield curve was collapsed to zero rate for interpolation.
    /// Only set by IVSolverInterpolated when a YieldCurve is passed.
    /// The solver uses zero_rate = -ln(D(T))/T as a flat-rate approximation.
    /// For full term structure dynamics, use IVSolverFDM instead.
    bool used_rate_approximation = false;
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

/// Convert ValidationError to IVError (shared by all IV solvers)
///
/// Maps ValidationErrorCode to IVErrorCode with consistent semantics:
/// - InvalidSpotPrice → NegativeSpot
/// - InvalidStrike → NegativeStrike
/// - InvalidMaturity → NegativeMaturity
/// - InvalidMarketPrice → NegativeMarketPrice (non-finite or negative) or ArbitrageViolation (positive but out of bounds)
/// - InvalidRate/InvalidDividend/InvalidVolatility → ArbitrageViolation (shouldn't occur in IV queries)
/// - default → ArbitrageViolation
///
/// @param ve Validation error from validate_iv_query()
/// @return Corresponding IVError with mapped error code
inline IVError validation_error_to_iv_error(const ValidationError& ve) {
    IVErrorCode code;
    switch (ve.code) {
        case ValidationErrorCode::InvalidSpotPrice:
            code = IVErrorCode::NegativeSpot;
            break;
        case ValidationErrorCode::InvalidStrike:
            code = IVErrorCode::NegativeStrike;
            break;
        case ValidationErrorCode::InvalidMaturity:
            code = IVErrorCode::NegativeMaturity;
            break;
        case ValidationErrorCode::InvalidMarketPrice:
            // InvalidMarketPrice covers negative prices, non-finite, and arbitrage violations
            // Priority: non-finite > negative > arbitrage
            if (!std::isfinite(ve.value)) {
                code = IVErrorCode::NegativeMarketPrice;  // Non-finite (NaN/inf) treated as invalid input
            } else if (ve.value <= 0.0) {
                code = IVErrorCode::NegativeMarketPrice;
            } else {
                code = IVErrorCode::ArbitrageViolation;  // Positive but violates bounds
            }
            break;
        case ValidationErrorCode::OptionTypeMismatch:
            code = IVErrorCode::OptionTypeMismatch;
            break;
        case ValidationErrorCode::DividendYieldMismatch:
            code = IVErrorCode::DividendYieldMismatch;
            break;
        case ValidationErrorCode::InvalidRate:
        case ValidationErrorCode::InvalidDividend:
        case ValidationErrorCode::InvalidVolatility:
            code = IVErrorCode::ArbitrageViolation;
            break;
        default:
            code = IVErrorCode::ArbitrageViolation;
            break;
    }

    return IVError{
        .code = code,
        .iterations = 0,
        .final_error = ve.value,  // Preserve the invalid value for diagnostics
        .last_vol = std::nullopt
    };
}

}  // namespace mango
