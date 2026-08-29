// SPDX-License-Identifier: MIT
/**
 * @file iv_error_mapping.hpp
 * @brief IVErrorCode -> MangoStatus mapping used by the C ABI shim
 *
 * Lives in a header rather than inside mango_c_api.cpp so the mapping is
 * testable on its own.  Driving every arm end-to-end through the public C
 * entry points would need a price surface engineered to fail in a specific
 * way for each code — for MultipleRoots, a surface that is non-monotone in
 * sigma, which the C API offers no way to construct.
 */

#pragma once

#include "mango/ffi/mango_c_api.h"
#include "mango/support/error_types.hpp"

namespace mango::ffi {

/// Map an IV solver error onto the C status enum.
///
/// Categories, not codes: bracketing failures and the multiple-root screen
/// both mean "no single root was isolated", so both report
/// MANGO_ERR_BRACKETING.  Validation-category codes must never fall through
/// to MANGO_ERR_SOLVER, or Rust callers see a solver failure for what is
/// really bad input.
inline MangoStatus map_iv_error(const IVError& e) {
  switch (e.code) {
    case IVErrorCode::ArbitrageViolation: return MANGO_ERR_ARBITRAGE;
    case IVErrorCode::BracketingFailed:
    case IVErrorCode::MultipleRoots: return MANGO_ERR_BRACKETING;
    case IVErrorCode::MaxIterationsExceeded: return MANGO_ERR_NO_CONVERGENCE;
    case IVErrorCode::NegativeSpot:
    case IVErrorCode::NegativeStrike:
    case IVErrorCode::NegativeMaturity:
    case IVErrorCode::NegativeMarketPrice:
    case IVErrorCode::InvalidGridConfig:
    case IVErrorCode::OptionTypeMismatch:
    case IVErrorCode::DividendYieldMismatch: return MANGO_ERR_VALIDATION;
    default: return MANGO_ERR_SOLVER;
  }
}

}  // namespace mango::ffi
