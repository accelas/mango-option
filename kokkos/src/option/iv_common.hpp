#pragma once

/// @file iv_common.hpp
/// @brief Common types for implied volatility solvers
///
/// Provides shared data structures used by both interpolated and FDM
/// IV solvers.

#include "kokkos/src/option/american_option.hpp"

namespace mango::kokkos {

/// IV query input (option parameters + market price)
///
/// Used by both IVSolverInterpolated and IVSolverFDM.
struct IVQuery {
    double strike;
    double spot;
    double maturity;
    double rate;
    double dividend_yield;
    OptionType type;
    double market_price;
};

}  // namespace mango::kokkos
