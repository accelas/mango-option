// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"

namespace mango {

/// Build-time helper: converts normalized American prices to EEP values.
/// EEP = American - European, with debiased softplus floor.
struct EEPDecomposer {
    OptionType option_type;
    double K_ref;
    double dividend_yield;

    /// Transform tensor from V/K_ref (normalized American prices) to EEP values.
    /// EEP = American - European, with debiased softplus floor for non-negativity.
    ///
    /// @param tensor In/out: tensor of normalized prices (V/K_ref), overwritten with EEP
    /// @param axes Grid axes (axis 0 = log-moneyness, 1 = tau, 2 = sigma, 3 = rate)
    void decompose(PriceTensor& tensor, const PriceTableAxes& axes) const;
};

}  // namespace mango
