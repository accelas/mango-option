// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/price_tensor.hpp"
#include "mango/option/table/price_table_axes.hpp"

namespace mango {

struct PriceQuery;  // Forward declare (defined in spliced_surface.hpp)

/// Transform that encapsulates Early Exercise Premium decomposition.
///
/// Build-time: decompose() converts American prices to EEP values.
/// Query-time: normalize_value() scales EEP by K/K_ref,
///             denormalize() adds European price to reconstruct American price.
///
/// Satisfies SliceTransform concept.
struct EEPTransform {
    OptionType option_type;
    double K_ref;
    double dividend_yield;

    // --- Query-time (SliceTransform interface) ---

    /// Identity: PriceTableInner handles ln(S/K) internally.
    [[nodiscard]] PriceQuery to_local(size_t, const PriceQuery& q) const noexcept;

    /// Scale EEP by K/K_ref for strike interpolation.
    [[nodiscard]] double normalize_value(size_t, const PriceQuery& q, double eep) const noexcept;

    /// Add European price to reconstruct American price.
    /// V_american = EEP * (K/K_ref) + V_european
    [[nodiscard]] double denormalize(double scaled_eep, const PriceQuery& q) const;

    // --- Build-time ---

    /// Transform tensor from V/K_ref (normalized American prices) to EEP values.
    /// EEP = American - European, with debiased softplus floor for non-negativity.
    ///
    /// @param tensor In/out: tensor of normalized prices (V/K_ref), overwritten with EEP
    /// @param axes Grid axes (axis 0 = log-moneyness, 1 = tau, 2 = sigma, 3 = rate)
    void decompose(PriceTensor<4>& tensor, const PriceTableAxes<4>& axes) const;
};

}  // namespace mango
