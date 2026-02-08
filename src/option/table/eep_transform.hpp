// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <memory>

#include "mango/option/option_spec.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/table/price_tensor.hpp"

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

/// Query-time Inner adapter for EEP surfaces. Satisfies SplicedInner.
///
/// Reconstructs full American prices/vega from stored EEP values:
///   price = EEP(ln(S/K)) * (K/K_ref) + V_european(S, K, tau, sigma, rate)
///   vega  = dEEP/dsigma  * (K/K_ref) + vega_european(S, K, tau, sigma, rate)
///
/// Encapsulates all query-time EEP math for the SplicedSurface framework.
class EEPPriceTableInner {
public:
    EEPPriceTableInner(std::shared_ptr<const PriceTableSurface> surface,
                       OptionType type, double K_ref, double dividend_yield)
        : surface_(std::move(surface))
        , type_(type)
        , K_ref_(K_ref)
        , dividend_yield_(dividend_yield)
    {}

    [[nodiscard]] double price(const PriceQuery& q) const;
    [[nodiscard]] double vega(const PriceQuery& q) const;

    [[nodiscard]] const PriceTableSurface& surface() const { return *surface_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }
    [[nodiscard]] OptionType option_type() const noexcept { return type_; }
    [[nodiscard]] double dividend_yield() const noexcept { return dividend_yield_; }

private:
    std::shared_ptr<const PriceTableSurface> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
