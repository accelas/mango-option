// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/option_spec.hpp"
#include <memory>

namespace mango {

/// Query adapter for 3D dimensionless EEP surface.
///
/// Maps physical queries (spot, strike, tau, sigma, rate) to
/// dimensionless coords (x, tau', ln kappa), reconstructs American price
/// from EEP + analytical European, and computes vega via chain rule.
///
/// Satisfies the SplicedInner concept.
class DimensionlessEEPInner {
public:
    DimensionlessEEPInner(std::shared_ptr<const PriceTableSurfaceND<3>> surface,
                          OptionType type, double K_ref, double dividend_yield);

    /// Reconstruct American price: EEP_norm * K + V_eu
    /// (Surface stores EEP/K_ref; multiply by K for dollar EEP at query strike.)
    [[nodiscard]] double price(const PriceQuery& q) const;

    /// Vega via chain rule:
    ///   K * [sigma*tau * dEEP/dtau' - (2/sigma) * dEEP/d(ln kappa)] + vega_eu
    [[nodiscard]] double vega(const PriceQuery& q) const;

    [[nodiscard]] const PriceTableSurfaceND<3>& surface() const noexcept { return *surface_; }
    [[nodiscard]] double K_ref() const noexcept { return K_ref_; }
    [[nodiscard]] OptionType option_type() const noexcept { return type_; }

private:
    std::shared_ptr<const PriceTableSurfaceND<3>> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
