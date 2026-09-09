// SPDX-License-Identifier: MIT
#pragma once
#include "mango/option/table/strike_bounds.hpp"
#include "mango/option/table/moneyness_bounds.hpp"
#include <optional>
namespace mango {
/// Bounds metadata for a price surface.
struct SurfaceBounds {
    double m_min, m_max;
    double tau_min, tau_max;
    double sigma_min, sigma_max;
    double rate_min, rate_max;
    /// Required for segmented publication; omitted for homogeneous tables.
    std::optional<StrikeBounds> strike_bounds = std::nullopt;
    /// Original ratio-input endpoints when known. Absent means a direct
    /// log-domain request, resolved once with exp(m_min/max).
    std::optional<MoneynessBounds> ratio_bounds = std::nullopt;
};

} // namespace mango
