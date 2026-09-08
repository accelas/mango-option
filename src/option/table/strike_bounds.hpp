// SPDX-License-Identifier: MIT
#pragma once

#include "mango/support/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <optional>
#include <span>

namespace mango {

/// Inclusive supported absolute-strike interval. A singleton is valid.
struct StrikeBounds {
    double min = 0.0;
    double max = 0.0;

    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(min) && std::isfinite(max) && min > 0.0 && min <= max;
    }
    [[nodiscard]] bool contains(double strike) const noexcept {
        return valid() && std::isfinite(strike) && strike >= min && strike <= max;
    }
};

/// Resolve the requested financial domain before numerical support expansion.
/// Ratio endpoints are S/K, not logarithms. Explicit intervals are preserved.
inline std::expected<StrikeBounds, ValidationError> resolve_strike_bounds(
    const std::optional<StrikeBounds>& requested, double spot,
    double min_ratio, double max_ratio)
{
    if (requested) {
        if (!requested->valid()) return std::unexpected(
            ValidationError{ValidationErrorCode::InvalidBounds, requested->min});
        return *requested;
    }
    if (!std::isfinite(spot) || spot <= 0.0 ||
        !std::isfinite(min_ratio) || !std::isfinite(max_ratio) ||
        min_ratio <= 0.0 || min_ratio > max_ratio) {
        return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds});
    }
    StrikeBounds bounds{spot / max_ratio, spot / min_ratio};
    if (!bounds.valid()) return std::unexpected(
        ValidationError{ValidationErrorCode::InvalidBounds});
    return bounds;
}

/// Adapter for segmented builders whose requested moneyness nodes are logs.
inline std::expected<StrikeBounds, ValidationError> resolve_strike_bounds_from_log_nodes(
    const std::optional<StrikeBounds>& requested, double spot,
    std::span<const double> log_nodes)
{
    if (log_nodes.empty() || !std::all_of(log_nodes.begin(), log_nodes.end(),
            [](double x) { return std::isfinite(x); })) {
        return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds});
    }
    const auto [lo, hi] = std::minmax_element(log_nodes.begin(), log_nodes.end());
    return resolve_strike_bounds(requested, spot, std::exp(*lo), std::exp(*hi));
}

}  // namespace mango
