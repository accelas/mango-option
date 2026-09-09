// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include "mango/option/table/fixed_expiry.hpp"
#include "mango/option/table/strike_bounds.hpp"
#include "mango/option/table/moneyness_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mango {

/// Serializable representation of any PriceTable surface.
/// Plain vectors — no I/O dependencies.
struct PriceTableData {
    std::string surface_type;

    OptionType option_type = OptionType::PUT;
    double dividend_yield = 0.0;
    /// Physical query domain and numerical model identity, never inferred
    /// from support reference strikes or the largest published maturity.
    std::optional<StrikeBounds> strike_bounds;
    /// Required numerical query-domain provenance in current payloads.
    std::optional<MoneynessBounds> ratio_bounds;
    std::optional<FixedExpiryMetadata> fixed_expiry;
    /// Largest published query maturity; not the numerical expiry anchor.
    double maturity = 0.0;

    /// Original SurfaceBounds (serialized directly, no heuristic inversion).
    double bounds_m_min = 0.0, bounds_m_max = 0.0;
    double bounds_tau_min = 0.0, bounds_tau_max = 0.0;
    double bounds_sigma_min = 0.0, bounds_sigma_max = 0.0;
    double bounds_rate_min = 0.0, bounds_rate_max = 0.0;

    struct Segment {
        int32_t segment_id = 0;
        double K_ref = 0.0;
        double tau_start = 0.0, tau_end = 0.0;
        double tau_min = 0.0, tau_max = 0.0;
        // "bspline": spline coefficients; "chebyshev": nodal values;
        // "chebyshev_modal": actual modal coefficients (no half factors).
        std::string interp_type;
        size_t ndim = 4;

        std::vector<double> domain_lo, domain_hi;
        std::vector<int32_t> num_pts;
        std::vector<std::vector<double>> grids;   // ndim vectors
        std::vector<std::vector<double>> knots;   // ndim vectors
        std::vector<double> values;               // coefficients or raw values
    };
    std::vector<Segment> segments;

    size_t n_pde_solves = 0;
    double precompute_time_seconds = 0.0;
};

namespace surface_types {
inline constexpr const char* kBSpline4D = "bspline_4d";
inline constexpr const char* kBSpline4DSegmented = "bspline_4d_segmented";
inline constexpr const char* kChebyshev4D = "chebyshev_4d";
inline constexpr const char* kChebyshev4DRaw = "chebyshev_4d_raw";
inline constexpr const char* kChebyshev4DSegmented = "chebyshev_4d_segmented";
inline constexpr const char* kBSpline3D = "bspline_3d";
inline constexpr const char* kChebyshev3D = "chebyshev_3d";
inline constexpr const char* kChebyshev3DRaw = "chebyshev_3d_raw";
}  // namespace surface_types

/// Validate model/domain provenance at both in-memory and file boundaries.
/// Reference support is allowed to exceed the declared strike interval.
[[nodiscard]] inline bool valid_price_table_metadata(const PriceTableData& data) {
    if (!data.ratio_bounds || !data.ratio_bounds->valid()) return false;
    if ((data.strike_bounds && !data.strike_bounds->valid()) ||
        (data.fixed_expiry && !data.fixed_expiry->valid(data.bounds_tau_max))) {
        return false;
    }
    const bool segmented = data.surface_type == surface_types::kBSpline4DSegmented ||
                           data.surface_type == surface_types::kChebyshev4DSegmented;
    if (!segmented) return true;
    if (!data.strike_bounds || !data.fixed_expiry || data.segments.empty()) return false;
    double min_ref = data.segments.front().K_ref;
    double max_ref = min_ref;
    for (const auto& segment : data.segments) {
        if (!std::isfinite(segment.K_ref) || segment.K_ref <= 0.0) return false;
        min_ref = std::min(min_ref, segment.K_ref);
        max_ref = std::max(max_ref, segment.K_ref);
    }
    return min_ref <= data.strike_bounds->min && max_ref >= data.strike_bounds->max;
}

}  // namespace mango
