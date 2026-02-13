// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/eep/analytical_eep.hpp"
#include "mango/option/table/shared_interp.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/transforms/standard_4d.hpp"
#include "mango/option/table/transforms/dimensionless_3d.hpp"
#include "mango/math/bspline_nd.hpp"
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"
#include "mango/support/error_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace mango {

// ============================================================================
// Level 1: Interpolant construction from Segment
// ============================================================================

/// Reconstruct a BSplineND<double, N> from a segment's grids, knots, and
/// coefficients. Returns a shared_ptr for use with SharedInterp.
template <size_t N>
[[nodiscard]] std::expected<std::shared_ptr<const BSplineND<double, N>>, PriceTableError>
make_bspline(const PriceTableData::Segment& seg) {
    if (seg.interp_type != "bspline") {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (seg.ndim != N) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, seg.ndim});
    }
    if (seg.grids.size() != N || seg.knots.size() != N) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }

    std::array<std::vector<double>, N> grids;
    std::array<std::vector<double>, N> knots;
    for (size_t d = 0; d < N; ++d) {
        grids[d] = seg.grids[d];
        knots[d] = seg.knots[d];
    }

    auto result = BSplineND<double, N>::create(
        std::move(grids), std::move(knots), std::vector<double>(seg.values));

    if (!result.has_value()) {
        return std::unexpected(convert_to_price_table_error(result.error()));
    }

    return std::make_shared<const BSplineND<double, N>>(
        std::move(result.value()));
}

/// Reconstruct a ChebyshevInterpolant<N, RawTensor<N>> from a segment's
/// domain, num_pts, and raw values.
template <size_t N>
[[nodiscard]] std::expected<ChebyshevInterpolant<N, RawTensor<N>>, PriceTableError>
make_chebyshev(const PriceTableData::Segment& seg) {
    if (seg.interp_type != "chebyshev") {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    if (seg.ndim != N) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, seg.ndim});
    }
    if (seg.domain_lo.size() != N || seg.domain_hi.size() != N ||
        seg.num_pts.size() != N) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }

    Domain<N> domain;
    std::array<size_t, N> num_pts;
    for (size_t d = 0; d < N; ++d) {
        domain.lo[d] = seg.domain_lo[d];
        domain.hi[d] = seg.domain_hi[d];
        num_pts[d] = static_cast<size_t>(seg.num_pts[d]);
    }

    return ChebyshevInterpolant<N, RawTensor<N>>::build_from_values(
        std::span<const double>(seg.values), domain, num_pts);
}

// ============================================================================
// Level 2: Leaf construction
// ============================================================================

/// Reconstruct a B-spline TransformLeaf from a segment.
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_bspline_leaf(const PriceTableData::Segment& seg)
    -> std::expected<TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>,
                     PriceTableError> {
    auto spline_ptr = make_bspline<N>(seg);
    if (!spline_ptr) return std::unexpected(spline_ptr.error());
    SharedInterp<BSplineND<double, N>, N> shared(std::move(*spline_ptr));
    return TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>(
        std::move(shared), Xform{}, seg.K_ref);
}

/// Reconstruct a Chebyshev Raw TransformLeaf from a segment.
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_chebyshev_leaf(const PriceTableData::Segment& seg)
    -> std::expected<TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>,
                     PriceTableError> {
    auto interp = make_chebyshev<N>(seg);
    if (!interp) return std::unexpected(interp.error());
    return TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>(
        std::move(*interp), Xform{}, seg.K_ref);
}

// ============================================================================
// Level 3: EEP layer wrapping
// ============================================================================

/// Wrap a leaf with AnalyticalEEP.
template <typename LeafType>
[[nodiscard]] auto reconstruct_eep(LeafType leaf, OptionType opt, double q)
    -> EEPLayer<LeafType, AnalyticalEEP> {
    AnalyticalEEP eep(opt, q);
    return EEPLayer<LeafType, AnalyticalEEP>(std::move(leaf), std::move(eep));
}

// ============================================================================
// Level 4: SurfaceBounds extraction from segment data
// ============================================================================

/// Derive SurfaceBounds from 4D segment domain (physical coordinates).
[[nodiscard]] inline SurfaceBounds bounds_from_4d_segment(
    const PriceTableData::Segment& seg) {
    return SurfaceBounds{
        .m_min = seg.domain_lo[0], .m_max = seg.domain_hi[0],
        .tau_min = seg.domain_lo[1], .tau_max = seg.domain_hi[1],
        .sigma_min = seg.domain_lo[2], .sigma_max = seg.domain_hi[2],
        .rate_min = seg.domain_lo[3], .rate_max = seg.domain_hi[3],
    };
}

/// Derive SurfaceBounds from 3D dimensionless segment domain.
///
/// The 3D axes are [x, tau_prime, ln_kappa] where:
///   x = ln(S/K)
///   tau_prime = sigma^2 * tau / 2
///   ln_kappa = ln(2r / sigma^2)
///
/// We reverse-engineer physical bounds from the dimensionless domain:
///   sigma_min^2 = 2 * tp_min / tau_max  (but we cannot know tau_max alone)
///
/// Since we cannot invert the mapping uniquely without the original physical
/// grid, we store conservative bounds. The SurfaceBounds for 3D surfaces are
/// set as: m from axis 0, tau/sigma/rate set to the max ranges that could
/// produce the dimensionless domain. In practice, the bounds are only used
/// for out-of-range checks, so we use the maturity from PriceTableData.
[[nodiscard]] inline SurfaceBounds bounds_from_3d_segment(
    const PriceTableData::Segment& seg, double maturity) {
    // Dimensionless domain: [m, tp, lk]
    double m_min = seg.domain_lo[0];
    double m_max = seg.domain_hi[0];
    double tp_min = seg.domain_lo[1];
    double tp_max = seg.domain_hi[1];
    double lk_min = seg.domain_lo[2];
    double lk_max = seg.domain_hi[2];

    // Reverse-engineer sigma bounds from tp and maturity:
    //   tp = sigma^2 * tau / 2
    //   tp_max = sigma_max^2 * maturity / 2  =>  sigma_max = sqrt(2*tp_max/maturity)
    //   tp_min = sigma_min^2 * tau_min / 2    (tau_min unknown, use a small value)
    // For ln_kappa = ln(2r/sigma^2):
    //   lk_max = ln(2*rate_max/sigma_min^2)
    //   lk_min = ln(2*rate_min/sigma_max^2)
    // We solve: sigma_max = sqrt(2*tp_max/maturity)
    double sigma_max = std::sqrt(2.0 * tp_max / maturity);
    // sigma_min: from tp_min = sigma_min^2 * tau_min / 2
    // tau_min is typically small; use the heuristic that tp_min corresponds to
    // the smallest sigma at some reasonable tau. As a conservative bound:
    double sigma_min = std::sqrt(2.0 * tp_min / maturity);
    // rate bounds from lk:
    //   lk_min = ln(2*rate_min/sigma_max^2) => rate_min = sigma_max^2/2 * exp(lk_min)
    //   lk_max = ln(2*rate_max/sigma_min^2) => rate_max = sigma_min^2/2 * exp(lk_max)
    double rate_min = sigma_max * sigma_max / 2.0 * std::exp(lk_min);
    double rate_max = sigma_min * sigma_min / 2.0 * std::exp(lk_max);

    // Ensure valid ordering
    if (sigma_min > sigma_max) std::swap(sigma_min, sigma_max);
    if (rate_min > rate_max) std::swap(rate_min, rate_max);

    return SurfaceBounds{
        .m_min = m_min, .m_max = m_max,
        .tau_min = 0.01, .tau_max = maturity,
        .sigma_min = sigma_min, .sigma_max = sigma_max,
        .rate_min = rate_min, .rate_max = rate_max,
    };
}

/// Derive SurfaceBounds for segmented surfaces from all segments.
[[nodiscard]] inline SurfaceBounds bounds_from_segments(
    const std::vector<PriceTableData::Segment>& segments) {
    if (segments.empty()) {
        return SurfaceBounds{};
    }

    // For segmented surfaces, all segments share the same coordinate space.
    // Compute the union of all segment domains.
    size_t ndim = segments[0].ndim;
    SurfaceBounds bounds{
        .m_min = segments[0].domain_lo[0],
        .m_max = segments[0].domain_hi[0],
        .tau_min = segments[0].tau_start,
        .tau_max = segments[0].tau_end,
        .sigma_min = ndim >= 3 ? segments[0].domain_lo[2] : 0.0,
        .sigma_max = ndim >= 3 ? segments[0].domain_hi[2] : 0.0,
        .rate_min = ndim >= 4 ? segments[0].domain_lo[3] : 0.0,
        .rate_max = ndim >= 4 ? segments[0].domain_hi[3] : 0.0,
    };

    for (size_t i = 1; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        bounds.m_min = std::min(bounds.m_min, seg.domain_lo[0]);
        bounds.m_max = std::max(bounds.m_max, seg.domain_hi[0]);
        bounds.tau_min = std::min(bounds.tau_min, seg.tau_start);
        bounds.tau_max = std::max(bounds.tau_max, seg.tau_end);
        if (ndim >= 3) {
            bounds.sigma_min = std::min(bounds.sigma_min, seg.domain_lo[2]);
            bounds.sigma_max = std::max(bounds.sigma_max, seg.domain_hi[2]);
        }
        if (ndim >= 4) {
            bounds.rate_min = std::min(bounds.rate_min, seg.domain_lo[3]);
            bounds.rate_max = std::max(bounds.rate_max, seg.domain_hi[3]);
        }
    }

    return bounds;
}

// ============================================================================
// Level 5: Segmented surface construction helpers
// ============================================================================

/// Group segments by K_ref value, preserving segment order within each group.
/// Returns groups sorted by K_ref.
[[nodiscard]] inline auto group_segments_by_kref(
    const std::vector<PriceTableData::Segment>& segments)
    -> std::map<double, std::vector<const PriceTableData::Segment*>> {
    std::map<double, std::vector<const PriceTableData::Segment*>> groups;
    for (const auto& seg : segments) {
        groups[seg.K_ref].push_back(&seg);
    }
    return groups;
}

/// Build a TauSegmentSplit + vector of B-spline leaves for one K_ref group.
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_bspline_tau_split(
    const std::vector<const PriceTableData::Segment*>& group, double K_ref)
    -> std::expected<
        SplitSurface<TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>,
                     TauSegmentSplit>,
        PriceTableError> {
    using LeafType = TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>;

    std::vector<LeafType> leaves;
    std::vector<double> tau_starts, tau_ends, tau_mins, tau_maxs;

    for (const auto* seg : group) {
        auto leaf = reconstruct_bspline_leaf<N, Xform>(*seg);
        if (!leaf) return std::unexpected(leaf.error());
        leaves.push_back(std::move(*leaf));
        tau_starts.push_back(seg->tau_start);
        tau_ends.push_back(seg->tau_end);
        tau_mins.push_back(seg->tau_min);
        tau_maxs.push_back(seg->tau_max);
    }

    TauSegmentSplit split(std::move(tau_starts), std::move(tau_ends),
                          std::move(tau_mins), std::move(tau_maxs), K_ref);
    return SplitSurface<LeafType, TauSegmentSplit>(
        std::move(leaves), std::move(split));
}

/// Build a TauSegmentSplit + vector of Chebyshev Raw leaves for one K_ref group.
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_chebyshev_tau_split(
    const std::vector<const PriceTableData::Segment*>& group, double K_ref)
    -> std::expected<
        SplitSurface<TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>,
                     TauSegmentSplit>,
        PriceTableError> {
    using LeafType = TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>;

    std::vector<LeafType> leaves;
    std::vector<double> tau_starts, tau_ends, tau_mins, tau_maxs;

    for (const auto* seg : group) {
        auto leaf = reconstruct_chebyshev_leaf<N, Xform>(*seg);
        if (!leaf) return std::unexpected(leaf.error());
        leaves.push_back(std::move(*leaf));
        tau_starts.push_back(seg->tau_start);
        tau_ends.push_back(seg->tau_end);
        tau_mins.push_back(seg->tau_min);
        tau_maxs.push_back(seg->tau_max);
    }

    TauSegmentSplit split(std::move(tau_starts), std::move(tau_ends),
                          std::move(tau_mins), std::move(tau_maxs), K_ref);
    return SplitSurface<LeafType, TauSegmentSplit>(
        std::move(leaves), std::move(split));
}

}  // namespace mango
