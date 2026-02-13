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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
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
    size_t expected_size = 1;
    for (size_t d = 0; d < N; ++d) {
        domain.lo[d] = seg.domain_lo[d];
        domain.hi[d] = seg.domain_hi[d];
        // Validate domain: finite and lo < hi
        if (!std::isfinite(domain.lo[d]) || !std::isfinite(domain.hi[d]) ||
            domain.lo[d] >= domain.hi[d]) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        // Chebyshev interpolation requires at least 2 points per axis
        if (seg.num_pts[d] < 2) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        num_pts[d] = static_cast<size_t>(seg.num_pts[d]);
        // Overflow check
        if (expected_size > 0 &&
            num_pts[d] > std::numeric_limits<size_t>::max() / expected_size) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        expected_size *= num_pts[d];
    }

    // Validate values size matches tensor dimensions
    if (seg.values.size() != expected_size) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }

    return ChebyshevInterpolant<N, RawTensor<N>>::build_from_values(
        std::span<const double>(seg.values), domain, num_pts);
}

// ============================================================================
// Level 2: Leaf construction
// ============================================================================

/// Reconstruct a B-spline TransformLeaf from a segment.
/// Rejects non-finite or non-positive K_ref (used as divisor at runtime).
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_bspline_leaf(const PriceTableData::Segment& seg)
    -> std::expected<TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>,
                     PriceTableError> {
    if (!std::isfinite(seg.K_ref) || seg.K_ref <= 0.0) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
    auto spline_ptr = make_bspline<N>(seg);
    if (!spline_ptr) return std::unexpected(spline_ptr.error());
    SharedInterp<BSplineND<double, N>, N> shared(std::move(*spline_ptr));
    return TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>(
        std::move(shared), Xform{}, seg.K_ref);
}

/// Reconstruct a Chebyshev Raw TransformLeaf from a segment.
/// Rejects non-finite or non-positive K_ref (used as divisor at runtime).
template <size_t N, typename Xform>
[[nodiscard]] auto reconstruct_chebyshev_leaf(const PriceTableData::Segment& seg)
    -> std::expected<TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>,
                     PriceTableError> {
    if (!std::isfinite(seg.K_ref) || seg.K_ref <= 0.0) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    }
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
// Level 4: Segmented surface construction helpers
// ============================================================================

/// Group segments by K_ref value, preserving segment order within each group.
/// Returns groups sorted by K_ref. Rejects non-finite K_ref values (NaN/Inf
/// would violate strict-weak-ordering for map keys).
[[nodiscard]] inline auto group_segments_by_kref(
    const std::vector<PriceTableData::Segment>& segments)
    -> std::expected<std::map<double, std::vector<const PriceTableData::Segment*>>,
                     PriceTableError> {
    std::map<double, std::vector<const PriceTableData::Segment*>> groups;
    for (const auto& seg : segments) {
        if (!std::isfinite(seg.K_ref)) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        groups[seg.K_ref].push_back(&seg);
    }
    return groups;
}

/// Validate tau-segment invariants:
/// - All tau values are finite
/// - start <= end, min <= max within each segment
/// - Monotonic ordering: each segment's start >= previous segment's start
/// - No interior gaps: each segment's start <= previous segment's end
///   (overlap is OK — bracket routing handles it — but gaps cause
///   silent fallback to segment 0 which would be silent mispricing)
[[nodiscard]] inline std::expected<void, PriceTableError>
validate_tau_segments(const std::vector<const PriceTableData::Segment*>& group) {
    double prev_start = -std::numeric_limits<double>::infinity();
    double max_end = -std::numeric_limits<double>::infinity();
    for (const auto* seg : group) {
        if (!std::isfinite(seg->tau_start) || !std::isfinite(seg->tau_end) ||
            !std::isfinite(seg->tau_min) || !std::isfinite(seg->tau_max)) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        if (seg->tau_start > seg->tau_end || seg->tau_min > seg->tau_max) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        if (seg->tau_start < prev_start) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        // Reject interior gaps: next segment must start at or before
        // the running maximum end to ensure full coverage. Uses max_end
        // (not prev_end) so overlapping segments that extend past earlier
        // ones don't cause false rejections.
        if (max_end > -std::numeric_limits<double>::infinity() &&
            seg->tau_start > max_end) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
        prev_start = seg->tau_start;
        if (seg->tau_end > max_end) max_end = seg->tau_end;
    }
    return {};
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

    auto tau_valid = validate_tau_segments(group);
    if (!tau_valid) return std::unexpected(tau_valid.error());

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

    auto tau_valid = validate_tau_segments(group);
    if (!tau_valid) return std::unexpected(tau_valid.error());

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
