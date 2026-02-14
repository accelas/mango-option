// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/serialization/price_table_data.hpp"
#include "mango/option/table/eep/eep_layer.hpp"
#include "mango/option/table/transform_leaf.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/table/shared_interp.hpp"
#include "mango/math/bspline/bspline_nd.hpp"
#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/raw_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mango {

// ============================================================================
// extract_segments: free-function overloads that walk the compositional type
// tree and populate PriceTableData::Segment entries.
//
// Dispatch hierarchy:
//   SplitSurface<Inner, MultiKRefSplit>  -- iterates K_ref groups
//   SplitSurface<Inner, TauSegmentSplit> -- iterates tau segments
//   EEPLayer<Leaf, EEP>                  -- delegates to leaf
//   TransformLeaf<Interp, Xform>         -- produces one Segment (leaf level)
// ============================================================================

// ---------------------------------------------------------------------------
// Leaf helpers: extract interpolant data into a Segment
// ---------------------------------------------------------------------------

namespace detail {

/// Populate a Segment from a BSplineND<double, N>.
template <size_t N>
void fill_bspline_segment(PriceTableData::Segment& seg,
                          const BSplineND<double, N>& bsp) {
    seg.interp_type = "bspline";
    seg.ndim = N;
    seg.domain_lo.resize(N);
    seg.domain_hi.resize(N);
    seg.num_pts.resize(N);
    seg.grids.resize(N);
    seg.knots.resize(N);

    for (size_t d = 0; d < N; ++d) {
        const auto& g = bsp.grid(d);
        seg.domain_lo[d] = g.front();
        seg.domain_hi[d] = g.back();
        seg.num_pts[d] = static_cast<int32_t>(g.size());
        seg.grids[d] = g;
        seg.knots[d] = bsp.knots(d);
    }
    seg.values = bsp.coefficients();
}

/// Populate a Segment from a ChebyshevInterpolant<N, RawTensor<N>>.
template <size_t N>
void fill_chebyshev_raw_segment(PriceTableData::Segment& seg,
                                const ChebyshevInterpolant<N, RawTensor<N>>& interp) {
    seg.interp_type = "chebyshev";
    seg.ndim = N;
    seg.domain_lo.resize(N);
    seg.domain_hi.resize(N);
    seg.num_pts.resize(N);
    seg.grids.clear();
    seg.knots.clear();

    const auto& dom = interp.domain();
    const auto& npts = interp.num_pts();
    for (size_t d = 0; d < N; ++d) {
        seg.domain_lo[d] = dom.lo[d];
        seg.domain_hi[d] = dom.hi[d];
        seg.num_pts[d] = static_cast<int32_t>(npts[d]);
    }
    seg.values = interp.storage().values();
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Leaf-level overloads: B-spline via SharedInterp
// ---------------------------------------------------------------------------

/// B-spline leaf: TransformLeaf<SharedInterp<BSplineND<double,N>,N>, Xform>
template <size_t N, typename Xform>
void extract_segments(const TransformLeaf<SharedInterp<BSplineND<double, N>, N>, Xform>& leaf,
                      std::vector<PriceTableData::Segment>& out,
                      double /*K_ref_hint*/, double tau_start, double tau_end,
                      double tau_min, double tau_max) {
    PriceTableData::Segment seg;
    seg.segment_id = static_cast<int32_t>(out.size());
    seg.K_ref = leaf.K_ref();
    seg.tau_start = tau_start;
    seg.tau_end = tau_end;
    seg.tau_min = tau_min;
    seg.tau_max = tau_max;

    detail::fill_bspline_segment<N>(seg, leaf.interpolant().get());
    out.push_back(std::move(seg));
}

// ---------------------------------------------------------------------------
// Leaf-level overloads: Chebyshev RawTensor
// ---------------------------------------------------------------------------

/// Chebyshev Raw leaf: TransformLeaf<ChebyshevInterpolant<N,RawTensor<N>>, Xform>
template <size_t N, typename Xform>
void extract_segments(const TransformLeaf<ChebyshevInterpolant<N, RawTensor<N>>, Xform>& leaf,
                      std::vector<PriceTableData::Segment>& out,
                      double /*K_ref_hint*/, double tau_start, double tau_end,
                      double tau_min, double tau_max) {
    PriceTableData::Segment seg;
    seg.segment_id = static_cast<int32_t>(out.size());
    seg.K_ref = leaf.K_ref();
    seg.tau_start = tau_start;
    seg.tau_end = tau_end;
    seg.tau_min = tau_min;
    seg.tau_max = tau_max;

    detail::fill_chebyshev_raw_segment<N>(seg, leaf.interpolant());
    out.push_back(std::move(seg));
}

// ---------------------------------------------------------------------------
// Recursive layer: EEPLayer<Leaf, EEP>
// ---------------------------------------------------------------------------

/// EEP layer: delegates to the underlying leaf.
template <typename Leaf, typename EEP>
void extract_segments(const EEPLayer<Leaf, EEP>& layer,
                      std::vector<PriceTableData::Segment>& out,
                      double K_ref_hint, double tau_start, double tau_end,
                      double tau_min, double tau_max) {
    extract_segments(layer.leaf(), out, K_ref_hint,
                     tau_start, tau_end, tau_min, tau_max);
}

// ---------------------------------------------------------------------------
// Recursive layer: SplitSurface<Inner, TauSegmentSplit>
// ---------------------------------------------------------------------------

/// Tau-segment split: iterates tau segments, passing tau boundaries to inner.
template <typename Inner>
void extract_segments(const SplitSurface<Inner, TauSegmentSplit>& surface,
                      std::vector<PriceTableData::Segment>& out,
                      double K_ref_hint, double /*tau_start*/, double /*tau_end*/,
                      double /*tau_min*/, double /*tau_max*/) {
    const auto& split = surface.split();
    const auto& pieces = surface.pieces();
    for (size_t i = 0; i < pieces.size(); ++i) {
        extract_segments(pieces[i], out, K_ref_hint,
                         split.tau_start()[i], split.tau_end()[i],
                         split.tau_min()[i], split.tau_max()[i]);
    }
}

// ---------------------------------------------------------------------------
// Recursive layer: SplitSurface<Inner, MultiKRefSplit>
// ---------------------------------------------------------------------------

/// Multi-K_ref split: iterates K_ref groups, passing K_ref to inner.
template <typename Inner>
void extract_segments(const SplitSurface<Inner, MultiKRefSplit>& surface,
                      std::vector<PriceTableData::Segment>& out,
                      double /*K_ref_hint*/, double tau_start, double tau_end,
                      double tau_min, double tau_max) {
    const auto& split = surface.split();
    const auto& pieces = surface.pieces();
    for (size_t i = 0; i < pieces.size(); ++i) {
        extract_segments(pieces[i], out, split.k_refs()[i],
                         tau_start, tau_end, tau_min, tau_max);
    }
}

}  // namespace mango
