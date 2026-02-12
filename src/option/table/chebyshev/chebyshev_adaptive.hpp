// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/split_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/table/splits/multi_kref.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace mango {

/// Tau-segmented Chebyshev surface (one leaf per inter-dividend interval)
using ChebyshevTauSegmented = SplitSurface<ChebyshevSegmentedLeaf, TauSegmentSplit>;

/// Multi-K_ref blended segmented Chebyshev surface
using ChebyshevMultiKRefInner = SplitSurface<ChebyshevTauSegmented, MultiKRefSplit>;

/// Multi-K_ref segmented Chebyshev price table (final queryable surface)
using ChebyshevMultiKRefSurface = PriceTable<ChebyshevMultiKRefInner>;

/// Result of adaptive Chebyshev surface construction (standard path)
struct ChebyshevAdaptiveResult {
    std::shared_ptr<ChebyshevRawSurface> surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

/// Build Chebyshev surface with adaptive CC-level refinement.
///
/// Uses CGL nodes for moneyness/tau and Clenshaw-Curtis levels for sigma/rate.
/// EEP decomposition is applied for better interpolation accuracy.
[[nodiscard]] std::expected<ChebyshevAdaptiveResult, PriceTableError>
build_adaptive_chebyshev(const AdaptiveGridParams& params,
                         const OptionGrid& chain,
                         OptionType type = OptionType::PUT);

/// Per-K_ref typed pieces for assembling a ChebyshevMultiKRefSurface.
struct ChebyshevSegmentedPieces {
    std::vector<ChebyshevSegmentedLeaf> leaves;  ///< One leaf per real segment
    TauSegmentSplit tau_split;                    ///< Gap-absorbed tau routing
};

/// Build typed Chebyshev segmented pieces from converged grids.
/// Each leaf stores V/K_ref (no EEP decomposition).
/// The TauSegmentSplit absorbs gap segments at construction time.
[[nodiscard]] std::expected<ChebyshevSegmentedPieces, PriceTableError>
build_chebyshev_segmented_pieces(
    double K_ref,
    OptionType option_type,
    double dividend_yield,
    const std::vector<Dividend>& discrete_dividends,
    const std::vector<double>& seg_bounds,
    const std::vector<bool>& seg_is_gap,
    std::span<const double> m_nodes,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes);

/// Result of adaptive segmented Chebyshev surface construction.
struct ChebyshevSegmentedAdaptiveResult {
    ChebyshevMultiKRefSurface surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

/// Build segmented Chebyshev surface with discrete dividend support.
/// Returns a ChebyshevMultiKRefSurface with multi-K_ref blending.
[[nodiscard]] std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(const AdaptiveGridParams& params,
                                   const SegmentedAdaptiveConfig& config,
                                   const IVGrid& domain);

/// Build typed segmented Chebyshev surface from explicit CC levels (no adaptive refinement).
/// Used for benchmarking with fixed grid sizes.
[[nodiscard]] std::expected<ChebyshevMultiKRefSurface, PriceTableError>
build_chebyshev_segmented_manual(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain,
    std::array<size_t, 4> cc_levels = {5, 3, 2, 1});

}  // namespace mango
