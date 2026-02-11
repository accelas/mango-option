// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/option/table/splits/tau_segment.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace mango {

/// Result of adaptive Chebyshev surface construction (standard path)
struct ChebyshevAdaptiveResult {
    std::shared_ptr<ChebyshevRawSurface> surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

/// Result of adaptive segmented Chebyshev surface construction
///
/// Segmented surfaces use multi-K_ref blending which can't be expressed
/// as a single typed PriceTable, so the result is a type-erased price_fn.
struct ChebyshevSegmentedAdaptiveResult {
    std::function<double(double, double, double, double, double)> price_fn;
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

/// Build segmented Chebyshev surface with discrete dividend support.
///
/// Stores V/K_ref directly per segment (no EEP decomposition).
/// Multi-K_ref blending is used when kref_config has multiple references.
[[nodiscard]] std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(const AdaptiveGridParams& params,
                                   const SegmentedAdaptiveConfig& config,
                                   const IVGrid& domain);

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

}  // namespace mango
