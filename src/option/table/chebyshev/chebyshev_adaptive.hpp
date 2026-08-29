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
#include <array>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace mango {

namespace detail {

/// State for Chebyshev CC-level refinement.
///
/// All 4 dimensions use Clenshaw-Curtis levels for nested node placement;
/// node count at level l = 2^l + 1.  The refiners mutate nothing outside this
/// struct and the four working grid vectors, so a copy of it is the complete
/// backend refinement state for the loop's snapshot/restore hooks (spec D6).
///
/// Exposed for testing the refiner/hook contract without a full table build.
struct ChebyshevRefinementState {
    size_t m_level = 5;       ///< CC level for moneyness (initial: 33 nodes)
    size_t tau_level = 3;     ///< CC level for tau (initial: 9 nodes)
    size_t sigma_level = 2;   ///< CC level for sigma (initial: 5 nodes)
    size_t rate_level = 1;    ///< CC level for rate (initial: 3 nodes)
    size_t max_level = 7;     ///< ceiling per dimension (2^7+1 = 129 nodes)
    // Frozen extended domain bounds
    double m_lo = 0.0, m_hi = 0.0, tau_lo = 0.0, tau_hi = 0.0;
    double sigma_lo = 0.0, sigma_hi = 0.0, rate_lo = 0.0, rate_hi = 0.0;
    std::vector<double> seg_boundaries;  ///< empty = vanilla (no segmentation)
    std::vector<bool> seg_is_gap;        ///< true for synthetic dividend gaps
};

/// Generate per-segment CC-level tau nodes, sorted and deduplicated.
/// Gap segments are skipped.
[[nodiscard]] std::vector<double> generate_segmented_tau_nodes(
    size_t tau_level,
    const std::vector<double>& seg_bounds,
    const std::vector<bool>& seg_is_gap);

/// RefineFn for Chebyshev CC-level refinement (spec D6).
///
/// Advances EXACTLY the requested axis by one CC level, or reports
/// `changed = false` when that axis is already at `state.max_level`.  No
/// redirection to another axis: the coordinate-descent walk owns axis
/// selection and must be able to measure the axis it picked.
/// `focus_intervals` is ignored -- CC nodes sit at fixed nested positions and
/// cannot be steered by physical intervals.
///
/// `state` is captured by reference: it must outlive the returned callable.
[[nodiscard]] RefineFn make_chebyshev_refine_fn(ChebyshevRefinementState& state);

/// RefineFn for segmented Chebyshev CC-level refinement.
/// Same contract as `make_chebyshev_refine_fn`; tau refinement generates
/// per-segment CC nodes instead of nodes over a single range.
///
/// `state` is captured by reference: it must outlive the returned callable.
[[nodiscard]] RefineFn make_segmented_chebyshev_refine_fn(
    ChebyshevRefinementState& state);

/// Snapshot/restore hooks over `state` for the loop's backtracking reset
/// (spec D6).  The snapshot is a full copy, so restoring reinstates the level
/// counters (and the segmentation metadata) exactly.
///
/// `state` is captured by reference: it must outlive the returned hooks.
[[nodiscard]] RefineStateHooks make_chebyshev_state_hooks(
    ChebyshevRefinementState& state);

}  // namespace detail

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
    size_t pde_solves = 0;                        ///< PDE solves used for this K_ref
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
///
/// The achieved errors and `target_met` describe the **assembled all-K_ref
/// surface** as measured by the mandatory final validation (spec D9), not the
/// single-K_ref sizing loop.
struct ChebyshevSegmentedAdaptiveResult {
    ChebyshevMultiKRefSurface surface;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;

    /// Diagnostics for the returned final surface (spec D7/D9), with the
    /// sizing-loop iterations appended for forensics.
    BuildDiagnostics diagnostics;
};

/// Builder for segmented Chebyshev surfaces (discrete dividends, multi-K_ref).
///
/// Performs shared setup (K_ref resolution, domain expansion, segment boundaries)
/// once in create(), then builds via fixed CC levels or adaptive refinement.
class ChebyshevSegmentedBuilder {
public:
    /// Create builder, performing shared setup.
    [[nodiscard]] static std::expected<ChebyshevSegmentedBuilder, PriceTableError>
    create(const SegmentedAdaptiveConfig& config, const IVGrid& domain);

    /// Build with fixed CC levels (no adaptive refinement).
    [[nodiscard]] std::expected<ChebyshevMultiKRefSurface, PriceTableError>
    build(std::array<size_t, 4> cc_levels = {5, 3, 2, 1}) const;

    /// Build with adaptive grid refinement.
    [[nodiscard]] std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
    build_adaptive(const AdaptiveGridParams& params) const;

private:
    struct AssembleResult {
        ChebyshevMultiKRefSurface surface;
        size_t pde_solves = 0;
    };

    ChebyshevSegmentedBuilder(
        SegmentedAdaptiveConfig config,
        std::vector<double> K_refs,
        SurfaceBounds domain,
        SurfaceBounds sample_domain,
        std::vector<double> seg_bounds,
        std::vector<bool> seg_is_gap);

    /// Build all K_ref surfaces (includes per-K_ref PDE solves) and compose.
    [[nodiscard]] std::expected<AssembleResult, PriceTableError>
    build_all_krefs(std::span<const double> m_nodes,
                    std::span<const double> tau_nodes,
                    std::span<const double> sigma_nodes,
                    std::span<const double> rate_nodes) const;

    [[nodiscard]] std::vector<double> generate_tau_nodes(size_t tau_level) const;

    struct ExtendedBounds {
        double m_lo, m_hi, sigma_lo, sigma_hi, rate_lo, rate_hi;
    };
    [[nodiscard]] ExtendedBounds compute_headroom(
        std::array<size_t, 4> cc_levels) const;

    SegmentedAdaptiveConfig config_;
    std::vector<double> K_refs_;
    SurfaceBounds domain_;         ///< node/support domain (incl. dividend span)
    SurfaceBounds sample_domain_;  ///< user-facing measurement domain (D2)
    std::vector<double> seg_bounds_;
    std::vector<bool> seg_is_gap_;
};

/// Build segmented Chebyshev surface with discrete dividend support.
/// Convenience wrapper around ChebyshevSegmentedBuilder.
[[nodiscard]] std::expected<ChebyshevSegmentedAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(const AdaptiveGridParams& params,
                                   const SegmentedAdaptiveConfig& config,
                                   const IVGrid& domain);

/// Build typed segmented Chebyshev surface from explicit CC levels.
/// Convenience wrapper around ChebyshevSegmentedBuilder.
[[nodiscard]] std::expected<ChebyshevMultiKRefSurface, PriceTableError>
build_chebyshev_segmented_manual(
    const SegmentedAdaptiveConfig& config,
    const IVGrid& domain,
    std::array<size_t, 4> cc_levels = {5, 3, 2, 1});

}  // namespace mango
