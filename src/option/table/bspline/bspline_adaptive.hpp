// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace mango {

/// Result of adaptive B-spline surface construction
struct BSplineAdaptiveResult {
    std::shared_ptr<const BSplineND<double, 4>> spline;
    PriceTableAxesND<4> axes;
    double K_ref = 0.0;
    double dividend_yield = 0.0;
    std::vector<IterationStats> iterations;
    double achieved_max_error = 0.0;
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
};

/// Result of adaptive segmented B-spline surface construction
struct BSplineSegmentedAdaptiveResult {
    BSplineMultiKRefInner surface;
    IVGrid grid;
    int tau_points_per_segment;

    // Convergence stats (aggregated across probe K_refs)
    std::vector<IterationStats> iterations;  ///< Per-probe worst-case iterations
    double achieved_max_error = 0.0;         ///< Max error from final LHS validation
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
    bool used_retry = false;                 ///< True if bumped-grid retry was used
};

/// Build B-spline price table with adaptive grid refinement.
///
/// Uses cached PDE solver (BSplinePDECache) for incremental builds.
/// Grid is iteratively refined via run_refinement() until target IV error is met.
[[nodiscard]] std::expected<BSplineAdaptiveResult, PriceTableError>
build_adaptive_bspline(const AdaptiveGridParams& params,
                       const OptionGrid& chain,
                       PDEGridSpec pde_grid,
                       OptionType type = OptionType::PUT);

/// Builder for segmented B-spline surfaces (discrete dividends, multi-K_ref).
///
/// Performs shared setup (K_ref resolution, domain expansion, headroom)
/// once in create(), then builds via adaptive refinement.
class BSplineSegmentedBuilder {
public:
    /// Create builder, performing shared setup.
    [[nodiscard]] static std::expected<BSplineSegmentedBuilder, PriceTableError>
    create(const SegmentedAdaptiveConfig& config, const IVGrid& domain);

    /// Build with adaptive grid refinement.
    [[nodiscard]] std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
    build_adaptive(const AdaptiveGridParams& params) const;

private:
    BSplineSegmentedBuilder(
        SegmentedAdaptiveConfig config,
        std::vector<double> K_refs,
        DomainBounds domain,
        IVGrid initial_grid);

    /// Assemble multi-K_ref surface from per-K_ref segmented surfaces.
    [[nodiscard]] std::expected<BSplineMultiKRefInner, PriceTableError>
    assemble(std::vector<BSplineSegmentedSurface> surfaces) const;

    SegmentedAdaptiveConfig config_;
    std::vector<double> K_refs_;
    DomainBounds domain_;
    IVGrid initial_grid_;
};

/// Build segmented multi-K_ref B-spline surface with adaptive grid refinement.
/// Convenience wrapper around BSplineSegmentedBuilder.
[[nodiscard]] std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain);

}  // namespace mango
