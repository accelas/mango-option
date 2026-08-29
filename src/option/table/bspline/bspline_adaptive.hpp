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

    // Convergence stats.  The achieved errors and `target_met` describe the
    // **returned** assembled surface as measured by the final validation
    // (spec D9), not the probe loops -- including when the bumped-grid retry
    // is the surface returned.
    std::vector<IterationStats> iterations;  ///< Per-probe loop forensics
    double achieved_max_error = 0.0;         ///< Max error from final LHS validation
    double achieved_avg_error = 0.0;
    bool target_met = false;
    size_t total_pde_solves = 0;
    bool used_retry = false;                 ///< True if bumped-grid retry was returned

    /// Diagnostics for the returned final surface (spec D7/D9), with the
    /// per-probe iterations appended for forensics.
    BuildDiagnostics diagnostics;
};

/// Create a RefineFn that does B-spline midpoint insertion, targeted at the
/// physical focus intervals the loop passes (empty => uniform refinement
/// over the whole axis). Returns changed=false when the requested axis is
/// at max_points_per_dim or no midpoint could be inserted; never redirects
/// to a different axis.
[[nodiscard]] RefineFn make_bspline_refine_fn(const AdaptiveGridParams& params);

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
/// Performs shared setup (K_ref resolution, domain expansion) once in
/// create(), then builds via adaptive refinement.  Support headroom is *not*
/// baked in at create() time: the headroom scale depends on
/// `AdaptiveGridParams::min_moneyness_points` (spec D3), so `build_adaptive()`
/// derives the fit domain from the sample domain.
class BSplineSegmentedBuilder {
public:
    /// Create builder, performing shared setup.
    [[nodiscard]] static std::expected<BSplineSegmentedBuilder, PriceTableError>
    create(const SegmentedAdaptiveConfig& config, const IVGrid& domain);

    /// Build with adaptive grid refinement.
    /// The fit domain (sample domain + D3 headroom) is derived from `params`
    /// locally, so the builder itself stays immutable.
    [[nodiscard]] std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
    build_adaptive(const AdaptiveGridParams& params) const;

private:
    BSplineSegmentedBuilder(
        SegmentedAdaptiveConfig config,
        std::vector<double> K_refs,
        SurfaceBounds sample_domain,
        SurfaceBounds support_domain,
        IVGrid initial_grid);

    /// Assemble multi-K_ref surface from per-K_ref segmented surfaces.
    [[nodiscard]] std::expected<BSplineMultiKRefInner, PriceTableError>
    assemble(std::vector<BSplineSegmentedSurface> surfaces) const;

    SegmentedAdaptiveConfig config_;
    std::vector<double> K_refs_;
    SurfaceBounds sample_domain_;   ///< user-facing measurement domain (D2)
    SurfaceBounds support_domain_;  ///< sample domain + discrete-dividend span
    IVGrid initial_grid_;
};

/// Build segmented multi-K_ref B-spline surface with adaptive grid refinement.
/// Convenience wrapper around BSplineSegmentedBuilder.
[[nodiscard]] std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain);

}  // namespace mango
