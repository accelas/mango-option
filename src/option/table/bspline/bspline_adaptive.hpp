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

struct SegmentedAdaptiveConfig;  // forward declare from adaptive_grid_builder.hpp

/// Result of adaptive B-spline surface construction
struct BSplineAdaptiveResult {
    std::shared_ptr<const PriceTableSurface> surface;
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
};

/// Build B-spline price table with adaptive grid refinement.
///
/// Uses cached PDE solver (SliceCache) for incremental builds.
/// Grid is iteratively refined via run_refinement() until target IV error is met.
[[nodiscard]] std::expected<BSplineAdaptiveResult, PriceTableError>
build_adaptive_bspline(const AdaptiveGridParams& params,
                       const OptionGrid& chain,
                       PDEGridSpec pde_grid,
                       OptionType type = OptionType::PUT);

/// Build segmented multi-K_ref B-spline surface with adaptive grid refinement.
///
/// Probes representative K_refs, runs adaptive refinement per probe,
/// aggregates grid sizes, then builds all segments.
[[nodiscard]] std::expected<BSplineSegmentedAdaptiveResult, PriceTableError>
build_adaptive_bspline_segmented(const AdaptiveGridParams& params,
                                 const SegmentedAdaptiveConfig& config,
                                 const IVGrid& domain);

}  // namespace mango
