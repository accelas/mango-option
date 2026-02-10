// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_slice_cache.hpp"
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <functional>

namespace mango {

/// Configuration for segmented adaptive grid building
struct SegmentedAdaptiveConfig {
    double spot;
    OptionType option_type;
    double dividend_yield;
    std::vector<Dividend> discrete_dividends;
    double maturity;
    MultiKRefConfig kref_config;
};

/// Type-erased surface handle for validation queries during adaptive refinement
struct SurfaceHandle {
    std::function<double(double spot, double strike, double tau,
                         double sigma, double rate)> price;
    size_t pde_solves = 0;
};

/// Adaptive grid builder for price tables
///
/// Iteratively refines grid density until target IV error is achieved.
/// Uses fresh FD solves for validation (not self-referential spline comparison).
///
/// **Usage:**
/// ```cpp
/// AdaptiveGridParams params;
/// params.target_iv_error = 0.0005;  // 5 bps
///
/// AdaptiveGridBuilder builder(params);
/// auto result = builder.build(chain, grid_spec, n_time, OptionType::PUT);
///
/// if (result->target_met) {
///     auto price = result->surface->value({m, tau, sigma, r});
/// }
/// ```
class AdaptiveGridBuilder {
public:
    /// Construct builder with configuration
    explicit AdaptiveGridBuilder(AdaptiveGridParams params);

    /// Build price table with adaptive grid refinement
    ///
    /// @param chain Option grid providing domain bounds
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps for PDE solver
    /// @param type Option type (default: PUT)
    /// @return AdaptiveResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionGrid& chain,
          GridSpec<double> grid_spec,
          size_t n_time,
          OptionType type = OptionType::PUT);

    /// Build price table with adaptive grid refinement (auto-estimated grid)
    ///
    /// @param chain Option grid providing domain bounds
    /// @param pde_grid PDE grid specification (PDEGridConfig or GridAccuracyParams)
    /// @param type Option type (default: PUT)
    /// @return AdaptiveResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionGrid& chain,
          PDEGridSpec pde_grid,
          OptionType type = OptionType::PUT);

    /// Build segmented multi-K_ref surface with adaptive grid refinement.
    /// Probes 2-3 representative K_refs, takes per-axis max grid sizes,
    /// then builds all segments with a uniform grid.
    ///
    /// `domain.moneyness` is interpreted as log-moneyness ln(S/K_ref).
    [[nodiscard]] std::expected<SegmentedAdaptiveResult, PriceTableError>
    build_segmented(const SegmentedAdaptiveConfig& config,
                    const IVGrid& domain);

    /// Build Chebyshev surface with adaptive CC-level refinement.
    /// Uses PDESliceCache for incremental PDE reuse across CC levels.
    ///
    /// @param chain Option grid providing domain bounds
    /// @param type Option type (default: PUT)
    /// @return AdaptiveResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build_chebyshev(const OptionGrid& chain,
                    OptionType type = OptionType::PUT);

    /// Build segmented Chebyshev surface with discrete dividend support.
    /// Uses TauSegmentSplit for tau segmentation at dividend dates and
    /// MultiKRefSplit for multi-K_ref interpolation.
    /// Stores V/K_ref directly (no EEP decomposition).
    ///
    /// @param config Segmented config with spot, dividends, maturity, K_refs
    /// @param domain IV grid providing domain bounds (moneyness in log S/K)
    /// @return AdaptiveResult with price_fn, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build_segmented_chebyshev(const SegmentedAdaptiveConfig& config,
                              const IVGrid& domain);

private:
    AdaptiveGridParams params_;
    SliceCache cache_;

    /// Compute IV error metric from price error and vega.
    /// Caller provides vega (FD American or BS European depending on path).
    /// Never returns nullopt — always counts the sample.
    double compute_error_metric(double price_error, double vega) const;

    /// Build BatchAmericanOptionResult by merging cached and fresh results
    /// @param all_params All (σ,r) parameter combos in full batch order
    /// @param fresh_indices Indices into all_params that were freshly solved
    /// @param fresh_results Fresh solve results (parallel to fresh_indices)
    /// @return Merged batch result with all_params.size() entries
    BatchAmericanOptionResult merge_results(
        const std::vector<PricingParams>& all_params,
        const std::vector<size_t>& fresh_indices,
        const BatchAmericanOptionResult& fresh_results) const;

    /// Build a cached price table surface from grids (standard adaptive path).
    /// Manages PDE batch solving, slice caching, tensor extraction, and surface construction.
    [[nodiscard]] std::expected<SurfaceHandle, PriceTableError>
    build_cached_surface(
        const std::vector<double>& m_grid,
        const std::vector<double>& tau_grid,
        const std::vector<double>& v_grid,
        const std::vector<double>& r_grid,
        double K_ref,
        double dividend_yield,
        const PDEGridSpec& pde_grid,
        OptionType type,
        size_t& build_iteration,
        std::shared_ptr<const PriceTableSurface>& last_surface,
        PriceTableAxes& last_axes);
};

}  // namespace mango
