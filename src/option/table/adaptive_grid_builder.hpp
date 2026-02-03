// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/slice_cache.hpp"
#include "mango/option/table/error_attribution.hpp"
#include "mango/option/table/segmented_multi_kref_builder.hpp"
#include "mango/option/table/segmented_multi_kref_surface.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <optional>

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

    /// Build segmented multi-K_ref surface with adaptive grid refinement.
    /// Probes 2-3 representative K_refs, takes per-axis max grid sizes,
    /// then builds all segments with a uniform grid.
    [[nodiscard]] std::expected<SegmentedMultiKRefSurface, PriceTableError>
    build_segmented(const SegmentedAdaptiveConfig& config,
                    const std::vector<double>& moneyness_domain,
                    const std::vector<double>& vol_domain,
                    const std::vector<double>& rate_domain);

private:
    AdaptiveGridParams params_;
    SliceCache cache_;

    /// Compute hybrid IV/price error metric with vega floor
    /// Returns nullopt for low-vega regions where price error is within tolerance
    std::optional<double> compute_error_metric(double interpolated_price, double reference_price,
                                               double spot, double strike, double tau,
                                               double sigma, double rate, double dividend_yield) const;

    /// Build BatchAmericanOptionResult by merging cached and fresh results
    /// @param all_params All (σ,r) parameter combos in full batch order
    /// @param fresh_indices Indices into all_params that were freshly solved
    /// @param fresh_results Fresh solve results (parallel to fresh_indices)
    /// @return Merged batch result with all_params.size() entries
    BatchAmericanOptionResult merge_results(
        const std::vector<PricingParams>& all_params,
        const std::vector<size_t>& fresh_indices,
        const BatchAmericanOptionResult& fresh_results) const;
};

}  // namespace mango
