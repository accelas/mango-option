#pragma once

#include "src/option/table/adaptive_grid_types.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/slice_cache.hpp"
#include "src/option/table/error_attribution.hpp"
#include "src/option/option_chain.hpp"
#include "src/pde/core/grid.hpp"
#include "src/support/error_types.hpp"
#include <expected>

namespace mango {

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
    /// @param chain Option chain providing domain bounds
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps for PDE solver
    /// @param type Option type (default: PUT)
    /// @return AdaptiveResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionChain& chain,
          GridSpec<double> grid_spec,
          size_t n_time,
          OptionType type = OptionType::PUT);

private:
    AdaptiveGridParams params_;
    SliceCache cache_;

    /// Compute hybrid IV/price error metric with vega floor
    double compute_error_metric(double interpolated_price, double reference_price,
                                double spot, double strike, double tau,
                                double sigma, double rate, double dividend_yield) const;

    /// Refine grid in the specified dimension
    std::vector<double> refine_dimension(const std::vector<double>& current_grid,
                                         const std::vector<size_t>& problematic_bins,
                                         size_t dim) const;
};

}  // namespace mango
