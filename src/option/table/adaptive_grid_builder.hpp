// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <functional>
#include <memory>

namespace mango {

class SliceCache;  // forward declaration (defined in bspline_slice_cache.hpp)

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
/// Uses fresh FD solves for validation (not self-referential comparison).
///
/// **Usage:**
/// ```cpp
/// AdaptiveGridParams params;
/// params.target_iv_error = 0.0005;  // 5 bps
///
/// AdaptiveGridBuilder builder(params);
/// auto result = builder.build(chain, accuracy, OptionType::PUT);
/// ```
class AdaptiveGridBuilder {
public:
    explicit AdaptiveGridBuilder(AdaptiveGridParams params);
    ~AdaptiveGridBuilder();

    AdaptiveGridBuilder(AdaptiveGridBuilder&&) noexcept;
    AdaptiveGridBuilder& operator=(AdaptiveGridBuilder&&) noexcept;

    /// Build price table with adaptive grid refinement
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionGrid& chain,
          GridSpec<double> grid_spec,
          size_t n_time,
          OptionType type = OptionType::PUT);

    /// Build price table with adaptive grid refinement (auto-estimated grid)
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build(const OptionGrid& chain,
          PDEGridSpec pde_grid,
          OptionType type = OptionType::PUT);

    /// Build segmented multi-K_ref surface with adaptive grid refinement.
    [[nodiscard]] std::expected<SegmentedAdaptiveResult, PriceTableError>
    build_segmented(const SegmentedAdaptiveConfig& config,
                    const IVGrid& domain);

    /// Build Chebyshev surface with adaptive CC-level refinement.
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build_chebyshev(const OptionGrid& chain,
                    OptionType type = OptionType::PUT);

    /// Build segmented Chebyshev surface with discrete dividend support.
    [[nodiscard]] std::expected<AdaptiveResult, PriceTableError>
    build_segmented_chebyshev(const SegmentedAdaptiveConfig& config,
                              const IVGrid& domain);

private:
    AdaptiveGridParams params_;
    std::unique_ptr<SliceCache> cache_;
};

}  // namespace mango
