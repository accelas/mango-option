// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <functional>
#include <vector>

namespace mango {

struct SegmentedAdaptiveConfig;  // forward declare (from adaptive_grid_builder.hpp)

/// Result of adaptive Chebyshev surface construction
struct ChebyshevAdaptiveResult {
    /// Price function (type-erased; underlying Chebyshev surface captured
    /// in the closure via shared_ptr)
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
[[nodiscard]] std::expected<ChebyshevAdaptiveResult, PriceTableError>
build_adaptive_chebyshev_segmented(const AdaptiveGridParams& params,
                                   const SegmentedAdaptiveConfig& config,
                                   const IVGrid& domain);

}  // namespace mango
