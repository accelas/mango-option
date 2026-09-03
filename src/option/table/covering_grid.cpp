// SPDX-License-Identifier: MIT
#include "mango/option/table/covering_grid.hpp"

#include <algorithm>
#include <cmath>

namespace mango::detail {

void ensure_moneyness_coverage(GridAccuracyParams& accuracy,
                               std::span<const PricingParams> batch,
                               std::span<const double> log_moneyness_nodes)
{
    if (batch.empty() || log_moneyness_nodes.empty()) return;

    const auto [lo_it, hi_it] = std::minmax_element(
        log_moneyness_nodes.begin(), log_moneyness_nodes.end());
    const double required_half_width =
        std::max(std::abs(*lo_it), std::abs(*hi_it));

    // Compute max σ√T across the batch (floor to avoid division by zero)
    double max_sigma_sqrt_T = 0.0;
    for (const auto& p : batch) {
        max_sigma_sqrt_T = std::max(max_sigma_sqrt_T,
                                    p.volatility * std::sqrt(p.maturity));
    }
    max_sigma_sqrt_T = std::max(max_sigma_sqrt_T, 1e-10);

    constexpr double MARGIN = 1.1;  // 10% margin for boundary effects
    double required_n_sigma = (required_half_width / max_sigma_sqrt_T) * MARGIN;
    accuracy.n_sigma = std::max(accuracy.n_sigma, required_n_sigma);
}

PDEGridSpec materialize_covering_grid(GridAccuracyParams accuracy,
                                      std::span<const PricingParams> batch,
                                      std::span<const double> log_moneyness_nodes)
{
    ensure_moneyness_coverage(accuracy, batch, log_moneyness_nodes);
    auto [grid_spec, time_domain] = estimate_batch_pde_grid(batch, accuracy);
    return PDEGridConfig{grid_spec, time_domain.n_steps(), {}};
}

}  // namespace mango::detail
