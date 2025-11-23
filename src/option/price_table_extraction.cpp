/**
 * @file price_table_extraction.cpp
 * @brief Implementation of price table extraction utility
 */

#include "src/option/price_table_extraction.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/support/parallel.hpp"
#include <experimental/mdspan>
#include <cmath>
#include <vector>
#include <ranges>
#include <algorithm>

namespace mango {

void extract_batch_results_to_4d(
    const BatchAmericanOptionResult& batch_result,
    std::span<double> prices_4d,
    const PriceTableGrid& grid,
    double K_ref)
{
    const size_t Nm = grid.moneyness.size();
    const size_t Nt = grid.maturity.size();
    const size_t Nv = grid.volatility.size();
    const size_t Nr = grid.rate.size();
    const double T_max = grid.maturity.back();

    // Get n_time from first successful result (all share same grid)
    size_t n_time = 0;
    for (const auto& result_expected : batch_result.results) {
        if (result_expected.has_value() && result_expected->converged) {
            n_time = result_expected->grid()->num_snapshots();
            break;
        }
    }

    // Guard against missing snapshots (would cause divide-by-zero)
    if (n_time == 0) {
        // This indicates snapshots were not registered during solver setup.
        // Price table extraction requires snapshot_times to be set via SetupCallback.
        throw std::runtime_error(
            "extract_batch_results_to_4d: No snapshots recorded. "
            "Price table construction requires snapshot_times to be registered "
            "via SetupCallback (call solver.set_snapshot_times() before solve).");
    }

    // Precompute step indices for each maturity using ranges pipeline
    const double dt = T_max / n_time;
    auto compute_step_index = [dt, n_time](double maturity) -> size_t {
        double step_exact = maturity / dt - 1.0;
        long long step_rounded = std::llround(step_exact);
        if (step_rounded < 0) return 0;
        if (step_rounded >= static_cast<long long>(n_time)) return n_time - 1;
        return static_cast<size_t>(step_rounded);
    };

    auto step_indices_view = grid.maturity | std::views::transform(compute_step_index);
    std::vector<size_t> step_indices(step_indices_view.begin(), step_indices_view.end());

    // Precompute log-moneyness values (vectorizable: simple affine math, no dependencies)
    std::vector<double> log_moneyness(Nm);
    MANGO_PRAGMA_SIMD
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(grid.moneyness[i]);
    }

    // Create mdspan view for type-safe 4D indexing
    // Layout: [moneyness, maturity, volatility, rate] in row-major order
    using std::experimental::mdspan;
    using std::experimental::dextents;
    mdspan<double, dextents<size_t, 4>> prices_view(prices_4d.data(), Nm, Nt, Nv, Nr);

    // Extract prices from surfaces for each (σ, r) result
    // Embarrassingly parallel: each (σ,r) slice writes to unique offset,
    // spline objects are thread-local, all shared inputs are read-only
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
        // Decode flat index to (vol_idx, r_idx) from cartesian_product ordering
        const size_t vol_idx = idx / Nr;
        const size_t r_idx = idx % Nr;
        const auto& result_expected = batch_result.results[idx];
        if (!result_expected.has_value() || !result_expected->converged) {
            continue;  // Leave zeros for failed solves
        }
        const auto& result = result_expected.value();

        // Extract grid from result
        auto result_grid = result.grid();
        auto x_grid = result_grid->x();  // Span of spatial grid points
        const size_t n_space = x_grid.size();
        const double x_min = x_grid.front();

        // For each maturity time step
        for (size_t j = 0; j < grid.maturity.size(); ++j) {
            size_t step_idx = step_indices[j];
            std::span<const double> spatial_solution = result.at_time(step_idx);

            if (spatial_solution.empty()) {
                continue;
            }

            // Build cubic spline for this time step
            CubicSpline<double> spline;
            auto build_error = spline.build(x_grid, spatial_solution);
            if (build_error.has_value()) {
                // Fall back to boundary values if spline build fails
                for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                    const double x = log_moneyness[m_idx];
                    double V_norm = (x <= x_min) ? spatial_solution[0] : spatial_solution[n_space - 1];
                    // Type-safe 4D indexing via mdspan
                    prices_view[m_idx, j, vol_idx, r_idx] = K_ref * V_norm;
                }
                continue;
            }

            // Interpolate spatial solution to moneyness grid using cubic spline
            for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                const double x = log_moneyness[m_idx];
                double V_norm = spline.eval(x);

                // Store denormalized price with type-safe 4D indexing via mdspan
                prices_view[m_idx, j, vol_idx, r_idx] = K_ref * V_norm;
            }
        }
    }
}

} // namespace mango
