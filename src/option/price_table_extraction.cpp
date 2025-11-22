/**
 * @file price_table_extraction.cpp
 * @brief Implementation of price table extraction utility
 */

#include "src/option/price_table_extraction.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include <cmath>
#include <vector>

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

    // Precompute step indices for each maturity
    const double dt = T_max / n_time;
    std::vector<size_t> step_indices(Nt);
    for (size_t j = 0; j < Nt; ++j) {
        double step_exact = grid.maturity[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        if (step_rounded < 0) {
            step_indices[j] = 0;
        } else if (step_rounded >= static_cast<long long>(n_time)) {
            step_indices[j] = n_time - 1;
        } else {
            step_indices[j] = static_cast<size_t>(step_rounded);
        }
    }

    // Precompute log-moneyness values
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(grid.moneyness[i]);
    }

    // Extract prices from surfaces for each (σ, r) result
    const size_t slice_stride = Nv * Nr;
    for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
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
                    size_t table_idx = (m_idx * Nt + j) * slice_stride + idx;
                    prices_4d[table_idx] = K_ref * V_norm;
                }
                continue;
            }

            // Interpolate spatial solution to moneyness grid using cubic spline
            for (size_t m_idx = 0; m_idx < Nm; ++m_idx) {
                const double x = log_moneyness[m_idx];
                double V_norm = spline.eval(x);

                // Store denormalized price
                size_t table_idx = (m_idx * Nt + j) * slice_stride + idx;
                prices_4d[table_idx] = K_ref * V_norm;
            }
        }
    }
}

} // namespace mango
