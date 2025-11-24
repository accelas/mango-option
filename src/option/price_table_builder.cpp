#include "src/option/price_table_builder.hpp"
#include "src/option/recursion_helpers.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/support/memory/aligned_arena.hpp"
#include "src/support/ivcalc_trace.h"
#include "src/pde/core/time_domain.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<PriceTableResult<N>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    if constexpr (N != 4) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, N, 0);
        return std::unexpected("build() only supports N=4");
    }

    // Step 1: Validate axes
    auto axes_valid = axes.validate();
    if (!axes_valid.has_value()) {
        auto err = axes_valid.error();
        return std::unexpected(
            "Invalid axes (error code " + std::to_string(static_cast<int>(err.code)) +
            ", value=" + std::to_string(err.value) + ")");
    }

    // Check minimum 4 points per axis (B-spline requirement)
    for (size_t i = 0; i < N; ++i) {
        if (axes.grids[i].size() < 4) {
            return std::unexpected("Axis " + std::to_string(i) +
                                   " has only " + std::to_string(axes.grids[i].size()) +
                                   " points (need >=4 for cubic B-splines)");
        }
    }

    // Check positive moneyness (needed for log)
    if (axes.grids[0].front() <= 0.0) {
        return std::unexpected("Moneyness must be positive (needed for log)");
    }

    // Check positive maturity (strict > 0)
    if (axes.grids[1].front() <= 0.0) {
        return std::unexpected("Maturity must be positive (tau > 0 required for PDE time domain)");
    }

    // Check positive volatility
    if (axes.grids[2].front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }

    // Check K_ref > 0
    if (config_.K_ref <= 0.0) {
        return std::unexpected("Reference strike K_ref must be positive");
    }

    // Check PDE domain coverage
    const double x_min_requested = std::log(axes.grids[0].front());
    const double x_max_requested = std::log(axes.grids[0].back());
    const double x_min = config_.grid_estimator.x_min();
    const double x_max = config_.grid_estimator.x_max();

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return std::unexpected(
            "Requested moneyness range [" + std::to_string(axes.grids[0].front()) + ", " +
            std::to_string(axes.grids[0].back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. "
            "Narrow the moneyness grid or expand the PDE domain."
        );
    }

    // Step 2: Generate batch (Nσ × Nr entries)
    auto batch_params = make_batch(axes);
    if (batch_params.empty()) {
        return std::unexpected("make_batch returned empty batch");
    }

    // Step 3: Solve batch with snapshot registration
    auto batch_result = solve_batch(batch_params, axes);
    if (batch_result.failed_count > 0) {
        return std::unexpected(
            "solve_batch had " + std::to_string(batch_result.failed_count) +
            " failures out of " + std::to_string(batch_result.results.size()));
    }

    // Count PDE solves (successful results)
    size_t n_pde_solves = batch_result.results.size() - batch_result.failed_count;

    // Step 4: Extract tensor via interpolation
    auto tensor_result = extract_tensor(batch_result, axes);
    if (!tensor_result.has_value()) {
        return std::unexpected("extract_tensor failed: " + tensor_result.error());
    }

    // Step 5: Fit B-spline coefficients
    auto coeffs_result = fit_coeffs(tensor_result.value(), axes);
    if (!coeffs_result.has_value()) {
        return std::unexpected("fit_coeffs failed: " + coeffs_result.error());
    }

    auto& fit_result = coeffs_result.value();
    auto coefficients = std::move(fit_result.coefficients);
    BSplineFittingStats fitting_stats = fit_result.stats;

    // Step 6: Create metadata
    PriceTableMetadata metadata{
        .K_ref = config_.K_ref,
        .dividend_yield = config_.dividend_yield,
        .discrete_dividends = config_.discrete_dividends
    };

    // Step 7: Build immutable surface
    auto surface_result = PriceTableSurface<N>::build(axes, std::move(coefficients), metadata);
    if (!surface_result.has_value()) {
        return std::unexpected("Surface build failed: " + surface_result.error());
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Return full result with diagnostics
    return PriceTableResult<N>{
        .surface = std::move(surface_result.value()),
        .n_pde_solves = n_pde_solves,
        .precompute_time_seconds = elapsed,
        .fitting_stats = fitting_stats
    };
}

template <size_t N>
std::vector<AmericanOptionParams>
PriceTableBuilder<N>::make_batch(const PriceTableAxes<N>& axes) const {
    if constexpr (N == 4) {
        std::vector<AmericanOptionParams> batch;

        // Iterate only over high-cost axes: axes[2] (σ) and axes[3] (r)
        // This creates Nσ × Nr batch entries, NOT Nm × Nt × Nσ × Nr
        // Each solve produces a surface over (m, τ) that gets reused
        const size_t Nσ = axes.grids[2].size();
        const size_t Nr = axes.grids[3].size();
        batch.reserve(Nσ * Nr);

        // Normalized parameters: Spot = Strike = K_ref
        // Moneyness and maturity are handled via grid interpolation in extract_tensor
        const double K_ref = config_.K_ref;

        for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
            for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
                double sigma = axes.grids[2][σ_idx];
                double r = axes.grids[3][r_idx];

                // Normalized solve: Spot = Strike = K_ref
                // Surface will be interpolated across m and τ in extract_tensor
                AmericanOptionParams params(
                    K_ref,                          // spot
                    K_ref,                          // strike
                    axes.grids[1].back(),           // maturity (max for this σ,r)
                    r,                              // rate
                    config_.dividend_yield,         // dividend_yield
                    config_.option_type,            // type
                    sigma,                          // volatility
                    config_.discrete_dividends      // discrete_dividends
                );

                batch.push_back(params);
            }
        }

        return batch;
    } else {
        // Return empty batch for N≠4
        return {};
    }
}

template <size_t N>
BatchAmericanOptionResult
PriceTableBuilder<N>::solve_batch(
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        // Return empty result for N≠4
        BatchAmericanOptionResult result;
        result.failed_count = batch.size();
        return result;
    } else {
        BatchAmericanOptionSolver solver;

        // Apply grid configuration from PriceTableConfig
        GridAccuracyParams accuracy;

        // Set spatial grid bounds (allow estimation within range)
        accuracy.min_spatial_points = std::min(config_.grid_estimator.n_points(), size_t(100));
        accuracy.max_spatial_points = std::max(config_.grid_estimator.n_points(), size_t(1200));
        accuracy.max_time_steps = config_.n_time;

        // Extract alpha parameter for sinh-spaced grids
        if (config_.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
            accuracy.alpha = config_.grid_estimator.concentration();
        }

        solver.set_grid_accuracy(accuracy);

        // Register maturity grid as snapshot times
        // This enables extract_tensor to access surfaces at each maturity point
        solver.set_snapshot_times(axes.grids[1]);  // axes.grids[1] = maturity axis

        // Solve batch with shared grid optimization (normalized chain solver)
        // NOTE: custom_grid parameter not used here - causes all solves to fail when
        // options have spot==strike (normalized case). Grid bounds are validated
        // above to ensure PDE domain covers requested moneyness range.
        return solver.solve_batch(batch, true);  // use_shared_grid = true
    }
}

template <size_t N>
std::expected<PriceTensor<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        return std::unexpected("extract_tensor only supports N=4");
    } else {
        const size_t Nm = axes.grids[0].size();  // moneyness
        const size_t Nt = axes.grids[1].size();  // maturity
        const size_t Nσ = axes.grids[2].size();  // volatility
        const size_t Nr = axes.grids[3].size();  // rate

    // Verify batch size matches (σ, r) grid
    const size_t expected_batch_size = Nσ * Nr;
    if (batch.results.size() != expected_batch_size) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE,
            batch.results.size(), expected_batch_size);
        return std::unexpected(
            "Batch size mismatch: expected " + std::to_string(expected_batch_size) +
            " results (Nσ × Nr), got " + std::to_string(batch.results.size()));
    }

    // Create tensor
    const size_t total_points = Nm * Nt * Nσ * Nr;
    const size_t tensor_bytes = total_points * sizeof(double);
    const size_t arena_bytes = tensor_bytes + 64;  // 64-byte alignment padding

    auto arena = memory::AlignedArena::create(arena_bytes);
    if (!arena.has_value()) {
        return std::unexpected("Failed to create arena: " + arena.error());
    }

    std::array<size_t, N> shape = {Nm, Nt, Nσ, Nr};
    auto tensor_result = PriceTensor<N>::create(shape, arena.value());
    if (!tensor_result.has_value()) {
        return std::unexpected("Failed to create tensor: " + tensor_result.error());
    }

    auto tensor = tensor_result.value();

    // Precompute log-moneyness for interpolation
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(axes.grids[0][i]);
    }

    // Extract prices from each (σ, r) surface
    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            size_t batch_idx = σ_idx * Nr + r_idx;
            const auto& result_expected = batch.results[batch_idx];

            if (!result_expected.has_value()) {
                // Fill with NaN for failed solves
                for (size_t i = 0; i < Nm; ++i) {
                    for (size_t j = 0; j < Nt; ++j) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                continue;
            }

            const auto& result = result_expected.value();
            auto grid = result.grid();
            auto x_grid = grid->x();  // Spatial grid (log-moneyness)

            // For each maturity snapshot
            for (size_t j = 0; j < Nt; ++j) {
                // Get spatial solution at this maturity
                std::span<const double> spatial_solution = result.at_time(j);

                // Interpolate across moneyness using cubic spline
                // This resamples the PDE solution onto our moneyness grid
                CubicSpline<double> spline;
                auto build_error = spline.build(x_grid, spatial_solution);

                if (build_error.has_value()) {
                    // Spline fitting failed, fill with NaN
                    for (size_t i = 0; i < Nm; ++i) {
                        tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                    continue;
                }

                // Evaluate spline at each moneyness point and scale by K_ref
                // PDE solves are normalized (Spot=Strike=K_ref), so V_normalized
                // needs to be scaled back to actual prices: V_actual = K_ref * V_norm
                const double K_ref = config_.K_ref;
                for (size_t i = 0; i < Nm; ++i) {
                    double normalized_price = spline.eval(log_moneyness[i]);
                    tensor.view[i, j, σ_idx, r_idx] = K_ref * normalized_price;
                }
            }
        }
    }

        return tensor;
    }
}

template <size_t N>
std::expected<typename PriceTableBuilder<N>::FitCoeffsResult, std::string>
PriceTableBuilder<N>::fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const
{
    if constexpr (N != 4) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, N, 0);
        return std::unexpected(
            "fit_coeffs only supports N=4 dimensions. Requested N=" +
            std::to_string(N));
    }

    // Extract grids for BSplineNDSeparable
    std::array<std::vector<double>, N> grids;
    for (size_t i = 0; i < N; ++i) {
        grids[i] = axes.grids[i];
    }

    // Create fitter
    auto fitter_result = BSplineNDSeparable<double, N>::create(std::move(grids));
    if (!fitter_result.has_value()) {
        return std::unexpected("Failed to create fitter: " + fitter_result.error());
    }

    // Extract values from tensor (convert mdspan to vector)
    size_t total_points = axes.total_points();
    std::vector<double> values;
    values.reserve(total_points);

    // Extract in row-major order using for_each_axis_index
    if constexpr (N == 4) {
        for_each_axis_index<0>(axes, [&](const std::array<size_t, N>& indices) {
            values.push_back(tensor.view[indices[0], indices[1], indices[2], indices[3]]);
        });
    }

    // Fit B-spline coefficients
    auto fit_result = fitter_result->fit(values);
    if (!fit_result.has_value()) {
        return std::unexpected("B-spline fitting failed: " + fit_result.error());
    }

    const auto& result = fit_result.value();

    // Map BSplineNDSeparableResult to BSplineFittingStats
    BSplineFittingStats stats;
    stats.max_residual_axis0 = result.max_residual_per_axis[0];
    stats.max_residual_axis1 = result.max_residual_per_axis[1];
    stats.max_residual_axis2 = result.max_residual_per_axis[2];
    stats.max_residual_axis3 = result.max_residual_per_axis[3];
    stats.max_residual_overall = *std::max_element(
        result.max_residual_per_axis.begin(),
        result.max_residual_per_axis.end()
    );

    stats.condition_axis0 = result.condition_per_axis[0];
    stats.condition_axis1 = result.condition_per_axis[1];
    stats.condition_axis2 = result.condition_per_axis[2];
    stats.condition_axis3 = result.condition_per_axis[3];
    stats.condition_max = *std::max_element(
        result.condition_per_axis.begin(),
        result.condition_per_axis.end()
    );

    stats.failed_slices_axis0 = result.failed_slices[0];
    stats.failed_slices_axis1 = result.failed_slices[1];
    stats.failed_slices_axis2 = result.failed_slices[2];
    stats.failed_slices_axis3 = result.failed_slices[3];
    stats.failed_slices_total = std::accumulate(
        result.failed_slices.begin(),
        result.failed_slices.end(),
        size_t(0)
    );

    return FitCoeffsResult{
        .coefficients = std::move(result.coefficients),
        .stats = stats
    };
}

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
