#include "src/option/price_table_builder.hpp"
#include "src/option/recursion_helpers.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/support/memory/aligned_arena.hpp"
#include "common/ivcalc_trace.h"
#include <cmath>
#include <limits>

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    if constexpr (N != 4) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE, N, 0);
        return std::unexpected("build() only supports N=4");
    }

    // Step 1: Validate axes
    auto axes_valid = axes.validate();
    if (!axes_valid.has_value()) {
        return std::unexpected("Invalid axes: " + axes_valid.error());
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

    // Step 6: Create metadata
    PriceTableMetadata metadata{
        .K_ref = config_.K_ref,
        .dividend_yield = config_.dividend_yield,
        .discrete_dividends = config_.discrete_dividends
    };

    // Step 7: Build immutable surface
    return PriceTableSurface<N>::build(axes, std::move(coeffs_result.value()), metadata);
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
        // Configure solver with default grid accuracy
        // The batch solver will use auto-estimation for grid sizing
        BatchAmericanOptionSolver solver;

        // Register maturity grid as snapshot times
        // This enables extract_tensor to access surfaces at each maturity point
        solver.set_snapshot_times(axes.grids[1]);  // axes.grids[1] = maturity axis

        // Solve batch with shared grid optimization (normalized chain solver)
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

                // Evaluate spline at each moneyness point
                for (size_t i = 0; i < Nm; ++i) {
                    double price = spline.eval(log_moneyness[i]);
                    tensor.view[i, j, σ_idx, r_idx] = price;
                }
            }
        }
    }

        return tensor;
    }
}

template <size_t N>
std::expected<std::vector<double>, std::string>
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

    return std::move(fit_result.value().coefficients);
}

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
