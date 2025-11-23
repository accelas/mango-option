#include "src/option/price_table_builder.hpp"
#include "src/option/recursion_helpers.hpp"

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<std::shared_ptr<const PriceTableSurface<N>>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    // TODO: Implement full pipeline (Phases 8-10)
    // This skeleton will be completed in subsequent phases:
    // - Phase 8: Parallel PDE solve batch
    // - Phase 9: N-dimensional B-spline fitting
    // - Phase 10: PriceTableSurface construction
    //
    // For now, return error to indicate incomplete implementation.
    // Tests explicitly document this is a skeleton.
    return std::unexpected("PriceTableBuilder::build() not yet implemented");
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
        // Configure solver with grid accuracy from config
        BatchAmericanOptionSolver solver;

        // Set grid accuracy parameters based on config's grid_estimator
        GridAccuracyParams accuracy;
        // Use the grid estimator's bounds and size to configure accuracy
        // The normalized chain solver will use these parameters
        accuracy.min_spatial_points = config_.grid_estimator.n_points();
        accuracy.max_spatial_points = config_.grid_estimator.n_points();
        accuracy.max_time_steps = config_.n_time;

        // Transfer alpha parameter if grid is sinh-spaced
        if (config_.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
            accuracy.alpha = config_.grid_estimator.concentration();
        }

        solver.set_grid_accuracy(accuracy);

        // Register maturity grid as snapshot times
        // This enables extract_tensor to access surfaces at each maturity point
        solver.set_snapshot_times(axes.grids[1]);  // axes.grids[1] = maturity axis

        // Solve batch with shared grid optimization (normalized chain solver)
        return solver.solve_batch(batch, true);  // use_shared_grid = true
    }
}

// Explicit instantiations
template class PriceTableBuilder<2>;
template class PriceTableBuilder<3>;
template class PriceTableBuilder<4>;
template class PriceTableBuilder<5>;

} // namespace mango
