/**
 * @file price_table_4d_builder.cpp
 * @brief Implementation of 4D price table builder
 */

#include "src/option/price_table_4d_builder.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/price_table_grid.hpp"
#include "src/option/price_table_extraction.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <ranges>

namespace mango {

std::expected<void, std::string> PriceTable4DBuilder::validate_grids() const {
    if (moneyness_.size() < 4) {
        return std::unexpected("Moneyness grid must have ≥4 points for cubic B-splines");
    }
    if (maturity_.size() < 4) {
        return std::unexpected("Maturity grid must have ≥4 points for cubic B-splines");
    }
    if (volatility_.size() < 4) {
        return std::unexpected("Volatility grid must have ≥4 points for cubic B-splines");
    }
    if (rate_.size() < 4) {
        return std::unexpected("Rate grid must have ≥4 points for cubic B-splines");
    }
    if (K_ref_ <= 0.0) {
        return std::unexpected("Reference strike K_ref must be positive");
    }

    // Verify sorted (using C++23 ranges)
    if (!std::ranges::is_sorted(moneyness_)) {
        return std::unexpected("Moneyness grid must be sorted");
    }
    if (!std::ranges::is_sorted(maturity_)) {
        return std::unexpected("Maturity grid must be sorted");
    }
    if (!std::ranges::is_sorted(volatility_)) {
        return std::unexpected("Volatility grid must be sorted");
    }
    if (!std::ranges::is_sorted(rate_)) {
        return std::unexpected("Rate grid must be sorted");
    }

    // Verify positive
    if (maturity_.front() <= 0.0) {
        return std::unexpected("Maturity must be positive");
    }
    if (volatility_.front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }

    // Verify moneyness values are positive (using C++23 ranges)
    // CRITICAL: PDE works in log-moneyness x = ln(m), so m must be > 0
    // Moneyness grid should represent S/K_ref ratios, not raw spots
    auto negative_it = std::ranges::find_if(moneyness_, [](double m) { return m <= 0.0; });
    if (negative_it != moneyness_.end()) {
        size_t i = std::distance(moneyness_.begin(), negative_it);
        return std::unexpected(
            "Moneyness values must be positive (m = S/K_ref > 0). "
            "Found m[" + std::to_string(i) + "] = " + std::to_string(*negative_it) + ". "
            "Note: moneyness represents spot ratios S/K_ref, not log-moneyness x = ln(S/K_ref)."
        );
    }

    return {};
}

std::expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    OptionType option_type,
    size_t n_space,
    size_t n_time,
    double dividend_yield)
{
    // Standard bounds: [-3.0, 3.0] log-moneyness
    constexpr double x_min = -3.0;
    constexpr double x_max = 3.0;
    return precompute(option_type, x_min, x_max, n_space, n_time, dividend_yield);
}

std::expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    const PriceTableConfig& config)
{
    if (config.x_bounds) {
        return precompute(
            config.option_type,
            config.x_bounds->first,
            config.x_bounds->second,
            config.n_space,
            config.n_time,
            config.dividend_yield);
    }

    return precompute(
        config.option_type,
        config.n_space,
        config.n_time,
        config.dividend_yield);
}

std::expected<PriceTable4DResult, std::string> PriceTable4DBuilder::precompute(
    OptionType option_type,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time,
    double dividend_yield)
{
    const size_t Nm = moneyness_.size();
    const size_t Nt = maturity_.size();
    const size_t Nv = volatility_.size();
    const size_t Nr = rate_.size();

    // Validate that requested moneyness range fits within PDE grid bounds
    // CRITICAL: PDE works in log-moneyness x = ln(m), and SnapshotInterpolator
    // uses natural cubic splines that extrapolate unpredictably outside knot domain.
    // If any requested ln(m) lies outside [x_min, x_max], the interpolation
    // will produce arbitrary extrapolation artifacts.

    const double x_min_requested = std::log(moneyness_.front());
    const double x_max_requested = std::log(moneyness_.back());

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return std::unexpected(
            "Requested moneyness range [" + std::to_string(moneyness_.front()) + ", " +
            std::to_string(moneyness_.back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. "
            "Either narrow the moneyness grid or expand the PDE x_min/x_max bounds. "
            "Example: for moneyness [0.7, 1.5], use x_min <= " +
            std::to_string(x_min_requested) + " and x_max >= " +
            std::to_string(x_max_requested) + "."
        );
    }

    // Allocate 4D price array
    std::vector<double> prices_4d(Nm * Nt * Nv * Nr, 0.0);

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Build batch parameters for all (σ, r) combinations using C++23 ranges
    const size_t batch_size = Nv * Nr;
    std::vector<AmericanOptionParams> batch_params;
    batch_params.reserve(batch_size);

    const double T_max = maturity_.back();

    // Cartesian product of volatility × rate grids
    for (auto [vol, r] : std::views::cartesian_product(volatility_, rate_)) {
        batch_params.emplace_back(
            K_ref_,           // spot
            K_ref_,           // strike
            T_max,            // maturity
            r,                // rate
            dividend_yield,   // dividend_yield
            option_type,      // type
            vol,              // volatility
            std::vector<std::pair<double, double>>{}  // discrete_dividends
        );
    }

    // Configure batch solver with explicit grid parameters
    BatchAmericanOptionSolver batch_solver;

    // Set grid accuracy to achieve desired n_space and n_time
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = n_space;
    accuracy.max_spatial_points = n_space;
    accuracy.max_time_steps = n_time;
    accuracy.alpha = 2.0;  // Keep default sinh clustering
    // Note: x_min, x_max are validated above but grid bounds are computed automatically
    // by BatchAmericanOptionSolver based on option parameters
    batch_solver.set_grid_accuracy(accuracy);

    // Setup callback to register snapshots (needed for price table extraction)
    auto setup_callback = [this]([[maybe_unused]] size_t idx, AmericanOptionSolver& solver) {
        // Register maturity grid as snapshot times so extract_batch_results_to_4d
        // can interpolate solutions at each maturity
        solver.set_snapshot_times(std::span{maturity_});
    };

    // Use BatchAmericanOptionSolver with shared grid (use_shared_grid=true)
    auto batch_result = batch_solver.solve_batch(batch_params, true, setup_callback);

    // Check for failures
    if (batch_result.failed_count > 0) {
        return std::unexpected("Failed to solve " + std::to_string(batch_result.failed_count) +
                         " out of " + std::to_string(batch_size) + " PDEs");
    }

    // Extract prices using shared helper
    PriceTableGrid grid{
        .moneyness = std::span{moneyness_},
        .maturity = std::span{maturity_},
        .volatility = std::span{volatility_},
        .rate = std::span{rate_},
        .K_ref = K_ref_
    };
    extract_batch_results_to_4d(batch_result, prices_4d, grid, K_ref_);

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);

    // Fit B-spline coefficients using generic N-D template
    auto fitter_result = BSplineNDSeparable<double, 4>::create(
        std::array<std::vector<double>, 4>{moneyness_, maturity_, volatility_, rate_});

    if (!fitter_result.has_value()) {
        return std::unexpected("B-spline fitter creation failed: " + fitter_result.error());
    }

    auto fit_result = fitter_result.value().fit(prices_4d, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});

    if (!fit_result.has_value()) {
        return std::unexpected("B-spline fitting failed: " + fit_result.error());
    }

    // Create workspace with all data
    auto workspace_result = PriceTableWorkspace::create(
        moneyness_, maturity_, volatility_, rate_,
        fit_result->coefficients,
        K_ref_, dividend_yield);

    if (!workspace_result.has_value()) {
        return std::unexpected("Workspace creation failed: " + workspace_result.error());
    }

    auto workspace = std::make_shared<PriceTableWorkspace>(std::move(workspace_result.value()));

    // Create evaluator for backward compatibility
    auto evaluator = std::make_shared<BSpline4D>(*workspace);

    // Populate fitting statistics from result (using C++23 ranges)
    BSplineFittingStats fitting_stats{
        .max_residual_axis0 = fit_result->max_residual_per_axis[0],
        .max_residual_axis1 = fit_result->max_residual_per_axis[1],
        .max_residual_axis2 = fit_result->max_residual_per_axis[2],
        .max_residual_axis3 = fit_result->max_residual_per_axis[3],
        .max_residual_overall = std::ranges::max(fit_result->max_residual_per_axis),
        .condition_axis0 = fit_result->condition_per_axis[0],
        .condition_axis1 = fit_result->condition_per_axis[1],
        .condition_axis2 = fit_result->condition_per_axis[2],
        .condition_axis3 = fit_result->condition_per_axis[3],
        .condition_max = std::ranges::max(fit_result->condition_per_axis),
        .failed_slices_axis0 = fit_result->failed_slices[0],
        .failed_slices_axis1 = fit_result->failed_slices[1],
        .failed_slices_axis2 = fit_result->failed_slices[2],
        .failed_slices_axis3 = fit_result->failed_slices[3],
        .failed_slices_total = fit_result->failed_slices[0] +
                               fit_result->failed_slices[1] +
                               fit_result->failed_slices[2] +
                               fit_result->failed_slices[3]
    };

    return PriceTable4DResult{
        .surface = PriceTableSurface(workspace),
        .evaluator = std::move(evaluator),
        .prices_4d = std::move(prices_4d),
        .n_pde_solves = Nv * Nr,  // Now correct: O(Nσ × Nr) not O(Nm × Nt × Nσ × Nr)
        .precompute_time_seconds = duration.count(),
        .fitting_stats = fitting_stats
    };
}

}  // namespace mango
