/**
 * @file price_table_4d_builder.cpp
 * @brief Implementation of 4D price table builder
 */

#include "price_table_4d_builder.hpp"
#include "price_table_snapshot_collector.hpp"
#include "american_option.hpp"
#include "trbdf2_config.hpp"
#include "root_finding.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mango {

void PriceTable4DBuilder::validate_grids() const {
    if (moneyness_.size() < 4) {
        throw std::invalid_argument("Moneyness grid must have ≥4 points for cubic B-splines");
    }
    if (maturity_.size() < 4) {
        throw std::invalid_argument("Maturity grid must have ≥4 points for cubic B-splines");
    }
    if (volatility_.size() < 4) {
        throw std::invalid_argument("Volatility grid must have ≥4 points for cubic B-splines");
    }
    if (rate_.size() < 4) {
        throw std::invalid_argument("Rate grid must have ≥4 points for cubic B-splines");
    }
    if (K_ref_ <= 0.0) {
        throw std::invalid_argument("Reference strike K_ref must be positive");
    }

    // Verify sorted
    auto is_sorted = [](const std::vector<double>& v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(moneyness_)) {
        throw std::invalid_argument("Moneyness grid must be sorted");
    }
    if (!is_sorted(maturity_)) {
        throw std::invalid_argument("Maturity grid must be sorted");
    }
    if (!is_sorted(volatility_)) {
        throw std::invalid_argument("Volatility grid must be sorted");
    }
    if (!is_sorted(rate_)) {
        throw std::invalid_argument("Rate grid must be sorted");
    }

    // Verify positive
    if (maturity_.front() <= 0.0) {
        throw std::invalid_argument("Maturity must be positive");
    }
    if (volatility_.front() <= 0.0) {
        throw std::invalid_argument("Volatility must be positive");
    }

    // Verify moneyness values are positive
    // CRITICAL: PDE works in log-moneyness x = ln(m), so m must be > 0
    // Moneyness grid should represent S/K_ref ratios, not raw spots
    for (size_t i = 0; i < moneyness_.size(); ++i) {
        if (moneyness_[i] <= 0.0) {
            throw std::invalid_argument(
                "Moneyness values must be positive (m = S/K_ref > 0). "
                "Found m[" + std::to_string(i) + "] = " + std::to_string(moneyness_[i]) + ". "
                "Note: moneyness represents spot ratios S/K_ref, not log-moneyness x = ln(S/K_ref)."
            );
        }
    }
}

PriceTable4DResult PriceTable4DBuilder::precompute(
    OptionType option_type,
    const AmericanOptionGrid& grid_config,
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

    if (x_min_requested < grid_config.x_min || x_max_requested > grid_config.x_max) {
        throw std::invalid_argument(
            "Requested moneyness range [" + std::to_string(moneyness_.front()) + ", " +
            std::to_string(moneyness_.back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(grid_config.x_min) + ", " +
            std::to_string(grid_config.x_max) + "]. "
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

    // Key insight: For each (σ, r) pair, solve ONE PDE at max maturity
    // and collect snapshots at all intermediate maturities.
    // This gives us the full 2D (m, τ) surface with O(Nσ × Nr) solves
    // instead of O(Nm × Nt × Nσ × Nr).

    const double T_max = maturity_.back();  // Maximum maturity
    const double dt = T_max / grid_config.n_time;

    // Precompute step indices for each maturity
    // CRITICAL: PDESolver calls process_snapshots(step, t) where t = (step+1)*dt
    // So to capture a snapshot at maturity τ, we need step k such that (k+1)*dt ≈ τ
    // Therefore: k = round(τ/dt) - 1
    std::vector<size_t> step_indices(Nt);
    for (size_t j = 0; j < Nt; ++j) {
        // Compute step index: k = round(τ/dt) - 1
        double step_exact = maturity_[j] / dt - 1.0;
        long long step_rounded = std::llround(step_exact);

        // Clamp to valid range [0, n_time-1]
        if (step_rounded < 0) {
            step_indices[j] = 0;  // Minimum maturity (at time dt)
        } else if (step_rounded >= static_cast<long long>(grid_config.n_time)) {
            step_indices[j] = grid_config.n_time - 1;  // Maximum maturity (at time n_time*dt)
        } else {
            step_indices[j] = static_cast<size_t>(step_rounded);
        }
    }

    // Loop over (σ, r) pairs only - this is the separable batch approach
    size_t failed_count = 0;

#pragma omp parallel for collapse(2) if(Nv * Nr > 4)
    for (size_t k = 0; k < Nv; ++k) {
        for (size_t l = 0; l < Nr; ++l) {
            try {
                // Set up AmericanOptionSolver for this (σ, r) pair
                AmericanOptionParams params{
                    .strike = K_ref_,
                    .spot = K_ref_,  // Will be overridden by spatial grid
                    .maturity = T_max,  // Solve to maximum maturity
                    .volatility = volatility_[k],
                    .rate = rate_[l],
                    .continuous_dividend_yield = dividend_yield,
                    .option_type = option_type
                };

                // Create snapshot collector for this (σ, r)
                PriceTableSnapshotCollectorConfig collector_config{
                    .moneyness = std::span{moneyness_},
                    .tau = std::span{maturity_},
                    .K_ref = K_ref_,
                    .exercise_type = ExerciseType::AMERICAN,
                    .option_type = option_type,  // CRITICAL: Pass option type for correct theta computation
                    .payoff_params = nullptr
                };
                PriceTableSnapshotCollector collector(collector_config);

                // Create American option solver
                AmericanOptionSolver solver(params, grid_config,
                                           TRBDF2Config{}, RootFindingConfig{});

                // Register snapshots at all maturity points
                for (size_t j = 0; j < Nt; ++j) {
                    solver.register_snapshot(step_indices[j], j, &collector);
                }

                // Solve PDE (this will collect all snapshots)
                auto result = solver.solve();

                if (!result.has_value()) {
#pragma omp atomic
                    ++failed_count;
                    continue;
                }

                // Extract the 2D (m, τ) surface from collector
                auto prices_2d = collector.prices();

                // Copy into 4D array at indices [:, :, k, l]
                for (size_t i = 0; i < Nm; ++i) {
                    for (size_t j = 0; j < Nt; ++j) {
                        size_t idx_2d = i * Nt + j;
                        size_t idx_4d = ((i * Nt + j) * Nv + k) * Nr + l;
                        prices_4d[idx_4d] = prices_2d[idx_2d];
                    }
                }

            } catch (const std::exception& e) {
#pragma omp atomic
                ++failed_count;
            }
        }
    }

    if (failed_count > 0) {
        throw std::runtime_error("Failed to solve " + std::to_string(failed_count) +
                                 " out of " + std::to_string(Nv * Nr) + " PDEs");
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);

    // Fit B-spline coefficients
    BSplineFitter4D fitter(moneyness_, maturity_, volatility_, rate_);
    auto fit_result = fitter.fit(prices_4d);

    if (!fit_result.success) {
        throw std::runtime_error("B-spline fitting failed: " + fit_result.error_message);
    }

    // Create evaluator
    auto evaluator = std::make_unique<BSpline4D_FMA>(
        moneyness_, maturity_, volatility_, rate_, fit_result.coefficients);

    return PriceTable4DResult{
        .evaluator = std::move(evaluator),
        .prices_4d = std::move(prices_4d),
        .n_pde_solves = Nv * Nr,  // Now correct: O(Nσ × Nr) not O(Nm × Nt × Nσ × Nr)
        .precompute_time_seconds = duration.count()
    };
}

}  // namespace mango
