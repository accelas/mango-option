/**
 * @file price_table_4d_builder.hpp
 * @brief Build 4D option price tables with B-spline interpolation
 *
 * Orchestrates multi-run PDE solves to build 4D price surfaces
 * (moneyness × maturity × volatility × rate) and fits B-spline
 * coefficients for fast interpolation.
 *
 * Usage:
 *   // Define 4D grids
 *   auto builder = PriceTable4DBuilder::create(
 *       {0.7, 0.8, ..., 1.3},   // 50 moneyness points
 *       {0.027, 0.1, ..., 2.0}, // 30 maturity points
 *       {0.10, 0.15, ..., 0.80},// 20 volatility points
 *       {0.0, 0.02, ..., 0.10}, // 10 rate points
 *       100.0                    // K_ref
 *   );
 *
 *   // Pre-compute prices (200 PDE solves, ~5 minutes on 16 cores)
 *   auto grid_config = AmericanOptionGrid{.n_space = 101, .n_time = 1000};
 *   builder.precompute(OptionType::PUT, grid_config);
 *
 *   // Get fast interpolator (~500ns per query)
 *   auto evaluator = builder.get_evaluator();
 *   double price = evaluator.eval(1.05, 0.25, 0.20, 0.05);
 *
 * Performance:
 * - Pre-computation: ~72 options/sec × 200 = ~3 minutes single-threaded
 * - With OpenMP: ~848 options/sec × 200 = ~24 seconds (16 cores)
 * - Query time: ~500ns per price lookup
 * - IV calculation: <30µs (vs 143ms FDM, 4800× speedup)
 */

#pragma once

#include "src/option/american_option.hpp"
#include "src/interpolation/bspline_4d.hpp"
#include "src/interpolation/bspline_fitter_4d.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace mango {

/// Statistics from B-spline fitting process
struct BSplineFittingStats {
    double max_residual_m = 0.0;       ///< Max residual along moneyness axis
    double max_residual_tau = 0.0;     ///< Max residual along maturity axis
    double max_residual_sigma = 0.0;   ///< Max residual along volatility axis
    double max_residual_r = 0.0;       ///< Max residual along rate axis
    double max_residual_overall = 0.0; ///< Max residual across all axes

    double condition_m = 0.0;          ///< Condition number estimate (moneyness)
    double condition_tau = 0.0;        ///< Condition number estimate (maturity)
    double condition_sigma = 0.0;      ///< Condition number estimate (volatility)
    double condition_r = 0.0;          ///< Condition number estimate (rate)
    double condition_max = 0.0;        ///< Maximum condition number

    size_t failed_slices_m = 0;        ///< Failed fits along moneyness
    size_t failed_slices_tau = 0;      ///< Failed fits along maturity
    size_t failed_slices_sigma = 0;    ///< Failed fits along volatility
    size_t failed_slices_r = 0;        ///< Failed fits along rate
    size_t failed_slices_total = 0;    ///< Total failed fits
};

/// Result of 4D price table building
struct PriceTable4DResult {
    std::unique_ptr<BSpline4D_FMA> evaluator;  ///< Fast B-spline evaluator
    std::vector<double> prices_4d;              ///< Raw 4D price array
    size_t n_pde_solves;                        ///< Number of PDE solves performed
    double precompute_time_seconds;             ///< Wall-clock time for pre-computation
    BSplineFittingStats fitting_stats;          ///< B-spline fitting diagnostics
};

/// 4D Price Table Builder
///
/// Builds interpolation-ready 4D option price surfaces by:
/// 1. Running PDE solver for each (σ, r) combination
/// 2. Evaluating prices at (m, τ) grid points
/// 3. Assembling into 4D array
/// 4. Fitting B-spline coefficients
///
/// Memory: O(Nm × Nτ × Nσ × Nr) for price storage
/// Time: O(Nσ × Nr × PDE_solve_time) for pre-computation
class PriceTable4DBuilder {
public:
    /// Create builder with 4D grids
    ///
    /// @param moneyness Moneyness grid m = S/K (sorted, ≥4 points)
    /// @param maturity Maturity grid τ in years (sorted, ≥4 points)
    /// @param volatility Volatility grid σ (sorted, ≥4 points)
    /// @param rate Risk-free rate grid r (sorted, ≥4 points)
    /// @param K_ref Reference strike price
    static PriceTable4DBuilder create(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref)
    {
        return PriceTable4DBuilder(
            std::move(moneyness),
            std::move(maturity),
            std::move(volatility),
            std::move(rate),
            K_ref
        );
    }

    /// Pre-compute all option prices on 4D grid
    ///
    /// Runs PDE solver for each (σ, r) combination and assembles
    /// prices into 4D array. Uses OpenMP for parallelization if available.
    ///
    /// @param option_type Call or Put
    /// @param grid_config PDE grid configuration
    /// @param dividend_yield Continuous dividend yield (default: 0)
    /// @return Result with fitted B-spline evaluator
    expected<PriceTable4DResult, std::string> precompute(
        OptionType option_type,
        const AmericanOptionGrid& grid_config,
        double dividend_yield = 0.0);

    /// Get grid dimensions
    std::tuple<size_t, size_t, size_t, size_t> dimensions() const {
        return {moneyness_.size(), maturity_.size(), volatility_.size(), rate_.size()};
    }

private:
    PriceTable4DBuilder(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref)
        : moneyness_(std::move(moneyness))
        , maturity_(std::move(maturity))
        , volatility_(std::move(volatility))
        , rate_(std::move(rate))
        , K_ref_(K_ref)
    {
        auto validation_result = validate_grids();
        if (!validation_result) {
            throw std::invalid_argument(validation_result.error());
        }
    }

    expected<void, std::string> validate_grids() const;

    std::vector<double> moneyness_;
    std::vector<double> maturity_;
    std::vector<double> volatility_;
    std::vector<double> rate_;
    double K_ref_;
};

}  // namespace mango
