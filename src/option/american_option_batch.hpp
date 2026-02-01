// SPDX-License-Identifier: MIT
/**
 * @file american_option_batch.hpp
 * @brief Batch American option pricing solver for parallel processing
 *
 * Includes both regular batch solver and normalized chain solver (scale-invariant).
 */

#ifndef MANGO_AMERICAN_OPTION_BATCH_HPP
#define MANGO_AMERICAN_OPTION_BATCH_HPP

#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include <vector>
#include <expected>
#include <span>
#include <functional>
#include <memory>
#include <memory_resource>
#include <tuple>
#include <optional>

namespace mango {

// Forward declaration for PDE parameter grouping
struct PDEParameterGroup;

/**
 * Batch solver result containing individual results and aggregate statistics.
 */
struct BatchAmericanOptionResult {
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    size_t failed_count;  ///< Number of failed solves

    /// Check if all solves succeeded
    bool all_succeeded() const { return failed_count == 0; }
};

/// Batch American Option Solver
///
/// Solves multiple American options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially
/// for embarrassingly parallel workloads.
///
/// **Basic usage (per-option grids):**
/// ```cpp
/// std::vector<PricingParams> batch;
/// batch.emplace_back(spot, strike, maturity, rate, dividend_yield, type, sigma);
///
/// BatchAmericanOptionSolver solver;
/// auto results = solver.solve_batch(batch);
/// ```
///
/// **Adjusting grid accuracy:**
/// ```cpp
/// BatchAmericanOptionSolver solver;
/// GridAccuracyParams accuracy;
/// accuracy.tol = 1e-6;  // High accuracy mode
/// solver.set_grid_accuracy(accuracy);
/// auto results = solver.solve_batch(batch);
/// ```
///
/// **Price table usage (shared grid):**
/// ```cpp
/// // use_shared_grid=true: all options share one global grid
/// BatchAmericanOptionSolver solver;
/// auto results = solver.solve_batch(batch, true);
///
/// // Results contain full surfaces for interpolation
/// for (const auto& result_expected : results.results) {
///     if (result_expected.has_value()) {
///         const auto& result = result_expected.value();
///         auto spatial_solution = result.at_time(step_idx);
///     }
/// }
/// ```
///
/// **RECOMMENDED: Price table with snapshots (preserves normalized optimization):**
/// ```cpp
/// // Fast path: set_snapshot_times() enables 19,000× normalized chain speedup
/// std::vector<double> maturities = {0.25, 0.5, 1.0};
/// BatchAmericanOptionSolver solver;
/// solver.set_grid_accuracy(accuracy)
///       .set_snapshot_times(std::span{maturities});  // Fluent API
///
/// auto results = solver.solve_batch(batch, true);
///
/// // Extract snapshots for price table construction
/// for (const auto& result : results.results) {
///     if (result.has_value()) {
///         auto grid = result.value().grid();
///         // grid->num_snapshots() == 3
///         // Use extract_batch_results_to_4d() for interpolation table
///     }
/// }
/// ```
///
/// **Alternative: SetupCallback (disables normalized optimization):**
/// ```cpp
/// // Use only when per-option customization is required
/// auto setup = [](size_t idx, AmericanOptionSolver& solver) {
///     solver.set_snapshot_times(...);  // Per-option snapshots
///     solver.set_tolerance(...);        // Per-option tolerance
/// };
/// auto results = solver.solve_batch(batch, true, setup);
/// // Note: Falls back to regular batch (no normalized speedup)
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid, tol=1e-3)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Setup callback: called before each solve() to configure solver
    /// @param index Index of current option in params vector
    /// @param solver Reference to solver for pre-solve configuration
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

    /// Set grid accuracy parameters
    /// @param accuracy Grid accuracy parameters controlling size/resolution tradeoff
    /// @return Reference to this solver for method chaining
    BatchAmericanOptionSolver& set_grid_accuracy(const GridAccuracyParams& accuracy) {
        grid_accuracy_ = accuracy;
        return *this;
    }

    /// Get current grid accuracy parameters
    const GridAccuracyParams& grid_accuracy() const {
        return grid_accuracy_;
    }

    /// Disable normalized chain optimization (for benchmarking/debugging)
    /// @return Reference to this solver for method chaining
    BatchAmericanOptionSolver& set_use_normalized(bool enable) {
        use_normalized_ = enable;
        return *this;
    }

    bool use_normalized() const {
        return use_normalized_;
    }

    /// Set snapshot times for all solvers in the batch
    ///
    /// This is the recommended way to register snapshot times when using
    /// normalized chain optimization. Using SetupCallback to register snapshots
    /// will disable the normalized path (see solve_batch documentation).
    ///
    /// @param times Snapshot times to register for all options
    /// @return Reference to this solver for method chaining
    BatchAmericanOptionSolver& set_snapshot_times(std::span<const double> times) {
        snapshot_times_.assign(times.begin(), times.end());
        return *this;
    }

    /// Set TR-BDF2 configuration for all solvers in the batch
    BatchAmericanOptionSolver& set_trbdf2_config(const TRBDF2Config& config) {
        trbdf2_config_ = config;
        return *this;
    }

    /// Clear snapshot times
    /// @return Reference to this solver for method chaining
    BatchAmericanOptionSolver& clear_snapshot_times() {
        snapshot_times_.clear();
        return *this;
    }

    /// Solve a batch of American options with automatic routing
    ///
    /// Automatically routes to normalized chain solver when eligible
    /// (varying strikes, same maturity, no discrete dividends).
    ///
    /// **Snapshot Registration:**
    /// Use `set_snapshot_times()` before calling solve_batch() to register
    /// snapshots for all options. This approach preserves the normalized
    /// chain optimization.
    ///
    /// **SetupCallback Limitation:**
    /// When a SetupCallback is provided, the normalized path is disabled
    /// and solve_regular_batch() is used instead. This is because the
    /// normalized solver creates one PDE for multiple options, making
    /// per-option callbacks ambiguous. For common configuration needs
    /// (snapshots, tolerances), use dedicated APIs instead.
    ///
    /// @param params Vector of option parameters
    /// @param use_shared_grid If true, all options share one global grid
    /// @param setup Optional callback invoked after solver creation
    ///              (disables normalized path - see documentation above)
    /// @param custom_grid Optional custom grid specification (GridSpec + TimeDomain).
    ///                    When provided, bypasses auto-estimation and uses exact grid.
    ///                    When nullopt, uses grid_accuracy_ member for estimation.
    /// @return Batch result with individual results and failure count
    BatchAmericanOptionResult solve_batch(
        std::span<const PricingParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr,
        std::optional<PDEGridSpec> custom_grid = std::nullopt);

    /// Solve a batch of American options (vector overload)
    BatchAmericanOptionResult solve_batch(
        const std::vector<PricingParams>& params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr,
        std::optional<PDEGridSpec> custom_grid = std::nullopt)
    {
        return solve_batch(std::span{params}, use_shared_grid, setup, custom_grid);
    }

private:
    GridAccuracyParams grid_accuracy_;  ///< Grid accuracy parameters for automatic estimation
    std::vector<double> snapshot_times_;  ///< Snapshot times for all solvers (preserves normalized optimization)
    std::optional<TRBDF2Config> trbdf2_config_;

    // Normalized chain solver eligibility constants
    static constexpr double MAX_WIDTH = 5.8;       ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;         ///< Von Neumann stability
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    bool use_normalized_ = true;  ///< Enable normalized chain optimization

    /// Regular batch solving (fallback path)
    BatchAmericanOptionResult solve_regular_batch(
        std::span<const PricingParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr,
        std::optional<PDEGridSpec> custom_grid = std::nullopt);

    /// Regular batch solving (vector overload)
    BatchAmericanOptionResult solve_regular_batch(
        const std::vector<PricingParams>& params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr,
        std::optional<PDEGridSpec> custom_grid = std::nullopt);

    /// Check if batch qualifies for normalized solving
    bool is_normalized_eligible(
        std::span<const PricingParams> params,
        bool use_shared_grid) const;

    /// Trace why normalized path wasn't used
    void trace_ineligibility_reason(
        std::span<const PricingParams> params,
        bool use_shared_grid) const;

    /// Group options by PDE parameters for normalized solving
    std::vector<PDEParameterGroup> group_by_pde_parameters(
        std::span<const PricingParams> params) const;

    /// Fast path: normalized chain solving with PDE grouping
    BatchAmericanOptionResult solve_normalized_chain(
        std::span<const PricingParams> params,
        SetupCallback setup,
        std::optional<PDEGridSpec> custom_grid = std::nullopt);
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP
