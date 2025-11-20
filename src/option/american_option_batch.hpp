/**
 * @file american_option_batch.hpp
 * @brief Batch American option pricing solver for parallel processing
 */

#ifndef MANGO_AMERICAN_OPTION_BATCH_HPP
#define MANGO_AMERICAN_OPTION_BATCH_HPP

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/option/grid_estimation.hpp"
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/pde/core/grid.hpp"
#include <vector>
#include <expected>
#include <span>
#include <functional>
#include <memory>
#include <memory_resource>
#include <tuple>

namespace mango {

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
/// std::vector<AmericanOptionParams> batch;
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
    void set_grid_accuracy(const GridAccuracyParams& accuracy) {
        grid_accuracy_ = accuracy;
    }

    /// Get current grid accuracy parameters
    const GridAccuracyParams& grid_accuracy() const {
        return grid_accuracy_;
    }

    /// Solve a batch of American options
    ///
    /// @param params Vector of option parameters
    /// @param use_shared_grid If true, all options share one global grid (required for price tables).
    ///                        If false (default), each option gets its own optimal grid.
    ///                        Shared grid enables at_time() access by populating surface_2d.
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Batch result with individual results and failure count
    BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        if (params.empty()) {
            return BatchAmericanOptionResult{.results = {}, .failed_count = 0};
        }

        std::vector<std::expected<AmericanOptionResult, SolverError>> results(params.size());
        size_t failed_count = 0;

        // Precompute shared grid if needed
        std::optional<std::tuple<GridSpec<double>, size_t>> shared_grid;
        if (use_shared_grid) {
            shared_grid = compute_global_grid_for_batch(params, grid_accuracy_);
        }

        MANGO_PRAGMA_PARALLEL
        {
            // Per-thread pool for cheap workspace reuse
            std::pmr::unsynchronized_pool_resource thread_pool;

            // Per-thread workspace (only for shared grid strategy)
            std::shared_ptr<AmericanSolverWorkspace> thread_workspace;
            if (use_shared_grid) {
                auto [grid_spec, n_time] = shared_grid.value();
                auto workspace_result = AmericanSolverWorkspace::create(
                    grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
                if (workspace_result.has_value()) {
                    thread_workspace = workspace_result.value();
                }
                // If creation failed, thread_workspace remains null and we'll fail in loop
            }

            MANGO_PRAGMA_FOR
            for (size_t i = 0; i < params.size(); ++i) {
                // Get or create workspace for this iteration
                std::shared_ptr<AmericanSolverWorkspace> workspace;
                if (use_shared_grid) {
                    // Shared grid: reuse thread workspace
                    workspace = thread_workspace;
                } else {
                    // Per-option grid: create workspace for this option
                    auto [grid_spec, n_time] = estimate_grid_for_option(params[i], grid_accuracy_);
                    auto workspace_result = AmericanSolverWorkspace::create(
                        grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
                    if (workspace_result.has_value()) {
                        workspace = workspace_result.value();
                    }
                }

                if (!workspace) {
                    // Workspace creation failed (either shared or per-option)
                    results[i] = std::unexpected(SolverError{
                        .code = SolverErrorCode::InvalidConfiguration,
                        .message = "Failed to create workspace",
                        .iterations = 0
                    });
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                    continue;
                }

                // Create solver (after CRTP revert: direct constructor, no factory)
                // Allocate surface buffer for full time history: (n_time + 1) × n_space
                std::vector<double> surface_buffer((workspace->n_time() + 1) * workspace->n_space());
                AmericanOptionSolver solver(params[i], workspace, surface_buffer);

                // Invoke setup callback if provided
                if (setup) {
                    setup(i, solver);
                }

                // Solve
                results[i] = solver.solve();
                if (!results[i].has_value()) {
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                }
            }
        }

        return BatchAmericanOptionResult{
            .results = std::move(results),
            .failed_count = failed_count
        };
    }

    /// Solve a batch of American options (vector overload)
    BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, use_shared_grid, setup);
    }

private:
    GridAccuracyParams grid_accuracy_;  ///< Grid accuracy parameters for automatic estimation
};

/// Solve a single American option with automatic grid determination
///
/// Convenience API that automatically determines optimal grid parameters
/// based on option characteristics, eliminating need for manual grid specification.
///
/// @param params Option parameters
/// @return Expected containing result on success, error on failure
inline std::expected<AmericanOptionResult, SolverError> solve_american_option_auto(
    const AmericanOptionParams& params)
{
    // Estimate grid for this option
    auto [grid_spec, n_time] = estimate_grid_for_option(params);

    // Create workspace with estimated grid
    auto workspace_result = AmericanSolverWorkspace::create(
        grid_spec.x_min(), grid_spec.x_max(), grid_spec.n_points(), n_time);
    if (!workspace_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Failed to create workspace: " + workspace_result.error(),
            .iterations = 0
        });
    }

    // Create and solve (after CRTP revert: direct constructor, no factory)
    AmericanOptionSolver solver(params, workspace_result.value());
    return solver.solve();
}

/// Solve a batch of American options with shared grid
///
/// Convenience function for batch solving with explicit grid parameters.
/// All options use the same PDE grid for efficiency.
///
/// @param params Vector or span of option parameters
/// @param x_min Minimum log-moneyness for grid
/// @param x_max Maximum log-moneyness for grid
/// @param n_space Number of spatial grid points
/// @param n_time Number of time steps
/// @return Batch result with individual results and failure count
inline BatchAmericanOptionResult solve_american_options_batch(
    std::span<const AmericanOptionParams> params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    if (params.empty()) {
        return BatchAmericanOptionResult{.results = {}, .failed_count = 0};
    }

    std::vector<std::expected<AmericanOptionResult, SolverError>> results(params.size());
    size_t failed_count = 0;

    MANGO_PRAGMA_PARALLEL
    {
        // Per-thread pool for workspace reuse
        std::pmr::unsynchronized_pool_resource thread_pool;

        // Per-thread workspace (shared by all options in this thread)
        auto workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace_result.has_value()) {
            // Workspace creation failed - all solves in this thread will fail
            MANGO_PRAGMA_FOR
            for (size_t i = 0; i < params.size(); ++i) {
                results[i] = std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = "Failed to create workspace: " + workspace_result.error(),
                    .iterations = 0
                });
                MANGO_PRAGMA_ATOMIC
                ++failed_count;
            }
        } else {
            auto workspace = workspace_result.value();

            MANGO_PRAGMA_FOR
            for (size_t i = 0; i < params.size(); ++i) {
                // Create solver (after CRTP revert: direct constructor, no factory)
                // Allocate surface buffer for full time history: (n_time + 1) × n_space
                std::vector<double> surface_buffer((workspace->n_time() + 1) * workspace->n_space());
                AmericanOptionSolver solver(params[i], workspace, surface_buffer);

                // Solve
                results[i] = solver.solve();
                if (!results[i].has_value()) {
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                }
            }
        }
    }

    return BatchAmericanOptionResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

/// Vector overload for solve_american_options_batch
inline BatchAmericanOptionResult solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    return solve_american_options_batch(std::span{params}, x_min, x_max, n_space, n_time);
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP
