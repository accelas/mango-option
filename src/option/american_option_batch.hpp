/**
 * @file american_option_batch.hpp
 * @brief Batch American option pricing solver for parallel processing
 */

#ifndef MANGO_AMERICAN_OPTION_BATCH_HPP
#define MANGO_AMERICAN_OPTION_BATCH_HPP

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
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
 * Compute conservative global maximum grid for heterogeneous option batch.
 *
 * Takes the union of spatial domains and maximum resolution across all options.
 * This ensures the grid is large enough for every option in the batch.
 *
 * @param params Vector of option parameters
 * @param n_sigma Domain half-width parameter (default: 5.0)
 * @param alpha Sinh clustering parameter (default: 2.0)
 * @param tol Target price tolerance (default: 1e-6)
 * @param c_t Time step safety factor (default: 0.75)
 * @return Tuple of (GridSpec, n_time)
 */
inline std::tuple<GridSpec<double>, size_t> compute_global_max_grid(
    std::span<const AmericanOptionParams> params,
    double n_sigma = 5.0,
    double alpha = 2.0,
    double tol = 1e-6,
    double c_t = 0.75)
{
    double global_x_min = 0.0;
    double global_x_max = 0.0;
    size_t global_Nx = 0;
    size_t global_Nt = 0;

    for (const auto& p : params) {
        auto [grid_spec, Nt] = estimate_grid_for_option(p, n_sigma, alpha, tol, c_t);
        global_x_min = std::min(global_x_min, grid_spec.x_min());
        global_x_max = std::max(global_x_max, grid_spec.x_max());
        global_Nx = std::max(global_Nx, grid_spec.n_points());
        global_Nt = std::max(global_Nt, Nt);
    }

    auto grid_spec = GridSpec<double>::uniform(global_x_min, global_x_max, global_Nx);
    return {grid_spec.value(), global_Nt};
}

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
/// **Standard API** (recommended):
/// ```cpp
/// std::vector<AmericanOptionParams> batch;
/// batch.emplace_back(spot, strike, maturity, rate, dividend_yield, type, sigma);
///
/// // Automatically determines optimal grid based on option characteristics
/// auto results = solve_american_options_batch(batch);
/// ```
///
/// **Advanced API** (for custom grid configuration):
/// ```cpp
/// // Manually specify grid parameters (for expert users)
/// auto results = solve_american_options_batch(batch, -3.0, 3.0, 101, 1000);
/// ```
///
/// **Accessing results**:
/// ```cpp
/// for (const auto& result_expected : batch_result.results) {
///     if (result_expected.has_value()) {
///         const auto& result = result_expected.value();
///         double price = result.value_at(spot);
///         auto spatial_solution = result.at_time(step_idx);
///     }
/// }
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Setup callback: called before each solve() to configure solver
    /// @param index Index of current option in params vector
    /// @param solver Reference to solver for pre-solve configuration
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

    /// Solve a batch of American options with automatic grid determination
    ///
    /// Automatically determines optimal grid parameters for EACH option based on
    /// its characteristics (volatility, maturity, moneyness). Each option gets
    /// its own workspace sized appropriately - no shared oversized grids.
    ///
    /// @param params Vector of option parameters
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        SetupCallback setup = nullptr)
    {
        std::vector<std::expected<AmericanOptionResult, SolverError>> results(params.size());
        size_t failed_count = 0;

        // Solve each option in parallel with its own optimal grid
        MANGO_PRAGMA_PARALLEL
        {
            // Per-thread pool for cheap workspace reuse within thread
            // Pool memory is reused across multiple workspace allocations
            std::pmr::unsynchronized_pool_resource thread_pool;

            MANGO_PRAGMA_FOR
            for (size_t i = 0; i < params.size(); ++i) {
                // Estimate grid for this specific option
                auto [grid_spec, n_time] = estimate_grid_for_option(params[i]);

                // Create workspace with estimated grid (uses thread pool for cheap reuse)
                auto workspace_result = AmericanSolverWorkspace::create(
                    grid_spec, n_time, &thread_pool);
                if (!workspace_result.has_value()) {
                    results[i] = std::unexpected(SolverError{
                        .code = SolverErrorCode::InvalidConfiguration,
                        .message = "Failed to create workspace: " + workspace_result.error(),
                        .iterations = 0
                    });
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                    continue;
                }

                // Create solver
                auto solver_result = AmericanOptionSolver::create(params[i], workspace_result.value());
                if (!solver_result.has_value()) {
                    results[i] = std::unexpected(SolverError{
                        .code = SolverErrorCode::InvalidConfiguration,
                        .message = solver_result.error(),
                        .iterations = 0
                    });
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                    continue;
                }

                // Invoke setup callback if provided
                if (setup) {
                    setup(i, solver_result.value());
                }

                // Solve
                results[i] = solver_result.value().solve();
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
    static BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, setup);
    }
};

/// Solve a batch of American options with automatic grid determination
///
/// This is the recommended API for solving multiple independent options.
/// Automatically determines optimal grid parameters based on option
/// characteristics (volatility, maturity, moneyness).
///
/// @param params Vector of option parameters
/// @return Batch result with individual results and failure count
inline BatchAmericanOptionResult solve_american_options_batch(
    std::span<const AmericanOptionParams> params)
{
    return BatchAmericanOptionSolver::solve_batch(params);
}

/// Solve a batch of American options (vector overload)
inline BatchAmericanOptionResult solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params)
{
    return BatchAmericanOptionSolver::solve_batch(params);
}

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
        grid_spec, n_time, std::pmr::get_default_resource());
    if (!workspace_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Failed to create workspace: " + workspace_result.error(),
            .iterations = 0
        });
    }

    // Create and solve
    auto solver_result = AmericanOptionSolver::create(params, workspace_result.value());
    if (!solver_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = solver_result.error(),
            .iterations = 0
        });
    }

    return solver_result.value().solve();
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP
