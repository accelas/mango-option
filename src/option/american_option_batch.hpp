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
#include <vector>
#include <expected>
#include <span>
#include <functional>
#include <memory>

namespace mango {

/**
 * Default grid configuration for batch American option solving.
 *
 * These values provide a good balance of accuracy and performance
 * for most American options:
 * - Covers 50% to 200% moneyness range (deep OTM to deep ITM)
 * - 101 spatial points provides good accuracy
 * - 1000 time steps provides fine temporal resolution
 */
struct DefaultBatchGrid {
    static constexpr double x_min = -3.0;      ///< Minimum log-moneyness
    static constexpr double x_max = 3.0;       ///< Maximum log-moneyness
    static constexpr size_t n_space = 101;     ///< Spatial grid points
    static constexpr size_t n_time = 1000;     ///< Time steps
};

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
/// **Simple API** (recommended for most use cases):
/// ```cpp
/// std::vector<AmericanOptionParams> batch;
/// batch.emplace_back(spot, strike, maturity, rate, dividend_yield, type, sigma);
///
/// // Uses default grid configuration (101 spatial points, 1000 time steps)
/// auto results = solve_american_options_batch(batch);
/// ```
///
/// **Advanced API** (for custom grid configuration):
/// ```cpp
/// // Specify custom grid parameters
/// auto results = solve_american_options_batch(batch, -3.0, 3.0, 101, 1000);
/// ```
///
/// **Advanced usage with snapshots**:
/// ```cpp
/// auto results = BatchAmericanOptionSolver::solve_batch(
///     batch, -3.0, 3.0, 101, 1000,
///     [&](size_t idx, AmericanOptionSolver& solver) {
///         // Register snapshots for this solve
///         solver.register_snapshot(step, user_idx, collector);
///     });
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Setup callback: called before each solve() to configure solver
    /// @param index Index of current option in params vector
    /// @param solver Reference to solver (can register snapshots, set configs, etc.)
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

    /// Solve a batch of American options with default grid configuration
    ///
    /// Uses sensible defaults (101 spatial points, 1000 time steps,
    /// log-moneyness range [-3, 3]) suitable for most American options.
    ///
    /// @param params Vector of option parameters
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @param collect_full_surface If true, collects full spatiotemporal surface (all time steps).
    ///                             If false, only stores final time step (faster, less memory).
    ///                             Default false for performance.
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        SetupCallback setup = nullptr,
        bool collect_full_surface = false)
    {
        return solve_batch_with_grid(
            params,
            DefaultBatchGrid::x_min,
            DefaultBatchGrid::x_max,
            DefaultBatchGrid::n_space,
            DefaultBatchGrid::n_time,
            setup,
            collect_full_surface);
    }

    /// Solve a batch of American options with default grid (vector overload)
    static BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        SetupCallback setup = nullptr,
        bool collect_full_surface = false)
    {
        return solve_batch(std::span{params}, setup, collect_full_surface);
    }

    /// Solve a batch of American options with custom grid configuration
    ///
    /// Use this when you need fine control over the grid parameters.
    /// All options in the batch will use the same grid configuration.
    ///
    /// @param params Vector of option parameters
    /// @param x_min Minimum log-moneyness
    /// @param x_max Maximum log-moneyness
    /// @param n_space Number of spatial grid points
    /// @param n_time Number of time steps
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @param collect_full_surface If true, collects full spatiotemporal surface (all time steps).
    ///                             If false, only stores final time step (faster, less memory).
    ///                             Default false for performance.
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch_with_grid(
        std::span<const AmericanOptionParams> params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr,
        bool collect_full_surface = false)
    {
        std::vector<std::expected<AmericanOptionResult, SolverError>> results(params.size());
        size_t failed_count = 0;

        // Validate workspace parameters once before parallel loop
        auto validation = AmericanSolverWorkspace::validate_params(x_min, x_max, n_space, n_time);
        if (!validation) {
            // If workspace validation fails, return error for all options
            SolverError error{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Invalid workspace parameters: " + validation.error(),
                .iterations = 0
            };
            for (size_t i = 0; i < params.size(); ++i) {
                results[i] = std::unexpected(error);
            }
            return BatchAmericanOptionResult{
                .results = std::move(results),
                .failed_count = params.size()
            };
        }

        // Common solve logic
        auto solve_one = [&](size_t i, std::shared_ptr<AmericanSolverWorkspace> workspace)
            -> std::expected<AmericanOptionResult, SolverError>
        {
            // Use factory method to avoid exceptions from constructor
            auto solver_result = AmericanOptionSolver::create(params[i], workspace);
            if (!solver_result) {
                return std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = solver_result.error(),
                    .iterations = 0
                });
            }

            // NEW: Invoke setup callback if provided
            if (setup) {
                setup(i, solver_result.value());
            }

            return solver_result.value().solve(collect_full_surface);
        };

        // Use parallel region + for to enable per-thread workspace reuse
        // Note: MANGO_PRAGMA_* macros expand to nothing in sequential mode
        MANGO_PRAGMA_PARALLEL
        {
            // Each thread creates ONE workspace and reuses it for all its iterations
            auto thread_workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);

            // If per-thread workspace creation fails (e.g., OOM), write error to all thread's results
            if (!thread_workspace_result) {
                SolverError error{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = "Failed to create per-thread workspace: " + thread_workspace_result.error(),
                    .iterations = 0
                };
                MANGO_PRAGMA_FOR
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = std::unexpected(error);
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                }
            } else {
                auto thread_workspace = thread_workspace_result.value();

                MANGO_PRAGMA_FOR
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = solve_one(i, thread_workspace);
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

    /// Solve a batch of American options with custom grid (vector overload)
    static BatchAmericanOptionResult solve_batch_with_grid(
        const std::vector<AmericanOptionParams>& params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr,
        bool collect_full_surface = false)
    {
        return solve_batch_with_grid(std::span{params}, x_min, x_max, n_space, n_time, setup, collect_full_surface);
    }
};

/// Solve a batch of American options with default grid configuration
///
/// This is the recommended API for solving multiple independent options.
/// Uses sensible default grid parameters (101 spatial points, 1000 time steps,
/// log-moneyness range [-3, 3]).
///
/// @param params Vector of option parameters
/// @return Batch result with individual results and failure count
inline BatchAmericanOptionResult solve_american_options_batch(
    std::span<const AmericanOptionParams> params)
{
    return BatchAmericanOptionSolver::solve_batch(params);
}

/// Solve a batch of American options with default grid configuration (vector overload)
inline BatchAmericanOptionResult solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params)
{
    return BatchAmericanOptionSolver::solve_batch(params);
}

/// Solve a batch of American options with custom grid configuration
///
/// Advanced API for when you need fine control over the grid parameters.
/// All options in the batch will use the same grid configuration.
///
/// @param params Vector of option parameters
/// @param x_min Minimum log-moneyness
/// @param x_max Maximum log-moneyness
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
    return BatchAmericanOptionSolver::solve_batch_with_grid(
        params, x_min, x_max, n_space, n_time);
}

/// Solve a batch of American options with custom grid configuration (vector overload)
inline BatchAmericanOptionResult solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    return BatchAmericanOptionSolver::solve_batch_with_grid(
        params, x_min, x_max, n_space, n_time);
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP
