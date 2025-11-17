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
/// Example usage:
/// ```cpp
/// std::vector<AmericanOptionParams> batch;
/// batch.emplace_back(spot, strike, maturity, rate, dividend_yield, type, sigma);
///
/// auto results = solve_american_options_batch(batch, -3.0, 3.0, 101, 1000);
/// ```
///
/// Advanced usage with snapshots:
/// ```cpp
/// auto results = BatchAmericanOptionSolver::solve_batch_with_setup(
///     params, -3.0, 3.0, 101, 1000,
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
    /// Solve a batch of American options in parallel
    ///
    /// Each thread creates its own workspace to avoid data races.
    /// The workspace parameters (grid configuration) are validated once.
    ///
    /// @param params Vector of option parameters
    /// @param x_min Minimum log-moneyness
    /// @param x_max Maximum log-moneyness
    /// @param n_space Number of spatial grid points
    /// @param n_time Number of time steps
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
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

            return solver_result.value().solve();
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

    /// Solve a batch of American options in parallel (vector overload)
    static BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, x_min, x_max, n_space, n_time, setup);
    }
};

/// Convenience function for batch solving
inline BatchAmericanOptionResult solve_american_options_batch(
    std::span<const AmericanOptionParams> params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    return BatchAmericanOptionSolver::solve_batch(params, x_min, x_max, n_space, n_time);
}

/// Convenience function for batch solving (vector overload)
inline BatchAmericanOptionResult solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    return BatchAmericanOptionSolver::solve_batch(params, x_min, x_max, n_space, n_time);
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP
