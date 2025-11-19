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
#include <tuple>

namespace mango {

/**
 * Estimate grid parameters for a single option using sinh-grid heuristics.
 *
 * Implements single-pass grid determination from sinh-grid specification:
 * - Spatial domain: x₀ ± n_sigma·σ√T (covers probability distribution)
 * - Spatial resolution: Δx ~ σ√tol (target truncation error)
 * - Temporal resolution: Δt ~ c_t·Δx_min (couples time/space errors)
 *
 * Grid is centered on current log-moneyness x₀ = ln(S/K), not at x=0.
 * This is appropriate for independent options (vs option chains).
 *
 * @param params Option parameters (spot, strike, maturity, volatility, etc.)
 * @param n_sigma Domain half-width in units of σ√T (default: 5.0)
 * @param alpha Sinh clustering strength (default: 2.0 for Europeans)
 * @param tol Target price tolerance (default: 1e-6)
 * @param c_t Time step safety factor (default: 0.75)
 * @return Tuple of (x_min, x_max, n_space, n_time)
 */
inline std::tuple<double, double, size_t, size_t> estimate_grid_for_option(
    const AmericanOptionParams& params,
    double n_sigma = 5.0,
    double alpha = 2.0,
    double tol = 1e-6,
    double c_t = 0.75)
{
    // Domain bounds (centered on current moneyness)
    double sigma_sqrt_T = params.volatility * std::sqrt(params.maturity);
    double x0 = std::log(params.spot / params.strike);

    double x_min = x0 - n_sigma * sigma_sqrt_T;
    double x_max = x0 + n_sigma * sigma_sqrt_T;

    // Spatial resolution (target truncation error)
    double dx_target = params.volatility * std::sqrt(tol);
    size_t Nx = static_cast<size_t>(std::ceil((x_max - x_min) / dx_target));
    Nx = std::clamp(Nx, size_t{200}, size_t{1200});

    // Ensure odd number of points (for centered stencils)
    if (Nx % 2 == 0) Nx++;

    // Temporal resolution (coupled to smallest spatial spacing)
    // For sinh grid with clustering α, dx_min ≈ dx_avg · exp(-α)
    double dx_avg = (x_max - x_min) / static_cast<double>(Nx);
    double dx_min = dx_avg * std::exp(-alpha);  // Sinh clustering factor

    double dt = c_t * dx_min;
    size_t Nt = static_cast<size_t>(std::ceil(params.maturity / dt));
    Nt = std::min(Nt, size_t{5000});  // Upper bound for stability

    return {x_min, x_max, Nx, Nt};
}

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
 * @return Tuple of (x_min, x_max, n_space, n_time)
 */
inline std::tuple<double, double, size_t, size_t> compute_global_max_grid(
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
        auto [x_min, x_max, Nx, Nt] = estimate_grid_for_option(p, n_sigma, alpha, tol, c_t);
        global_x_min = std::min(global_x_min, x_min);
        global_x_max = std::max(global_x_max, x_max);
        global_Nx = std::max(global_Nx, Nx);
        global_Nt = std::max(global_Nt, Nt);
    }

    return {global_x_min, global_x_max, global_Nx, global_Nt};
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
    /// Automatically determines optimal grid parameters for each option based on
    /// option characteristics (volatility, maturity, moneyness), then uses the
    /// conservative global maximum grid for the entire batch.
    ///
    /// @param params Vector of option parameters
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        SetupCallback setup = nullptr)
    {
        auto [x_min, x_max, n_space, n_time] = compute_global_max_grid(params);
        return solve_batch_with_grid(params, x_min, x_max, n_space, n_time, setup);
    }

    /// Solve a batch of American options (vector overload)
    static BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, setup);
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
    /// @param output_buffer Optional buffer for full surface. If provided, must be large enough
    ///                      for (n_time + 1) * n_space * params.size() doubles.
    ///                      Buffer layout: option 0 surface | option 1 surface | ...
    /// @return Batch result with individual results and failure count
    static BatchAmericanOptionResult solve_batch_with_grid(
        std::span<const AmericanOptionParams> params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr,
        std::span<double> output_buffer = {})
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

        // Calculate buffer slice size for each option (if buffer provided)
        const size_t slice_size = (n_time + 1) * n_space;

        // Validate buffer size if provided
        if (!output_buffer.empty()) {
            const size_t required_size = slice_size * params.size();
            if (output_buffer.size() < required_size) {
                SolverError error{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = "Output buffer too small: need " +
                               std::to_string(required_size) + " but got " +
                               std::to_string(output_buffer.size()),
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
        }

        // Common solve logic
        auto solve_one = [&](size_t i, std::shared_ptr<AmericanSolverWorkspace> workspace)
            -> std::expected<AmericanOptionResult, SolverError>
        {
            // Calculate buffer slice for this option (if buffer provided)
            std::span<double> option_buffer;
            if (!output_buffer.empty()) {
                const size_t offset = i * slice_size;
                option_buffer = output_buffer.subspan(offset, slice_size);
            }

            // Use factory method to avoid exceptions from constructor
            auto solver_result = AmericanOptionSolver::create(params[i], workspace, option_buffer);
            if (!solver_result) {
                return std::unexpected(SolverError{
                    .code = SolverErrorCode::InvalidConfiguration,
                    .message = solver_result.error(),
                    .iterations = 0
                });
            }

            // Invoke setup callback if provided
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

    /// Solve a batch of American options with custom grid (vector overload)
    static BatchAmericanOptionResult solve_batch_with_grid(
        const std::vector<AmericanOptionParams>& params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
    {
        return solve_batch_with_grid(std::span{params}, x_min, x_max, n_space, n_time, setup);
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
    auto [x_min, x_max, n_space, n_time] = estimate_grid_for_option(params);

    // Create workspace with estimated grid
    auto workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
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
