// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_fdm.hpp
 * @brief FDM-based implied volatility solver with std::expected error handling
 *
 * The FDMIVSolver uses finite difference methods (FDM) to price American options
 * and Brent's root-finding method to solve for implied volatility.
 *
 * API (C++23):
 * - solve_impl() → std::expected<IVSuccess, IVError>
 * - solve_batch_impl() → BatchIVResult
 *
 * Error Handling:
 * - Type-safe error codes via IVErrorCode enum
 * - Detailed diagnostics (iterations, final_error, last_vol)
 * - Monadic validation chains with .and_then()
 *
 * Example:
 * @code
 * IVQuery query{...};
 * FDMIVSolver solver(config);
 * auto result = solver.solve_impl(query);
 *
 * if (result.has_value()) {
 *     std::cout << "IV: " << result->implied_vol << "\n";
 * } else {
 *     std::cerr << "Error: " <<  "Error code: " << static_cast<int>(result.error().code) << "\n";
 * }
 * @endcode
 */

#pragma once

#include "src/option/option_spec.hpp"
#include "src/option/iv_result.hpp"
#include "src/option/american_option.hpp"
#include "src/math/root_finding.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <span>
#include <optional>
#include <unordered_map>

namespace mango {

/// Configuration for FDM-based IV solver
struct IVSolverFDMConfig {
    /// Root-finding configuration (Brent's method parameters)
    RootFindingConfig root_config;

    /// Parallelization threshold for batch solving
    ///
    /// Batches smaller than this threshold use serial execution to avoid
    /// parallel overhead. IV solves are expensive (multiple PDE solves),
    /// so even small batches benefit from parallelization once overhead
    /// is amortized.
    ///
    /// **Tuning guidance:**
    /// - Default (4): Good balance for most workloads
    /// - Larger (8-16): For systems with high parallel overhead
    /// - Smaller (1-2): For systems with low parallel overhead or large batches
    /// - Set to SIZE_MAX to force serial execution for all batch sizes
    size_t batch_parallel_threshold = 4;

    /// Use manual grid specification instead of auto-estimation
    ///
    /// When false (default): Automatically estimate optimal grid based on option parameters
    /// When true: Use grid_n_space, grid_n_time, grid_x_min, grid_x_max, grid_alpha exactly
    ///
    /// **Advanced usage only** (for benchmarks or custom grid control)
    bool use_manual_grid = false;

    /// Number of spatial grid points (used when use_manual_grid = true)
    size_t grid_n_space = 101;

    /// Number of time steps (used when use_manual_grid = true)
    size_t grid_n_time = 1000;

    /// Minimum log-moneyness (used when use_manual_grid = true)
    double grid_x_min = -3.0;

    /// Maximum log-moneyness (used when use_manual_grid = true)
    double grid_x_max = 3.0;

    /// Sinh clustering parameter (used when use_manual_grid = true)
    double grid_alpha = 2.0;

    /// Grid accuracy parameters for automatic grid estimation
    ///
    /// Controls the resolution of the PDE grid used internally for pricing
    /// during root-finding iterations. Higher accuracy reduces IV error
    /// at the cost of slower solves.
    ///
    /// Only used when use_manual_grid = false (default).
    GridAccuracyParams grid_accuracy;
};

/// FDM-based Implied Volatility Solver for American Options
///
/// Finds the volatility parameter that makes the American option's
/// theoretical price (from PDE solver) match the observed market price.
///
/// **Algorithm:**
/// Uses Brent's method for root-finding, with each iteration solving
/// the American option PDE for a candidate volatility. This nested
/// approach is robust but relatively slow (~250ms per IV calculation).
///
/// **Usage:**
/// ```cpp
/// OptionSpec spec{
///     .spot = 100.0,
///     .strike = 100.0,
///     .maturity = 1.0,
///     .rate = 0.05,
///     .dividend_yield = 0.02,
///     .type = OptionType::PUT
/// };
///
/// IVQuery query{.option = spec, .market_price = 10.45};
///
/// IVSolverFDMConfig config{
///     .root_config = RootFindingConfig{.max_iter = 100, .tolerance = 1e-6}
/// };
///
/// IVSolverFDM solver(config);
/// auto result = solver.solve_impl(query);
///
/// if (result.has_value()) {
///     std::cout << "IV: " << result->implied_vol << "\n";
/// } else {
///     std::cerr << "Error: " <<  "Error code: " << static_cast<int>(result.error().code) << "\n";
/// }
/// ```
///
/// **Batch Usage:**
/// ```cpp
/// std::vector<IVQuery> queries = { ... };
/// auto batch = solver.solve_batch_impl(queries);
///
/// for (size_t i = 0; i < batch.results.size(); ++i) {
///     if (batch.results[i].has_value()) {
///         std::cout << "Query " << i << ": σ = " << batch.results[i]->implied_vol << "\n";
///     }
/// }
/// ```
///
/// **Performance:**
/// - Single query: ~143ms (FDM ground truth)
/// - Batch: ~107 IVs/sec on 32 cores (15.3x speedup)
///
/// **Thread Safety:**
/// - Single query (solve): Not thread-safe (stateful objective function)
/// - Batch (solve_batch): Thread-safe (creates thread-local solvers)
///
/// **USDT Tracing:**
/// Emits MODULE_IMPLIED_VOL traces for monitoring:
/// - algo_start: IV calculation begins
/// - algo_progress: Root-finding progress
/// - algo_complete: IV calculation completes
/// - validation_error: Input validation failures
/// - convergence_failed: Non-convergence diagnostics
class IVSolverFDM {
public:
    /// Construct solver with configuration
    ///
    /// @param config Solver configuration (root-finding and grid settings)
    explicit IVSolverFDM(const IVSolverFDMConfig& config);

    /// Solve for implied volatility (single query)
    /// Uses Brent's method to find the volatility that makes the
    /// American option's theoretical price match the market price.
    ///
    /// @param query Option specification and market price
    /// @return std::expected<IVSuccess, IVError>
    ///         - Success: IVSuccess with implied_vol, iterations, final_error, vega (optional)
    ///         - Failure: IVError with error code, message, and diagnostics
    ///
    /// Error codes:
    /// - NegativeSpot, NegativeStrike, NegativeMaturity, NegativeMarketPrice: Validation errors
    /// - ArbitrageViolation: Price violates arbitrage bounds or intrinsic value
    /// - MaxIterationsExceeded: Brent solver did not converge
    /// - BracketingFailed: Root not bracketed by initial bounds
    /// - InvalidGridConfig: FDM grid parameters invalid (manual mode only)
    ///
    /// @note Uses monadic validation: params → arbitrage → grid → Brent solving
    std::expected<IVSuccess, IVError> solve_impl(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    ///
    /// Implementation for batch solving with std::expected.
    /// Creates thread-local solver instances in OpenMP parallel region
    /// to ensure thread safety.
    ///
    /// @param queries Input queries (as vector for convenience)
    /// @return BatchIVResult with individual results and failure count
    BatchIVResult solve_batch_impl(const std::vector<IVQuery>& queries) const;

private:
    IVSolverFDMConfig config_;
    mutable std::optional<SolverError> last_solver_error_;

    /// Estimate upper bound for volatility search using intrinsic value approximation
    /// @return Upper bound estimate (typically 2.0-3.0 for reasonable markets)
    double estimate_upper_bound(const IVQuery& query) const;

    /// Estimate lower bound for volatility search
    /// @return Lower bound (typically 0.01 or 1%)
    double estimate_lower_bound() const;

    /// Objective function for root-finding: f(σ) = V(σ) - V_market
    /// @param query Option specification and market price
    /// @param volatility Candidate volatility
    /// @return Difference between theoretical and market price
    double objective_function(const IVQuery& query, double volatility) const;

    // Grid validators (FDM-specific, only used when manual grid mode enabled)
    /// Validate grid n_space is positive (manual grid mode)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_n_space_positive() const;

    /// Validate grid n_time is positive (manual grid mode)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_n_time_positive() const;

    /// Validate grid x_min < x_max (manual grid mode)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_x_bounds() const;

    /// Validate grid alpha >= 0 (manual grid mode)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_alpha_nonnegative() const;

    /// Validate grid parameters (FDM-specific, only when manual grid mode enabled)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_grid_params() const;

    /// Validate query using centralized validation + grid params
    /// Uses validate_iv_query() from option_spec.cpp for consistency
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_query(const IVQuery& query) const;

    /// Run Brent solver to find implied volatility
    /// @return IVSuccess with implied volatility or IVError on failure
    std::expected<IVSuccess, IVError> solve_brent(const IVQuery& query) const;
};

} // namespace mango
