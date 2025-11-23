/**
 * @file iv_solver_fdm.hpp
 * @brief FDM-based implied volatility solver
 */

#pragma once

#include "src/option/iv_solver_base.hpp"
#include "src/option/option_spec.hpp"
#include "src/option/iv_types.hpp"
#include "src/option/iv_result.hpp"
#include "src/math/root_finding.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <span>
#include <optional>

namespace mango {

/// Configuration for FDM-based IV solver
struct IVSolverFDMConfig {
    /// Root-finding configuration (Brent's method parameters)
    RootFindingConfig root_config;

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
/// IVResult result = solver.solve(query);
///
/// if (result.converged) {
///     std::cout << "IV: " << result.implied_vol << "\n";
/// }
/// ```
///
/// **Batch Usage:**
/// ```cpp
/// std::vector<IVQuery> queries = { ... };
/// std::vector<IVResult> results(queries.size());
///
/// solver.solve_batch(queries, results);  // Uses OpenMP internally
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
class IVSolverFDM : public IVSolverBase {
public:
    /// Construct solver with configuration
    ///
    /// @param config Solver configuration (root-finding and grid settings)
    explicit IVSolverFDM(const IVSolverFDMConfig& config);

    /// Solve for implied volatility (single query)
    ///
    /// Implementation for IVSolverBase::solve().
    /// Uses Brent's method to find the volatility that makes the
    /// American option's theoretical price match the market price.
    ///
    /// @param query Option specification and market price
    /// @return std::expected<IVSuccess, IVError> with convergence result or error
    std::expected<IVSuccess, IVError> solve_impl(const IVQuery& query);

    /// Solve for implied volatility (batch with OpenMP)
    ///
    /// Implementation for IVSolverBase::solve_batch().
    /// Creates thread-local solver instances in OpenMP parallel region
    /// to ensure thread safety.
    ///
    /// @param queries Input queries
    /// @param results Output buffer (must match queries.size())
    void solve_batch_impl(std::span<const IVQuery> queries,
                         std::span<IVResult> results);

private:
    IVSolverFDMConfig config_;
    mutable std::optional<SolverError> last_solver_error_;

    /// Validate input parameters
    /// @return expected success or validation error message
    std::expected<void, std::string> validate_query(const IVQuery& query) const;

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

    // Atomic validators (C++23 monadic) - uniform API: all take const IVQuery&
    /// Validate spot price is positive
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_spot_positive(const IVQuery& query) const;

    /// Validate strike price is positive
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_strike_positive(const IVQuery& query) const;

    /// Validate time to maturity is positive
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_maturity_positive(const IVQuery& query) const;

    /// Validate market price is positive
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_price_positive(const IVQuery& query) const;

    /// Validate call price <= spot price (arbitrage check)
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_call_price_bound(const IVQuery& query) const;

    /// Validate put price <= strike price (arbitrage check)
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_put_price_bound(const IVQuery& query) const;

    /// Validate market price >= intrinsic value (arbitrage check)
    /// @param query Option specification and market price
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_intrinsic_value(const IVQuery& query) const;

    // Composite validators (C++23 monadic)
    /// Validate positive parameters (spot, strike, maturity, market_price)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_positive_parameters(const IVQuery& query) const;

    /// Validate arbitrage bounds (call/put constraints, intrinsic value)
    /// @return std::monostate on success, IVError on validation failure
    std::expected<std::monostate, IVError> validate_arbitrage_bounds(const IVQuery& query) const;

    /// Run Brent solver to find implied volatility
    /// @return IVSuccess with implied volatility or IVError on failure
    std::expected<IVSuccess, IVError> solve_brent(const IVQuery& query) const;
};

} // namespace mango
