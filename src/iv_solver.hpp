#pragma once

#include "root_finding.hpp"
#include "common/ivcalc_trace.h"
#include "src/expected.hpp"
#include <optional>
#include <string>
#include <vector>
#include <span>

namespace mango {

/// Parameters for implied volatility calculation
///
/// Describes the option contract and market conditions for which
/// we want to find the implied volatility.
struct IVParams {
    /// Current stock price (S)
    double spot_price;

    /// Strike price (K)
    double strike;

    /// Time to expiration in years (T)
    double time_to_maturity;

    /// Risk-free interest rate (r)
    double risk_free_rate;

    /// Observed market price of the option
    double market_price;

    /// Option type: true for call, false for put
    bool is_call;
};

/// Configuration for implied volatility solver
///
/// Controls both the root-finding algorithm and the PDE grid
/// used for American option pricing during IV calculation.
struct IVConfig {
    /// Root-finding configuration (Brent's method parameters)
    RootFindingConfig root_config;

    /// Number of spatial grid points for PDE solver
    size_t grid_n_space = 101;

    /// Number of time steps for PDE solver
    size_t grid_n_time = 1000;

    /// Maximum spot price for grid (S_max)
    double grid_s_max = 200.0;
};

/// Result from implied volatility calculation
///
/// Extends RootFindingResult with IV-specific information.
struct IVResult {
    /// Whether the solver converged to a solution
    bool converged;

    /// Number of iterations performed
    size_t iterations;

    /// The calculated implied volatility (if converged)
    /// Value is only meaningful if converged == true
    double implied_vol;

    /// Final error measure (difference between theoretical and market price)
    double final_error;

    /// Optional failure diagnostic message
    std::optional<std::string> failure_reason;

    /// Optional vega (∂V/∂σ) at the solution
    /// Useful for sensitivity analysis and Newton-based methods
    std::optional<double> vega;
};

/// Implied Volatility Solver for American Options
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
/// IVParams params{
///     .spot_price = 100.0,
///     .strike = 100.0,
///     .time_to_maturity = 1.0,
///     .risk_free_rate = 0.05,
///     .market_price = 10.45,
///     .is_call = false
/// };
///
/// IVConfig config{
///     .root_config = RootFindingConfig{.max_iter = 100, .tolerance = 1e-6}
/// };
///
/// IVSolver solver(params, config);
/// IVResult result = solver.solve();
///
/// if (result.converged) {
///     std::cout << "IV: " << result.implied_vol << "\n";
/// }
/// ```
///
/// **Performance:**
/// - FDM-based: ~250ms per calculation (ground truth)
/// - For production use cases requiring many queries, consider
///   precomputed price table with interpolation (~7.5µs per query)
///
/// **USDT Tracing:**
/// Emits MODULE_IMPLIED_VOL traces for monitoring:
/// - algo_start: IV calculation begins
/// - algo_progress: Root-finding progress
/// - algo_complete: IV calculation completes
/// - validation_error: Input validation failures
/// - convergence_failed: Non-convergence diagnostics
class IVSolver {
public:
    /// Construct solver with problem parameters and configuration
    ///
    /// @param params Option parameters and market price
    /// @param config Solver configuration (root-finding and grid settings)
    explicit IVSolver(const IVParams& params, const IVConfig& config);

    /// Solve for implied volatility
    ///
    /// Uses Brent's method to find the volatility that makes the
    /// American option's theoretical price match the market price.
    ///
    /// **Note:** This is a STUB implementation that returns "Not implemented"
    /// error. Full implementation will be added in subsequent tasks.
    ///
    /// @return IVResult with convergence status and implied volatility
    IVResult solve();

private:
    IVParams params_;
    IVConfig config_;
    mutable std::optional<SolverError> last_solver_error_;

    /// Validate input parameters
    /// @return expected success or validation error message
    expected<void, std::string> validate_params() const;

    /// Estimate upper bound for volatility search using intrinsic value approximation
    /// @return Upper bound estimate (typically 2.0-3.0 for reasonable markets)
    double estimate_upper_bound() const;

    /// Estimate lower bound for volatility search
    /// @return Lower bound (typically 0.01 or 1%)
    double estimate_lower_bound() const;

    /// Objective function for root-finding: f(σ) = V(σ) - V_market
    /// @param volatility Candidate volatility
    /// @return Difference between theoretical and market price
    double objective_function(double volatility) const;
};

/// Batch Implied Volatility Solver
///
/// Solves implied volatility for multiple options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially.
///
/// Example usage:
/// ```cpp
/// std::vector<IVParams> batch = { ... };
/// IVConfig config;  // Shared configuration
///
/// auto results = solve_implied_vol_batch(batch, config);
/// ```
///
/// Performance:
/// - Single-threaded: ~7 IVs/sec (101x1000 grid)
/// - Parallel (32 cores): ~107 IVs/sec (15.3x speedup)
///
/// Use cases:
/// - Volatility surface construction: Calculate IV for entire grid of strikes/maturities
/// - Market data processing: Batch-process option chains
/// - Risk calculations: Compute sensitivities across multiple scenarios
/// - Model calibration: Evaluate objective function for optimization
class BatchIVSolver {
public:
    /// Solve implied volatility for a batch of options in parallel
    ///
    /// @param params Vector of IV parameters (spot, strike, maturity, price)
    /// @param config Shared configuration (grid size, tolerances)
    /// @return Vector of IV results (same order as input)
    static std::vector<IVResult> solve_batch(
        std::span<const IVParams> params,
        const IVConfig& config)
    {
        std::vector<IVResult> results(params.size());

        #pragma omp parallel for
        for (size_t i = 0; i < params.size(); ++i) {
            IVSolver solver(params[i], config);
            results[i] = solver.solve();
        }

        return results;
    }

    /// Solve implied volatility for a batch of options (vector overload)
    static std::vector<IVResult> solve_batch(
        const std::vector<IVParams>& params,
        const IVConfig& config)
    {
        return solve_batch(std::span{params}, config);
    }
};

/// Convenience function for batch IV solving
inline std::vector<IVResult> solve_implied_vol_batch(
    std::span<const IVParams> params,
    const IVConfig& config)
{
    return BatchIVSolver::solve_batch(params, config);
}

/// Convenience function for batch IV solving (vector overload)
inline std::vector<IVResult> solve_implied_vol_batch(
    const std::vector<IVParams>& params,
    const IVConfig& config)
{
    return BatchIVSolver::solve_batch(params, config);
}

}  // namespace mango
