#pragma once

#include "root_finding.hpp"
#include "src/ivcalc_trace.h"
#include <optional>
#include <string>

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

    /// Validate input parameters
    /// @return Error message if invalid, std::nullopt if valid
    std::optional<std::string> validate_params() const;

    /// Objective function for root-finding: f(σ) = V(σ) - V_market
    /// @param volatility Candidate volatility
    /// @return Difference between theoretical and market price
    double objective_function(double volatility) const;
};

}  // namespace mango
