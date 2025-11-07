/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde_solver.hpp"
#include "src/spatial_operators.hpp"
#include "src/expected.hpp"
#include "parallel.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>

namespace mango {

/**
 * Option type enumeration.
 */
enum class OptionType {
    CALL,
    PUT
};

/**
 * American option pricing parameters.
 */
struct AmericanOptionParams {
    double strike;                      ///< Strike price (dollars)
    double spot;                        ///< Current stock price (dollars)
    double maturity;                    ///< Time to maturity (years)
    double volatility;                  ///< Implied volatility (fraction)
    double rate;                        ///< Risk-free rate (fraction)
    double continuous_dividend_yield;   ///< Continuous dividend yield (fraction, affects PDE drift)
    OptionType option_type;             ///< Call or Put

    /// Discrete dividend schedule: (time, amount) pairs
    /// Time is in years from now, amount is in dollars
    /// Can be used simultaneously with continuous_dividend_yield
    std::vector<std::pair<double, double>> discrete_dividends;

    /// Validate parameters
    void validate() const {
        if (strike <= 0.0) throw std::invalid_argument("Strike must be positive");
        if (spot <= 0.0) throw std::invalid_argument("Spot must be positive");
        if (maturity <= 0.0) throw std::invalid_argument("Maturity must be positive");
        if (volatility <= 0.0) throw std::invalid_argument("Volatility must be positive");
        // Note: rate can be negative (EUR, JPY, CHF markets)
        if (continuous_dividend_yield < 0.0) throw std::invalid_argument("Continuous dividend yield must be non-negative");

        // Validate discrete dividends
        for (const auto& [time, amount] : discrete_dividends) {
            if (time < 0.0 || time > maturity) {
                throw std::invalid_argument("Discrete dividend time must be in [0, maturity]");
            }
            if (amount < 0.0) {
                throw std::invalid_argument("Discrete dividend amount must be non-negative");
            }
        }
    }
};

/**
 * Numerical grid parameters for PDE solver.
 */
struct AmericanOptionGrid {
    size_t n_space;    ///< Number of spatial grid points
    size_t n_time;     ///< Number of time steps
    double x_min;      ///< Minimum log-moneyness (default: -3.0)
    double x_max;      ///< Maximum log-moneyness (default: +3.0)

    /// Default constructor with sensible defaults
    AmericanOptionGrid()
        : n_space(101)
        , n_time(1000)
        , x_min(-3.0)
        , x_max(3.0) {}

    void validate() const {
        if (n_space < 10) throw std::invalid_argument("n_space must be >= 10");
        if (n_time < 10) throw std::invalid_argument("n_time must be >= 10");
        if (x_min >= x_max) throw std::invalid_argument("x_min must be < x_max");
    }
};

/**
 * Solver result containing option value and Greeks.
 */
struct AmericanOptionResult {
    double value;    ///< Option value (dollars)
    double delta;    ///< V/S (first derivative wrt spot)
    double gamma;    ///< �V/S� (second derivative wrt spot)
    double theta;    ///< V/t (time decay)
    bool converged;  ///< Solver convergence status

    /// Default constructor
    AmericanOptionResult()
        : value(0.0), delta(0.0), gamma(0.0), theta(0.0), converged(false) {}
};

// Forward declaration
class SliceSolverWorkspace;

/**
 * American option pricing solver using finite difference method.
 *
 * Solves the Black-Scholes PDE with obstacle constraints in log-moneyness
 * coordinates using TR-BDF2 time stepping and projection method for
 * early exercise boundary.
 */
class AmericanOptionSolver {
public:
    /**
     * Constructor (standard mode - creates own grid).
     *
     * @param params Option pricing parameters (including discrete dividends)
     * @param grid Numerical grid parameters
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration for Newton solver
     */
    AmericanOptionSolver(const AmericanOptionParams& params,
                        const AmericanOptionGrid& grid,
                        const TRBDF2Config& trbdf2_config = {},
                        const RootFindingConfig& root_config = {});

    /**
     * Constructor (workspace mode - reuses grid and spacing).
     *
     * This constructor enables efficient batch solving by reusing
     * grid allocations across multiple solver instances. Use when
     * solving many options with same grid but different coefficients.
     *
     * IMPORTANT: The workspace must outlive the solver. Use std::shared_ptr
     * to ensure proper lifetime management.
     *
     * @param params Option pricing parameters (including discrete dividends)
     * @param grid Numerical grid parameters (must match workspace)
     * @param workspace Shared workspace with pre-allocated grid (keeps workspace alive)
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration for Newton solver
     */
    AmericanOptionSolver(const AmericanOptionParams& params,
                        const AmericanOptionGrid& grid,
                        std::shared_ptr<const SliceSolverWorkspace> workspace,
                        const TRBDF2Config& trbdf2_config = {},
                        const RootFindingConfig& root_config = {});

    /**
     * Solve for option value and Greeks.
     *
     * @return Result containing option value and Greeks
     */
    expected<AmericanOptionResult, SolverError> solve();

    /**
     * Register snapshot collection at specific step index.
     *
     * Must be called before solve(). Snapshots will be collected
     * during the solve() call.
     *
     * @param step_index Step number (0-based) to collect snapshot
     * @param user_index User-provided index for matching
     * @param collector Callback to receive snapshot (must outlive solve())
     */
    void register_snapshot(size_t step_index, size_t user_index, SnapshotCollector* collector) {
        snapshot_requests_.push_back({step_index, user_index, collector});
    }

    /**
     * Get the full solution surface (for debugging/analysis).
     *
     * @return Vector of option values across the spatial grid
     */
    std::vector<double> get_solution() const;

private:
    // Parameters
    AmericanOptionParams params_;
    AmericanOptionGrid grid_;
    TRBDF2Config trbdf2_config_;
    RootFindingConfig root_config_;

    // Workspace (optional - nullptr means standalone mode)
    // Uses shared_ptr to keep workspace alive for the solver's lifetime
    std::shared_ptr<const SliceSolverWorkspace> workspace_;

    // Solution state
    std::vector<double> solution_;
    bool solved_ = false;

    // Snapshot requests
    struct SnapshotRequest {
        size_t step_index;
        size_t user_index;
        SnapshotCollector* collector;
    };
    std::vector<SnapshotRequest> snapshot_requests_;

    // Helper methods
    double compute_delta() const;
    double compute_gamma() const;
    double compute_theta() const;
    double interpolate_solution(double x_target, std::span<const double> x_grid) const;
};

/// Batch American Option Solver
///
/// Solves multiple American options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially
/// for embarrassingly parallel workloads.
///
/// Example usage:
/// ```cpp
/// std::vector<AmericanOptionParams> batch = { ... };
/// AmericanOptionGrid grid{.n_space = 101, .n_time = 1000};
///
/// auto results = solve_american_options_batch(batch, grid);
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Solve a batch of American options in parallel
    ///
    /// @param params Vector of option parameters
    /// @param grid Shared grid configuration (same for all options)
    /// @return Vector of results (same order as input)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        std::span<const AmericanOptionParams> params,
        const AmericanOptionGrid& grid)
    {
        std::vector<expected<AmericanOptionResult, SolverError>> results(params.size());

        MANGO_PRAGMA_PARALLEL_FOR
        for (size_t i = 0; i < params.size(); ++i) {
            AmericanOptionSolver solver(params[i], grid);
            results[i] = solver.solve();
        }

        return results;
    }

    /// Solve a batch of American options in parallel (vector overload)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        const std::vector<AmericanOptionParams>& params,
        const AmericanOptionGrid& grid)
    {
        return solve_batch(std::span{params}, grid);
    }
};

/// Convenience function for batch solving
inline std::vector<expected<AmericanOptionResult, SolverError>> solve_american_options_batch(
    std::span<const AmericanOptionParams> params,
    const AmericanOptionGrid& grid)
{
    return BatchAmericanOptionSolver::solve_batch(params, grid);
}

/// Convenience function for batch solving (vector overload)
inline std::vector<expected<AmericanOptionResult, SolverError>> solve_american_options_batch(
    const std::vector<AmericanOptionParams>& params,
    const AmericanOptionGrid& grid)
{
    return BatchAmericanOptionSolver::solve_batch(params, grid);
}

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
