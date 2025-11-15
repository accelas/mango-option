/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include "src/option/american_solver_workspace.hpp"
#include "src/option/option_spec.hpp"  // For OptionType enum
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>

namespace mango {

/**
 * BlackScholesPDE: Black-Scholes PDE operator in log-moneyness coordinates
 *
 * Implements the Black-Scholes PDE in log-moneyness coordinates x = ln(S/K):
 *   ∂V/∂t = L(V)
 *   L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V
 *
 * This is a lightweight PDE formula class used with operators::SpatialOperator.
 */
template<typename T = double>
class BlackScholesPDE {
public:
    /**
     * Construct Black-Scholes operator
     * @param sigma Volatility (σ)
     * @param r Risk-free rate
     * @param d Continuous dividend yield
     */
    BlackScholesPDE(T sigma, T r, T d)
        : half_sigma_sq_(T(0.5) * sigma * sigma)
        , drift_(r - d - half_sigma_sq_)
        , discount_rate_(r)
    {}

    /**
     * Apply operator: L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V
     *
     * @param d2v_dx2 Second derivative ∂²V/∂x²
     * @param dv_dx First derivative ∂V/∂x
     * @param v Value V
     * @return L(V)
     */
    T operator()(T d2v_dx2, T dv_dx, T v) const {
        return half_sigma_sq_ * d2v_dx2 + drift_ * dv_dx - discount_rate_ * v;
    }

    /**
     * Compute first derivative coefficient: (r - d - σ²/2)
     * Used for analytical Jacobian construction
     */
    T first_derivative_coeff() const { return drift_; }

    /**
     * Compute second derivative coefficient: σ²/2
     * Used for analytical Jacobian construction
     */
    T second_derivative_coeff() const { return half_sigma_sq_; }

    /**
     * Compute discount rate: r
     * Used for analytical Jacobian construction
     */
    T discount_rate() const { return discount_rate_; }

private:
    T half_sigma_sq_;    // σ²/2
    T drift_;            // r - d - σ²/2
    T discount_rate_;    // r
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

    /// Validate parameters (exception-based)
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

    /// Validate parameters (expected-based)
    static std::expected<void, std::string> validate_expected(const AmericanOptionParams& params) {
        // Check strike
        if (params.strike <= 0.0) {
            return std::unexpected("Strike must be positive");
        }

        // Check spot
        if (params.spot <= 0.0) {
            return std::unexpected("Spot must be positive");
        }

        // Check maturity
        if (params.maturity <= 0.0) {
            return std::unexpected("Maturity must be positive");
        }

        // Check volatility
        if (params.volatility <= 0.0) {
            return std::unexpected("Volatility must be positive");
        }

        // Check continuous dividend yield (rate can be negative)
        if (params.continuous_dividend_yield < 0.0) {
            return std::unexpected("Continuous dividend yield must be non-negative");
        }

        // Validate discrete dividends
        for (const auto& [time, amount] : params.discrete_dividends) {
            if (time < 0.0 || time > params.maturity) {
                return std::unexpected("Discrete dividend time must be in [0, maturity]");
            }
            if (amount < 0.0) {
                return std::unexpected("Discrete dividend amount must be non-negative");
            }
        }

        return {};
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

/**
 * Batch solver result containing individual results and aggregate statistics.
 */
struct BatchAmericanOptionResult {
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    size_t failed_count;  ///< Number of failed solves

    /// Check if all solves succeeded
    bool all_succeeded() const { return failed_count == 0; }
};

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
     * Constructor with workspace.
     *
     * This constructor enables efficient batch solving by reusing
     * grid allocations across multiple solver instances. Use when
     * solving many options with same grid but different coefficients.
     *
     * IMPORTANT: The workspace must outlive the solver. Use std::shared_ptr
     * to ensure proper lifetime management.
     *
     * @param params Option pricing parameters (including discrete dividends)
     * @param workspace Shared workspace with grid configuration and pre-allocated storage
     */
    AmericanOptionSolver(const AmericanOptionParams& params,
                        std::shared_ptr<AmericanSolverWorkspace> workspace);

    /**
     * Factory method with expected-based validation.
     *
     * Creates an AmericanOptionSolver with validation returning std::expected<void, std::string>.
     * This provides a non-throwing alternative to the constructor.
     *
     * IMPORTANT: The workspace must outlive the solver. Use std::shared_ptr
     * to ensure proper lifetime management.
     *
     * @param params Option pricing parameters (including discrete dividends)
     * @param workspace Shared workspace with grid configuration and pre-allocated storage
     * @return Expected containing solver on success, error message on failure
     */
    static std::expected<AmericanOptionSolver, std::string> create(
        const AmericanOptionParams& params,
        std::shared_ptr<AmericanSolverWorkspace> workspace);

    /**
     * Solve for option value and Greeks.
     *
     * @return Result containing option value and Greeks
     */
    std::expected<AmericanOptionResult, SolverError> solve();

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
     * Set TR-BDF2 solver configuration (advanced).
     *
     * Allows fine-tuning of the time-stepping scheme and Newton solver.
     * Most users should use the default configuration.
     *
     * @param config TR-BDF2 solver configuration (includes Newton parameters)
     */
    void set_trbdf2_config(const TRBDF2Config& config) {
        trbdf2_config_ = config;
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
    TRBDF2Config trbdf2_config_;

    // Workspace (contains grid configuration and pre-allocated storage)
    // Uses shared_ptr to keep workspace alive for the solver's lifetime
    std::shared_ptr<AmericanSolverWorkspace> workspace_;

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

#endif  // MANGO_AMERICAN_OPTION_HPP
