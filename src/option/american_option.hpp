/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/spatial_operators.hpp"
#include "src/support/expected.hpp"
#include "src/support/parallel.hpp"
#include "src/option/american_solver_workspace.hpp"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>

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
    static expected<void, std::string> validate_expected(const AmericanOptionParams& params) {
        // Check strike
        if (params.strike <= 0.0) {
            return unexpected("Strike must be positive");
        }

        // Check spot
        if (params.spot <= 0.0) {
            return unexpected("Spot must be positive");
        }

        // Check maturity
        if (params.maturity <= 0.0) {
            return unexpected("Maturity must be positive");
        }

        // Check volatility
        if (params.volatility <= 0.0) {
            return unexpected("Volatility must be positive");
        }

        // Check continuous dividend yield (rate can be negative)
        if (params.continuous_dividend_yield < 0.0) {
            return unexpected("Continuous dividend yield must be non-negative");
        }

        // Validate discrete dividends
        for (const auto& [time, amount] : params.discrete_dividends) {
            if (time < 0.0 || time > params.maturity) {
                return unexpected("Discrete dividend time must be in [0, maturity]");
            }
            if (amount < 0.0) {
                return unexpected("Discrete dividend amount must be non-negative");
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
     * Creates an AmericanOptionSolver with validation returning expected<void, std::string>.
     * This provides a non-throwing alternative to the constructor.
     *
     * IMPORTANT: The workspace must outlive the solver. Use std::shared_ptr
     * to ensure proper lifetime management.
     *
     * @param params Option pricing parameters (including discrete dividends)
     * @param workspace Shared workspace with grid configuration and pre-allocated storage
     * @return Expected containing solver on success, error message on failure
     */
    static expected<AmericanOptionSolver, std::string> create(
        const AmericanOptionParams& params,
        std::shared_ptr<AmericanSolverWorkspace> workspace);

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
     * Set TR-BDF2 solver configuration (advanced).
     *
     * Allows fine-tuning of the time-stepping scheme. Most users
     * should use the default configuration.
     *
     * @param config TR-BDF2 solver configuration
     */
    void set_trbdf2_config(const TRBDF2Config& config) {
        trbdf2_config_ = config;
    }

    /**
     * Set root-finding configuration (advanced).
     *
     * Allows fine-tuning of the Newton solver for early exercise boundary.
     * Most users should use the default configuration.
     *
     * @param config Root finding configuration
     */
    void set_root_config(const RootFindingConfig& config) {
        root_config_ = config;
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
    RootFindingConfig root_config_;

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
    /// @return Vector of results (same order as input)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
        std::span<const AmericanOptionParams> params,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        SetupCallback setup = nullptr)
    {
        std::vector<expected<AmericanOptionResult, SolverError>> results(params.size());

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
                results[i] = unexpected(error);
            }
            return results;
        }

        // Common solve logic
        auto solve_one = [&](size_t i, std::shared_ptr<AmericanSolverWorkspace> workspace)
            -> expected<AmericanOptionResult, SolverError>
        {
            // Use factory method to avoid exceptions from constructor
            auto solver_result = AmericanOptionSolver::create(params[i], workspace);
            if (!solver_result) {
                return unexpected(SolverError{
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
#ifdef _OPENMP
#pragma omp parallel
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
#pragma omp for
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = unexpected(error);
                }
            } else {
                auto thread_workspace = thread_workspace_result.value();

#pragma omp for
                for (size_t i = 0; i < params.size(); ++i) {
                    results[i] = solve_one(i, thread_workspace);
                }
            }
        }
#else
        // Sequential: create workspace once and reuse for all options
        auto workspace_result = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
        if (!workspace_result) {
            SolverError error{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Failed to create workspace: " + workspace_result.error(),
                .iterations = 0
            };
            for (size_t i = 0; i < params.size(); ++i) {
                results[i] = unexpected(error);
            }
            return results;
        }

        auto workspace = workspace_result.value();
        for (size_t i = 0; i < params.size(); ++i) {
            results[i] = solve_one(i, workspace);
        }
#endif

        return results;
    }

    /// Solve a batch of American options in parallel (vector overload)
    static std::vector<expected<AmericanOptionResult, SolverError>> solve_batch(
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
inline std::vector<expected<AmericanOptionResult, SolverError>> solve_american_options_batch(
    std::span<const AmericanOptionParams> params,
    double x_min,
    double x_max,
    size_t n_space,
    size_t n_time)
{
    return BatchAmericanOptionSolver::solve_batch(params, x_min, x_max, n_space, n_time);
}

/// Convenience function for batch solving (vector overload)
inline std::vector<expected<AmericanOptionResult, SolverError>> solve_american_options_batch(
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
