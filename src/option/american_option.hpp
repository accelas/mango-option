/**
 * @file american_option.hpp
 * @brief American option pricing solver using finite difference method
 */

#ifndef MANGO_AMERICAN_OPTION_HPP
#define MANGO_AMERICAN_OPTION_HPP

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"
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
 * @brief Backward compatibility alias for PricingParams
 * @deprecated Use PricingParams from option_spec.hpp instead
 */
using AmericanOptionParams = PricingParams;


/**
 * Solver result containing solution surface (interpolate on-demand for specific prices).
 */
struct AmericanOptionResult {
    double value;                      ///< Option value at current spot (dollars)
    std::vector<double> solution;      ///< Final spatial solution V/K (always present, for value_at())
    std::vector<double> surface_2d;   ///< Full spatiotemporal surface V/K [time][space] (optional, for at_time())
    size_t n_space;                    ///< Number of spatial grid points
    size_t n_time;                     ///< Number of time steps
    double x_min;                      ///< Minimum log-moneyness
    double x_max;                      ///< Maximum log-moneyness
    double strike;                     ///< Strike price K (for denormalization)
    bool converged;                    ///< Solver convergence status

    /// Default constructor
    AmericanOptionResult()
        : value(0.0), solution(), surface_2d(), n_space(0), n_time(0),
          x_min(0.0), x_max(0.0), strike(1.0), converged(false) {}

    /**
     * Interpolate to get option value at specific spot price (at final time).
     *
     * @param spot Spot price S
     * @return Option value in dollars (denormalized)
     */
    double value_at(double spot) const;

    /**
     * Get solution at specific time step.
     *
     * @param time_idx Time step index (0 = maturity, n_time-1 = present)
     * @return Span of spatial solution at that time
     */
    std::span<const double> at_time(size_t time_idx) const {
        if (surface_2d.empty() || time_idx >= n_time) {
            return {};
        }
        return std::span<const double>{surface_2d.data() + time_idx * n_space, n_space};
    }
};

/**
 * Option Greeks (sensitivities).
 * Computed on-demand via AmericanOptionSolver::compute_greeks().
 */
struct AmericanOptionGreeks {
    double delta;    ///< ∂V/∂S (first derivative wrt spot)
    double gamma;    ///< ∂²V/∂S² (second derivative wrt spot)
    double theta;    ///< ∂V/∂t (time decay)

    /// Default constructor
    AmericanOptionGreeks()
        : delta(0.0), gamma(0.0), theta(0.0) {}
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
     * Solve for option value.
     *
     * Always stores final solution for value_at(). If output_buffer was provided
     * at construction, collects full spatiotemporal surface enabling at_time().
     *
     * @return Result containing option value (compute Greeks separately via compute_greeks())
     */
    std::expected<AmericanOptionResult, SolverError> solve();

    /**
     * Compute Greeks (sensitivities) for the current solution.
     *
     * Must be called after solve() has succeeded. Computes delta, gamma, and theta
     * based on the current solution state.
     *
     * @return Greeks on success, error if solve() hasn't been called yet
     */
    std::expected<AmericanOptionGreeks, SolverError> compute_greeks() const;

    /**
     * Get the full solution surface (for debugging/analysis).
     *
     * @return Vector of option values across the spatial grid
     */
    std::vector<double> get_solution() const;

private:
    // Parameters
    AmericanOptionParams params_;

    // Workspace (contains grid configuration and pre-allocated storage)
    // Uses shared_ptr to keep workspace alive for the solver's lifetime
    std::shared_ptr<AmericanSolverWorkspace> workspace_;

    // Solution state
    std::vector<double> solution_;
    bool solved_ = false;

    // Helper methods
    double compute_delta() const;
    double compute_gamma() const;
    double compute_theta() const;
    double interpolate_solution(double x_target, std::span<const double> x_grid) const;
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
