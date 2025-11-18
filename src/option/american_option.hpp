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
 * @brief Backward compatibility alias for PricingParams
 * @deprecated Use PricingParams from option_spec.hpp instead
 */
using AmericanOptionParams = PricingParams;


/**
 * Solver result containing solution surface (interpolate on-demand for specific prices).
 */
struct AmericanOptionResult {
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
        : solution(), surface_2d(), n_space(0), n_time(0),
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
     * @param output_buffer Optional buffer for full spatiotemporal surface.
     *                      If provided, solver writes all time steps to this buffer
     *                      enabling at_time() access. Buffer layout:
     *                      [u_old_initial][step0][step1]...[step(n_time-1)]
     *                      Required size: (n_time + 1) * n_space doubles
     */
    AmericanOptionSolver(const AmericanOptionParams& params,
                        std::shared_ptr<AmericanSolverWorkspace> workspace,
                        std::span<double> output_buffer = {});

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
     * @param output_buffer Optional buffer for full spatiotemporal surface (see constructor)
     * @return Expected containing solver on success, error message on failure
     */
    static std::expected<AmericanOptionSolver, std::string> create(
        const AmericanOptionParams& params,
        std::shared_ptr<AmericanSolverWorkspace> workspace,
        std::span<double> output_buffer = {});

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

    // Optional output buffer for full surface
    std::span<double> output_buffer_;

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
