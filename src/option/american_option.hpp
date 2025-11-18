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

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_HPP
