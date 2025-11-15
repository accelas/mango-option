/**
 * @file normalized_chain_solver.hpp
 * @brief Dimensionless American option solver exploiting scale invariance
 */

#ifndef MANGO_NORMALIZED_CHAIN_SOLVER_HPP
#define MANGO_NORMALIZED_CHAIN_SOLVER_HPP

#include "src/option/american_option.hpp"
#include "src/option/american_solver_workspace.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include <span>
#include <memory>
#include <vector>

namespace mango {

/**
 * Request for normalized PDE solve.
 *
 * Solves dimensionless PDE: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
 * where u = V/K, x = ln(S/K)
 *
 * Output: u(x,τ) on specified grid
 * Caller converts to prices: V = K·u(ln(S/K), τ)
 */
struct NormalizedSolveRequest {
    // PDE coefficients
    double sigma;              ///< Volatility
    double rate;               ///< Risk-free rate
    double dividend;           ///< Continuous dividend yield
    OptionType option_type;    ///< Call or Put

    // Grid configuration
    double x_min;              ///< Minimum log-moneyness
    double x_max;              ///< Maximum log-moneyness
    size_t n_space;            ///< Spatial grid points
    size_t n_time;             ///< Time steps
    double T_max;              ///< Maximum maturity (years)

    // Snapshot collection
    std::span<const double> tau_snapshots;  ///< Maturities to collect

    /// Validate request parameters
    std::expected<void, std::string> validate() const;
};

/**
 * View of normalized solution surface u(x,τ).
 *
 * Provides interpolation interface for querying u at arbitrary (x,τ).
 * Caller scales results: V = K·u
 */
class NormalizedSurfaceView {
public:
    NormalizedSurfaceView(
        std::span<const double> x_grid,
        std::span<const double> tau_grid,
        std::span<const double> values)
        : x_grid_(x_grid)
        , tau_grid_(tau_grid)
        , values_(values)
    {}

    /// Interpolate u(x,τ) using bilinear interpolation
    double interpolate(double x, double tau) const;

    /// Access raw data (for testing)
    std::span<const double> x_grid() const { return x_grid_; }
    std::span<const double> tau_grid() const { return tau_grid_; }
    std::span<const double> values() const { return values_; }

private:
    std::span<const double> x_grid_;
    std::span<const double> tau_grid_;
    std::span<const double> values_;
};

/**
 * Reusable workspace for normalized solves.
 *
 * Allocates all buffers needed for PDE solve + interpolation surface.
 * Thread-safe: each thread creates its own workspace instance.
 */
class NormalizedWorkspace {
public:
    /// Create workspace for given request parameters
    static std::expected<NormalizedWorkspace, std::string> create(
        const NormalizedSolveRequest& request);

    /// Get view of solution surface (after solve completes)
    NormalizedSurfaceView surface_view();

    // No copying (expensive)
    NormalizedWorkspace(const NormalizedWorkspace&) = delete;
    NormalizedWorkspace& operator=(const NormalizedWorkspace&) = delete;

    // Moving OK
    NormalizedWorkspace(NormalizedWorkspace&&) = default;
    NormalizedWorkspace& operator=(NormalizedWorkspace&&) = default;

private:
    NormalizedWorkspace() = default;

    friend class NormalizedChainSolver;

    std::shared_ptr<AmericanSolverWorkspace> pde_workspace_;
    std::vector<double> x_grid_;
    std::vector<double> tau_grid_;
    std::vector<double> values_;  // u(x,τ) [row-major: Nx × Ntau]
};

/**
 * Eligibility limits for normalized solver.
 *
 * Thresholds derived from numerical stability and convergence analysis:
 * - Margin: ≥6 ghost cells to avoid boundary reflection (<0.5bp error)
 * - Width: ≤5.8 log-units for convergence (empirical from sweeps)
 * - Grid spacing: ≤0.05 for Von Neumann stability at σ=200%, τ=2y
 */
struct EligibilityLimits {
    static constexpr double MAX_WIDTH = 5.8;      ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;        ///< Truncation error O(dx²)
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    /// Minimum margin (6 ghost cells or 0.35, whichever is larger)
    static double min_margin(double dx) {
        return std::max(MIN_MARGIN_ABS, 6.0 * dx);
    }

    /// Maximum ratio K_max/K_min given dx
    static double max_ratio(double dx) {
        double margin = min_margin(dx);
        return std::exp(MAX_WIDTH - 2.0 * margin);
    }
};

/**
 * Normalized chain solver.
 *
 * Solves American option PDE in dimensionless coordinates exploiting
 * scale invariance: V(S,K,τ) = K·u(ln(S/K), τ)
 *
 * One PDE solve yields prices for all (S,K) combinations via interpolation.
 */
class NormalizedChainSolver {
public:
    /**
     * Solve normalized PDE.
     *
     * Solves: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
     * Terminal condition: u(x,0) = max(eˣ-1, 0) (call) or max(1-eˣ, 0) (put)
     * Boundary conditions: American exercise constraint u ≥ intrinsic
     *
     * @param request PDE configuration and snapshot times
     * @param workspace Pre-allocated buffers (reusable across solves)
     * @param surface_view Output view (references workspace.values_)
     * @return Success or solver error
     */
    static std::expected<void, SolverError> solve(
        const NormalizedSolveRequest& request,
        NormalizedWorkspace& workspace,
        NormalizedSurfaceView& surface_view);

    /**
     * Check eligibility for normalized solving.
     *
     * Criteria:
     * 1. Grid spacing dx ≤ MAX_DX (0.05)
     * 2. Domain width ≤ MAX_WIDTH (5.8)
     * 3. Margins ≥ min_margin(dx) on both boundaries
     *
     * @param request Request to validate
     * @param moneyness_grid Moneyness values m = K/S from price table
     * @return Success or reason for ineligibility
     */
    static std::expected<void, std::string> check_eligibility(
        const NormalizedSolveRequest& request,
        std::span<const double> moneyness_grid);
};

}  // namespace mango

#endif  // MANGO_NORMALIZED_CHAIN_SOLVER_HPP
