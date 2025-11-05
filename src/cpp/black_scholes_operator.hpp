#pragma once

#include <span>
#include <cmath>
#include <cstddef>

namespace mango {

/**
 * Black-Scholes spatial operator in log-moneyness coordinates.
 *
 * Implements L(V) for the Black-Scholes PDE in coordinates x = ln(S/K):
 *   ∂V/∂τ = (σ²/2)·∂²V/∂x² + (r - d - σ²/2)·∂V/∂x - r·V
 *
 * where:
 *   - τ = T - t (time to maturity, backward time)
 *   - σ = volatility
 *   - r = risk-free rate
 *   - d = dividend yield
 *   - x = ln(S/K) (log-moneyness)
 *
 * Since our solver uses forward time t (not backward time τ = T - t),
 * we return -L(V) to match the sign convention for ∂V/∂t = L(V).
 *
 * Uses centered finite differences:
 *   ∂²V/∂x² ≈ (V[i-1] - 2V[i] + V[i+1]) / dx²
 *   ∂V/∂x ≈ (V[i+1] - V[i-1]) / (2dx)
 *
 * Boundary points are set to zero (handled by boundary conditions).
 *
 * The operator supports both full-array and cache-blocked evaluation:
 *   - operator(): Full-array evaluation for small grids
 *   - apply_block(): Block-aware evaluation for large grids (cache optimization)
 *
 * Cache blocking is automatically selected by PDESolver based on grid size.
 */
class LogMoneynessBlackScholesOperator {
public:
    /**
     * Create Black-Scholes operator in log-moneyness coordinates
     *
     * @param sigma Volatility (σ > 0)
     * @param r Risk-free rate
     * @param d Dividend yield (default: 0.0)
     */
    LogMoneynessBlackScholesOperator(double sigma, double r, double d = 0.0)
        : sigma_(sigma), r_(r), d_(d) {}

    /**
     * Apply spatial operator: Lu = -[(σ²/2)·∂²u/∂x² + (r-d-σ²/2)·∂u/∂x - r·u]
     *
     * The negative sign converts from backward time (∂V/∂τ) to forward time (∂V/∂t).
     *
     * @param t Current time (unused for time-independent coefficients)
     * @param x Grid points (log-moneyness values)
     * @param u Solution at current time
     * @param Lu Output: L(u) applied to entire grid
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void operator()([[maybe_unused]] double t,
                    std::span<const double> x,
                    std::span<const double> u,
                    std::span<double> Lu,
                    std::span<const double> dx) const {
        const size_t n = x.size();

        // Boundary points set to zero (handled by BCs)
        Lu[0] = Lu[n-1] = 0.0;

        const double half_sigma_sq = 0.5 * sigma_ * sigma_;
        const double drift = r_ - d_ - half_sigma_sq;

        // Interior points: apply Black-Scholes operator
        #pragma omp simd
        for (size_t i = 1; i < n - 1; ++i) {
            // Use pre-computed dx for non-uniform grids
            const double dx_left = dx[i-1];   // x[i] - x[i-1]
            const double dx_right = dx[i];    // x[i+1] - x[i]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative: ∂²V/∂x² (centered finite difference)
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dx2 = d2u / dx_center;

            // First derivative: ∂V/∂x (centered finite difference)
            const double du_dx = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Black-Scholes operator in forward time (negated)
            // L(u) = -(σ²/2·∂²u/∂x² + (r-d-σ²/2)·∂u/∂x - r·u)
            Lu[i] = -(half_sigma_sq * d2u_dx2 + drift * du_dx - r_ * u[i]);
        }
    }

    /// Compute first derivative ∂u/∂x using centered finite differences
    void compute_first_derivative([[maybe_unused]] std::span<const double> x,
                                  std::span<const double> u,
                                  std::span<double> du,
                                  std::span<const double> dx) const {
        const size_t n = u.size();
        if (n < 2) return;

        // Interior: centered difference
        for (size_t i = 1; i < n - 1; ++i) {
            double dx_total = dx[i] + dx[i-1];
            du[i] = (u[i+1] - u[i-1]) / dx_total;
        }

        // Boundaries: one-sided
        du[0] = (u[1] - u[0]) / dx[0];
        du[n-1] = (u[n-1] - u[n-2]) / dx[n-2];
    }

    /// Compute second derivative ∂²u/∂x² using centered finite differences
    void compute_second_derivative([[maybe_unused]] std::span<const double> x,
                                   std::span<const double> u,
                                   std::span<double> d2u,
                                   std::span<const double> dx) const {
        const size_t n = u.size();
        if (n < 3) {
            std::fill(d2u.begin(), d2u.end(), 0.0);
            return;
        }

        // Interior: centered difference
        for (size_t i = 1; i < n - 1; ++i) {
            double left_slope = (u[i] - u[i-1]) / dx[i-1];
            double right_slope = (u[i+1] - u[i]) / dx[i];
            double dx_avg = 0.5 * (dx[i] + dx[i-1]);
            d2u[i] = (right_slope - left_slope) / dx_avg;
        }

        // Boundaries: zero (needs ghost points for accuracy)
        d2u[0] = 0.0;
        d2u[n-1] = 0.0;
    }

    /**
     * Cache-blocked version for large grids (n ≥ 5000).
     *
     * Applies operator to a contiguous block of interior points using halos
     * for stencil dependencies. This improves cache locality by processing
     * data in smaller chunks that fit in L1 cache.
     *
     * @param t Current time (unused)
     * @param base_idx Starting global index of interior
     * @param halo_left Left halo size (must be >= 1 for 3-point stencil)
     * @param halo_right Right halo size (must be >= 1 for 3-point stencil)
     * @param x_with_halo Grid points including halos
     * @param u_with_halo Solution values including halos
     * @param Lu_interior Output: operator applied (interior only, no halos)
     * @param dx Pre-computed grid spacing (global array, size n-1)
     */
    void apply_block([[maybe_unused]] double t,
                     size_t base_idx,
                     size_t halo_left,
                     [[maybe_unused]] size_t halo_right,
                     [[maybe_unused]] std::span<const double> x_with_halo,
                     std::span<const double> u_with_halo,
                     std::span<double> Lu_interior,
                     std::span<const double> dx) const {

        const size_t interior_count = Lu_interior.size();
        const double half_sigma_sq = 0.5 * sigma_ * sigma_;
        const double drift = r_ - d_ - half_sigma_sq;

        #pragma omp simd
        for (size_t i = 0; i < interior_count; ++i) {
            const size_t j = i + halo_left;  // Index in u_with_halo
            const size_t global_idx = base_idx + i;

            // Access pre-computed dx at global index
            const double dx_left = dx[global_idx - 1];   // x[global] - x[global-1]
            const double dx_right = dx[global_idx];      // x[global+1] - x[global]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative using halo points
            const double d2u = (u_with_halo[j+1] - u_with_halo[j]) / dx_right
                             - (u_with_halo[j] - u_with_halo[j-1]) / dx_left;
            const double d2u_dx2 = d2u / dx_center;

            // First derivative using halo points
            const double du_dx = (u_with_halo[j+1] - u_with_halo[j-1]) / (dx_left + dx_right);

            // Black-Scholes operator (negated for forward time)
            Lu_interior[i] = -(half_sigma_sq * d2u_dx2 + drift * du_dx - r_ * u_with_halo[j]);
        }
    }

private:
    double sigma_;  ///< Volatility
    double r_;      ///< Risk-free rate
    double d_;      ///< Dividend yield
};

}  // namespace mango
