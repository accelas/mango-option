#pragma once

#include <span>
#include <cstddef>
#include <vector>

namespace mango {

/**
 * LaplacianOperator: Simple diffusion operator for testing
 *
 * PDE: dV/dt = D * d2V/dx2
 *
 * This is a simplified operator for heat equation and basic PDE testing.
 */
class LaplacianOperator {
public:
    /**
     * Create operator for simple diffusion
     * @param D Diffusion coefficient
     */
    explicit LaplacianOperator(double D) : D_(D) {}

    /**
     * Apply spatial operator: Lu = D * d2u/dx2
     * @param t Current time (unused for Laplacian)
     * @param x Grid points
     * @param u Solution values
     * @param Lu Output: operator applied to u
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void operator()([[maybe_unused]] double t, std::span<const double> x,
                   std::span<const double> u, std::span<double> Lu,
                   std::span<const double> dx) const {
        const size_t n = x.size();

        // Boundaries handled by boundary conditions
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: centered finite differences
        for (size_t i = 1; i < n - 1; ++i) {
            const double dx_left = dx[i-1];   // x[i] - x[i-1]
            const double dx_right = dx[i];     // x[i+1] - x[i]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative: d2u/dx2
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dx2 = d2u / dx_center;

            // Laplacian operator: L(u) = D * d2u/dx2
            Lu[i] = D_ * d2u_dx2;
        }
    }

    /**
     * Apply spatial operator to a single block with halos
     * @param t Current time (unused for Laplacian)
     * @param base_idx Starting global index of interior
     * @param halo_left Left halo size (must be >= 1 for 3-point stencil)
     * @param halo_right Right halo size (must be >= 1 for 3-point stencil)
     * @param x_with_halo Grid points including halos
     * @param u_with_halo Solution values including halos
     * @param Lu_interior Output: operator applied (interior only, no halos)
     * @param dx Pre-computed grid spacing (size n-1)
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

        for (size_t i = 0; i < interior_count; ++i) {
            const size_t j = i + halo_left;  // Index in u_with_halo
            const size_t global_idx = base_idx + i;

            // Access pre-computed dx at global index
            const double dx_left = dx[global_idx - 1];   // x[global] - x[global-1]
            const double dx_right = dx[global_idx];      // x[global+1] - x[global]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // 3-point stencil using halo
            const double d2u = (u_with_halo[j+1] - u_with_halo[j]) / dx_right
                             - (u_with_halo[j] - u_with_halo[j-1]) / dx_left;

            Lu_interior[i] = D_ * d2u / dx_center;
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

private:
    double D_;  // Diffusion coefficient
};

/**
 * EquityBlackScholesOperator: Black-Scholes PDE for equity options
 *
 * PDE: dV/dt = L(V) where
 * L(V) = 0.5*sigma^2*S^2*d2V/dS2 + (r - q)*S*dV/dS - r*V
 *
 * For equity options, dividends are discrete (not part of PDE term).
 * The drift term uses (r - 0) = r since continuous dividend yield q=0.
 */
class EquityBlackScholesOperator {
public:
    /**
     * Create operator for equity option
     * @param r Risk-free rate
     * @param sigma Volatility
     */
    EquityBlackScholesOperator(double r, double sigma)
        : r_(r), sigma_(sigma), sigma_sq_half_(0.5 * sigma * sigma) {}

    /**
     * Apply spatial operator: Lu = L(u)
     * @param t Current time
     * @param S Grid of stock prices
     * @param u Solution values
     * @param Lu Output: operator applied to u
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void apply([[maybe_unused]] double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu,
               std::span<const double> dx) const {

        const size_t n = S.size();

        // Boundaries are handled by boundary conditions
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: centered finite differences using pre-computed dx
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];

            // Use PRE-COMPUTED grid spacing
            const double dx_left = dx[i-1];   // S[i] - S[i-1]
            const double dx_right = dx[i];     // S[i+1] - S[i]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative: d2u/dS2 (centered difference on non-uniform grid)
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // First derivative: du/dS (centered difference)
            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Black-Scholes operator
            // L(u) = 0.5*sigma^2*S^2*d2u/dS2 + r*S*du/dS - r*u
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + r_ * S_i * du_dS
                  - r_ * u[i];
        }
    }

    /**
     * Apply spatial operator to a single block with halos
     * @param t Current time
     * @param base_idx Starting global index of interior
     * @param halo_left Left halo size (must be >= 1 for 3-point stencil)
     * @param halo_right Right halo size (must be >= 1 for 3-point stencil)
     * @param x_with_halo Grid points including halos
     * @param u_with_halo Solution values including halos
     * @param Lu_interior Output: operator applied (interior only, no halos)
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void apply_block([[maybe_unused]] double t,
                     size_t base_idx,
                     size_t halo_left,
                     [[maybe_unused]] size_t halo_right,
                     std::span<const double> x_with_halo,
                     std::span<const double> u_with_halo,
                     std::span<double> Lu_interior,
                     std::span<const double> dx) const {

        const size_t interior_count = Lu_interior.size();

        for (size_t i = 0; i < interior_count; ++i) {
            const size_t j = i + halo_left;
            const size_t global_idx = base_idx + i;

            const double S_i = x_with_halo[j];

            // Grid spacing from pre-computed dx array
            const double dx_left = dx[global_idx - 1];
            const double dx_right = dx[global_idx];
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative: d²u/dS²
            const double d2u = (u_with_halo[j+1] - u_with_halo[j]) / dx_right
                             - (u_with_halo[j] - u_with_halo[j-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // First derivative: weighted three-point (2nd order on non-uniform grids)
            const double w_left = dx_right / (dx_left + dx_right);
            const double w_right = dx_left / (dx_left + dx_right);
            const double du_dS = w_left * (u_with_halo[j] - u_with_halo[j-1]) / dx_left
                               + w_right * (u_with_halo[j+1] - u_with_halo[j]) / dx_right;

            // Black-Scholes operator
            Lu_interior[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                           + r_ * S_i * du_dS
                           - r_ * u_with_halo[j];
        }
    }

private:
    double r_;              // Risk-free rate
    double sigma_;          // Volatility
    double sigma_sq_half_;  // 0.5 * sigma^2 (cached)
};

/**
 * IndexBlackScholesOperator: Black-Scholes PDE for index options
 *
 * PDE: dV/dt = L(V) where
 * L(V) = 0.5*sigma^2*S^2*d2V/dS2 + (r - q)*S*dV/dS - r*V
 *
 * For index options, continuous dividend yield q appears in drift term.
 */
class IndexBlackScholesOperator {
public:
    /**
     * Create operator for index option
     * @param r Risk-free rate
     * @param sigma Volatility
     * @param q Continuous dividend yield
     */
    IndexBlackScholesOperator(double r, double sigma, double q)
        : r_(r), sigma_(sigma), q_(q), sigma_sq_half_(0.5 * sigma * sigma) {}

    /**
     * Apply spatial operator: Lu = L(u)
     * @param t Current time
     * @param S Grid of stock prices
     * @param u Solution values
     * @param Lu Output: operator applied to u
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void apply([[maybe_unused]] double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu,
               std::span<const double> dx) const {

        const size_t n = S.size();
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: use pre-computed dx
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];

            // Use PRE-COMPUTED grid spacing
            const double dx_left = dx[i-1];   // S[i] - S[i-1]
            const double dx_right = dx[i];     // S[i+1] - S[i]
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // First derivative
            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Black-Scholes operator with continuous dividend
            // L(u) = 0.5*sigma^2*S^2*d2u/dS2 + (r - q)*S*du/dS - r*u
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + (r_ - q_) * S_i * du_dS
                  - r_ * u[i];
        }
    }

    /**
     * Apply spatial operator to a single block with halos
     * @param t Current time
     * @param base_idx Starting global index of interior
     * @param halo_left Left halo size (must be >= 1 for 3-point stencil)
     * @param halo_right Right halo size (must be >= 1 for 3-point stencil)
     * @param x_with_halo Grid points including halos
     * @param u_with_halo Solution values including halos
     * @param Lu_interior Output: operator applied (interior only, no halos)
     * @param dx Pre-computed grid spacing (size n-1)
     */
    void apply_block([[maybe_unused]] double t,
                     size_t base_idx,
                     size_t halo_left,
                     [[maybe_unused]] size_t halo_right,
                     std::span<const double> x_with_halo,
                     std::span<const double> u_with_halo,
                     std::span<double> Lu_interior,
                     std::span<const double> dx) const {

        const size_t interior_count = Lu_interior.size();

        for (size_t i = 0; i < interior_count; ++i) {
            const size_t j = i + halo_left;
            const size_t global_idx = base_idx + i;

            const double S_i = x_with_halo[j];

            const double dx_left = dx[global_idx - 1];
            const double dx_right = dx[global_idx];
            const double dx_center = 0.5 * (dx_left + dx_right);

            const double d2u = (u_with_halo[j+1] - u_with_halo[j]) / dx_right
                             - (u_with_halo[j] - u_with_halo[j-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            const double w_left = dx_right / (dx_left + dx_right);
            const double w_right = dx_left / (dx_left + dx_right);
            const double du_dS = w_left * (u_with_halo[j] - u_with_halo[j-1]) / dx_left
                               + w_right * (u_with_halo[j+1] - u_with_halo[j]) / dx_right;

            // Black-Scholes with dividend yield
            Lu_interior[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                           + (r_ - q_) * S_i * du_dS
                           - r_ * u_with_halo[j];
        }
    }

private:
    double r_;              // Risk-free rate
    double sigma_;          // Volatility
    double q_;              // Continuous dividend yield
    double sigma_sq_half_;  // 0.5 * sigma^2 (cached)
};

/**
 * LogMoneynessBlackScholesOperator: Black-Scholes PDE in log-moneyness coordinates
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
 * The operator returns L(V) for the PDE ∂V/∂t = L(V) where t is solver time
 * (t=0 at maturity, t=T at present). This matches backward calendar time.
 *
 * This coordinate system is ideal for American options as it:
 * - Normalizes strike to K=1 (simplifies obstacles)
 * - Makes the operator have constant coefficients
 * - Naturally handles wide ranges of spot prices
 */
class LogMoneynessBlackScholesOperator {
public:
    /**
     * Create Black-Scholes operator in log-moneyness coordinates
     * @param sigma Volatility (σ > 0)
     * @param r Risk-free rate
     * @param d Dividend yield (default: 0.0)
     */
    LogMoneynessBlackScholesOperator(double sigma, double r, double d = 0.0)
        : sigma_(sigma), r_(r), d_(d) {}

    /**
     * Apply spatial operator: Lu = (σ²/2)·∂²u/∂x² + (r-d-σ²/2)·∂u/∂x - r·u
     *
     * This is the Black-Scholes PDE operator in log-moneyness coordinates.
     * PDESolver integrates forward in solver time where t=0 is at maturity.
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

            // Black-Scholes operator (negated for backward-time PDE)
            // ∂V/∂t = -[(σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V]
            Lu[i] = -(half_sigma_sq * d2u_dx2 + drift * du_dx - r_ * u[i]);
        }
    }

    /**
     * Cache-blocked version for large grids (n ≥ 5000).
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
            const size_t j = i + halo_left;
            const size_t global_idx = base_idx + i;

            const double dx_left = dx[global_idx - 1];
            const double dx_right = dx[global_idx];
            const double dx_center = 0.5 * (dx_left + dx_right);

            const double d2u = (u_with_halo[j+1] - u_with_halo[j]) / dx_right
                             - (u_with_halo[j] - u_with_halo[j-1]) / dx_left;
            const double d2u_dx2 = d2u / dx_center;

            const double du_dx = (u_with_halo[j+1] - u_with_halo[j-1]) / (dx_left + dx_right);

            // Black-Scholes operator (negated for backward-time PDE)
            Lu_interior[i] = -(half_sigma_sq * d2u_dx2 + drift * du_dx - r_ * u_with_halo[j]);
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

private:
    double sigma_;  ///< Volatility
    double r_;      ///< Risk-free rate
    double d_;      ///< Dividend yield
};

} // namespace mango
