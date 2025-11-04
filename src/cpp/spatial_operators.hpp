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
    void operator()(double t, std::span<const double> x,
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
    void apply_block(double t,
                     size_t base_idx,
                     size_t halo_left,
                     size_t halo_right,
                     std::span<const double> x_with_halo,
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
    void apply(double t, std::span<const double> S,
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
    void apply(double t, std::span<const double> S,
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

private:
    double r_;              // Risk-free rate
    double sigma_;          // Volatility
    double q_;              // Continuous dividend yield
    double sigma_sq_half_;  // 0.5 * sigma^2 (cached)
};

} // namespace mango
