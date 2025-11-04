#pragma once

#include <span>
#include <cstddef>
#include <vector>

namespace mango {

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
     */
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu) const {

        const size_t n = S.size();

        // Boundaries are handled by boundary conditions
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: centered finite differences
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];
            const double dx_left = S[i] - S[i-1];
            const double dx_right = S[i+1] - S[i];
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
     */
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu) const {

        const size_t n = S.size();
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];
            const double dx_left = S[i] - S[i-1];
            const double dx_right = S[i+1] - S[i];
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
