#pragma once

/// @file black_scholes.hpp
/// @brief Black-Scholes spatial operator using Kokkos

#include <Kokkos_Core.hpp>
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Black-Scholes spatial operator in log-moneyness coordinates
///
/// Implements: L(u) = 0.5*sigma^2*u_xx + (r-q-0.5*sigma^2)*u_x - r*u
/// where x = log(S/K) is log-moneyness.
///
/// @note Current implementation assumes uniform grid spacing.
///       Non-uniform grid support may be added in future versions.
template <typename MemSpace>
class BlackScholesOperator {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Construct Black-Scholes operator
    /// @param sigma Volatility (annualized, e.g., 0.20 for 20%)
    /// @param r Risk-free interest rate (annualized)
    /// @param q Continuous dividend yield (annualized)
    BlackScholesOperator(double sigma, double r, double q)
        : sigma_(sigma), r_(r), q_(q),
          half_sigma_sq_(0.5 * sigma * sigma),
          drift_(r - q - 0.5 * sigma * sigma) {}

    /// Apply operator: Lu = L(u)
    ///
    /// Uses second-order centered differences.
    /// Boundary values in Lu are undefined (caller handles BCs).
    void apply(view_type x, view_type u, view_type Lu, double dx) const {
        const size_t n = u.extent(0);
        const double half_sigma_sq = half_sigma_sq_;
        const double drift = drift_;
        const double r = r_;
        const double dx_sq = dx * dx;
        const double two_dx = 2.0 * dx;

        Kokkos::parallel_for("black_scholes_apply",
            Kokkos::RangePolicy<typename MemSpace::execution_space>(1, n - 1),
            KOKKOS_LAMBDA(const size_t i) {
                // Second derivative: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
                double u_xx = (u(i + 1) - 2.0 * u(i) + u(i - 1)) / dx_sq;

                // First derivative: (u[i+1] - u[i-1]) / (2*dx)
                double u_x = (u(i + 1) - u(i - 1)) / two_dx;

                // L(u) = 0.5*sigma^2*u_xx + drift*u_x - r*u
                Lu(i) = half_sigma_sq * u_xx + drift * u_x - r * u(i);
            });

        Kokkos::fence();
    }

    /// Assemble Jacobian for implicit time stepping
    ///
    /// For u_t = L(u), implicit: (I - dt*L)u^{n+1} = u^n
    /// Jacobian J = I - dt*L in tridiagonal form.
    void assemble_jacobian(double dt, double dx,
                           view_type lower, view_type diag, view_type upper) const {
        const size_t n = diag.extent(0);
        const double half_sigma_sq = half_sigma_sq_;
        const double drift = drift_;
        const double r = r_;
        const double dx_sq = dx * dx;
        const double two_dx = 2.0 * dx;

        // Coefficients for L in tridiagonal form
        // L_lower = 0.5*sigma^2/dx^2 - drift/(2*dx)
        // L_diag = -sigma^2/dx^2 - r
        // L_upper = 0.5*sigma^2/dx^2 + drift/(2*dx)

        const double L_lower = half_sigma_sq / dx_sq - drift / two_dx;
        const double L_diag = -2.0 * half_sigma_sq / dx_sq - r;
        const double L_upper = half_sigma_sq / dx_sq + drift / two_dx;

        Kokkos::parallel_for("assemble_jacobian", n,
            KOKKOS_LAMBDA(const size_t i) {
                diag(i) = 1.0 - dt * L_diag;
                if (i > 0) {
                    lower(i - 1) = -dt * L_lower;
                }
                if (i < n - 1) {
                    upper(i) = -dt * L_upper;
                }
            });

        Kokkos::fence();
    }

    // Accessor methods for coefficients (useful for testing)
    double half_sigma_sq() const { return half_sigma_sq_; }
    double drift() const { return drift_; }
    double discount_rate() const { return r_; }

private:
    double sigma_;
    double r_;
    double q_;
    double half_sigma_sq_;
    double drift_;
};

}  // namespace mango::kokkos
