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

    /// Apply operator: Lu = L(u) with non-uniform grid spacing
    ///
    /// Uses second-order centered differences with local grid spacing.
    /// Boundary values are set to 0 (BCs handle them separately).
    void apply(view_type x, view_type u, view_type Lu, double /*dx_avg*/) const {
        const size_t n = u.extent(0);
        const double half_sigma_sq = half_sigma_sq_;
        const double drift = drift_;
        const double r = r_;

        Kokkos::parallel_for("black_scholes_apply", n,
            KOKKOS_LAMBDA(const size_t i) {
                if (i == 0 || i == n - 1) {
                    // Boundary values: set to 0 (BCs handle separately)
                    Lu(i) = 0.0;
                } else {
                    // Local grid spacings for non-uniform grid
                    double dx_minus = x(i) - x(i - 1);
                    double dx_plus = x(i + 1) - x(i);

                    // Second derivative: non-uniform centered difference
                    // u_xx ≈ 2 * [(u[i+1] - u[i])/dx_plus - (u[i] - u[i-1])/dx_minus] / (dx_plus + dx_minus)
                    double u_xx = 2.0 * ((u(i + 1) - u(i)) / dx_plus - (u(i) - u(i - 1)) / dx_minus) / (dx_minus + dx_plus);

                    // First derivative: non-uniform centered difference
                    // u_x ≈ (u[i+1] - u[i-1]) / (dx_plus + dx_minus)
                    double u_x = (u(i + 1) - u(i - 1)) / (dx_minus + dx_plus);

                    // L(u) = 0.5*sigma^2*u_xx + drift*u_x - r*u
                    Lu(i) = half_sigma_sq * u_xx + drift * u_x - r * u(i);
                }
            });

        Kokkos::fence();
    }

    /// Assemble Jacobian for implicit time stepping with non-uniform grid
    ///
    /// For u_t = L(u), implicit: (I - dt*L)u^{n+1} = u^n
    /// Jacobian J = I - dt*L in tridiagonal form.
    ///
    /// @param dt Time step coefficient
    /// @param x Grid coordinates (for non-uniform spacing)
    /// @param lower Sub-diagonal (size n-1)
    /// @param diag Main diagonal (size n)
    /// @param upper Super-diagonal (size n-1)
    void assemble_jacobian(double dt, view_type x,
                           view_type lower, view_type diag, view_type upper) const {
        const size_t n = diag.extent(0);
        const double half_sigma_sq = half_sigma_sq_;
        const double drift = drift_;
        const double r = r_;

        Kokkos::parallel_for("assemble_jacobian", n,
            KOKKOS_LAMBDA(const size_t i) {
                if (i == 0 || i == n - 1) {
                    // Boundary rows: identity (BCs set separately)
                    diag(i) = 1.0;
                    if (i > 0) {
                        lower(i - 1) = 0.0;
                    }
                    if (i < n - 1) {
                        upper(i) = 0.0;
                    }
                } else {
                    // Local grid spacings for non-uniform grid
                    double dx_minus = x(i) - x(i - 1);
                    double dx_plus = x(i + 1) - x(i);
                    double dx_sum = dx_minus + dx_plus;

                    // Non-uniform FD coefficients for L
                    // Second derivative: coeff_i-1 = 2/(dx_minus * dx_sum)
                    //                    coeff_i = -2/(dx_minus * dx_plus)
                    //                    coeff_i+1 = 2/(dx_plus * dx_sum)
                    // First derivative:  coeff_i-1 = -1/dx_sum
                    //                    coeff_i+1 = 1/dx_sum

                    double c_xx_lower = 2.0 / (dx_minus * dx_sum);
                    double c_xx_diag = -2.0 / (dx_minus * dx_plus);
                    double c_xx_upper = 2.0 / (dx_plus * dx_sum);

                    double c_x_lower = -1.0 / dx_sum;
                    double c_x_upper = 1.0 / dx_sum;

                    // L coefficients: L(u) = half_sigma_sq * u_xx + drift * u_x - r * u
                    double L_lower = half_sigma_sq * c_xx_lower + drift * c_x_lower;
                    double L_diag = half_sigma_sq * c_xx_diag - r;
                    double L_upper = half_sigma_sq * c_xx_upper + drift * c_x_upper;

                    // Jacobian J = I - dt * L
                    diag(i) = 1.0 - dt * L_diag;
                    lower(i - 1) = -dt * L_lower;
                    upper(i) = -dt * L_upper;
                }
            });

        Kokkos::fence();
    }

    /// Legacy interface for uniform grids (backward compatibility)
    void assemble_jacobian(double dt, double dx,
                           view_type lower, view_type diag, view_type upper) const {
        const size_t n = diag.extent(0);
        const double half_sigma_sq = half_sigma_sq_;
        const double drift = drift_;
        const double r = r_;
        const double dx_sq = dx * dx;
        const double two_dx = 2.0 * dx;

        const double L_lower = half_sigma_sq / dx_sq - drift / two_dx;
        const double L_diag = -2.0 * half_sigma_sq / dx_sq - r;
        const double L_upper = half_sigma_sq / dx_sq + drift / two_dx;

        Kokkos::parallel_for("assemble_jacobian_uniform", n,
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
