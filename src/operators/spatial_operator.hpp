#pragma once

#include "grid_spacing.hpp"
#include "centered_difference.hpp"
#include "../jacobian_view.hpp"
#include <span>
#include <memory>
#include <concepts>
#include <cassert>

namespace mango::operators {

/// Helper to describe stencil interior range
struct StencilInterior {
    size_t start;  // First interior point
    size_t end;    // One past last interior point
};

/// Concept to detect time-dependent PDEs
///
/// A PDE is time-dependent if it accepts a time parameter in its operator().
/// Time-dependent: operator()(double t, double d2u, double du, double u)
/// Time-independent: operator()(double d2u, double du, double u)
template<typename PDE>
concept TimeDependentPDE = requires(PDE pde, double t, double d2u, double du, double u) {
    { pde(t, d2u, du, u) } -> std::convertible_to<double>;
};

/// Concept to detect PDEs with analytical Jacobian coefficients
///
/// A PDE supports analytical Jacobian if it exposes constant coefficient methods
/// for the linear operator: L(u) = a·∂²u/∂x² + b·∂u/∂x + c·u
template<typename PDE>
concept HasJacobianCoefficients = requires(const PDE pde) {
    { pde.second_derivative_coeff() } -> std::convertible_to<double>;  // a
    { pde.first_derivative_coeff() } -> std::convertible_to<double>;   // b
    { pde.discount_rate() } -> std::convertible_to<double>;            // r (c = -r)
};

/// SpatialOperator: Composes PDE, GridSpacing, and CenteredDifference
template<typename PDE, typename T = double>
class SpatialOperator {
public:
    SpatialOperator(PDE pde, std::shared_ptr<GridSpacing<T>> spacing)
        : pde_(std::move(pde))
        , spacing_(std::move(spacing))
        , stencil_(*spacing_)
    {}

    /// Get interior range for this stencil (3-point: [1, n-1))
    /// Precondition: n >= GridSpacing<T>::min_stencil_size() (i.e., n >= 3)
    StencilInterior interior_range(size_t n) const {
        assert(n >= GridSpacing<T>::min_stencil_size() && "Grid too small for stencil");
        return {1, n - 1};  // 3-point stencil width
    }

    /// Apply operator to full grid (convenience)
    void apply(double t, std::span<const T> u, std::span<T> Lu) const {
        const auto range = interior_range(u.size());
        apply_interior(t, u, Lu, range.start, range.end);
    }

    /// Apply operator to interior points only [start, end)
    /// Used by both full-grid and cache-blocked evaluation
    void apply_interior(double t,
                       std::span<const T> u,
                       std::span<T> Lu,
                       size_t start,
                       size_t end) const {
        // Create evaluator lambda that handles time parameter
        auto eval = [&](T d2u, T du, T val) -> T {
            if constexpr (TimeDependentPDE<PDE>) {
                return pde_(t, d2u, du, val);  // Time-dependent PDE
            } else {
                return pde_(d2u, du, val);     // Time-independent PDE
            }
        };

        // Dispatch to appropriate stencil strategy
        if (spacing_->is_uniform()) {
            stencil_.apply_uniform(u, Lu, start, end, eval);
        } else {
            stencil_.apply_non_uniform(u, Lu, start, end, eval);
        }
    }

    /// Greeks computation (delegates to stencil)
    void compute_first_derivative(std::span<const T> u,
                                 std::span<T> du_dx) const {
        const auto range = interior_range(u.size());
        stencil_.compute_all_first(u, du_dx, range.start, range.end);
    }

    void compute_second_derivative(std::span<const T> u,
                                  std::span<T> d2u_dx2) const {
        const auto range = interior_range(u.size());
        stencil_.compute_all_second(u, d2u_dx2, range.start, range.end);
    }

    /// Assemble analytical Jacobian for PDEs with constant coefficients
    ///
    /// For linear PDEs of the form L(u) = a·∂²u/∂x² + b·∂u/∂x + c·u,
    /// computes the Jacobian matrix ∂L/∂u analytically in O(n) time.
    ///
    /// Available only for PDEs satisfying HasJacobianCoefficients concept.
    ///
    /// @param coeff_dt TR-BDF2 weight coefficient
    /// @param jac Jacobian view to populate
    void assemble_jacobian([[maybe_unused]] double coeff_dt,
                          JacobianView& jac) const
        requires HasJacobianCoefficients<PDE>
    {
        // Get PDE coefficients
        const T a = pde_.second_derivative_coeff();  // σ²/2
        const T b = pde_.first_derivative_coeff();   // r - d - σ²/2
        const T c = -pde_.discount_rate();           // -r

        const size_t n = jac.size();

        if (spacing_->is_uniform()) {
            // Uniform grid: constant coefficients (O(1) compute + O(n) fill)
            const T dx = spacing_->spacing();
            const T dx_sq = dx * dx;

            // Jacobian of L(u): ∂L/∂u
            const T jac_lower_coeff = a / dx_sq - b / (2.0 * dx);
            const T jac_diag_coeff = -2.0 * a / dx_sq + c;
            const T jac_upper_coeff = a / dx_sq + b / (2.0 * dx);

            // F(u) = u - rhs - coeff_dt·L(u)
            // ∂F/∂u = I - coeff_dt·∂L/∂u
            const T lower = -coeff_dt * jac_lower_coeff;
            const T diag = 1.0 - coeff_dt * jac_diag_coeff;
            const T upper = -coeff_dt * jac_upper_coeff;

            // Fill interior points
            for (size_t i = 1; i < n - 1; ++i) {
                jac.lower()[i - 1] = lower;
                jac.diag()[i] = diag;
                jac.upper()[i] = upper;
            }
        } else {
            // Non-uniform grid: per-point coefficients (O(n))
            for (size_t i = 1; i < n - 1; ++i) {
                // Get local grid spacing
                const T dx_left = spacing_->left_spacing(i);    // x[i] - x[i-1]
                const T dx_right = spacing_->right_spacing(i);  // x[i+1] - x[i]
                const T dx_avg = (dx_left + dx_right) / 2.0;

                // Second derivative: (u[i-1]/dx_left - u[i]*(1/dx_left + 1/dx_right) + u[i+1]/dx_right) / dx_avg
                const T d2_coeff_im1 = a / (dx_left * dx_avg);
                const T d2_coeff_i = -a * (1.0 / dx_left + 1.0 / dx_right) / dx_avg;
                const T d2_coeff_ip1 = a / (dx_right * dx_avg);

                // First derivative: (u[i+1] - u[i-1]) / (dx_left + dx_right)
                const T d1_denom = dx_left + dx_right;
                const T d1_coeff_im1 = -b / d1_denom;
                const T d1_coeff_ip1 = b / d1_denom;

                // Jacobian of L(u): ∂L_i/∂u
                const T jac_lower_i = d2_coeff_im1 + d1_coeff_im1;
                const T jac_diag_i = d2_coeff_i + c;
                const T jac_upper_i = d2_coeff_ip1 + d1_coeff_ip1;

                // F(u) = u - rhs - coeff_dt·L(u), so ∂F/∂u = I - coeff_dt·∂L/∂u
                jac.lower()[i - 1] = -coeff_dt * jac_lower_i;
                jac.diag()[i] = 1.0 - coeff_dt * jac_diag_i;
                jac.upper()[i] = -coeff_dt * jac_upper_i;
            }
        }

        // Note: Boundary rows (i=0, i=n-1) are NOT filled here.
        // They must be handled separately based on boundary condition types.
    }

private:
    PDE pde_;  // Owned by value (PDEs are typically small)
    std::shared_ptr<GridSpacing<T>> spacing_;
    CenteredDifference<T> stencil_;
};

} // namespace mango::operators
