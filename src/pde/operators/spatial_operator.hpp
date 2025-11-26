#pragma once

#include "src/pde/core/grid.hpp"
#include "centered_difference_facade.hpp"
#include "src/math/tridiagonal_matrix_view.hpp"
#include <span>
#include <memory>
#include <concepts>
#include <cassert>
#include <cmath>

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
///
/// Note: Uses std::fma in Jacobian assembly for improved precision and performance.
/// Therefore T must be a standard floating-point type (float, double, long double).
template<typename PDE, std::floating_point T = double>
class SpatialOperator {
public:
    SpatialOperator(PDE pde, std::shared_ptr<GridSpacing<T>> spacing)
        : pde_(std::move(pde))
        , spacing_(std::move(spacing))
        , stencil_(std::make_shared<CenteredDifference<T>>(*spacing_))
    {}

    // Default copy/move (shared_ptr makes it copyable)
    SpatialOperator(const SpatialOperator&) = default;
    SpatialOperator& operator=(const SpatialOperator&) = default;
    SpatialOperator(SpatialOperator&&) noexcept = default;
    SpatialOperator& operator=(SpatialOperator&&) noexcept = default;

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
        // Temporary buffers for derivatives
        thread_local std::vector<T> d2u_dx2_buf;
        thread_local std::vector<T> du_dx_buf;

        const size_t n = u.size();
        d2u_dx2_buf.resize(n);
        du_dx_buf.resize(n);

        // Zero only the active range to avoid stale values (optimization)
        std::fill(d2u_dx2_buf.begin() + start, d2u_dx2_buf.begin() + end, T(0));
        std::fill(du_dx_buf.begin() + start, du_dx_buf.begin() + end, T(0));

        // Compute derivatives using facade
        stencil_->compute_second_derivative(u, std::span<T>(d2u_dx2_buf), start, end);
        stencil_->compute_first_derivative(u, std::span<T>(du_dx_buf), start, end);

        // Apply PDE operator to combine derivatives
        for (size_t i = start; i < end; ++i) {
            if constexpr (TimeDependentPDE<PDE>) {
                Lu[i] = pde_(t, d2u_dx2_buf[i], du_dx_buf[i], u[i]);
            } else {
                Lu[i] = pde_(d2u_dx2_buf[i], du_dx_buf[i], u[i]);
            }
        }
    }

    /// Greeks computation (delegates to stencil)
    void compute_first_derivative(std::span<const T> u,
                                 std::span<T> du_dx) const {
        const auto range = interior_range(u.size());
        stencil_->compute_first_derivative(u, du_dx, range.start, range.end);
    }

    void compute_second_derivative(std::span<const T> u,
                                  std::span<T> d2u_dx2) const {
        const auto range = interior_range(u.size());
        stencil_->compute_second_derivative(u, d2u_dx2, range.start, range.end);
    }

    /// Assemble analytical Jacobian for PDEs with time-varying coefficients
    ///
    /// For linear PDEs of the form L(u) = a·∂²u/∂x² + b·∂u/∂x + c·u,
    /// computes the Jacobian matrix ∂L/∂u analytically in O(n) time.
    ///
    /// Available only for PDEs satisfying HasJacobianCoefficients concept.
    ///
    /// @param t Current time (for time-varying rates)
    /// @param coeff_dt TR-BDF2 weight coefficient
    /// @param jac Tridiagonal matrix view to populate
    void assemble_jacobian([[maybe_unused]] double t,
                          [[maybe_unused]] double coeff_dt,
                          TridiagonalMatrixView& jac) const
        requires HasJacobianCoefficients<PDE>
    {
        // Get PDE coefficients at current time t
        const T a = pde_.second_derivative_coeff();   // σ²/2 (time-independent)
        const T b = pde_.first_derivative_coeff(t);   // r(t) - d - σ²/2
        const T c = -pde_.discount_rate(t);           // -r(t)

        const size_t n = jac.size();

        if (spacing_->is_uniform()) {
            // Uniform grid: constant coefficients (O(1) compute + O(n) fill)
            const T dx = spacing_->spacing();
            const T dx_sq = dx * dx;

            // Jacobian of L(u): ∂L/∂u
            // Use FMA for coefficient computations
            const T jac_lower_coeff = std::fma(a, T(1) / dx_sq, -b / (T(2) * dx));
            const T jac_diag_coeff = std::fma(T(-2) * a, T(1) / dx_sq, c);
            const T jac_upper_coeff = std::fma(a, T(1) / dx_sq, b / (T(2) * dx));

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
            const auto& grid = spacing_->grid();
            for (size_t i = 1; i < n - 1; ++i) {
                // Get local grid spacing directly from grid
                const T dx_left = grid[i] - grid[i-1];      // x[i] - x[i-1]
                const T dx_right = grid[i+1] - grid[i];     // x[i+1] - x[i]
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
    std::shared_ptr<CenteredDifference<T>> stencil_;  // Shared ownership of templated facade
};

} // namespace mango::operators
