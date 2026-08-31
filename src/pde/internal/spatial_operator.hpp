// SPDX-License-Identifier: MIT
#pragma once

#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/boundary_conditions.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/operators/centered_difference.hpp"
#include "mango/math/tridiagonal_matrix_view.hpp"
#include <memory>
#include <concepts>
#include <cassert>
#include <span>

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
/// A PDE supports analytical Jacobian if it exposes coefficient methods
/// for the linear operator: L(u) = a·∂²u/∂x² + b(t)·∂u/∂x + c(t)·u.
/// The first-derivative coefficient and discount rate are time-dependent
/// call forms, matching how assemble_jacobian() (and the boundary-row
/// methods below) actually invoke them.
template<typename PDE>
concept HasJacobianCoefficients = requires(const PDE pde, double t) {
    { pde.second_derivative_coeff() } -> std::convertible_to<double>;  // a
    { pde.first_derivative_coeff(t) } -> std::convertible_to<double>;  // b(t)
    { pde.discount_rate(t) } -> std::convertible_to<double>;           // r(t) (c = -r)
};

/// Jacobian coefficients (diag, offdiag) of the ghost-eliminated boundary
/// row's spatial operator ∂L/∂u at the boundary node and its interior
/// neighbor. See SpatialOperator::boundary_row_jacobian().
struct BoundaryRowJacobian { double diag; double offdiag; };

/// SpatialOperator: Composes PDE, GridSpacing, and CenteredDifference
template<typename PDE, std::floating_point T = double>
class SpatialOperator {
public:
    SpatialOperator(PDE pde, std::shared_ptr<GridSpacing<T>> spacing,
                    PDEWorkspace& workspace)
        : pde_(std::move(pde))
        , spacing_(std::move(spacing))
        , stencil_(std::make_shared<CenteredDifference<T>>(*spacing_))
        , workspace_(&workspace)
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
    /// Uses scratch buffers from the workspace
    void apply_interior(double t,
                       std::span<const T> u,
                       std::span<T> Lu,
                       size_t start,
                       size_t end) const {
        auto d2u = workspace_->d2u_scratch();
        auto du = workspace_->du_scratch();

        // Zero only the active range to avoid stale values
        std::fill(d2u.begin() + start, d2u.begin() + end, T(0));
        std::fill(du.begin() + start, du.begin() + end, T(0));

        // Compute derivatives using facade
        stencil_->compute_second_derivative(u, d2u, start, end);
        stencil_->compute_first_derivative(u, du, start, end);

        // Apply PDE operator to combine derivatives
        for (size_t i = start; i < end; ++i) {
            if constexpr (TimeDependentPDE<PDE>) {
                Lu[i] = pde_(t, d2u[i], du[i], u[i]);
            } else {
                Lu[i] = pde_(d2u[i], du[i], u[i]);
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
        const auto& grid = spacing_->grid();

        for (size_t i = 1; i < n - 1; ++i) {
            const T dx_left = grid[i] - grid[i-1];
            const T dx_right = grid[i+1] - grid[i];
            const T dx_avg = (dx_left + dx_right) / 2.0;

            // Second derivative coefficients
            const T d2_coeff_im1 = a / (dx_left * dx_avg);
            const T d2_coeff_i = -a * (1.0 / dx_left + 1.0 / dx_right) / dx_avg;
            const T d2_coeff_ip1 = a / (dx_right * dx_avg);

            // First derivative coefficients (weighted central difference)
            const T d1_denom = dx_left + dx_right;
            const T d1_coeff_im1 = -b * dx_right / (dx_left * d1_denom);
            const T d1_coeff_i   =  b * (dx_right - dx_left) / (dx_left * dx_right);
            const T d1_coeff_ip1 =  b * dx_left / (dx_right * d1_denom);

            // F(u) = u - rhs - coeff_dt·L(u), so ∂F/∂u = I - coeff_dt·∂L/∂u
            const T jac_lower_i = d2_coeff_im1 + d1_coeff_im1;
            const T jac_diag_i = d2_coeff_i + d1_coeff_i + c;
            const T jac_upper_i = d2_coeff_ip1 + d1_coeff_ip1;

            jac.lower()[i - 1] = -coeff_dt * jac_lower_i;
            jac.diag()[i] = 1.0 - coeff_dt * jac_diag_i;
            jac.upper()[i] = -coeff_dt * jac_upper_i;
        }

        // Note: Boundary rows (i=0, i=n-1) are NOT filled here.
        // They must be handled separately based on boundary condition types.
    }

    /// Jacobian coefficients (diag, offdiag) of the ghost-eliminated boundary
    /// row's spatial operator ∂L/∂u at the boundary node and its interior
    /// neighbor. Both boundary rows collapse to the same closed form once
    /// expressed in terms of the adjacent spacing h:
    ///
    ///   left:  L_0     = (2a/h²)·(u_1 − u_0) + c·u_0 + affine
    ///   right: L_{n−1} = (2a/h²)·(u_{n−2} − u_{n−1}) + c·u_{n−1} + affine
    ///
    /// so diag = c − 2a/h² and offdiag = 2a/h² on both sides; only h (and
    /// the affine term, computed separately) differ by side.
    BoundaryRowJacobian boundary_row_jacobian(double t, bc::BoundarySide side) const
        requires HasJacobianCoefficients<PDE>
    {
        const double a = pde_.second_derivative_coeff();
        const double c = -pde_.discount_rate(t);
        const double h = boundary_spacing(side);
        const double coeff = 2.0 * a / (h * h);
        return BoundaryRowJacobian{.diag = c - coeff, .offdiag = coeff};
    }

    /// Affine (gradient-dependent, u-independent) term of the ghost-eliminated
    /// boundary row: g·(b − 2a/h) on the left, g·(b + 2a/h) on the right.
    double boundary_row_affine(double t, bc::BoundarySide side, double g) const
        requires HasJacobianCoefficients<PDE>
    {
        const double a = pde_.second_derivative_coeff();
        const double b = pde_.first_derivative_coeff(t);
        const double h = boundary_spacing(side);
        const double sign = (side == bc::BoundarySide::Left) ? -1.0 : 1.0;
        return g * (b + sign * 2.0 * a / h);
    }

    /// Evaluate the analytic ghost-eliminated boundary row L at the
    /// boundary node, for a Neumann condition with gradient g.
    ///
    /// Defined in terms of boundary_row_jacobian()/boundary_row_affine() so
    /// the eval == jacobian·u + affine identity holds by construction.
    double eval_boundary_row(double t, bc::BoundarySide side, double g,
                             std::span<const T> u) const
        requires HasJacobianCoefficients<PDE>
    {
        const auto jac = boundary_row_jacobian(t, side);
        const double affine = boundary_row_affine(t, side, g);
        const size_t n = u.size();
        const size_t node = (side == bc::BoundarySide::Left) ? 0 : n - 1;
        const size_t neighbor = (side == bc::BoundarySide::Left) ? 1 : n - 2;
        return jac.diag * u[node] + jac.offdiag * u[neighbor] + affine;
    }

private:
    /// Adjacent interior spacing used as the ghost spacing h: dx[0] on the
    /// left, dx[n-2] on the right. Valid for both uniform and nonuniform
    /// grids since it reads directly from the grid points.
    double boundary_spacing(bc::BoundarySide side) const {
        const auto& grid = spacing_->grid();
        const size_t n = grid.size();
        return (side == bc::BoundarySide::Left)
                 ? (grid[1] - grid[0])
                 : (grid[n - 1] - grid[n - 2]);
    }

    PDE pde_;  // Owned by value (PDEs are typically small)
    std::shared_ptr<GridSpacing<T>> spacing_;
    std::shared_ptr<CenteredDifference<T>> stencil_;  // Shared ownership of templated facade
    PDEWorkspace* workspace_;  // Non-owning; workspace outlives operator
};

/// Concept to detect spatial operators exposing analytic ghost-eliminated
/// Neumann boundary rows (SpatialOperator::eval_boundary_row(),
/// boundary_row_jacobian(), boundary_row_affine() above). PDESolver uses
/// this to select the analytic boundary path, with a static_assert
/// diagnostic when a Neumann BC is paired with an operator that lacks it.
template<typename Op>
concept HasBoundaryRows = requires(const Op op, double t, bc::BoundarySide s,
                                   double g, std::span<const double> u) {
    { op.eval_boundary_row(t, s, g, u) } -> std::convertible_to<double>;
    { op.boundary_row_jacobian(t, s) };
    { op.boundary_row_affine(t, s, g) } -> std::convertible_to<double>;
};

} // namespace mango::operators
