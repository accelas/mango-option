// SPDX-License-Identifier: MIT
#pragma once

#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/boundary_conditions.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/internal/fitted_diffusion.hpp"
#include "mango/pde/operators/centered_difference.hpp"
#include "mango/math/tridiagonal_matrix_view.hpp"
#include <memory>
#include <concepts>
#include <cassert>
#include <limits>
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
///
/// CONTRACT (#472): for any PDE satisfying this concept, the coefficient
/// methods are the AUTHORITATIVE definition of the interior operator:
/// SpatialOperator evaluates L(u) = a_f·u'' + b(t)·u' − r(t)·u from them
/// (with Il'in-fitted a_f; see fitted_diffusion.hpp) and never calls
/// operator() on interior nodes. Accessors must be pure (deterministic,
/// side-effect free) at fixed t, and a = second_derivative_coeff() must
/// be >= 0 unconditionally (a < 0 is outside the fitting contract; a == 0
/// is supported as the convection limit — see fitted_diffusion.hpp).
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
        if constexpr (HasJacobianCoefficients<PDE>) {
            // Coefficient-combine path (#472): coefficients are the
            // authoritative operator definition (concept contract), and the
            // fitted diffusion keeps this residual/RHS evaluation the SAME
            // operator the assembled Jacobian represents — the projected
            // LCP stage solves A·u = rhs with A from assemble_jacobian, so
            // the two paths must not diverge.
            const T a = pde_.second_derivative_coeff();
            const T b = pde_.first_derivative_coeff(t);
            const T r = pde_.discount_rate(t);
            ensure_fitted_cache(a, b);
            const auto a_f_cache = workspace_->a_f_cache();
            for (size_t i = start; i < end; ++i) {
                Lu[i] = a_f_cache[i] * d2u[i] + b * du[i] - r * u[i];
            }
        } else {
            for (size_t i = start; i < end; ++i) {
                if constexpr (TimeDependentPDE<PDE>) {
                    Lu[i] = pde_(t, d2u[i], du[i], u[i]);
                } else {
                    Lu[i] = pde_(d2u[i], du[i], u[i]);
                }
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
        // Sample coefficients ONCE per invocation (concept contract: pure
        // at fixed t). The per-node quantity is the fitted diffusion only.
        const T a = pde_.second_derivative_coeff();   // σ²/2
        const T b = pde_.first_derivative_coeff(t);   // r(t) - d - σ²/2
        const T c = -pde_.discount_rate(t);           // -r(t)

        const size_t n = jac.size();
        assert(n == spacing_->grid().size() &&
               "Jacobian view size must match the grid this operator was built over");
        const auto& grid = spacing_->grid();
        // Uniform grids: use the canonical stored spacing, NOT per-cell
        // coordinate differences — CenteredDifference's uniform stencil
        // uses spacing_->spacing() for every derivative, and per-cell
        // diffs differ from it in the last ulp, which would break the
        // exact apply/Jacobian identity the fitted scheme requires.
        const bool uniform = spacing_->is_uniform();
        const T h_uniform = uniform ? spacing_->spacing() : T(0);

        ensure_fitted_cache(a, b);
        const auto a_f_cache = workspace_->a_f_cache();

        for (size_t i = 1; i < n - 1; ++i) {
            const T dx_left = uniform ? h_uniform : grid[i] - grid[i-1];
            const T dx_right = uniform ? h_uniform : grid[i+1] - grid[i];
            const T dx_avg = (dx_left + dx_right) / 2.0;

            const T a_f = a_f_cache[i];
            // z (binding half-cell drift mass) is cheap arithmetic, not a
            // transcendental — recomputed per node directly rather than
            // cached, matching fitted_diffusion()'s own z formula exactly.
            const T h_binding = (b > 0.0) ? dx_right : dx_left;
            const T z = T(0.5) * std::abs(b) * h_binding;

            // Sign-preserving reduced assembly (#472): the binding-side
            // numerator is computed literally as a_f − z, which
            // fitted_diffusion()'s clamp keeps >= 0 exactly in floating
            // point. The non-binding numerator is a sum of non-negatives.
            // This replaces the separate d2+d1 coefficient accumulation,
            // which could round the binding entry to a tiny negative at
            // high cell Péclet.
            T num_lower, num_upper;
            if (b >= 0.0) {
                num_lower = a_f - z;               // binding (z = b·dx_r/2)
                num_upper = a_f + T(0.5) * b * dx_left;
            } else {
                num_lower = a_f - T(0.5) * b * dx_right;  // adds |b|·dx_r/2
                num_upper = a_f - z;               // binding (z = |b|·dx_l/2)
            }
            const T lower = num_lower / (dx_left * dx_avg);
            const T upper = num_upper / (dx_right * dx_avg);
            const T diag = c - lower - upper;  // row-sum identity by construction

            jac.lower()[i - 1] = -coeff_dt * lower;
            jac.diag()[i] = 1.0 - coeff_dt * diag;
            jac.upper()[i] = -coeff_dt * upper;
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
    ///
    /// NOTE (#472): boundary rows deliberately use the raw (unfitted)
    /// diffusion coefficient a. Their ghost-eliminated off-diagonal
    /// +2a/h² already has the Z-matrix sign for any drift b (drift enters
    /// only the affine term), and each row's eval == jacobian·u + affine
    /// identity is internal to the row, so residual/Jacobian consistency
    /// is unaffected by the interior fitting.
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

    /// Ensure workspace_->a_f_cache()[1 .. n-2] holds the Il'in-fitted
    /// diffusion coefficient for the CURRENTLY sampled (a, b, grid) (#472
    /// perf follow-up: measured ~1.85x slower on the default sinh-spaced
    /// grid without this cache — std::tanh landed on every interior node
    /// of every apply()/assemble_jacobian() call). Recomputes only when
    /// (a, b, grid) differ from the last sampled triple, via exact double
    /// comparison against cached_a_/cached_b_ and pointer comparison
    /// against cached_grid_ (initialized to NaN/nullptr so the first call
    /// always misses). For a constant-rate PDE, b never changes across a
    /// whole solve, so std::tanh runs once per grid instead of once per
    /// node per call. For a callable-rate PDE, b(t) changes only at
    /// time-step/stage boundaries, so the cache still amortizes std::tanh
    /// across the several apply() calls (residual, Newton line search,
    /// ...) that share one (t, a, b) sample. The grid key exists so a
    /// cache filled for one GridSpacing can never be silently read back
    /// for a different one that happens to sample the same (a, b) — e.g.
    /// two SpatialOperator instances sharing a PDEWorkspace across grids
    /// (not a pattern used today, but (a, b) alone cannot distinguish it).
    ///
    /// THREAD-SAFETY: cached_a_/cached_b_/cached_grid_ are `mutable`, and
    /// the cache array lives in *workspace_ (non-owning). This is safe
    /// only because a SpatialOperator/PDEWorkspace pair is owned 1:1 by a
    /// single solver instance and never shared across threads: grepping
    /// create_spatial_operator() call sites under src/ shows exactly two
    /// (src/option/american_option.cpp, both inside a solver class that
    /// constructs its own workspace_local_ and spatial_op_ once, in its
    /// constructor), and the OpenMP batch loop
    /// (src/option/american_option_batch.cpp) constructs one full solver
    /// per loop iteration rather than sharing one across iterations/
    /// threads. Do not add a call site that shares one SpatialOperator
    /// (or its workspace) across threads without revisiting this cache.
    void ensure_fitted_cache(T a, T b) const {
        const void* grid_key = spacing_.get();
        if (a == cached_a_ && b == cached_b_ && grid_key == cached_grid_) {
            return;
        }
        cached_a_ = a;
        cached_b_ = b;
        cached_grid_ = grid_key;
        const auto& grid = spacing_->grid();
        const size_t n = grid.size();
        assert(n == workspace_->a_f_cache().size() &&
               "a_f_cache size must match the grid this operator was built over");
        auto cache = workspace_->a_f_cache();
        if (spacing_->is_uniform()) {
            const T h = spacing_->spacing();
            const T a_f = detail::fitted_diffusion(a, b, h, h).a_f;
            std::fill(cache.begin() + 1, cache.begin() + (n - 1), a_f);
        } else {
            for (size_t i = 1; i < n - 1; ++i) {
                const T dx_left = grid[i] - grid[i - 1];
                const T dx_right = grid[i + 1] - grid[i];
                cache[i] = detail::fitted_diffusion(a, b, dx_left, dx_right).a_f;
            }
        }
    }

    PDE pde_;  // Owned by value (PDEs are typically small)
    std::shared_ptr<GridSpacing<T>> spacing_;
    std::shared_ptr<CenteredDifference<T>> stencil_;  // Shared ownership of templated facade
    PDEWorkspace* workspace_;  // Non-owning; workspace outlives operator
    // Fitted-diffusion cache validity (#472 perf follow-up): keyed on
    // (a, b, grid) -- NaN/nullptr so the first ensure_fitted_cache() call
    // always (correctly) misses. See ensure_fitted_cache() for why the
    // grid identity is part of the key.
    mutable T cached_a_ = std::numeric_limits<T>::quiet_NaN();
    mutable T cached_b_ = std::numeric_limits<T>::quiet_NaN();
    mutable const void* cached_grid_ = nullptr;
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
