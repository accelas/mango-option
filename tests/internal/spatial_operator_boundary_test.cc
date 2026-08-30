// SPDX-License-Identifier: MIT
//
// Tests for analytic ghost-eliminated Neumann boundary rows on
// SpatialOperator (issue #455). See docs/plans/
// 2026-08-30-boundary-correctness-439-455-design.md section C1-C2 for the
// closed forms:
//
//   Left:  L_0     = (2a/h^2)*(u_1 - u_0)         + c*u_0     + g*(b - 2a/h)
//   Right: L_{n-1} = (2a/h^2)*(u_{n-2} - u_{n-1})  + c*u_{n-1} + g*(b + 2a/h)
//
// where a = second_derivative_coeff(), b = first_derivative_coeff(t),
// c = -discount_rate(t), h = adjacent interior spacing, g = Neumann
// gradient. The closed-form expectations below are hand-coded directly
// from this formula (not derived by calling boundary_row_jacobian /
// boundary_row_affine), so they are independent of the implementation's
// own jac-plus-affine decomposition.

#include "mango/pde/internal/spatial_operator.hpp"
#include "mango/pde/internal/operator_factory.hpp"
#include "mango/pde/internal/pde_workspace.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/pde/operators/laplacian_pde.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/boundary_conditions.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace mango::operators {
namespace {

// ===========================================================================
// TEST 1: Left boundary closed form, Black-Scholes coefficients
// ===========================================================================
TEST(BoundaryRow, LeftClosedFormBlackScholes) {
    const double sigma = 0.30;
    const double r = 0.05;
    const double q = 0.02;
    const double g = 1.7;
    const double t = 0.0;

    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 11).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const auto& x = grid_view.span();
    const size_t n = grid_view.size();

    auto pde = BlackScholesPDE<double>(sigma, r, q);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto op = create_spatial_operator(std::move(pde), grid_view, workspace);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = 1.0 + 0.3 * static_cast<double>(i) - 0.05 * static_cast<double>(i * i);
    }

    const double actual = op.eval_boundary_row(t, bc::BoundarySide::Left, g, u);

    // Independently hand-coded expectation (not via boundary_row_jacobian/affine).
    const double a = 0.5 * sigma * sigma;
    const double b = r - q - a;
    const double c = -r;
    const double h = x[1] - x[0];
    const double expected = (2.0 * a / (h * h)) * (u[1] - u[0]) + c * u[0]
                           + g * (b - 2.0 * a / h);

    EXPECT_NEAR(actual, expected, 1e-12);
}

// ===========================================================================
// TEST 2: Right boundary closed form, Black-Scholes coefficients
// ===========================================================================
TEST(BoundaryRow, RightClosedFormBlackScholes) {
    const double sigma = 0.30;
    const double r = 0.05;
    const double q = 0.02;
    const double g = 1.7;
    const double t = 0.0;

    auto grid_spec = GridSpec<double>::uniform(-1.0, 1.0, 11).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const auto& x = grid_view.span();
    const size_t n = grid_view.size();

    auto pde = BlackScholesPDE<double>(sigma, r, q);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto op = create_spatial_operator(std::move(pde), grid_view, workspace);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = 2.0 - 0.4 * static_cast<double>(i) + 0.02 * static_cast<double>(i * i);
    }

    const double actual = op.eval_boundary_row(t, bc::BoundarySide::Right, g, u);

    const double a = 0.5 * sigma * sigma;
    const double b = r - q - a;
    const double c = -r;
    const double h = x[n - 1] - x[n - 2];
    const double expected = (2.0 * a / (h * h)) * (u[n - 2] - u[n - 1]) + c * u[n - 1]
                           + g * (b + 2.0 * a / h);

    EXPECT_NEAR(actual, expected, 1e-12);
}

// ===========================================================================
// TEST 3: eval_boundary_row must equal jac.diag*u[node] + jac.offdiag*u[neighbor]
//         + affine by construction, on both sides, for "random" u.
// ===========================================================================
TEST(BoundaryRow, EvalEqualsJacobianDotUPlusAffine) {
    const double sigma = 0.22;
    const double r = 0.04;
    const double q = 0.015;
    const double g = -0.9;
    const double t = 0.37;

    auto grid_spec = GridSpec<double>::uniform(0.5, 2.5, 9).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const size_t n = grid_view.size();

    auto pde = BlackScholesPDE<double>(sigma, r, q);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto op = create_spatial_operator(std::move(pde), grid_view, workspace);

    // Deterministic pseudo-random-looking values (no <random> needed).
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(1.7 * static_cast<double>(i) + 0.4)
             + 0.3 * std::cos(2.3 * static_cast<double>(i));
    }

    for (auto side : {bc::BoundarySide::Left, bc::BoundarySide::Right}) {
        const double eval = op.eval_boundary_row(t, side, g, u);
        const auto jac = op.boundary_row_jacobian(t, side);
        const double affine = op.boundary_row_affine(t, side, g);

        const size_t node = (side == bc::BoundarySide::Left) ? 0 : n - 1;
        const size_t neighbor = (side == bc::BoundarySide::Left) ? 1 : n - 2;
        const double reconstructed = jac.diag * u[node] + jac.offdiag * u[neighbor] + affine;

        EXPECT_NEAR(eval, reconstructed, 1e-13)
            << "side=" << (side == bc::BoundarySide::Left ? "Left" : "Right");
    }
}

// ===========================================================================
// TEST 4: Nonuniform (geometric/log-spaced) grid uses the adjacent interior
//         spacing, not e.g. the average grid spacing.
// ===========================================================================
TEST(BoundaryRow, NonuniformGridUsesAdjacentSpacing) {
    const double sigma = 0.28;
    const double r = 0.06;
    const double q = 0.01;
    const double g = 1.1;
    const double t = 0.0;

    // Log-spaced (geometric) grid: adjacent spacings differ from each other
    // and from the naive (x_max - x_min) / (n - 1) average.
    auto grid_spec = GridSpec<double>::log_spaced(1.0, 100.0, 7).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const auto& x = grid_view.span();
    const size_t n = grid_view.size();

    // Sanity: the grid really is nonuniform, and the average spacing differs
    // measurably from either adjacent spacing so the test can distinguish.
    const double h_left = x[1] - x[0];
    const double h_right = x[n - 1] - x[n - 2];
    const double h_avg = (x[n - 1] - x[0]) / static_cast<double>(n - 1);
    ASSERT_GT(std::abs(h_left - h_avg), 1e-6 * h_avg);
    ASSERT_GT(std::abs(h_right - h_avg), 1e-6 * h_avg);

    auto pde = BlackScholesPDE<double>(sigma, r, q);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto op = create_spatial_operator(std::move(pde), grid_view, workspace);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = 0.5 + 0.1 * static_cast<double>(i);
    }

    const double a = 0.5 * sigma * sigma;
    const double b = r - q - a;
    const double c = -r;

    const double left_actual = op.eval_boundary_row(t, bc::BoundarySide::Left, g, u);
    const double left_expected = (2.0 * a / (h_left * h_left)) * (u[1] - u[0]) + c * u[0]
                                + g * (b - 2.0 * a / h_left);
    EXPECT_NEAR(left_actual, left_expected, 1e-11);

    const double right_actual = op.eval_boundary_row(t, bc::BoundarySide::Right, g, u);
    const double right_expected = (2.0 * a / (h_right * h_right)) * (u[n - 2] - u[n - 1])
                                 + c * u[n - 1] + g * (b + 2.0 * a / h_right);
    EXPECT_NEAR(right_actual, right_expected, 1e-11);
}

// ===========================================================================
// TEST 5: LaplacianPDE satisfies the (tightened) HasJacobianCoefficients
//         concept and produces the correct (b=0, c=0) boundary row.
// ===========================================================================
TEST(BoundaryRow, LaplacianSatisfiesConcept) {
    static_assert(HasJacobianCoefficients<LaplacianPDE<double>>,
                  "LaplacianPDE must satisfy the tightened HasJacobianCoefficients concept");

    const double D = 1.5;
    const double g = 0.8;
    const double t = 0.0;

    auto grid_spec = GridSpec<double>::uniform(0.0, 1.0, 6).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const auto& x = grid_view.span();
    const size_t n = grid_view.size();

    auto pde = LaplacianPDE<double>(D);
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();
    auto op = create_spatial_operator(std::move(pde), grid_view, workspace);

    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = static_cast<double>(i) * static_cast<double>(i);
    }

    const double h_left = x[1] - x[0];
    const double left_actual = op.eval_boundary_row(t, bc::BoundarySide::Left, g, u);
    const double left_expected = (2.0 * D / (h_left * h_left)) * (u[1] - u[0])
                                - g * (2.0 * D / h_left);
    EXPECT_NEAR(left_actual, left_expected, 1e-12);

    const double h_right = x[n - 1] - x[n - 2];
    const double right_actual = op.eval_boundary_row(t, bc::BoundarySide::Right, g, u);
    const double right_expected = (2.0 * D / (h_right * h_right)) * (u[n - 2] - u[n - 1])
                                 + g * (2.0 * D / h_right);
    EXPECT_NEAR(right_actual, right_expected, 1e-12);
}

} // namespace
} // namespace mango::operators
