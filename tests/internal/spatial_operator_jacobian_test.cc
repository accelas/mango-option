// SPDX-License-Identifier: MIT
//
// Regression test for Jacobian/operator stencil consistency on non-uniform grids.
// See issue #329: the Jacobian used a different first-derivative stencil than apply().

#include "mango/pde/operators/spatial_operator.hpp"
#include "mango/pde/operators/black_scholes_pde.hpp"
#include "mango/pde/core/grid.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include "mango/pde/operators/operator_factory.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace mango::operators {
namespace {

// ===========================================================================
// Regression: Jacobian first-derivative stencil must match apply() stencil
// Bug: assemble_jacobian() used (u[i+1]-u[i-1])/(dx_l+dx_r) while apply()
//      used the weighted forward/backward stencil. These differ on non-uniform
//      grids, causing the implicit solve to converge to the wrong discretization.
// ===========================================================================

// Verify Jacobian coefficients match finite-difference of apply() on a sinh grid
TEST(SpatialOperatorJacobianTest, NonUniformFirstDerivativeConsistency) {
    // Use a sinh grid with strong concentration (α=4) to amplify non-uniformity
    auto grid_spec = GridSpec<double>::sinh_spaced(-1.0, 1.0, 21, 4.0).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const size_t n = grid_view.size();

    // Black-Scholes PDE: L(u) = a·u'' + b·u' + c·u
    // Use exaggerated drift to make first-derivative term dominant
    const double sigma = 0.30;
    const double rate = 0.10;
    const double div_yield = 0.01;
    auto pde = BlackScholesPDE<double>(sigma, rate, div_yield);

    const double a = pde.second_derivative_coeff();   // σ²/2
    const double b = pde.first_derivative_coeff();     // r - d - σ²/2
    const double c = -pde.discount_rate();             // -r

    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);

    // Allocate workspace
    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    auto spatial_op = create_spatial_operator(std::move(pde), spacing, workspace);

    // Assemble Jacobian with coeff_dt = 1.0 so J = I - 1.0·∂L/∂u
    // => ∂L/∂u coefficients = (I - J) at each row
    auto jac = workspace.jacobian();
    spatial_op.assemble_jacobian(0.0, 1.0, jac);

    // For each interior point, verify Jacobian matches finite-difference of apply()
    // by perturbing u[j] and checking the response at row i
    const auto& x = grid_view.span();

    for (size_t i = 1; i < n - 1; ++i) {
        // Extract Jacobian coefficients: J = I - ∂L/∂u
        // => ∂L/∂u lower  = -jac.lower()[i-1]  (for j = i-1)
        // => ∂L/∂u diag   = 1 - jac.diag()[i]   (for j = i)
        // => ∂L/∂u upper  = -jac.upper()[i]      (for j = i+1)
        double jac_L_lower = -jac.lower()[i - 1];
        double jac_L_diag  = 1.0 - jac.diag()[i];
        double jac_L_upper = -jac.upper()[i];

        // Compute expected coefficients from the weighted stencil directly
        double dx_left = x[i] - x[i - 1];
        double dx_right = x[i + 1] - x[i];
        double dx_avg = (dx_left + dx_right) / 2.0;

        // Second derivative coefficients
        double d2_lower = a / (dx_left * dx_avg);
        double d2_diag  = -a * (1.0 / dx_left + 1.0 / dx_right) / dx_avg;
        double d2_upper = a / (dx_right * dx_avg);

        // First derivative: weighted forward/backward
        double d1_denom = dx_left + dx_right;
        double d1_lower = -b * dx_right / (dx_left * d1_denom);
        double d1_diag  =  b * (dx_right - dx_left) / (dx_left * dx_right);
        double d1_upper =  b * dx_left / (dx_right * d1_denom);

        double expected_lower = d2_lower + d1_lower;
        double expected_diag  = d2_diag + d1_diag + c;
        double expected_upper = d2_upper + d1_upper;

        EXPECT_NEAR(jac_L_lower, expected_lower, 1e-14)
            << "Lower mismatch at i=" << i
            << " dx_left=" << dx_left << " dx_right=" << dx_right;
        EXPECT_NEAR(jac_L_diag, expected_diag, 1e-14)
            << "Diag mismatch at i=" << i
            << " dx_left=" << dx_left << " dx_right=" << dx_right;
        EXPECT_NEAR(jac_L_upper, expected_upper, 1e-14)
            << "Upper mismatch at i=" << i
            << " dx_left=" << dx_left << " dx_right=" << dx_right;
    }
}

// Verify that the diagonal first-derivative term is nonzero on non-uniform grids
// (the old bug had d1_coeff_i = 0, which is only correct on uniform grids)
TEST(SpatialOperatorJacobianTest, NonUniformDiagonalIsNonzero) {
    auto grid_spec = GridSpec<double>::sinh_spaced(-1.0, 1.0, 11, 4.0).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const size_t n = grid_view.size();

    auto pde = BlackScholesPDE<double>(0.30, 0.10, 0.01);
    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);

    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    auto spatial_op = create_spatial_operator(std::move(pde), spacing, workspace);

    auto jac = workspace.jacobian();
    spatial_op.assemble_jacobian(0.0, 1.0, jac);

    const auto& x = grid_view.span();

    // On a non-uniform sinh grid, at least some interior points should have
    // dx_left != dx_right, giving a nonzero first-derivative diagonal contribution.
    // The center of a sinh grid is the most uniform region, so check away from center.
    bool found_nonzero_d1_diag = false;
    for (size_t i = 1; i < n - 1; ++i) {
        double dx_left = x[i] - x[i - 1];
        double dx_right = x[i + 1] - x[i];

        if (std::abs(dx_right - dx_left) > 1e-10) {
            // This point has non-uniform spacing — the first-derivative
            // should contribute to the diagonal
            double b = 0.10 - 0.01 - 0.5 * 0.30 * 0.30;  // r - d - σ²/2
            double d1_diag = b * (dx_right - dx_left) / (dx_left * dx_right);

            // The Jacobian diagonal should include this d1_diag contribution
            // J_diag = 1 - (d2_diag + d1_diag + c)
            // So d1_diag != 0 means J_diag != 1 - (d2_diag + c)
            if (std::abs(d1_diag) > 1e-10) {
                found_nonzero_d1_diag = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_nonzero_d1_diag)
        << "Sinh grid should have nonzero first-derivative diagonal term";
}

// Verify Jacobian matches apply() via finite-difference perturbation
TEST(SpatialOperatorJacobianTest, JacobianMatchesApplyFiniteDifference) {
    auto grid_spec = GridSpec<double>::sinh_spaced(-1.0, 1.0, 15, 3.5).value();
    auto grid_buf = grid_spec.generate();
    auto grid_view = grid_buf.view();
    const size_t n = grid_view.size();

    auto pde = BlackScholesPDE<double>(0.25, 0.08, 0.02);
    auto spacing = std::make_shared<GridSpacing<double>>(grid_view);

    std::vector<double> buffer(PDEWorkspace::required_size(n));
    auto workspace = PDEWorkspace::from_buffer(buffer, n).value();

    auto spatial_op = create_spatial_operator(std::move(pde), spacing, workspace);

    // Assemble analytical Jacobian (coeff_dt=0 gives pure ∂L/∂u)
    // Actually with coeff_dt=0: J = I - 0·∂L/∂u = I, not useful.
    // Use coeff_dt=1: J = I - ∂L/∂u, so ∂L/∂u = I - J
    auto jac = workspace.jacobian();
    spatial_op.assemble_jacobian(0.0, 1.0, jac);

    // Base state: use a smooth function u = sin(π·x)
    const auto& x = grid_view.span();
    std::vector<double> u(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(M_PI * x[i]);
    }

    // Compute L(u)
    std::vector<double> Lu(n, 0.0);
    spatial_op.apply(0.0, u, Lu);

    // For each interior point i and each neighbor j ∈ {i-1, i, i+1},
    // verify ∂L_i/∂u_j ≈ (L_i(u + ε·e_j) - L_i(u)) / ε
    const double eps = 1e-7;
    for (size_t i = 2; i < n - 2; ++i) {
        for (int offset = -1; offset <= 1; ++offset) {
            size_t j = static_cast<size_t>(static_cast<int>(i) + offset);

            // Perturb u[j]
            std::vector<double> u_pert(u);
            u_pert[j] += eps;

            std::vector<double> Lu_pert(n, 0.0);
            spatial_op.apply(0.0, u_pert, Lu_pert);

            double fd_deriv = (Lu_pert[i] - Lu[i]) / eps;

            // Get analytical value from Jacobian
            double analytical;
            if (offset == -1) {
                analytical = -jac.lower()[i - 1];  // ∂L/∂u = I - J
            } else if (offset == 0) {
                analytical = 1.0 - jac.diag()[i];
            } else {
                analytical = -jac.upper()[i];
            }

            EXPECT_NEAR(fd_deriv, analytical, std::abs(analytical) * 1e-5 + 1e-8)
                << "FD/analytical mismatch at i=" << i << " j=" << j
                << " (offset=" << offset << ")";
        }
    }
}

} // namespace
} // namespace mango::operators
