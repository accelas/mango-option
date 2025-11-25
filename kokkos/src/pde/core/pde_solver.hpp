#pragma once

/// @file pde_solver.hpp
/// @brief PDE solver base infrastructure for Kokkos
///
/// This file provides foundation infrastructure for PDE solving with Kokkos.
/// The HeatEquationSolver is a TEST SOLVER for validating infrastructure
/// before implementing the full Black-Scholes American option solver.

#include <Kokkos_Core.hpp>
#include <cmath>
#include "kokkos/src/pde/core/grid.hpp"
#include "kokkos/src/pde/core/workspace.hpp"
#include "kokkos/src/math/thomas_solver.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// TR-BDF2 configuration
struct TRBDF2Config {
    double gamma = 2.0 - std::sqrt(2.0);  // L-stable choice
    size_t max_newton_iter = 10;
    double newton_tol = 1e-10;
};

/// Simple heat equation solver (for testing PDE infrastructure)
///
/// Solves u_t = u_xx with Dirichlet BCs u(0,t)=u(1,t)=0.
/// Uses implicit Euler (unconditionally stable, O(dt) accuracy).
///
/// @note This is a TEST SOLVER for validating Kokkos infrastructure.
///       Production solver (Task 8) will use TR-BDF2 with Newton iteration.
///
/// Stability: Unconditionally stable (implicit diffusion)
/// Accuracy: O(dt) + O(dx^2)
/// Memory: O(n) with n = grid points
template <typename MemSpace>
class HeatEquationSolver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    HeatEquationSolver(Grid<MemSpace>& grid, PDEWorkspace<MemSpace>& workspace)
        : grid_(grid), workspace_(workspace), n_(grid.n_points()) {}

    void solve(double T, size_t n_steps) {
        double dt = T / static_cast<double>(n_steps);
        double dx = (grid_.x_max() - grid_.x_min()) / static_cast<double>(n_ - 1);
        double r = dt / (dx * dx);  // Fourier number

        auto u = grid_.u_current();
        auto lower = workspace_.jacobian_lower();
        auto diag = workspace_.jacobian_diag();
        auto upper = workspace_.jacobian_upper();
        auto rhs = workspace_.rhs();
        auto solution = workspace_.delta_u();  // Reuse buffer

        // Pre-allocated temporary buffers for Thomas solver
        auto lower_temp = workspace_.thomas_lower_temp();
        auto diag_temp = workspace_.thomas_diag_temp();
        auto upper_temp = workspace_.thomas_upper_temp();

        // Assemble tridiagonal system for implicit Euler with BCs
        // (I - dt*L)u^{n+1} = u^n where L = d^2/dx^2
        // Interior: -r*u_{i-1} + (1+2r)*u_i - r*u_{i+1} = u_i^n
        // Boundaries: u_0 = 0, u_{n-1} = 0 (Dirichlet)
        const size_t n = n_;
        Kokkos::parallel_for("assemble_heat_with_bc", n,
            KOKKOS_LAMBDA(const size_t i) {
                if (i == 0) {
                    // Left boundary: u_0 = 0
                    diag(i) = 1.0;
                    upper(i) = 0.0;
                } else if (i == n - 1) {
                    // Right boundary: u_{n-1} = 0
                    diag(i) = 1.0;
                    lower(i - 1) = 0.0;
                } else {
                    // Interior points
                    diag(i) = 1.0 + 2.0 * r;
                    lower(i - 1) = -r;
                    upper(i) = -r;
                }
            });
        Kokkos::fence();

        ThomasSolver<MemSpace> thomas;

        // Time stepping
        for (size_t step = 0; step < n_steps; ++step) {
            // Set up RHS = u^n with boundary conditions
            Kokkos::deep_copy(rhs, u);

            // Apply Dirichlet BCs to RHS (u=0 at boundaries)
            // Use parallel_for instead of host mirror for GPU efficiency
            Kokkos::parallel_for("apply_bc_rhs", n,
                KOKKOS_LAMBDA(const size_t i) {
                    if (i == 0 || i == n - 1) {
                        rhs(i) = 0.0;
                    }
                });
            Kokkos::fence();

            // Copy matrix to temporary buffers (Thomas solver modifies in-place)
            Kokkos::deep_copy(lower_temp, lower);
            Kokkos::deep_copy(diag_temp, diag);
            Kokkos::deep_copy(upper_temp, upper);

            // Solve tridiagonal system
            auto result = thomas.solve(lower_temp, diag_temp, upper_temp, rhs, solution);

            // Copy solution back to u
            Kokkos::deep_copy(u, solution);
        }
    }

private:
    Grid<MemSpace>& grid_;
    PDEWorkspace<MemSpace>& workspace_;
    size_t n_;
};

}  // namespace mango::kokkos
