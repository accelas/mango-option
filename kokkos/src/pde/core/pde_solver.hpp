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
    double gamma = 2.0 - std::sqrt(2.0);  // L-stable choice (~0.586)
    size_t max_newton_iter = 10;
    double newton_tol = 1e-10;

    /// Stage 1 implicit weight for trapezoidal: w1 = gamma * dt / 2
    double stage1_implicit_weight(double dt) const {
        return gamma * dt / 2.0;
    }

    /// Stage 2 implicit weight: (1-gamma)/(2-gamma) * dt
    double stage2_implicit_weight(double dt) const {
        return (1.0 - gamma) / (2.0 - gamma) * dt;
    }

    /// BDF2 combination coefficients
    /// u^{n+1} = alpha * u^{n+gamma} + beta * u^n
    double alpha() const {
        return 1.0 / (gamma * (2.0 - gamma));  // ~1.207
    }

    double beta() const {
        return -(1.0 - gamma) * (1.0 - gamma) / (gamma * (2.0 - gamma));  // ~-0.207
    }
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

/// TR-BDF2 solver infrastructure
///
/// Base infrastructure for TR-BDF2 time stepping with obstacle projection.
/// Derived solvers provide specific spatial operators and boundary conditions.
///
/// TR-BDF2 is a two-stage method:
/// - Stage 1 (Trapezoidal): u^{n+γ} = u^n + (γ*dt/2)*(L(u^n) + L(u^{n+γ}))
/// - Stage 2 (BDF2): u^{n+1} = α*u^{n+γ} + β*u^n + w2*dt*L(u^{n+1})
///
/// Where:
/// - γ = 2 - √2 ≈ 0.586 (L-stable choice)
/// - α = 1/(γ*(2-γ)) ≈ 1.207
/// - β = -(1-γ)²/(γ*(2-γ)) ≈ -0.207
/// - w2 = (1-γ)/(2-γ) ≈ 0.293
///
/// Stability: A-stable (both stages unconditionally stable)
/// Accuracy: O(dt²) + O(dx²)
template <typename MemSpace>
class TRBDF2Solver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    TRBDF2Solver(Grid<MemSpace>& grid, PDEWorkspace<MemSpace>& workspace,
                 TRBDF2Config config = TRBDF2Config{})
        : grid_(grid), workspace_(workspace), config_(config), n_(grid.n_points()) {}

    /// Solve from t=0 to t=T with n_steps time steps
    ///
    /// @param T Final time
    /// @param n_steps Number of time steps
    /// @param assemble_jacobian Callback to assemble Jacobian: (dt, lower, diag, upper) -> void
    /// @param apply_operator Callback to apply spatial operator L(u): (u, Lu) -> void
    /// @param apply_boundary_to_matrix Callback to apply BCs to matrix: (lower, diag, upper) -> void
    /// @param apply_boundary_to_rhs Callback to apply BCs to RHS: (rhs, t, dt, step) -> void
    /// @param compute_obstacle Callback to compute obstacle values: (x, psi) -> void
    template <typename AssembleJacobian, typename ApplyOperator, typename ApplyBCMatrix,
              typename ApplyBCRHS, typename ComputeObstacle = std::nullptr_t>
    void solve(double T, size_t n_steps,
               AssembleJacobian&& assemble_jacobian,
               ApplyOperator&& apply_operator,
               ApplyBCMatrix&& apply_boundary_to_matrix,
               ApplyBCRHS&& apply_boundary_to_rhs,
               ComputeObstacle&& compute_obstacle = nullptr) {
        double dt = T / static_cast<double>(n_steps);

        auto u_current = grid_.u_current();
        auto u_prev = grid_.u_prev();
        auto u_stage = workspace_.u_stage();
        auto rhs = workspace_.rhs();
        auto lu = workspace_.lu();  // Buffer for L(u)
        auto psi = workspace_.psi();  // Buffer for obstacle values
        auto solution = workspace_.delta_u();

        auto lower = workspace_.jacobian_lower();
        auto diag = workspace_.jacobian_diag();
        auto upper = workspace_.jacobian_upper();

        auto lower_temp = workspace_.thomas_lower_temp();
        auto diag_temp = workspace_.thomas_diag_temp();
        auto upper_temp = workspace_.thomas_upper_temp();

        // Use projected Thomas for obstacle problems, regular Thomas otherwise
        constexpr bool has_obstacle = !std::is_same_v<ComputeObstacle, std::nullptr_t>;
        ThomasSolver<MemSpace> thomas;
        ProjectedThomasSolver<MemSpace> projected_thomas;

        // Time stepping
        for (size_t step = 0; step < n_steps; ++step) {
            double t = static_cast<double>(step) * dt;

            // Copy u_current to u_prev
            Kokkos::deep_copy(u_prev, u_current);

            // ===================================================================
            // Stage 1: Trapezoidal rule to t + γ*dt
            // ===================================================================
            // Equation: (I - w1*L)*u^{n+γ} = u^n + w1*L(u^n)
            // where w1 = γ*dt/2
            double w1 = config_.stage1_implicit_weight(dt);
            double t_stage1 = t + config_.gamma * dt;

            // Compute L(u^n)
            apply_operator(u_prev, lu);

            // RHS = u^n + w1*L(u^n)  (Full trapezoidal rule)
            const size_t n = n_;
            Kokkos::parallel_for("compute_tr_rhs", n,
                KOKKOS_LAMBDA(const size_t i) {
                    rhs(i) = u_prev(i) + w1 * lu(i);
                });
            Kokkos::fence();

            // Assemble Jacobian: (I - w1*L)
            assemble_jacobian(w1, lower, diag, upper);
            apply_boundary_to_matrix(lower, diag, upper);
            apply_boundary_to_rhs(rhs, t_stage1, dt, step);

            // Solve stage 1 with or without obstacle
            Kokkos::deep_copy(lower_temp, lower);
            Kokkos::deep_copy(diag_temp, diag);
            Kokkos::deep_copy(upper_temp, upper);

            if constexpr (has_obstacle) {
                // Compute obstacle at current time
                compute_obstacle(grid_.x(), psi);
                // Use Projected Thomas (enforces u ≥ ψ during backward substitution)
                auto result1 = projected_thomas.solve(lower_temp, diag_temp, upper_temp, rhs, psi, solution);
                (void)result1;
            } else {
                auto result1 = thomas.solve(lower_temp, diag_temp, upper_temp, rhs, solution);
                (void)result1;
            }

            // Store stage 1 result
            Kokkos::deep_copy(u_stage, solution);

            // ===================================================================
            // Stage 2: BDF2 from t to t + dt
            // ===================================================================
            // Equation: (I - w2*L)*u^{n+1} = α*u^{n+γ} + β*u^n
            // where:
            //   w2 = (1-γ)*dt/(2-γ)
            //   α = 1/(γ*(2-γ))
            //   β = -(1-γ)²/(γ*(2-γ))
            double w2 = config_.stage2_implicit_weight(dt);
            double t_next = t + dt;

            double gamma = config_.gamma;
            double alpha = 1.0 / (gamma * (2.0 - gamma));
            double beta = -(1.0 - gamma) * (1.0 - gamma) / (gamma * (2.0 - gamma));

            // Assemble Jacobian: (I - w2*L)
            assemble_jacobian(w2, lower, diag, upper);
            apply_boundary_to_matrix(lower, diag, upper);

            // RHS = α*u^{n+γ} + β*u^n
            Kokkos::parallel_for("compute_bdf2_rhs", n,
                KOKKOS_LAMBDA(const size_t i) {
                    rhs(i) = alpha * u_stage(i) + beta * u_prev(i);
                });
            Kokkos::fence();

            apply_boundary_to_rhs(rhs, t_next, dt, step);

            // Solve stage 2 with or without obstacle
            Kokkos::deep_copy(lower_temp, lower);
            Kokkos::deep_copy(diag_temp, diag);
            Kokkos::deep_copy(upper_temp, upper);

            if constexpr (has_obstacle) {
                // Compute obstacle at next time (same as current since obstacle is time-independent)
                compute_obstacle(grid_.x(), psi);
                // Use Projected Thomas (enforces u ≥ ψ during backward substitution)
                auto result2 = projected_thomas.solve(lower_temp, diag_temp, upper_temp, rhs, psi, solution);
                (void)result2;
            } else {
                auto result2 = thomas.solve(lower_temp, diag_temp, upper_temp, rhs, solution);
                (void)result2;
            }

            // Update current solution
            Kokkos::deep_copy(u_current, solution);
        }
    }

protected:
    Grid<MemSpace>& grid_;
    PDEWorkspace<MemSpace>& workspace_;
    TRBDF2Config config_;
    size_t n_;
};

}  // namespace mango::kokkos
