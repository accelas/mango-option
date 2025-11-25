#pragma once

#include <Kokkos_Core.hpp>
#include <expected>
#include <cmath>
#include <vector>
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Thomas solver error codes
enum class ThomasError {
    SingularMatrix,
    SizeMismatch
};

/// Thomas algorithm for tridiagonal systems
///
/// Solves Ax = d where A is tridiagonal.
/// Uses in-place forward elimination and back substitution.
template <typename MemSpace>
class ThomasSolver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Solve tridiagonal system
    ///
    /// @param lower Lower diagonal (size n-1)
    /// @param diag Main diagonal (size n)
    /// @param upper Upper diagonal (size n-1)
    /// @param rhs Right-hand side (size n)
    /// @param solution Output solution (size n)
    [[nodiscard]] std::expected<void, ThomasError>
    solve(view_type lower, view_type diag, view_type upper,
          view_type rhs, view_type solution) const {

        const size_t n = diag.extent(0);

        // Validate sizes
        if (lower.extent(0) != n - 1 || upper.extent(0) != n - 1 ||
            rhs.extent(0) != n || solution.extent(0) != n) {
            return std::unexpected(ThomasError::SizeMismatch);
        }

        // Optimized path for HostSpace: avoid mirror copies
        if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
            return solve_host_impl(lower, diag, upper, rhs, solution, n);
        } else {
            // For device memory, copy to host, solve, copy back
            auto lower_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lower);
            auto diag_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, diag);
            auto upper_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upper);
            auto rhs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rhs);
            auto solution_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, solution);

            auto result = solve_host_impl(lower_h, diag_h, upper_h, rhs_h, solution_h, n);
            if (result.has_value()) {
                Kokkos::deep_copy(solution, solution_h);
            }
            return result;
        }
    }

private:
    /// Host implementation of Thomas algorithm
    template <typename ViewType>
    [[nodiscard]] std::expected<void, ThomasError>
    solve_host_impl(ViewType lower, ViewType diag, ViewType upper,
                    ViewType rhs, ViewType solution, size_t n) const {

        // Workspace for modified coefficients
        std::vector<double> c_prime(n);
        std::vector<double> d_prime(n);

        // Forward elimination
        constexpr double tol = 1e-15;

        if (std::abs(diag(0)) < tol) {
            return std::unexpected(ThomasError::SingularMatrix);
        }

        c_prime[0] = upper(0) / diag(0);
        d_prime[0] = rhs(0) / diag(0);

        for (size_t i = 1; i < n; ++i) {
            double denom = diag(i) - lower(i - 1) * c_prime[i - 1];
            if (std::abs(denom) < tol) {
                return std::unexpected(ThomasError::SingularMatrix);
            }

            if (i < n - 1) {
                c_prime[i] = upper(i) / denom;
            }
            d_prime[i] = (rhs(i) - lower(i - 1) * d_prime[i - 1]) / denom;
        }

        // Back substitution
        solution(n - 1) = d_prime[n - 1];
        for (size_t i = n - 1; i > 0; --i) {
            solution(i - 1) = d_prime[i - 1] - c_prime[i - 1] * solution(i);
        }

        return {};
    }
};

/// Batched Thomas solver for GPU execution
///
/// Solves many independent tridiagonal systems in parallel.
template <typename MemSpace>
class BatchedThomasSolver {
public:
    using view_2d = Kokkos::View<double**, MemSpace>;

    /// Solve batch of tridiagonal systems
    ///
    /// Each row of the 2D views is one system.
    /// @param lower Lower diagonals [batch_size, n-1]
    /// @param diag Main diagonals [batch_size, n]
    /// @param upper Upper diagonals [batch_size, n-1]
    /// @param rhs Right-hand sides [batch_size, n]
    /// @param solutions Output solutions [batch_size, n]
    /// @param workspace Workspace for c_prime and d_prime [batch_size, 2*n]
    void solve_batch(view_2d lower, view_2d diag, view_2d upper,
                     view_2d rhs, view_2d solutions, view_2d workspace) const {

        const size_t batch_size = diag.extent(0);
        const size_t n = diag.extent(1);

        // Parallel over batch dimension
        Kokkos::parallel_for("thomas_batch", batch_size,
            KOKKOS_LAMBDA(const size_t batch) {
                // Thomas algorithm for this system
                constexpr double tol = 1e-15;

                // Use workspace for c_prime and d_prime
                // workspace layout: [c_prime[0..n-1], d_prime[0..n-1]]
                auto get_c = [&](size_t i) -> double& {
                    return workspace(batch, i);
                };
                auto get_d = [&](size_t i) -> double& {
                    return workspace(batch, n + i);
                };

                // Forward elimination
                double diag_val = diag(batch, 0);
                if (diag_val != 0.0) {
                    workspace(batch, 0) = upper(batch, 0) / diag_val;  // c_prime[0]
                    workspace(batch, n) = rhs(batch, 0) / diag_val;    // d_prime[0]

                    for (size_t i = 1; i < n; ++i) {
                        double denom = diag(batch, i) - lower(batch, i - 1) * workspace(batch, i - 1);
                        if (i < n - 1) {
                            workspace(batch, i) = upper(batch, i) / denom;  // c_prime[i]
                        }
                        workspace(batch, n + i) = (rhs(batch, i) - lower(batch, i - 1) * workspace(batch, n + i - 1)) / denom;  // d_prime[i]
                    }

                    // Back substitution
                    solutions(batch, n - 1) = workspace(batch, 2 * n - 1);  // d_prime[n-1]
                    for (size_t i = n - 1; i > 0; --i) {
                        solutions(batch, i - 1) = workspace(batch, n + i - 1) - workspace(batch, i - 1) * solutions(batch, i);
                    }
                }
            });

        Kokkos::fence();
    }
};

/// Projected Thomas solver for American options (Brennan-Schwartz algorithm)
///
/// Solves the Linear Complementarity Problem:
///   A·x = d, subject to x ≥ ψ (obstacle constraint)
///
/// Key difference from standard Thomas: enforces x ≥ ψ DURING backward
/// substitution, not after. This is critical for correct American option pricing.
template <typename MemSpace>
class ProjectedThomasSolver {
public:
    using view_type = Kokkos::View<double*, MemSpace>;

    /// Solve tridiagonal system with obstacle constraint
    ///
    /// @param lower Lower diagonal (size n-1)
    /// @param diag Main diagonal (size n)
    /// @param upper Upper diagonal (size n-1)
    /// @param rhs Right-hand side (size n)
    /// @param psi Obstacle (lower bound) (size n)
    /// @param solution Output solution (size n), will satisfy x ≥ ψ
    [[nodiscard]] std::expected<void, ThomasError>
    solve(view_type lower, view_type diag, view_type upper,
          view_type rhs, view_type psi, view_type solution) const {

        const size_t n = diag.extent(0);

        // Validate sizes
        if (lower.extent(0) != n - 1 || upper.extent(0) != n - 1 ||
            rhs.extent(0) != n || psi.extent(0) != n || solution.extent(0) != n) {
            return std::unexpected(ThomasError::SizeMismatch);
        }

        // Optimized path for HostSpace: avoid mirror copies
        if constexpr (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
            return solve_host_impl(lower, diag, upper, rhs, psi, solution, n);
        } else {
            // For device memory, copy to host, solve, copy back
            auto lower_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lower);
            auto diag_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, diag);
            auto upper_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upper);
            auto rhs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rhs);
            auto psi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, psi);
            auto solution_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, solution);

            auto result = solve_host_impl(lower_h, diag_h, upper_h, rhs_h, psi_h, solution_h, n);
            if (result.has_value()) {
                Kokkos::deep_copy(solution, solution_h);
            }
            return result;
        }
    }

private:
    /// Host implementation of projected Thomas algorithm
    template <typename ViewType>
    [[nodiscard]] std::expected<void, ThomasError>
    solve_host_impl(ViewType lower, ViewType diag, ViewType upper,
                    ViewType rhs, ViewType psi, ViewType solution, size_t n) const {

        // Workspace for modified coefficients
        std::vector<double> c_prime(n);
        std::vector<double> d_prime(n);

        // ========== Forward Elimination (identical to standard Thomas) ==========
        constexpr double tol = 1e-15;

        if (std::abs(diag(0)) < tol) {
            return std::unexpected(ThomasError::SingularMatrix);
        }

        c_prime[0] = upper(0) / diag(0);
        d_prime[0] = rhs(0) / diag(0);

        for (size_t i = 1; i < n; ++i) {
            double denom = diag(i) - lower(i - 1) * c_prime[i - 1];
            if (std::abs(denom) < tol) {
                return std::unexpected(ThomasError::SingularMatrix);
            }

            if (i < n - 1) {
                c_prime[i] = upper(i) / denom;
            }
            d_prime[i] = (rhs(i) - lower(i - 1) * d_prime[i - 1]) / denom;
        }

        // ========== Projected Back Substitution (KEY DIFFERENCE) ==========
        // Apply obstacle constraint x ≥ ψ at EACH step

        // Last element with projection
        solution(n - 1) = std::max(d_prime[n - 1], psi(n - 1));

        // Backward iteration with projection at each step
        for (size_t i = n - 1; i > 0; --i) {
            double unconstrained = d_prime[i - 1] - c_prime[i - 1] * solution(i);
            solution(i - 1) = std::max(unconstrained, psi(i - 1));
        }

        return {};
    }
};

}  // namespace mango::kokkos
