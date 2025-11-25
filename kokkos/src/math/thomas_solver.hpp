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

        // For host execution, run serial Thomas algorithm
        // For device, this would be called per-system in a batched context

        auto lower_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lower);
        auto diag_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, diag);
        auto upper_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upper);
        auto rhs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rhs);
        auto solution_h = Kokkos::create_mirror_view(Kokkos::HostSpace{}, solution);

        // Workspace for modified coefficients
        std::vector<double> c_prime(n);
        std::vector<double> d_prime(n);

        // Forward elimination
        constexpr double tol = 1e-15;

        if (std::abs(diag_h(0)) < tol) {
            return std::unexpected(ThomasError::SingularMatrix);
        }

        c_prime[0] = upper_h(0) / diag_h(0);
        d_prime[0] = rhs_h(0) / diag_h(0);

        for (size_t i = 1; i < n; ++i) {
            double denom = diag_h(i) - lower_h(i - 1) * c_prime[i - 1];
            if (std::abs(denom) < tol) {
                return std::unexpected(ThomasError::SingularMatrix);
            }

            if (i < n - 1) {
                c_prime[i] = upper_h(i) / denom;
            }
            d_prime[i] = (rhs_h(i) - lower_h(i - 1) * d_prime[i - 1]) / denom;
        }

        // Back substitution
        solution_h(n - 1) = d_prime[n - 1];
        for (size_t i = n - 1; i > 0; --i) {
            solution_h(i - 1) = d_prime[i - 1] - c_prime[i - 1] * solution_h(i);
        }

        Kokkos::deep_copy(solution, solution_h);

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

}  // namespace mango::kokkos
