#pragma once

#include <concepts>
#include <cmath>
#include <cstddef>
#include <span>

namespace mango {

/// Fixed-point iteration solver with under-relaxation
///
/// Solves: x = G(x) using iteration x_{k+1} = x_k + ω·(G(x_k) - x_k)
/// where ω is the under-relaxation parameter (0 < ω ≤ 1).
///
/// @tparam T Value type (e.g., double)
/// @tparam Func Callable that computes G(x)
/// @param x Initial guess (updated to solution on success)
/// @param iterate Function G: x → G(x)
/// @param max_iter Maximum number of iterations
/// @param tolerance Convergence tolerance (relative error)
/// @param omega Under-relaxation parameter (0 < ω ≤ 1)
/// @param iterations_taken Output: number of iterations performed
/// @return true if converged, false if max_iter reached
template<typename T, std::invocable<T> Func>
    requires std::convertible_to<std::invoke_result_t<Func, T>, T>
bool fixed_point_solve(
    T& x,
    Func&& iterate,
    size_t max_iter,
    double tolerance,
    double omega,
    size_t& iterations_taken)
{
    iterations_taken = 0;

    for (size_t k = 0; k < max_iter; ++k) {
        iterations_taken = k + 1;

        // Compute next iterate: G(x)
        T x_next = iterate(x);

        // Under-relaxation: x_{k+1} = x_k + ω·(G(x_k) - x_k)
        T x_new = x + omega * (x_next - x);

        // Check convergence (relative error)
        T error = std::abs(x_new - x);
        T scale = std::max(std::abs(x_new), T{1.0});

        if (error / scale < tolerance) {
            x = x_new;
            return true;  // Converged
        }

        x = x_new;
    }

    return false;  // Failed to converge
}

/// Vectorized fixed-point solver for PDE time stepping
///
/// Solves: u = G(u) element-wise using fixed-point iteration with
/// under-relaxation. Used for implicit stages in TR-BDF2.
///
/// @param u Initial guess (updated in-place to solution)
/// @param iterate Function that computes G(u) and stores result in output buffer
/// @param temp Temporary buffer (same size as u)
/// @param max_iter Maximum iterations
/// @param tolerance Convergence tolerance
/// @param omega Under-relaxation parameter
/// @param iterations_taken Output: number of iterations performed
/// @return true if converged, false if max_iter reached
template<typename Func>
bool fixed_point_solve_vector(
    std::span<double> u,
    Func&& iterate,
    std::span<double> temp,
    size_t max_iter,
    double tolerance,
    double omega,
    size_t& iterations_taken)
{
    const size_t n = u.size();
    iterations_taken = 0;

    for (size_t k = 0; k < max_iter; ++k) {
        iterations_taken = k + 1;

        // Compute G(u) → temp
        iterate(u, temp);

        // Under-relaxation and convergence check
        double max_error = 0.0;
        double max_scale = 1.0;

        for (size_t i = 0; i < n; ++i) {
            // x_{k+1} = x_k + ω·(G(x_k) - x_k)
            double u_new = u[i] + omega * (temp[i] - u[i]);

            double error = std::abs(u_new - u[i]);
            double scale = std::max(std::abs(u_new), 1.0);

            max_error = std::max(max_error, error);
            max_scale = std::max(max_scale, scale);

            u[i] = u_new;
        }

        // Check convergence (max relative error)
        if (max_error / max_scale < tolerance) {
            return true;  // Converged
        }
    }

    return false;  // Failed to converge
}

}  // namespace mango
