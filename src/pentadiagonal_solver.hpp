/**
 * @file pentadiagonal_solver.hpp
 * @brief Pentadiagonal linear system solver for B-spline fitting
 *
 * Extends ThomasSolver pattern to width-5 band systems.
 * Optimized for symmetric positive definite systems arising from
 * cubic B-spline least-squares fitting.
 *
 * System form (5 diagonals):
 *   e[i]: 2nd subdiagonal (i >= 2)
 *   a[i]: 1st subdiagonal (i >= 1)
 *   b[i]: main diagonal
 *   c[i]: 1st superdiagonal (i < n-1)
 *   d[i]: 2nd superdiagonal (i < n-2)
 *
 * Algorithm: Gaussian elimination without pivoting
 * Complexity: O(n)
 * Workspace: O(n)
 *
 * References:
 * - Extension of Thomas algorithm to pentadiagonal case
 * - de Boor, "A Practical Guide to Splines" (2001), Appendix
 */

#pragma once

#include "solver_common.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <optional>
#include <string_view>
#include <algorithm>

namespace mango {

/// Result type for pentadiagonal solver
template<FloatingPoint T>
struct PentadiagonalResult {
    bool success;
    std::optional<std::string_view> error;

    constexpr explicit operator bool() const noexcept { return success; }
    [[nodiscard]] constexpr bool ok() const noexcept { return success; }
    [[nodiscard]] constexpr std::string_view message() const noexcept {
        return error.value_or("");
    }

    [[nodiscard]] static constexpr PentadiagonalResult ok_result() noexcept {
        return PentadiagonalResult{.success = true, .error = std::nullopt};
    }

    [[nodiscard]] static constexpr PentadiagonalResult error_result(std::string_view msg) noexcept {
        return PentadiagonalResult{.success = false, .error = msg};
    }
};

/// Configuration for pentadiagonal solver
template<FloatingPoint T>
struct PentadiagonalConfig {
    T singularity_tol = static_cast<T>(1e-15);
    bool check_diagonal_dominance = false;
};

/// Pentadiagonal System Solver
///
/// Solves Ax = rhs where A has width-5 band structure.
///
/// Matrix structure (n=6 example):
///   [b0 c0 d0  0  0  0]
///   [a0 b1 c1 d1  0  0]
///   [e0 a1 b2 c2 d2  0]
///   [ 0 e1 a2 b3 c3 d3]
///   [ 0  0 e2 a3 b4 c4]
///   [ 0  0  0 e3 a4 b5]
///
/// Array sizes:
///   e: length n-2 (2nd subdiagonal)
///   a: length n-1 (1st subdiagonal)
///   b: length n   (main diagonal)
///   c: length n-1 (1st superdiagonal)
///   d: length n-2 (2nd superdiagonal)
///
/// @tparam T Floating point type
template<FloatingPoint T>
[[nodiscard]] constexpr PentadiagonalResult<T> solve_pentadiagonal(
    std::span<const T> e,     // 2nd subdiagonal (length n-2)
    std::span<const T> a,     // 1st subdiagonal (length n-1)
    std::span<const T> b,     // Main diagonal (length n)
    std::span<const T> c,     // 1st superdiagonal (length n-1)
    std::span<const T> d,     // 2nd superdiagonal (length n-2)
    std::span<const T> rhs,   // Right-hand side (length n)
    std::span<T> solution,    // Output solution (length n)
    std::span<T> workspace,   // Workspace (length 5n)
    const PentadiagonalConfig<T>& config = {}) noexcept
{
    using Result = PentadiagonalResult<T>;

    const size_t n = b.size();

    // Validate dimensions
    if (e.size() != n - 2 && n > 2) {
        return Result::error_result("2nd subdiagonal size must be n-2");
    }
    if (a.size() != n - 1 && n > 1) {
        return Result::error_result("1st subdiagonal size must be n-1");
    }
    if (c.size() != n - 1 && n > 1) {
        return Result::error_result("1st superdiagonal size must be n-1");
    }
    if (d.size() != n - 2 && n > 2) {
        return Result::error_result("2nd superdiagonal size must be n-2");
    }
    if (rhs.size() != n) {
        return Result::error_result("RHS size must be n");
    }
    if (solution.size() != n) {
        return Result::error_result("Solution size must be n");
    }
    if (workspace.size() < 5 * n) {
        return Result::error_result("Workspace size must be at least 5n");
    }

    // Handle trivial cases
    if (n == 0) {
        return Result::ok_result();
    }

    if (n == 1) {
        if (std::abs(b[0]) < config.singularity_tol) {
            return Result::error_result("Singular matrix");
        }
        solution[0] = rhs[0] / b[0];
        return Result::ok_result();
    }

    // Split workspace into modified diagonals
    std::span<T> alpha = workspace.subspan(0, n);      // Modified main diagonal
    std::span<T> beta = workspace.subspan(n, n);       // Modified 1st super
    std::span<T> gamma = workspace.subspan(2*n, n);    // Modified 2nd super
    std::span<T> mu = workspace.subspan(3*n, n);       // Modified 1st sub
    std::span<T> z = workspace.subspan(4*n, n);        // Modified RHS

    // Optional: Check diagonal dominance
    if (config.check_diagonal_dominance) {
        for (size_t i = 0; i < n; ++i) {
            T diag_abs = std::abs(b[i]);
            T off_diag = T{0};

            if (i >= 2) off_diag += std::abs(e[i-2]);
            if (i >= 1) off_diag += std::abs(a[i-1]);
            if (i < n - 1) off_diag += std::abs(c[i]);
            if (i < n - 2) off_diag += std::abs(d[i]);

            if (diag_abs <= off_diag) {
                return Result::error_result("Matrix not diagonally dominant");
            }
        }
    }

    // ========== Forward Elimination ==========

    // Initialize first row
    if (std::abs(b[0]) < config.singularity_tol) {
        return Result::error_result("Singular matrix at row 0");
    }

    alpha[0] = b[0];
    if (n > 1) beta[0] = c[0];
    if (n > 2) gamma[0] = d[0];
    z[0] = rhs[0];

    // Second row (special case - only one subdiagonal)
    if (n > 1) {
        mu[0] = a[0] / alpha[0];
        alpha[1] = b[1] - mu[0] * beta[0];

        if (std::abs(alpha[1]) < config.singularity_tol) {
            return Result::error_result("Singular matrix at row 1");
        }

        if (n > 2) {
            beta[1] = c[1] - mu[0] * gamma[0];
            gamma[1] = d[1];
        }

        z[1] = rhs[1] - mu[0] * z[0];
    }

    // Remaining rows (full pentadiagonal structure)
    for (size_t i = 2; i < n; ++i) {
        // Compute multipliers
        const T mu1 = a[i-1] / alpha[i-1];
        mu[i-1] = mu1;

        T mu2 = T{0};
        if (i >= 2) {
            mu2 = (e[i-2] - mu1 * mu[i-2] * alpha[i-2]) / alpha[i-1];
        }

        // Update main diagonal
        alpha[i] = b[i] - mu1 * beta[i-1];
        if (i >= 2) {
            alpha[i] -= mu2 * beta[i-1];
        }

        if (std::abs(alpha[i]) < config.singularity_tol) {
            return Result::error_result("Singular matrix during elimination");
        }

        // Update superdiagonals
        if (i < n - 1) {
            beta[i] = c[i];
            if (i >= 1) {
                beta[i] -= mu1 * gamma[i-1];
            }
        }

        if (i < n - 2) {
            gamma[i] = d[i];
        }

        // Update RHS
        z[i] = rhs[i] - mu1 * z[i-1];
        if (i >= 2 && i - 2 < z.size()) {
            z[i] -= mu2 * z[i-2];
        }
    }

    // ========== Back Substitution ==========

    solution[n-1] = z[n-1] / alpha[n-1];

    if (n > 1) {
        solution[n-2] = (z[n-2] - beta[n-2] * solution[n-1]) / alpha[n-2];
    }

    for (size_t i = n - 2; i > 0; --i) {
        const size_t idx = i - 1;
        solution[idx] = z[idx] - beta[idx] * solution[idx+1];

        if (idx + 2 < n) {
            solution[idx] -= gamma[idx] * solution[idx+2];
        }

        solution[idx] /= alpha[idx];
    }

    return Result::ok_result();
}

/// Convenience wrapper with automatic workspace allocation
template<FloatingPoint T>
[[nodiscard]] inline PentadiagonalResult<T> solve_pentadiagonal_alloc(
    std::span<const T> e,
    std::span<const T> a,
    std::span<const T> b,
    std::span<const T> c,
    std::span<const T> d,
    std::span<const T> rhs,
    std::span<T> solution,
    const PentadiagonalConfig<T>& config = {})
{
    const size_t n = b.size();
    std::vector<T> workspace(5 * n);
    return solve_pentadiagonal(e, a, b, c, d, rhs, solution,
                               std::span{workspace}, config);
}

/// RAII workspace manager for pentadiagonal solver
template<FloatingPoint T>
class PentadiagonalWorkspace {
public:
    explicit PentadiagonalWorkspace(size_t n) : workspace_(5 * n), n_(n) {}

    [[nodiscard]] std::span<T> get() noexcept {
        return std::span{workspace_};
    }

    [[nodiscard]] constexpr size_t size() const noexcept { return n_; }

    void resize(size_t new_n) {
        n_ = new_n;
        workspace_.resize(5 * new_n);
    }

private:
    std::vector<T> workspace_;
    size_t n_;
};

}  // namespace mango
