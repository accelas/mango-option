/**
 * @file thomas_solver.hpp
 * @brief Modern C++20 Thomas algorithm for tridiagonal systems
 *
 * Features C++20 concepts, SIMD hints, and improved error handling.
 */

#pragma once

#include <span>
#include <vector>
#include <cmath>
#include <optional>
#include <concepts>
#include <string_view>
#include <limits>

namespace mango {

/// Floating point concept for template constraints
template<typename T>
concept FloatingPoint = std::floating_point<T>;

/// Result type for Thomas solver
template<FloatingPoint T>
struct ThomasResult {
    bool success;
    std::optional<std::string_view> error;

    /// Implicit conversion to bool for easy checking
    constexpr explicit operator bool() const noexcept { return success; }

    /// Check if solver succeeded
    [[nodiscard]] constexpr bool ok() const noexcept { return success; }

    /// Get error message (empty if successful)
    [[nodiscard]] constexpr std::string_view message() const noexcept {
        return error.value_or("");
    }

    /// Create success result
    [[nodiscard]] static constexpr ThomasResult ok_result() noexcept {
        return ThomasResult{.success = true, .error = std::nullopt};
    }

    /// Create error result
    [[nodiscard]] static constexpr ThomasResult error_result(std::string_view msg) noexcept {
        return ThomasResult{.success = false, .error = msg};
    }
};

/// Configuration for Thomas solver
template<FloatingPoint T>
struct ThomasConfig {
    /// Tolerance for singularity detection
    T singularity_tol = static_cast<T>(1e-15);

    /// Whether to check for diagonal dominance (stricter condition)
    bool check_diagonal_dominance = false;
};

/// Thomas Algorithm for Tridiagonal Systems (Modern C++20)
///
/// Solves Ax = d where A is a tridiagonal matrix using Thomas algorithm.
///
/// System form:
///   b[0]·x[0] + c[0]·x[1] = d[0]
///   a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]  for i=1..n-2
///   a[n-1]·x[n-2] + b[n-1]·x[n-1] = d[n-1]
///
/// Algorithm:
///   1. Forward elimination (lower diagonal elimination)
///   2. Back substitution
///
/// Time:  O(n)
/// Space: O(n) for workspace
///
/// @tparam T Floating point type (float, double, long double)
/// @param lower Lower diagonal (a), size n-1
/// @param diag Main diagonal (b), size n
/// @param upper Upper diagonal (c), size n-1
/// @param rhs Right-hand side (d), size n
/// @param solution Output solution (x), size n (modified in-place)
/// @param workspace Temporary storage, size 2n
/// @param config Solver configuration
/// @return Result indicating success/failure with error message
template<FloatingPoint T>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<T> solution,
    std::span<T> workspace,
    const ThomasConfig<T>& config = {}) noexcept
{
    using Result = ThomasResult<T>;

    const size_t n = diag.size();

    // Validate dimensions
    if (lower.size() != n - 1) {
        return Result::error_result("Lower diagonal size must be n-1");
    }
    if (upper.size() != n - 1) {
        return Result::error_result("Upper diagonal size must be n-1");
    }
    if (rhs.size() != n) {
        return Result::error_result("RHS size must be n");
    }
    if (solution.size() != n) {
        return Result::error_result("Solution size must be n");
    }
    if (workspace.size() < 2 * n) {
        return Result::error_result("Workspace size must be at least 2n");
    }

    // Handle trivial cases
    if (n == 0) {
        return Result::ok_result();
    }

    if (n == 1) {
        if (std::abs(diag[0]) < config.singularity_tol) {
            return Result::error_result("Singular matrix (diagonal[0] ≈ 0)");
        }
        solution[0] = rhs[0] / diag[0];
        return Result::ok_result();
    }

    // Split workspace into c' and d' arrays
    std::span<T> c_prime = workspace.subspan(0, n);
    std::span<T> d_prime = workspace.subspan(n, n);

    // Optional: Check diagonal dominance for stability
    if (config.check_diagonal_dominance) {
        for (size_t i = 0; i < n; ++i) {
            T row_sum = std::abs(diag[i]);
            if (i > 0) row_sum -= std::abs(lower[i-1]);
            if (i < n - 1) row_sum -= std::abs(upper[i]);

            if (row_sum < config.singularity_tol) {
                return Result::error_result("Matrix not diagonally dominant");
            }
        }
    }

    // ========== Forward Elimination ==========

    // First row
    if (std::abs(diag[0]) < config.singularity_tol) {
        return Result::error_result("Singular matrix (diagonal[0] ≈ 0)");
    }

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    // Middle rows (vectorizable loop)
    // Modern compilers can auto-vectorize this with appropriate flags
    #pragma omp simd
    for (size_t i = 1; i < n - 1; ++i) {
        const T denom = diag[i] - lower[i-1] * c_prime[i-1];

        // Singularity check (note: breaks vectorization, but necessary)
        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular or ill-conditioned matrix");
        }

        const T inv_denom = static_cast<T>(1) / denom;
        c_prime[i] = upper[i] * inv_denom;
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) * inv_denom;
    }

    // Last row (no upper diagonal term)
    {
        const size_t i = n - 1;
        const T denom = diag[i] - lower[i-1] * c_prime[i-1];

        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular matrix (at last row)");
        }

        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom;
    }

    // ========== Back Substitution ==========

    solution[n-1] = d_prime[n-1];

    // Reverse iteration (vectorization limited by data dependency)
    for (size_t i = n - 1; i > 0; --i) {
        solution[i-1] = d_prime[i-1] - c_prime[i-1] * solution[i];
    }

    return Result::ok_result();
}

/// Convenience wrapper with automatic workspace allocation
///
/// For performance-critical code, prefer the workspace version above
/// to avoid repeated allocations.
///
/// @tparam T Floating point type
/// @param lower Lower diagonal, size n-1
/// @param diag Main diagonal, size n
/// @param upper Upper diagonal, size n-1
/// @param rhs Right-hand side, size n
/// @param solution Output solution, size n
/// @param config Solver configuration
/// @return Result indicating success/failure
template<FloatingPoint T>
[[nodiscard]] inline ThomasResult<T> solve_thomas_alloc(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<T> solution,
    const ThomasConfig<T>& config = {})
{
    const size_t n = diag.size();
    std::vector<T> workspace(2 * n);
    return solve_thomas(lower, diag, upper, rhs, solution,
                       std::span{workspace}, config);
}

/// RAII workspace manager for Thomas solver
///
/// Manages workspace lifetime and provides convenient access.
/// Useful for repeated solves with the same matrix size.
///
/// Example:
///   ThomasWorkspace<double> ws(1000);  // For n=1000 system
///   for (auto& rhs : many_rhs_vectors) {
///       solve_thomas(lower, diag, upper, rhs, solution, ws.get());
///   }
template<FloatingPoint T>
class ThomasWorkspace {
public:
    explicit ThomasWorkspace(size_t n) : workspace_(2 * n), n_(n) {}

    /// Get workspace span for Thomas solver
    [[nodiscard]] std::span<T> get() noexcept {
        return std::span{workspace_};
    }

    /// Get system size
    [[nodiscard]] constexpr size_t size() const noexcept { return n_; }

    /// Resize workspace for different system size
    void resize(size_t new_n) {
        n_ = new_n;
        workspace_.resize(2 * new_n);
    }

private:
    std::vector<T> workspace_;
    size_t n_;
};

}  // namespace mango
