// SPDX-License-Identifier: MIT
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
#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>
#include <limits>
#include "mango/math/tridiagonal_matrix_view.hpp"

namespace mango {

/// Result type for Thomas solver
template<std::floating_point T>
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
template<std::floating_point T>
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
template<std::floating_point T>
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
    const size_t offdiag_size = (n == 0) ? 0 : n - 1;

    // Validate dimensions (offdiag_size avoids n-1 wrap-around at n==0)
    if (lower.size() != offdiag_size) {
        return Result::error_result("Lower diagonal size must be n-1");
    }
    if (upper.size() != offdiag_size) {
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

    // Middle rows
    // Note: Singularity checking prevents SIMD vectorization (early return)
    for (size_t i = 1; i < n - 1; ++i) {
        // Use FMA for denominator calculation
        const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

        // Singularity check
        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular or ill-conditioned matrix");
        }

        const T inv_denom = static_cast<T>(1) / denom;
        c_prime[i] = upper[i] * inv_denom;
        // Use FMA: (rhs[i] - lower[i-1]*d_prime[i-1]) * inv_denom
        d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) * inv_denom;
    }

    // Last row (no upper diagonal term)
    {
        const size_t i = n - 1;
        // Use FMA for denominator calculation
        const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular matrix (at last row)");
        }

        // Use FMA for numerator calculation
        d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) / denom;
    }

    // ========== Back Substitution ==========

    solution[n-1] = d_prime[n-1];

    // Reverse iteration (vectorization limited by data dependency)
    // Use FMA for back substitution: d_prime[i-1] - c_prime[i-1]*solution[i]
    for (size_t i = n - 1; i > 0; --i) {
        solution[i-1] = std::fma(-c_prime[i-1], solution[i], d_prime[i-1]);
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
template<std::floating_point T>
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
template<std::floating_point T>
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

/// Projected Thomas Algorithm (Brennan-Schwartz) for American Options
///
/// **Purpose:**
/// Solves the Linear Complementarity Problem (LCP) arising in American option pricing:
///   A·x = d,  subject to x ≥ ψ (obstacle constraint)
/// where A is a tridiagonal M-matrix from TR-BDF2 time-stepping.
///
/// **Mathematical Background:**
/// American options require solving a PDE with obstacle constraint:
///   ∂V/∂t + L(V) = 0,  V ≥ ψ (payoff)
/// Implicit time-stepping gives: (I - dt·L)·V = rhs,  V ≥ ψ
/// This is an LCP - linear system with inequality constraint.
///
/// **Algorithm Overview:**
/// The Brennan-Schwartz (1977) algorithm enforces the obstacle constraint
/// DURING backward substitution, not after. This respects the tridiagonal
/// coupling between nodes and provably converges in a single pass.
///
/// **Key Difference from "Solve then Project":**
///   WRONG approach (breaks tridiagonal coupling):
///     1. Solve Ax = d unconstrained
///     2. Project: x[i] = max(x[i], ψ[i]) for all i
///     → This violates Ax = d at projected nodes!
///
///   CORRECT approach (Projected Thomas):
///     1. Forward elimination: build c', d' (standard Thomas)
///     2. Projected backward substitution:
///        x[n-1] = max(d'[n-1], ψ[n-1])
///        x[i] = max(d'[i] - c'[i]·x[i+1], ψ[i])  for i = n-2, ..., 0
///     → Constraint enforced at EACH STEP, preserving tridiagonal structure
///
/// **Why This Works:**
/// For M-matrices (which TR-BDF2 produces with proper dt):
///   - A has positive diagonal and non-positive off-diagonals
///   - The projection max(·, ψ[i]) is monotone
///   - Backward substitution propagates constraints correctly
///   - Result: Single-pass convergence, no iteration needed
///
/// **Performance:**
/// - Time: O(n) (same as standard Thomas)
/// - Space: O(n) workspace
/// - Iterations: 1 (always, for well-posed problems)
///
/// **Contrast with Newton + Active Set:**
/// - Newton solves J·δx = -F(x), updates x ← x + δx (iterative)
/// - Active set guesses which nodes are on obstacle (heuristic)
/// - Projected Thomas solves A·x = d directly, enforces constraint exactly (single-pass)
///
/// @tparam T Floating point type (float or double)
/// @param lower Lower diagonal a[0..n-2], where A[i,i-1] = a[i-1]
/// @param diag Main diagonal b[0..n-1], where A[i,i] = b[i]
/// @param upper Upper diagonal c[0..n-2], where A[i,i+1] = c[i]
/// @param rhs Right-hand side d[0..n-1]
/// @param psi Obstacle (lower bound) ψ[0..n-1], must have ψ ≤ d for consistency
/// @param solution OUTPUT: Solution x[0..n-1], satisfies A·x = d and x ≥ ψ
/// @param workspace Temporary storage, size ≥ 2n (for c', d' arrays)
/// @param config Optional solver configuration (tolerances, max iterations)
/// @return ThomasResult with success/failure status and diagnostics
template<std::floating_point T>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas_projected(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<T> solution,
    std::span<T> workspace,
    const ThomasConfig<T>& config = {}) noexcept
{
    using Result = ThomasResult<T>;

    const size_t n = diag.size();
    const size_t offdiag_size = (n == 0) ? 0 : n - 1;

    // Validate dimensions (offdiag_size avoids n-1 wrap-around at n==0)
    if (lower.size() != offdiag_size) {
        return Result::error_result("Lower diagonal size must be n-1");
    }
    if (upper.size() != offdiag_size) {
        return Result::error_result("Upper diagonal size must be n-1");
    }
    if (rhs.size() != n) {
        return Result::error_result("RHS size must be n");
    }
    if (psi.size() != n) {
        return Result::error_result("Obstacle size must be n");
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
        solution[0] = std::max(rhs[0] / diag[0], psi[0]);
        return Result::ok_result();
    }

    // Split workspace into c' and d' arrays
    std::span<T> c_prime = workspace.subspan(0, n);
    std::span<T> d_prime = workspace.subspan(n, n);

    // ========== Forward Elimination (IDENTICAL to standard Thomas) ==========

    // First row
    if (std::abs(diag[0]) < config.singularity_tol) {
        return Result::error_result("Singular matrix (diagonal[0] ≈ 0)");
    }

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    // Middle rows
    for (size_t i = 1; i < n - 1; ++i) {
        const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular or ill-conditioned matrix");
        }

        const T inv_denom = static_cast<T>(1) / denom;
        c_prime[i] = upper[i] * inv_denom;
        d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) * inv_denom;
    }

    // Last row
    {
        const size_t i = n - 1;
        const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

        if (std::abs(denom) < config.singularity_tol) {
            return Result::error_result("Singular matrix (at last row)");
        }

        d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) / denom;
    }

    // ========== Projected Back Substitution (KEY DIFFERENCE) ==========

    // Last element with projection
    solution[n-1] = std::max(d_prime[n-1], psi[n-1]);

    // Backward iteration with projection at each step
    // This couples the obstacle constraint with the tridiagonal structure
    for (size_t i = n - 1; i > 0; --i) {
        T unconstrained = std::fma(-c_prime[i-1], solution[i], d_prime[i-1]);
        solution[i-1] = std::max(unconstrained, psi[i-1]);
    }

    return Result::ok_result();
}

/// Which side of the grid the active (constrained) interval touches.
///
/// The projected Thomas substitution enforces the obstacle exactly, in a
/// single pass, only when the active set is a contiguous interval touching
/// the side the substitution sweeps *from* first. `Right` sweeps back
/// (right-to-left, matching `solve_thomas_projected` above) and is exact for
/// RIGHT-interval active sets (e.g. call-like obstacles). `Left` sweeps
/// forward (left-to-right, via a UL elimination) and is exact for
/// LEFT-interval active sets (e.g. put-like obstacles).
enum class LcpActiveSide { Left, Right };

/// Oriented projected Thomas Algorithm (Brennan-Schwartz) with active mask.
///
/// Same LCP as `solve_thomas_projected` (A·x = d, x ≥ ψ), but:
///   - `Side` selects which end the projected substitution sweeps from,
///     so that interval-active obstacles on either side can be solved
///     exactly (see `LcpActiveSide` above).
///   - `active_mask[i]` is written 1 where node i was clamped to ψ[i],
///     0 otherwise — this is the mask `validate_lcp_kkt` needs.
///
/// `Side::Right` reproduces `solve_thomas_projected` exactly (identical
/// FMA sequence), only adding the mask writes.
///
/// @tparam T Floating point type (float or double)
/// @tparam Side Which end the active interval is expected to touch
/// @param lower Lower diagonal a[0..n-2]
/// @param diag Main diagonal b[0..n-1]
/// @param upper Upper diagonal c[0..n-2]
/// @param rhs Right-hand side d[0..n-1]
/// @param psi Obstacle (lower bound) ψ[0..n-1]
/// @param solution OUTPUT: solution x[0..n-1], satisfies A·x = d and x ≥ ψ
/// @param workspace Temporary storage, size ≥ 2n
/// @param active_mask OUTPUT: size n, 1 where clamped to ψ[i], else 0
/// @param config Optional solver configuration
/// @return ThomasResult with success/failure status and diagnostics
template<std::floating_point T, LcpActiveSide Side = LcpActiveSide::Right>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas_projected2(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<T> solution,
    std::span<T> workspace,
    std::span<uint8_t> active_mask,
    const ThomasConfig<T>& config = {}) noexcept
{
    using Result = ThomasResult<T>;

    const size_t n = diag.size();
    const size_t offdiag_size = (n == 0) ? 0 : n - 1;

    // Validate dimensions (offdiag_size avoids n-1 wrap-around at n==0)
    if (lower.size() != offdiag_size) {
        return Result::error_result("Lower diagonal size must be n-1");
    }
    if (upper.size() != offdiag_size) {
        return Result::error_result("Upper diagonal size must be n-1");
    }
    if (rhs.size() != n) {
        return Result::error_result("RHS size must be n");
    }
    if (psi.size() != n) {
        return Result::error_result("Obstacle size must be n");
    }
    if (solution.size() != n) {
        return Result::error_result("Solution size must be n");
    }
    if (workspace.size() < 2 * n) {
        return Result::error_result("Workspace size must be at least 2n");
    }
    if (active_mask.size() != n) {
        return Result::error_result("Active mask size must be n");
    }

    // Handle trivial cases
    if (n == 0) {
        return Result::ok_result();
    }

    if (n == 1) {
        if (std::abs(diag[0]) < config.singularity_tol) {
            return Result::error_result("Singular matrix (diagonal[0] ≈ 0)");
        }
        const T unconstrained = rhs[0] / diag[0];
        active_mask[0] = (unconstrained < psi[0]) ? uint8_t{1} : uint8_t{0};
        solution[0] = std::max(unconstrained, psi[0]);
        return Result::ok_result();
    }

    // Split workspace into c' and d' arrays
    std::span<T> c_prime = workspace.subspan(0, n);
    std::span<T> d_prime = workspace.subspan(n, n);

    if constexpr (Side == LcpActiveSide::Right) {
        // ===== Forward Elimination (IDENTICAL to solve_thomas_projected) =====

        // First row
        if (std::abs(diag[0]) < config.singularity_tol) {
            return Result::error_result("Singular matrix (diagonal[0] ≈ 0)");
        }

        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];

        // Middle rows
        for (size_t i = 1; i < n - 1; ++i) {
            const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

            if (std::abs(denom) < config.singularity_tol) {
                return Result::error_result("Singular or ill-conditioned matrix");
            }

            const T inv_denom = static_cast<T>(1) / denom;
            c_prime[i] = upper[i] * inv_denom;
            d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) * inv_denom;
        }

        // Last row
        {
            const size_t i = n - 1;
            const T denom = std::fma(-lower[i-1], c_prime[i-1], diag[i]);

            if (std::abs(denom) < config.singularity_tol) {
                return Result::error_result("Singular matrix (at last row)");
            }

            d_prime[i] = std::fma(-lower[i-1], d_prime[i-1], rhs[i]) / denom;
        }

        // ===== Projected Back Substitution (IDENTICAL, plus mask writes) =====

        active_mask[n-1] = (d_prime[n-1] < psi[n-1]) ? uint8_t{1} : uint8_t{0};
        solution[n-1] = std::max(d_prime[n-1], psi[n-1]);

        for (size_t i = n - 1; i > 0; --i) {
            T unconstrained = std::fma(-c_prime[i-1], solution[i], d_prime[i-1]);
            active_mask[i-1] = (unconstrained < psi[i-1]) ? uint8_t{1} : uint8_t{0};
            solution[i-1] = std::max(unconstrained, psi[i-1]);
        }
    } else {
        // ===== UL elimination from row n-1 upward; c_prime[i] couples LEFT =====
        if (std::abs(diag[n-1]) < config.singularity_tol) {
            return Result::error_result("Singular matrix (last diagonal ≈ 0)");
        }
        c_prime[n-1] = lower[n-2] / diag[n-1];
        d_prime[n-1] = rhs[n-1] / diag[n-1];

        for (size_t i = n - 2; i >= 1; --i) {
            const T denom = std::fma(-upper[i], c_prime[i+1], diag[i]);
            if (std::abs(denom) < config.singularity_tol) {
                return Result::error_result("Singular or ill-conditioned matrix");
            }
            const T inv = T(1) / denom;
            c_prime[i] = lower[i-1] * inv;
            d_prime[i] = std::fma(-upper[i], d_prime[i+1], rhs[i]) * inv;
        }

        const T denom0 = std::fma(-upper[0], c_prime[1], diag[0]);
        if (std::abs(denom0) < config.singularity_tol) {
            return Result::error_result("Singular matrix (at first row)");
        }
        d_prime[0] = std::fma(-upper[0], d_prime[1], rhs[0]) / denom0;

        // ===== Projected substitution top-down (starts on the LEFT side) =====
        active_mask[0] = (d_prime[0] < psi[0]) ? uint8_t{1} : uint8_t{0};
        solution[0] = std::max(d_prime[0], psi[0]);
        for (size_t i = 1; i < n; ++i) {
            T unconstrained = std::fma(-c_prime[i], solution[i-1], d_prime[i]);
            active_mask[i] = (unconstrained < psi[i]) ? uint8_t{1} : uint8_t{0};
            solution[i] = std::max(unconstrained, psi[i]);
        }
    }

    return Result::ok_result();
}

/// Diagnostic report from `validate_lcp_kkt`: how badly (if at all) a
/// candidate LCP solution violates the full KKT conditions.
struct LcpKktReport {
    /// Number of grid points that violate a KKT condition beyond tolerance.
    size_t violation_count = 0;
    /// Largest raw KKT defect observed (0 if no violations).
    double max_violation = 0.0;
    /// Kind of the worst violation: 0=primal/complementarity (u vs ψ: either
    /// u < ψ anywhere, or u != ψ on an active node), 1=dual
    /// ((Au)_i < rhs_i on an active node), 2=residual (|(Au)_i - rhs_i| too
    /// large on an inactive node, or a non-finite input). -1 if no
    /// violations were recorded.
    int worst_kind = -1;
};

/// Validates a candidate LCP solution against the full KKT conditions:
///   - primal feasibility: u[i] >= psi[i] for all i
///   - complementarity:    u[i] == psi[i] for active i (active_mask[i]==1)
///   - dual feasibility:   (A u)[i] >= rhs[i] for active i (active_mask[i]==1)
///   - complementarity/residual: (A u)[i] == rhs[i] for inactive i
///
/// This is an independent check — it does not assume `u`/`active_mask` came
/// from any particular solver, so it can catch a solver applying the wrong
/// sweep orientation (Task 1's motivating bug) as well as any other
/// LCP-violating candidate.
///
/// Dimension-mismatched input (e.g., u smaller than diag) yields a sentinel report:
/// violation_count=1, max_violation=infinity, worst_kind=2, without attempting
/// any validation that would risk out-of-bounds reads.
///
/// @tparam T Floating point type (float or double)
/// @param lower Lower diagonal a[0..n-2]
/// @param diag Main diagonal b[0..n-1]
/// @param upper Upper diagonal c[0..n-2]
/// @param rhs Right-hand side d[0..n-1]
/// @param psi Obstacle (lower bound) ψ[0..n-1]
/// @param u Candidate solution to validate, size n
/// @param active_mask Candidate active set, size n (1 = clamped to ψ[i])
/// @param atol Absolute tolerance floor
/// @param rtol Relative tolerance, scaled by the row's term magnitudes
/// @return Report summarizing the worst violation found (if any), or sentinel
///         report if dimensions are mismatched
template<std::floating_point T>
[[nodiscard]] LcpKktReport validate_lcp_kkt(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<const T> u,
    std::span<const uint8_t> active_mask,
    T atol = T(1e-12), T rtol = T(1e-10))
{
    LcpKktReport rep;
    const size_t n = diag.size();

    // Validate input dimensions before any access
    const size_t offdiag_size = (n == 0) ? 0 : n - 1;
    if (lower.size() != offdiag_size || upper.size() != offdiag_size ||
        rhs.size() != n || psi.size() != n || u.size() != n || active_mask.size() != n) {
        return LcpKktReport{
            .violation_count = 1,
            .max_violation = std::numeric_limits<double>::infinity(),
            .worst_kind = 2
        };
    }
    auto note = [&](double defect, int kind) {
        rep.violation_count++;
        if (defect > rep.max_violation) { rep.max_violation = defect; rep.worst_kind = kind; }
    };
    for (size_t i = 0; i < n; ++i) {
        const T lo = (i > 0) ? lower[i-1] * u[i-1] : T(0);
        const T hi = (i + 1 < n) ? upper[i] * u[i+1] : T(0);
        const T Au = lo + diag[i] * u[i] + hi;
        const T scale = std::abs(lo) + std::abs(diag[i] * u[i]) + std::abs(hi) + std::abs(rhs[i]);
        const T tol = atol + rtol * scale;
        if (!std::isfinite(Au) || !std::isfinite(u[i]) || !std::isfinite(tol) ||
            !std::isfinite(psi[i])) {
            note(std::numeric_limits<double>::infinity(), 2);
            continue;
        }
        if (u[i] < psi[i] - tol) note(double(psi[i] - u[i]), 0);
        if (active_mask[i]) {
            if (std::abs(u[i] - psi[i]) > tol) note(std::abs(double(u[i] - psi[i])), 0);
            if (Au < rhs[i] - tol) note(double(rhs[i] - Au), 1);
        }
        else                { if (std::abs(Au - rhs[i]) > tol) note(std::abs(double(Au - rhs[i])), 2); }
    }
    return rep;
}

/// Convenience wrapper for projected Thomas with automatic workspace
///
/// @tparam T Floating point type
/// @param lower Lower diagonal, size n-1
/// @param diag Main diagonal, size n
/// @param upper Upper diagonal, size n-1
/// @param rhs Right-hand side, size n
/// @param psi Lower bound (obstacle), size n
/// @param solution Output solution, size n
/// @param config Solver configuration
/// @return Result indicating success/failure
template<std::floating_point T>
[[nodiscard]] inline ThomasResult<T> solve_thomas_projected_alloc(
    std::span<const T> lower,
    std::span<const T> diag,
    std::span<const T> upper,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<T> solution,
    const ThomasConfig<T>& config = {})
{
    const size_t n = diag.size();
    std::vector<T> workspace(2 * n);
    return solve_thomas_projected(lower, diag, upper, rhs, psi, solution,
                                  std::span{workspace}, config);
}

// ============================================================================
// TridiagonalMatrixView Overloads (Convenience API)
// ============================================================================
//
// These overloads accept TridiagonalMatrixView instead of three separate spans,
// providing better type safety and clearer intent. They simply forward
// to the span-based implementations.

/// Thomas solver accepting TridiagonalMatrixView (convenience overload)
///
/// @param matrix Tridiagonal matrix view containing lower, diag, upper bands
/// @param rhs Right-hand side vector
/// @param solution Output solution vector
/// @param workspace Temporary storage (2n)
/// @param config Solver configuration
/// @return Result indicating success/failure
template<std::floating_point T>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas(
    const TridiagonalMatrixView& matrix,
    std::span<const T> rhs,
    std::span<T> solution,
    std::span<T> workspace,
    const ThomasConfig<T>& config = {}) noexcept
{
    return solve_thomas(matrix.lower(), matrix.diag(), matrix.upper(),
                       rhs, solution, workspace, config);
}

/// Projected Thomas solver accepting TridiagonalMatrixView (convenience overload)
///
/// @param matrix Tridiagonal matrix view containing lower, diag, upper bands
/// @param rhs Right-hand side vector
/// @param psi Obstacle constraint vector
/// @param solution Output solution vector
/// @param workspace Temporary storage (2n)
/// @param config Solver configuration
/// @return Result indicating success/failure
template<std::floating_point T>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas_projected(
    const TridiagonalMatrixView& matrix,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<T> solution,
    std::span<T> workspace,
    const ThomasConfig<T>& config = {}) noexcept
{
    return solve_thomas_projected(matrix.lower(), matrix.diag(), matrix.upper(),
                                  rhs, psi, solution, workspace, config);
}

/// Oriented projected Thomas solver accepting TridiagonalMatrixView
/// (convenience overload)
///
/// @param matrix Tridiagonal matrix view containing lower, diag, upper bands
/// @param rhs Right-hand side vector
/// @param psi Obstacle constraint vector
/// @param solution Output solution vector
/// @param workspace Temporary storage (2n)
/// @param active_mask OUTPUT: size n, 1 where clamped to psi[i], else 0
/// @param config Solver configuration
/// @return Result indicating success/failure
template<std::floating_point T, LcpActiveSide Side = LcpActiveSide::Right>
[[nodiscard]] constexpr ThomasResult<T> solve_thomas_projected2(
    const TridiagonalMatrixView& matrix,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<T> solution,
    std::span<T> workspace,
    std::span<uint8_t> active_mask,
    const ThomasConfig<T>& config = {}) noexcept
{
    return solve_thomas_projected2<T, Side>(matrix.lower(), matrix.diag(), matrix.upper(),
                                            rhs, psi, solution, workspace, active_mask,
                                            config);
}

/// KKT validator accepting TridiagonalMatrixView (convenience overload)
///
/// @param matrix Tridiagonal matrix view containing lower, diag, upper bands
/// @param rhs Right-hand side vector
/// @param psi Obstacle constraint vector
/// @param u Candidate solution to validate
/// @param active_mask Candidate active set (1 = clamped to psi[i])
/// @param atol Absolute tolerance floor
/// @param rtol Relative tolerance, scaled by the row's term magnitudes
/// @return Report summarizing the worst violation found (if any)
template<std::floating_point T>
[[nodiscard]] LcpKktReport validate_lcp_kkt(
    const TridiagonalMatrixView& matrix,
    std::span<const T> rhs,
    std::span<const T> psi,
    std::span<const T> u,
    std::span<const uint8_t> active_mask,
    T atol = T(1e-12), T rtol = T(1e-10))
{
    return validate_lcp_kkt<T>(matrix.lower(), matrix.diag(), matrix.upper(),
                               rhs, psi, u, active_mask, atol, rtol);
}

}  // namespace mango
