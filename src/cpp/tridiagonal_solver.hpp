#pragma once

#include <span>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace mango {

/// Thomas Algorithm for Tridiagonal Systems
///
/// Solves Ax = d where A is a tridiagonal matrix:
///   a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]
///
/// The system is:
///   b[0]·x[0] + c[0]·x[1] = d[0]
///   a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]  for i=1..n-2
///   a[n-1]·x[n-2] + b[n-1]·x[n-1] = d[n-1]
///
/// Algorithm:
///   Forward sweep: eliminate lower diagonal
///   Back substitution: solve for x from bottom up
///
/// Time complexity: O(n)
/// Space complexity: O(n) for workspace
///
/// @param lower Lower diagonal (a), size n-1, indices [0..n-2]
/// @param diag Main diagonal (b), size n
/// @param upper Upper diagonal (c), size n-1, indices [0..n-2]
/// @param rhs Right-hand side (d), size n
/// @param solution Output solution vector (x), size n
/// @param workspace Temporary storage, size 2n (c_prime, d_prime)
/// @return true if successful, false if diagonal becomes zero (singular)
inline bool solve_tridiagonal(
    std::span<const double> lower,
    std::span<const double> diag,
    std::span<const double> upper,
    std::span<const double> rhs,
    std::span<double> solution,
    std::span<double> workspace)
{
    const size_t n = diag.size();

    // Validate sizes
    if (lower.size() != n - 1 || upper.size() != n - 1 ||
        rhs.size() != n || solution.size() != n ||
        workspace.size() < 2 * n) {
        throw std::invalid_argument("solve_tridiagonal: size mismatch");
    }

    if (n == 0) {
        return true;
    }

    if (n == 1) {
        if (std::abs(diag[0]) < 1e-15) {
            return false;  // Singular
        }
        solution[0] = rhs[0] / diag[0];
        return true;
    }

    // Workspace: c_prime[0..n-2], d_prime[0..n-1]
    std::span<double> c_prime = workspace.subspan(0, n);
    std::span<double> d_prime = workspace.subspan(n, n);

    // Forward sweep: eliminate lower diagonal
    // Row 0: b[0]·x[0] + c[0]·x[1] = d[0]
    //        x[0] + c'[0]·x[1] = d'[0]
    //        where c'[0] = c[0]/b[0], d'[0] = d[0]/b[0]

    if (std::abs(diag[0]) < 1e-15) {
        return false;  // Singular
    }

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    // Rows 1..n-1: eliminate a[i]
    // a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]
    // After substituting x[i-1] = d'[i-1] - c'[i-1]·x[i]:
    //   (b[i] - a[i]·c'[i-1])·x[i] + c[i]·x[i+1] = d[i] - a[i]·d'[i-1]
    //   x[i] + c'[i]·x[i+1] = d'[i]

    for (size_t i = 1; i < n - 1; ++i) {
        double denom = diag[i] - lower[i-1] * c_prime[i-1];
        if (std::abs(denom) < 1e-15) {
            return false;  // Singular or ill-conditioned
        }
        c_prime[i] = upper[i] / denom;
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom;
    }

    // Last row (i = n-1): no upper diagonal
    // a[n-1]·x[n-2] + b[n-1]·x[n-1] = d[n-1]
    {
        size_t i = n - 1;
        double denom = diag[i] - lower[i-1] * c_prime[i-1];
        if (std::abs(denom) < 1e-15) {
            return false;  // Singular or ill-conditioned
        }
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom;
    }

    // Back substitution
    solution[n-1] = d_prime[n-1];
    for (size_t i = n - 1; i > 0; --i) {
        solution[i-1] = d_prime[i-1] - c_prime[i-1] * solution[i];
    }

    return true;
}

/// Tridiagonal solver with automatic workspace allocation
///
/// Convenience wrapper that allocates workspace internally.
/// For performance-critical code, use the workspace version above.
///
/// @param lower Lower diagonal, size n-1
/// @param diag Main diagonal, size n
/// @param upper Upper diagonal, size n-1
/// @param rhs Right-hand side, size n
/// @param solution Output solution, size n
/// @return true if successful, false if singular
inline bool solve_tridiagonal_alloc(
    std::span<const double> lower,
    std::span<const double> diag,
    std::span<const double> upper,
    std::span<const double> rhs,
    std::span<double> solution)
{
    const size_t n = diag.size();
    std::vector<double> workspace(2 * n);
    return solve_tridiagonal(lower, diag, upper, rhs, solution, std::span{workspace});
}

}  // namespace mango
