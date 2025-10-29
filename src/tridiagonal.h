#ifndef TRIDIAGONAL_H
#define TRIDIAGONAL_H

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>

// Tridiagonal matrix solver using Thomas algorithm (header-only)
// Solves: A*x = b where A is tridiagonal
// Parameters:
//   n: matrix size
//   lower: lower diagonal (n-1 elements): lower[0] is A[1,0], lower[i] is A[i+1,i]
//   diag: main diagonal (n elements): diag[i] is A[i,i]
//   upper: upper diagonal (n-1 elements): upper[0] is A[0,1], upper[i] is A[i,i+1]
//   rhs: right-hand side vector (n elements)
//   solution: output solution vector (n elements)
//   workspace: temporary workspace (2n doubles) for c_prime and d_prime arrays
//              If NULL, will use malloc (slower but backward compatible)
// Time complexity: O(n), Space complexity: O(n)
static inline void solve_tridiagonal(size_t n, const double *lower, const double *diag,
                                    const double *upper, const double *rhs, double *solution,
                                    double *workspace) {
    double *c_prime;
    double *d_prime;
    bool allocated = false;

    if (workspace != NULL) {
        // Use provided workspace (zero-allocation path)
        c_prime = workspace;
        d_prime = workspace + n;
    } else {
        // Fallback to allocation for backward compatibility
        c_prime = (double *)malloc(n * sizeof(double));
        d_prime = (double *)malloc(n * sizeof(double));
        allocated = true;
    }

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    #pragma omp simd
    for (size_t i = 1; i < n; i++) {
        double m = 1.0 / (diag[i] - lower[i-1] * c_prime[i - 1]);
        c_prime[i] = (i < n - 1) ? upper[i] * m : 0.0;
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i - 1]) * m;
    }

    // Back substitution
    solution[n - 1] = d_prime[n - 1];
    for (int i = (int)n - 2; i >= 0; i--) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    if (allocated) {
        free(c_prime);
        free(d_prime);
    }
}

#endif // TRIDIAGONAL_H
