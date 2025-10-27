#ifndef TRIDIAGONAL_H
#define TRIDIAGONAL_H

#include <stddef.h>
#include <stdlib.h>

// Tridiagonal matrix solver using Thomas algorithm (header-only)
// Solves: A*x = b where A is tridiagonal
// Parameters:
//   n: matrix size
//   lower: lower diagonal (n elements, lower[0] unused)
//   diag: main diagonal (n elements)
//   upper: upper diagonal (n elements, upper[n-1] unused)
//   rhs: right-hand side vector (n elements)
//   solution: output solution vector (n elements)
// Time complexity: O(n), Space complexity: O(n)
static inline void solve_tridiagonal(size_t n, const double *lower, const double *diag,
                                    const double *upper, const double *rhs, double *solution) {
    double *c_prime = malloc(n * sizeof(double));
    double *d_prime = malloc(n * sizeof(double));

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (size_t i = 1; i < n; i++) {
        double m = 1.0 / (diag[i] - lower[i] * c_prime[i - 1]);
        c_prime[i] = (i < n - 1) ? upper[i] * m : 0.0;
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) * m;
    }

    // Back substitution
    solution[n - 1] = d_prime[n - 1];
    for (int i = (int)n - 2; i >= 0; i--) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }

    free(c_prime);
    free(d_prime);
}

#endif // TRIDIAGONAL_H
