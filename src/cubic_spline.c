#include "cubic_spline.h"
#include "tridiagonal.h"
#include "ivcalc_trace.h"
#include <stdlib.h>

// Binary search to find interval containing x_eval
static size_t find_interval(const double *x, size_t n, double x_eval) {
    // Handle boundary cases
    if (x_eval <= x[0]) return 0;
    if (x_eval >= x[n - 1]) return n - 2;

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        size_t mid = (left + right) / 2;
        if (x_eval < x[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    return left;
}

CubicSpline* pde_spline_create(const double *x, const double *y, size_t n_points) {
    if (n_points < 2) {
        // Trace error condition
        IVCALC_TRACE_SPLINE_ERROR(n_points, 2);
        return nullptr;
    }

    CubicSpline *spline = malloc(sizeof(CubicSpline));
    spline->n_points = n_points;
    spline->x = x;  // Store pointer (not owned)
    spline->y = y;  // Store pointer (not owned)

    // Allocate single workspace buffer for all coefficient arrays
    // Need 4 arrays of size n (a, b, c, d)
    const size_t n = n_points;
    spline->workspace = malloc(4 * n * sizeof(double));

    // Slice workspace into coefficient arrays
    spline->coeffs_a = spline->workspace;
    spline->coeffs_b = spline->workspace + n;
    spline->coeffs_c = spline->workspace + 2 * n;
    spline->coeffs_d = spline->workspace + 3 * n;

    // Compute spline coefficients using natural cubic spline
    // Sᵢ(x) = aᵢ + bᵢ·(x - xᵢ) + cᵢ·(x - xᵢ)² + dᵢ·(x - xᵢ)³
    // for x ∈ [xᵢ, xᵢ₊₁]

    // aᵢ = yᵢ
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        spline->coeffs_a[i] = y[i];
    }

    // Allocate single temporary workspace for all temporary arrays
    // Need: h(n-1), alpha(n-1), lower(n), diag(n), upper(n), rhs(n)
    // Total: 2*(n-1) + 4*n = 6n - 2 doubles
    double *temp_workspace = malloc((6 * n) * sizeof(double));

    // Slice temporary workspace
    double *h = temp_workspace;                    // n-1 (but allocate n for simplicity)
    double *alpha = temp_workspace + n;            // n-1 (but allocate n)
    double *lower = temp_workspace + 2 * n;
    double *diag = temp_workspace + 3 * n;
    double *upper = temp_workspace + 4 * n;
    double *rhs = temp_workspace + 5 * n;

    #pragma omp simd
    for (size_t i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];
    }

    #pragma omp simd
    for (size_t i = 1; i < n - 1; i++) {
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) -
                   (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Set up tridiagonal system: A*c = rhs
    // For natural spline: c[0] = 0, c[n-1] = 0
    // Interior equations: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = alpha[i]

    // Boundary conditions for natural spline
    diag[0] = 1.0;
    upper[0] = 0.0;
    rhs[0] = 0.0;

    for (size_t i = 1; i < n - 1; i++) {
        lower[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        rhs[i] = alpha[i];
    }

    lower[n - 1] = 0.0;
    diag[n - 1] = 1.0;
    rhs[n - 1] = 0.0;

    // Solve tridiagonal system using shared solver
    // Pass NULL for workspace (not in hot path, allocation is acceptable)
    solve_tridiagonal(n, lower, diag, upper, rhs, spline->coeffs_c, NULL);

    // Compute b and d coefficients from c
    #pragma omp simd
    for (size_t j = 0; j < n - 1; j++) {
        spline->coeffs_b[j] = (y[j + 1] - y[j]) / h[j] -
                             h[j] * (spline->coeffs_c[j + 1] + 2.0 * spline->coeffs_c[j]) / 3.0;
        spline->coeffs_d[j] = (spline->coeffs_c[j + 1] - spline->coeffs_c[j]) / (3.0 * h[j]);
    }

    // Free temporary workspace (single free for all temporary arrays)
    free(temp_workspace);

    return spline;
}

void pde_spline_destroy(CubicSpline *spline) {
    if (spline == nullptr) return;

    // Single free for all coefficient arrays
    free(spline->workspace);
    free(spline);
}

double pde_spline_eval(const CubicSpline *spline, double x_eval) {
    // Find the interval containing x_eval
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    // Evaluate spline in interval i
    // Sᵢ(x) = aᵢ + bᵢ·(x - xᵢ) + cᵢ·(x - xᵢ)² + dᵢ·(x - xᵢ)³
    double dx = x_eval - spline->x[i];
    double result = spline->coeffs_a[i] +
                   spline->coeffs_b[i] * dx +
                   spline->coeffs_c[i] * dx * dx +
                   spline->coeffs_d[i] * dx * dx * dx;

    return result;
}

double pde_spline_eval_derivative(const CubicSpline *spline, double x_eval) {
    // Find the interval containing x_eval
    size_t i = find_interval(spline->x, spline->n_points, x_eval);

    // Evaluate derivative: S'ᵢ(x) = bᵢ + 2·cᵢ·(x - xᵢ) + 3·dᵢ·(x - xᵢ)²
    double dx = x_eval - spline->x[i];
    double result = spline->coeffs_b[i] +
                   2.0 * spline->coeffs_c[i] * dx +
                   3.0 * spline->coeffs_d[i] * dx * dx;

    return result;
}
