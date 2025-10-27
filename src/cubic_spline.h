#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

#include <stddef.h>

// Cubic spline structure
typedef struct {
    size_t n_points;      // Number of data points
    const double *x;      // Grid points (not owned)
    const double *y;      // Function values (not owned)
    double *workspace;    // Single buffer for all coefficients
    double *coeffs_a;     // Spline coefficients (sliced from workspace)
    double *coeffs_b;
    double *coeffs_c;
    double *coeffs_d;
} CubicSpline;

// Create and compute cubic spline interpolation
// Uses natural boundary conditions (second derivative = 0 at endpoints)
CubicSpline* pde_spline_create(const double *x, const double *y, size_t n_points);

// Destroy spline and free memory
void pde_spline_destroy(CubicSpline *spline);

// Evaluate spline at arbitrary point x_eval
// Returns interpolated value
double pde_spline_eval(const CubicSpline *spline, double x_eval);

// Evaluate spline derivative at arbitrary point x_eval
double pde_spline_eval_derivative(const CubicSpline *spline, double x_eval);

#endif // CUBIC_SPLINE_H
