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

// Create and compute cubic spline interpolation (malloc-based)
// Uses natural boundary conditions (second derivative = 0 at endpoints)
// This version allocates memory internally - use pde_spline_init() for zero-malloc version
CubicSpline* pde_spline_create(const double *x, const double *y, size_t n_points);

// Initialize cubic spline with caller-provided workspace (zero-malloc version)
// Uses natural boundary conditions (second derivative = 0 at endpoints)
//
// This version eliminates malloc overhead by using caller-provided buffers.
// Ideal for hot paths where splines are created/destroyed repeatedly.
//
// Parameters:
//   spline: Pointer to CubicSpline struct (can be stack-allocated)
//   x: Grid points (must remain valid while spline is in use)
//   y: Function values (must remain valid while spline is in use)
//   n_points: Number of data points (must be >= 2)
//   workspace: Buffer for coefficient storage (must be >= 4*n_points doubles)
//              Layout: [a_coeffs(n), b_coeffs(n), c_coeffs(n), d_coeffs(n)]
//   temp_workspace: Buffer for temporary computation (must be >= 6*n_points doubles)
//                   This can be reused across multiple pde_spline_init() calls
//
// Returns: 0 on success, -1 on error
//
// Example usage:
//   CubicSpline spline;
//   double workspace[4 * N];
//   double temp_workspace[6 * N];
//   pde_spline_init(&spline, x, y, N, workspace, temp_workspace);
//   double val = pde_spline_eval(&spline, x_query);
//   // No destroy needed - workspace managed by caller
//
// Total workspace requirement: 10*n_points doubles
int pde_spline_init(CubicSpline *spline, const double *x, const double *y,
                    size_t n_points, double *workspace, double *temp_workspace);

// Destroy spline and free memory (only for splines created with pde_spline_create)
// Do NOT call this for splines initialized with pde_spline_init()
void pde_spline_destroy(CubicSpline *spline);

// Evaluate spline at arbitrary point x_eval
// Returns interpolated value
double pde_spline_eval(const CubicSpline *spline, double x_eval);

// Evaluate spline derivative at arbitrary point x_eval
double pde_spline_eval_derivative(const CubicSpline *spline, double x_eval);

#endif // CUBIC_SPLINE_H
