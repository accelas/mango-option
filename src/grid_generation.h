#ifndef MANGO_GRID_GENERATION_H
#define MANGO_GRID_GENERATION_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file grid_generation.h
 * @brief Grid generation utilities for non-uniform spacing
 *
 * Provides flexible grid generation strategies optimized for option pricing:
 * - Uniform spacing (baseline)
 * - Logarithmic spacing (current default for moneyness)
 * - Chebyshev nodes (optimal for polynomial interpolation)
 * - Tanh-based concentration (flexible, centered at specific point)
 * - Sinh-based concentration (one-sided, e.g., short maturities)
 *
 * Non-uniform grids concentrate points where price surfaces have high curvature,
 * achieving better accuracy with fewer total points (5-10× reduction typical).
 *
 * Usage:
 *   // Concentrate moneyness points near ATM (m = 1.0)
 *   double *m_grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
 *
 *   // Concentrate maturity points near short-term (tau = 0)
 *   double *tau_grid = grid_sinh_onesided(0.027, 2.0, 15, 2.5);
 *
 *   // Free when done
 *   free(m_grid);
 *   free(tau_grid);
 */

/**
 * Grid spacing strategies
 */
typedef enum {
    GRID_UNIFORM,           ///< Uniform spacing (baseline)
    GRID_LOG,               ///< Logarithmic spacing (existing)
    GRID_CHEBYSHEV,         ///< Chebyshev nodes (optimal for polynomials)
    GRID_TANH_CENTER,       ///< Tanh concentration at center point
    GRID_SINH_ONESIDED,     ///< Sinh concentration at one end
    GRID_CUSTOM             ///< User-provided spacing function
} GridSpacingType;

/**
 * Grid generation specification
 */
typedef struct {
    GridSpacingType type;
    double min;              ///< Minimum value
    double max;              ///< Maximum value
    size_t n_points;         ///< Number of points

    /// Type-specific parameters
    union {
        struct {
            double center;      ///< Concentration center (TANH_CENTER)
            double strength;    ///< Concentration strength (0-10, default 3)
        } tanh_params;

        struct {
            double strength;    ///< Concentration strength (SINH_ONESIDED)
        } sinh_params;
    };
} GridSpec;

/**
 * Grid quality metrics
 */
typedef struct {
    double min_spacing;      ///< Minimum spacing between consecutive points
    double max_spacing;      ///< Maximum spacing
    double avg_spacing;      ///< Average spacing
    double spacing_ratio;    ///< max_spacing / min_spacing (uniformity metric)
} GridMetrics;

/**
 * Generate grid points according to specification
 *
 * @param spec Grid specification
 * @return Newly allocated array of grid points (caller must free), or NULL on error
 *
 * Grid is guaranteed to be sorted in ascending order.
 */
double* grid_generate(const GridSpec *spec);

/**
 * Generate uniformly spaced grid
 *
 * @param min Minimum value
 * @param max Maximum value
 * @param n Number of points
 * @return Newly allocated array (caller must free), or NULL on error
 *
 * Example:
 *   double *grid = grid_uniform(0.0, 1.0, 11);
 *   // grid = [0.0, 0.1, 0.2, ..., 1.0]
 */
double* grid_uniform(double min, double max, size_t n);

/**
 * Generate logarithmically spaced grid
 *
 * @param min Minimum value (must be > 0)
 * @param max Maximum value (must be > min)
 * @param n Number of points
 * @return Newly allocated array (caller must free), or NULL on error
 *
 * Generates points uniformly spaced in log-space:
 *   x_i = min * (max/min)^(i/(n-1))
 *
 * Example:
 *   double *grid = grid_log(0.7, 1.3, 10);
 *   // Points concentrated near min, spreading out toward max
 */
double* grid_log(double min, double max, size_t n);

/**
 * Generate Chebyshev nodes
 *
 * @param min Minimum value
 * @param max Maximum value
 * @param n Number of points
 * @return Newly allocated array (caller must free), or NULL on error
 *
 * Chebyshev nodes minimize Runge's phenomenon in polynomial interpolation.
 * Points are concentrated at both boundaries.
 *
 * Formula:
 *   x_i = center + radius * cos((2i + 1)π / (2n))
 *   where center = (min + max)/2, radius = (max - min)/2
 *
 * Example:
 *   double *grid = grid_chebyshev(0.0, 1.0, 10);
 *   // Points concentrated at x=0 and x=1
 */
double* grid_chebyshev(double min, double max, size_t n);

/**
 * Generate grid with tanh-based concentration at a center point
 *
 * @param min Minimum value
 * @param max Maximum value
 * @param n Number of points
 * @param center Concentration center (typically ATM for moneyness)
 * @param strength Concentration strength (0-10, typical: 2-4)
 * @return Newly allocated array (caller must free), or NULL on error
 *
 * Uses hyperbolic tangent to create smooth concentration around a target point.
 * Higher strength values create tighter concentration.
 *
 * Formula:
 *   t_i = i / (n-1)
 *   s = tanh(α(t - 0.5)) / tanh(α/2)
 *   x_i = center + s * (max - center)  [if s >= 0]
 *         center + s * (center - min)  [if s < 0]
 *
 * Example:
 *   // Concentrate moneyness points near ATM (m = 1.0)
 *   double *grid = grid_tanh_center(0.7, 1.3, 20, 1.0, 3.0);
 */
double* grid_tanh_center(double min, double max, size_t n,
                         double center, double strength);

/**
 * Generate grid with sinh-based concentration at one end
 *
 * @param min Minimum value (concentration end)
 * @param max Maximum value
 * @param n Number of points
 * @param strength Concentration strength (typical: 2-4)
 * @return Newly allocated array (caller must free), or NULL on error
 *
 * Concentrates points at the minimum end using sinh function.
 * Ideal for time-like dimensions where short maturities need more resolution.
 *
 * Formula:
 *   t_i = i / (n-1)
 *   s = sinh(α * t) / sinh(α)
 *   x_i = min + s * (max - min)
 *
 * Example:
 *   // Concentrate maturity points near short-term (tau = 0)
 *   double *grid = grid_sinh_onesided(0.027, 2.0, 15, 2.5);
 */
double* grid_sinh_onesided(double min, double max, size_t n,
                           double strength);

/**
 * Validate grid properties
 *
 * @param grid Grid points to validate
 * @param n Number of points
 * @param min Expected minimum value
 * @param max Expected maximum value
 * @return true if grid is valid, false otherwise
 *
 * Checks:
 * - Grid is sorted in ascending order
 * - No duplicate values
 * - First point equals min (within tolerance)
 * - Last point equals max (within tolerance)
 * - All points in [min, max]
 */
bool grid_validate(const double *grid, size_t n, double min, double max);

/**
 * Compute grid quality metrics
 *
 * @param grid Grid points
 * @param n Number of points
 * @return Grid metrics
 *
 * Spacing ratio (max_spacing / min_spacing):
 * - ratio ≈ 1: Uniform grid
 * - ratio > 2: Non-uniform (concentrated)
 * - ratio > 5: Highly concentrated
 */
GridMetrics grid_compute_metrics(const double *grid, size_t n);

#ifdef __cplusplus
}
#endif

#endif // MANGO_GRID_GENERATION_H
