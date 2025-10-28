#ifndef IVCALC_INTERP_MULTILINEAR_H
#define IVCALC_INTERP_MULTILINEAR_H

#include "interp_strategy.h"

/**
 * @file interp_multilinear.h
 * @brief Multi-linear interpolation strategy implementation
 *
 * Implements separable multi-linear interpolation in 2D, 4D, and 5D.
 *
 * Algorithm:
 * - Find bracketing indices for each dimension (binary search)
 * - Perform linear interpolation along each dimension sequentially
 * - For d dimensions: 2^d lookups + (2^d - 1) linear interpolations
 *
 * Performance:
 * - 2D: ~50-100ns (4 lookups + 3 lerps)
 * - 4D: ~200-500ns (16 lookups + 15 lerps)
 * - 5D: ~1-2µs (32 lookups + 31 lerps)
 *
 * Properties:
 * - C0 continuous (not smooth)
 * - Guaranteed to stay within bounds
 * - Fast and cache-friendly
 * - No overshoot (unlike cubic splines)
 */

/**
 * Global multi-linear interpolation strategy instance
 *
 * Usage:
 *   IVSurface *surface = iv_surface_create_with_strategy(
 *       moneyness, n_m, maturity, n_tau, &INTERP_MULTILINEAR);
 */
extern const InterpolationStrategy INTERP_MULTILINEAR;

/**
 * Helper function: Find bracketing interval for query point
 *
 * Given a sorted grid and a query point, find the index i such that:
 *   grid[i] <= query < grid[i+1]
 *
 * Uses binary search for O(log n) performance.
 *
 * @param grid: sorted array of grid points
 * @param n: number of grid points
 * @param query: query point to bracket
 * @return index i (clamped to [0, n-2])
 */
size_t find_bracket(const double *grid, size_t n, double query);

/**
 * Helper function: Linear interpolation between two points
 *
 * @param x0: left grid point
 * @param x1: right grid point
 * @param y0: value at x0
 * @param y1: value at x1
 * @param x: query point (should be in [x0, x1])
 * @return interpolated value at x
 */
double lerp(double x0, double x1, double y0, double y1, double x);

#endif // IVCALC_INTERP_MULTILINEAR_H
