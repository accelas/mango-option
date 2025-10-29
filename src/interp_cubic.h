#ifndef IVCALC_INTERP_CUBIC_H
#define IVCALC_INTERP_CUBIC_H

#include "interp_strategy.h"

/**
 * @file interp_cubic.h
 * @brief Tensor-product cubic spline interpolation strategy
 *
 * Provides C2-continuous (smooth) interpolation using natural cubic splines.
 * Allows accurate calculation of second derivatives (gamma, vanna, volga).
 *
 * Algorithm:
 * - Pre-computes cubic spline coefficients for each dimension
 * - At query time: recursively evaluates cubic splines along each dimension
 * - Similar to multilinear but uses cubic evaluation instead of linear lerp
 *
 * Performance:
 * - 3-5x slower than multilinear (~500ns vs ~100ns for 2D)
 * - Still sub-microsecond for real-time queries
 * - One-time pre-computation cost: O(n_total)
 *
 * Memory:
 * - ~4x more storage than multilinear (stores 4 coefficients per point)
 * - Worth it for accurate Greeks calculations
 *
 * Benefits:
 * - C2 continuous (smooth second derivatives)
 * - Accurate gamma via analytical derivatives
 * - Better accuracy for smooth functions (<0.1% error with coarser grids)
 *
 * Limitations:
 * - Can overshoot (may need clamping for option prices >= 0)
 * - More complex than multilinear
 * - Slower queries (still sub-microsecond)
 */

// Global strategy instance
extern const InterpolationStrategy INTERP_CUBIC;

#endif // IVCALC_INTERP_CUBIC_H
