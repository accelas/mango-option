#ifndef IVCALC_INTERP_CUBIC_H
#define IVCALC_INTERP_CUBIC_H

#include "interp_strategy.h"
#include <stddef.h>

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

// Workspace structure for cubic interpolation queries
// This eliminates all malloc calls in hot path by using caller-provided buffers
typedef struct {
    // Spline computation workspace (reused across all stages)
    double *spline_coeff_workspace;  // 4 * max_grid_size doubles
    double *spline_temp_workspace;   // 6 * max_grid_size doubles

    // Intermediate arrays for tensor-product interpolation
    double *intermediate_arrays;     // Sum of all intermediate array sizes

    // Slice extraction buffers
    double *slice_buffers;           // max_grid_size doubles

    // Internal bookkeeping (do not modify)
    size_t max_grid_size;
    size_t total_size;
} CubicInterpWorkspace;

// Calculate required workspace size for 2D interpolation
// Returns total number of doubles needed
size_t cubic_interp_workspace_size_2d(size_t n_moneyness, size_t n_maturity);

// Calculate required workspace size for 4D interpolation
size_t cubic_interp_workspace_size_4d(size_t n_moneyness, size_t n_maturity,
                                       size_t n_volatility, size_t n_rate);

// Calculate required workspace size for 5D interpolation
size_t cubic_interp_workspace_size_5d(size_t n_moneyness, size_t n_maturity,
                                       size_t n_volatility, size_t n_rate,
                                       size_t n_dividend);

// Initialize workspace from caller-provided buffer
// buffer must have at least cubic_interp_workspace_size_*() doubles allocated
// Returns 0 on success, -1 on error
int cubic_interp_workspace_init(CubicInterpWorkspace *workspace,
                                 double *buffer,
                                 size_t n_moneyness, size_t n_maturity,
                                 size_t n_volatility, size_t n_rate,
                                 size_t n_dividend);

#endif // IVCALC_INTERP_CUBIC_H
