#ifndef IVCALC_INTERP_STRATEGY_H
#define IVCALC_INTERP_STRATEGY_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @file interp_strategy.h
 * @brief Interpolation strategy interface using dependency injection
 *
 * This module defines a strategy pattern for interpolation algorithms,
 * allowing runtime selection of linear, cubic, or custom interpolation methods.
 *
 * Design Philosophy:
 * - Strategy Pattern: Uses function pointers (vtable) for polymorphism
 * - Runtime Selection: Switch algorithms without recompilation
 * - Extensibility: Users can implement custom interpolation strategies
 * - Consistency: Follows existing callback-based architecture from PDESolver
 */

// Forward declarations
typedef struct InterpolationStrategy InterpolationStrategy;
typedef struct OptionPriceTable OptionPriceTable;
typedef struct IVSurface IVSurface;

/**
 * Interpolation context: opaque scratch space for algorithm-specific data
 * Each strategy can allocate and manage its own context
 */
typedef void* InterpContext;

/**
 * Strategy interface: function pointers for interpolation operations
 *
 * All strategies must implement these functions.
 * Follows "vtable" pattern common in C for polymorphism.
 */
struct InterpolationStrategy {
    // Name of the strategy (for logging/debugging)
    const char *name;

    // Short description
    const char *description;

    // ---------- 2D Interpolation (IV Surfaces) ----------

    /**
     * Interpolate on 2D grid (moneyness, maturity)
     *
     * @param surface: IV surface data
     * @param moneyness: query point (m = S/K)
     * @param maturity: query point (tau = T-t)
     * @param context: algorithm-specific scratch space
     * @return interpolated IV value
     */
    double (*interpolate_2d)(const IVSurface *surface,
                             double moneyness,
                             double maturity,
                             InterpContext context);

    // ---------- 4D Interpolation (Price Tables) ----------

    /**
     * Interpolate on 4D grid (moneyness, maturity, volatility, rate)
     *
     * @param table: option price table
     * @param moneyness: query point
     * @param maturity: query point
     * @param volatility: query point
     * @param rate: query point
     * @param context: algorithm-specific scratch space
     * @return interpolated option price
     */
    double (*interpolate_4d)(const OptionPriceTable *table,
                             double moneyness,
                             double maturity,
                             double volatility,
                             double rate,
                             InterpContext context);

    /**
     * Interpolate on 5D grid (adds dividend dimension)
     * Optional: can be NULL if not supported
     */
    double (*interpolate_5d)(const OptionPriceTable *table,
                             double moneyness,
                             double maturity,
                             double volatility,
                             double rate,
                             double dividend,
                             InterpContext context);

    // ---------- Context Management ----------

    /**
     * Create algorithm-specific context (scratch space)
     * Called once when strategy is initialized
     *
     * @param dimensions: number of dimensions (2, 4, or 5)
     * @param grid_sizes: array of grid sizes for each dimension
     * @return opaque context pointer (owned by caller), or NULL if no context needed
     */
    InterpContext (*create_context)(size_t dimensions,
                                     const size_t *grid_sizes);

    /**
     * Destroy context and free resources
     * Can be NULL if no context cleanup needed
     *
     * @param context: context created by create_context()
     */
    void (*destroy_context)(InterpContext context);

    // ---------- Optional: Pre-computation ----------

    /**
     * Optional: Pre-compute coefficients or data structures
     * For cubic splines: compute spline coefficients
     * For linear: no-op (can be NULL)
     *
     * @param grid_data: raw grid data (strategy-specific format)
     * @param context: context to store pre-computed data
     * @return 0 on success, -1 on error
     */
    int (*precompute)(const void *grid_data, InterpContext context);
};

#endif // IVCALC_INTERP_STRATEGY_H
