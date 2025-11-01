#ifndef MANGO_IV_SURFACE_H
#define MANGO_IV_SURFACE_H

#include <stddef.h>
#include <time.h>
#include "interp_strategy.h"

/**
 * @file iv_surface.h
 * @brief 2D Implied Volatility Surface with pluggable interpolation
 *
 * Manages a 2D grid of implied volatility values indexed by:
 * - Moneyness (m = S/K, spot/strike ratio)
 * - Maturity (τ = T - t, time to expiration)
 *
 * Features:
 * - Fast 2D interpolation (<100ns typical)
 * - Small memory footprint (12KB for 50×30 grid)
 * - Runtime interpolation strategy selection
 * - Binary save/load for persistence
 *
 * Typical Usage:
 *   // Create surface with default (cubic) interpolation
 *   IVSurface *surface = iv_surface_create(moneyness, n_m, maturity, n_tau);
 *
 *   // Set IV data
 *   iv_surface_set(surface, iv_data);
 *
 *   // Query interpolated IV
 *   double iv = iv_surface_interpolate(surface, 1.05, 0.25);
 *
 *   // Save for later
 *   iv_surface_save(surface, "spx_iv.bin");
 *
 *   // Cleanup
 *   iv_surface_destroy(surface);
 */

/**
 * IV Surface data structure
 *
 * Memory layout: row-major (moneyness varies fastest)
 *   iv_surface[i_tau * n_moneyness + i_m]
 */
typedef struct IVSurface {
    // Grid definition
    size_t n_moneyness;        // Number of moneyness points
    size_t n_maturity;          // Number of maturity points
    double *moneyness_grid;     // Moneyness values (m = S/K), sorted
    double *maturity_grid;      // Maturity values (τ = T - t), sorted

    // IV data (row-major: moneyness varies fastest)
    double *iv_surface;         // n_moneyness × n_maturity values

    // Metadata
    char underlying[32];        // Underlying symbol (e.g., "SPX")
    time_t last_update;         // Timestamp of last update

    // Interpolation strategy (dependency injection)
    const InterpolationStrategy *strategy;  // Strategy vtable (not owned)
    InterpContext interp_context;           // Algorithm-specific context (owned)
} IVSurface;

// ---------- Creation and Destruction ----------

/**
 * Create IV surface with specified interpolation strategy
 *
 * @param moneyness: array of moneyness values (must be sorted)
 * @param n_m: number of moneyness points
 * @param maturity: array of maturity values (must be sorted)
 * @param n_tau: number of maturity points
 * @param strategy: interpolation strategy (e.g., &INTERP_CUBIC)
 *                  If NULL, defaults to cubic
 * @return newly created surface, or NULL on error
 *
 * Note: Takes ownership of grid arrays (caller should not free them)
 */
IVSurface* iv_surface_create_with_strategy(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const InterpolationStrategy *strategy);

/**
 * Create IV surface with default (cubic) interpolation
 */
IVSurface* iv_surface_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau);

/**
 * Destroy IV surface and free all resources
 */
void iv_surface_destroy(IVSurface *surface);

// ---------- Data Access ----------

/**
 * Set IV values for the entire surface
 *
 * @param surface: IV surface
 * @param iv_data: array of IV values (n_m × n_tau), row-major layout
 * @return 0 on success, -1 on error
 *
 * Note: Copies data, caller retains ownership of iv_data
 */
int iv_surface_set(IVSurface *surface, const double *iv_data);

/**
 * Get IV value at specific grid point
 *
 * @param surface: IV surface
 * @param i_m: moneyness index
 * @param i_tau: maturity index
 * @return IV value at grid point, or NaN if indices out of bounds
 */
double iv_surface_get(const IVSurface *surface, size_t i_m, size_t i_tau);

/**
 * Set IV value at specific grid point
 *
 * @param surface: IV surface
 * @param i_m: moneyness index
 * @param i_tau: maturity index
 * @param iv: IV value to set
 * @return 0 on success, -1 on error
 */
int iv_surface_set_point(IVSurface *surface, size_t i_m, size_t i_tau, double iv);

// ---------- Interpolation ----------

/**
 * Interpolate IV at arbitrary point using injected strategy
 *
 * @param surface: IV surface
 * @param moneyness: query moneyness (m = S/K)
 * @param maturity: query maturity (τ = T - t)
 * @return interpolated IV value
 *
 * Note: If query is outside grid bounds, clamps to boundary
 */
double iv_surface_interpolate(const IVSurface *surface,
                               double moneyness, double maturity);

/**
 * Change interpolation strategy at runtime
 *
 * Destroys old context, creates new one with specified strategy.
 *
 * @param surface: IV surface
 * @param strategy: new interpolation strategy
 * @return 0 on success, -1 on error
 */
int iv_surface_set_strategy(IVSurface *surface,
                             const InterpolationStrategy *strategy);

// ---------- Metadata ----------

/**
 * Set underlying symbol
 */
void iv_surface_set_underlying(IVSurface *surface, const char *underlying);

/**
 * Get underlying symbol
 */
const char* iv_surface_get_underlying(const IVSurface *surface);

/**
 * Update timestamp to current time
 */
void iv_surface_touch(IVSurface *surface);

// ---------- I/O ----------

/**
 * Save IV surface to binary file
 *
 * File format:
 *   - Header (128 bytes): magic, version, dimensions, metadata
 *   - Grid arrays: moneyness[], maturity[]
 *   - IV data: iv_surface[]
 *   - Footer: checksum
 *
 * @param surface: IV surface to save
 * @param filename: output file path
 * @return 0 on success, -1 on error
 */
int iv_surface_save(const IVSurface *surface, const char *filename);

/**
 * Load IV surface from binary file
 *
 * @param filename: input file path
 * @return loaded surface, or NULL on error
 *
 * Note: Always loads with default (cubic) strategy
 *       Call iv_surface_set_strategy() to change if needed
 */
IVSurface* iv_surface_load(const char *filename);

#endif // MANGO_IV_SURFACE_H
