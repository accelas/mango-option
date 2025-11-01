#ifndef MANGO_GRID_PRESETS_H
#define MANGO_GRID_PRESETS_H

#include <stddef.h>
#include "grid_generation.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file grid_presets.h
 * @brief Preset grid configurations optimized for option pricing
 *
 * Provides ready-to-use grid configurations that balance accuracy and memory usage.
 * Three presets target different use cases:
 *
 * - **FAST**: ~5K points, ~10% error, rapid prototyping/backtesting
 * - **BALANCED**: ~15K points, ~3% error, production-ready
 * - **ACCURATE**: ~30K points, ~1% error, high-accuracy applications
 *
 * All presets use non-uniform spacing concentrated at:
 * - Moneyness: ATM (m = 1.0)
 * - Maturity: Short-term (near tau = 0)
 * - Volatility: Typical trading range (σ ≈ 0.20)
 * - Rate: Uniform (low curvature)
 *
 * Usage:
 *   GridConfig config = grid_preset_get(
 *       GRID_PRESET_BALANCED,
 *       0.7, 1.3,      // moneyness
 *       0.027, 2.0,    // maturity
 *       0.10, 0.80,    // volatility
 *       0.0, 0.10);    // rate
 *
 *   GeneratedGrids grids = grid_generate_all(&config);
 *   // Use grids.moneyness, grids.maturity, etc.
 *   grid_free_all(&grids);
 */

/**
 * Preset grid configurations
 */
typedef enum {
    GRID_PRESET_UNIFORM,        ///< Uniform spacing (baseline, 112K points)
    GRID_PRESET_LOG_STANDARD,   ///< Log-spaced moneyness (current default, 112K)
    GRID_PRESET_ADAPTIVE_FAST,  ///< Fast: ~5K points, ~10% error
    GRID_PRESET_ADAPTIVE_BALANCED, ///< Balanced: ~15K points, ~3% error
    GRID_PRESET_ADAPTIVE_ACCURATE, ///< Accurate: ~30K points, ~1% error
    GRID_PRESET_CUSTOM          ///< User-defined configuration
} GridPreset;

/**
 * Grid configuration for all dimensions
 */
typedef struct {
    GridSpec moneyness;
    GridSpec maturity;
    GridSpec volatility;
    GridSpec rate;
    GridSpec dividend;  ///< Only used if n_dividend > 0
} GridConfig;

/**
 * Generated grids for all dimensions
 */
typedef struct {
    double *moneyness;
    size_t n_moneyness;
    double *maturity;
    size_t n_maturity;
    double *volatility;
    size_t n_volatility;
    double *rate;
    size_t n_rate;
    double *dividend;
    size_t n_dividend;
    size_t total_points;  ///< Product of all dimensions
} GeneratedGrids;

/**
 * Get preset grid configuration
 *
 * @param preset Preset type
 * @param m_min Minimum moneyness
 * @param m_max Maximum moneyness
 * @param tau_min Minimum maturity
 * @param tau_max Maximum maturity
 * @param sigma_min Minimum volatility
 * @param sigma_max Maximum volatility
 * @param r_min Minimum rate
 * @param r_max Maximum rate
 * @return Grid configuration for all dimensions
 *
 * For 4D tables (no dividend), pass q_min = q_max = 0.0.
 * For 5D tables, specify dividend range.
 *
 * Example:
 *   GridConfig config = grid_preset_get(
 *       GRID_PRESET_BALANCED,
 *       0.7, 1.3,      // moneyness
 *       0.027, 2.0,    // maturity
 *       0.10, 0.80,    // volatility
 *       0.0, 0.10,     // rate
 *       0.0, 0.0);     // no dividend
 */
GridConfig grid_preset_get(
    GridPreset preset,
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    double q_min, double q_max);

/**
 * Generate all grids from configuration
 *
 * @param config Grid configuration
 * @return Generated grids (caller must free with grid_free_all)
 *
 * Allocates and generates grid points for all dimensions.
 * Use grid_free_all() to free all allocated memory.
 */
GeneratedGrids grid_generate_all(const GridConfig *config);

/**
 * Free all generated grids
 *
 * @param grids Generated grids structure
 *
 * Frees all allocated grid arrays and resets pointers to NULL.
 */
void grid_free_all(GeneratedGrids *grids);

/**
 * Get preset name as string
 *
 * @param preset Preset type
 * @return Human-readable name
 */
const char* grid_preset_name(GridPreset preset);

/**
 * Get preset description
 *
 * @param preset Preset type
 * @return Description of preset characteristics
 */
const char* grid_preset_description(GridPreset preset);

#ifdef __cplusplus
}
#endif

#endif // MANGO_GRID_PRESETS_H
