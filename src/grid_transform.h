#ifndef MANGO_GRID_TRANSFORM_H
#define MANGO_GRID_TRANSFORM_H

#include "grid_presets.h"
#include "price_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file grid_transform.h
 * @brief Coordinate transformation for grid arrays
 *
 * Transforms grid arrays between raw coordinates and transformed coordinates
 * (log-sqrt, log-variance) as required by OptionPriceTable.
 *
 * When using COORD_LOG_SQRT or COORD_LOG_VARIANCE with price_table_create_ex(),
 * grids must be pre-transformed before passing to the table.
 */

/**
 * Transform grids to specified coordinate system (in-place)
 *
 * @param grids Generated grids (modified in-place)
 * @param coord_system Target coordinate system
 *
 * Transforms moneyness and maturity grids according to the coordinate system:
 * - COORD_RAW: No transformation (no-op)
 * - COORD_LOG_SQRT: m → log(m), tau → sqrt(tau)
 * - COORD_LOG_VARIANCE: m → log(m), tau → sigma²*tau
 *
 * Note: Volatility and rate grids are never transformed.
 *
 * Example:
 *   GridConfig config = grid_preset_get(GRID_PRESET_BALANCED, ...);
 *   GeneratedGrids grids = grid_generate_all(&config);
 *
 *   // Transform for LOG_SQRT coordinate system
 *   grid_transform_coordinates(&grids, COORD_LOG_SQRT);
 *
 *   // Now grids are ready for price_table_create_ex with COORD_LOG_SQRT
 *   OptionPriceTable *table = price_table_create_ex(
 *       grids.moneyness, grids.n_moneyness,
 *       grids.maturity, grids.n_maturity,
 *       ..., COORD_LOG_SQRT, ...);
 */
void grid_transform_coordinates(GeneratedGrids *grids, CoordinateSystem coord_system);

#ifdef __cplusplus
}
#endif

#endif // MANGO_GRID_TRANSFORM_H
