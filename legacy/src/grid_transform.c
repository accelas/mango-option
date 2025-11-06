#include "grid_transform.h"
#include <math.h>

void grid_transform_coordinates(GeneratedGrids *grids, CoordinateSystem coord_system) {
    if (!grids) return;

    switch (coord_system) {
        case COORD_RAW:
            // No transformation needed
            break;

        case COORD_LOG_SQRT:
            // Transform moneyness: m → log(m)
            for (size_t i = 0; i < grids->n_moneyness; i++) {
                grids->moneyness[i] = log(grids->moneyness[i]);
            }

            // Transform maturity: tau → sqrt(tau)
            for (size_t i = 0; i < grids->n_maturity; i++) {
                grids->maturity[i] = sqrt(grids->maturity[i]);
            }
            break;

        case COORD_LOG_VARIANCE:
            // Transform moneyness: m → log(m)
            for (size_t i = 0; i < grids->n_moneyness; i++) {
                grids->moneyness[i] = log(grids->moneyness[i]);
            }

            // Transform maturity: tau → tau (no transform here, sigma²*tau applied during query)
            // Note: COORD_LOG_VARIANCE uses w = sigma²*tau, but grid stores just tau
            // The transformation happens in transform_query_to_grid()
            break;
    }

    // Volatility and rate grids are never transformed
}
