// SPDX-License-Identifier: MIT
/**
 * @file price_table_extraction.hpp
 * @brief Utility for extracting batch results into 4D price arrays
 */

#pragma once

#include "mango/option/american_option_batch.hpp"
#include "mango/option/table/price_table_grid.hpp"

namespace mango {

/**
 * @brief Extract prices from batch results into 4D array
 *
 * Interpolates spatial solutions to moneyness grid using cubic splines.
 * Shared logic for all price table solvers.
 *
 * @param batch_result Batch solve results containing surfaces
 * @param prices_4d Output array (Nm × Nt × Nv × Nr), must be pre-sized
 * @param grid Grid specification with parameter arrays
 * @param K_ref Reference strike for denormalization
 *
 * @pre Batch results must contain recorded snapshots. Snapshots are registered
 *      via solver.set_snapshot_times() before calling solve().
 */
void extract_batch_results_to_4d(
    const BatchAmericanOptionResult& batch_result,
    std::span<double> prices_4d,
    const PriceTableGrid& grid,
    double K_ref);

} // namespace mango
