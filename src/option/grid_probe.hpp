// SPDX-License-Identifier: MIT
/**
 * @file grid_probe.hpp
 * @brief Richardson-style probe for grid adequacy estimation
 *
 * Provides adaptive grid calibration by solving at Nx and 2*Nx and comparing
 * results to estimate discretization error. Used by IVSolver to automatically
 * select grid parameters that achieve target accuracy.
 */

#pragma once

#include <expected>
#include <cstddef>
#include "src/option/american_option.hpp"
#include "src/option/grid_spec_types.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/support/error_types.hpp"

namespace mango {

/**
 * Result of grid adequacy probe
 *
 * Contains the calibrated grid specification and time domain along with
 * convergence diagnostics. If converged is false, the returned grid
 * represents the best effort after max_iterations.
 */
struct ProbeResult {
    GridSpec<double> grid;        ///< Calibrated spatial grid specification
    TimeDomain time_domain;       ///< Calibrated time domain
    double estimated_error;       ///< Richardson error estimate (|P_2Nx - P_Nx|)
    size_t probe_iterations;      ///< Number of probe iterations performed
    bool converged;               ///< True if target_error was achieved
};

/**
 * Probe grid adequacy using Richardson-style extrapolation
 *
 * Iteratively solves at Nx and 2*Nx, comparing prices and deltas to estimate
 * discretization error. Doubles Nx until convergence or max_iterations.
 *
 * Convergence criteria:
 *   - Price difference <= max(target_error, 0.001 * max(|P1|, |P2|, 0.10))
 *   - Delta difference <= 0.01
 *
 * The relative tolerance floor (0.1% of price, minimum $0.10) prevents
 * over-refinement for cheap options while maintaining absolute accuracy
 * for expensive ones.
 *
 * @param params Option pricing parameters (spot, strike, maturity, vol, etc.)
 * @param target_error Target absolute price error (must be > 0)
 * @param initial_Nx Starting number of spatial grid points (default: 100)
 * @param max_iterations Maximum probe iterations before giving up (default: 3)
 * @return ProbeResult on success, ValidationError if target_error <= 0
 *
 * @note TR-BDF2 is L-stable, so no CFL constraint is needed. The Nt floor
 *       (minimum 50) is for accuracy only, not stability.
 *
 * Example:
 * @code
 * PricingParams params(...);
 * auto result = probe_grid_adequacy(params, 0.01);
 * if (result && result->converged) {
 *     // Use result->grid and result->time_domain for solving
 * }
 * @endcode
 */
std::expected<ProbeResult, ValidationError> probe_grid_adequacy(
    const PricingParams& params,
    double target_error,
    size_t initial_Nx = 100,
    size_t max_iterations = 3);

}  // namespace mango
