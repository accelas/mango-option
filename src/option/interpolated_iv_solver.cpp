// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver.cpp
 * @brief Explicit template instantiations for InterpolatedIVSolver
 *
 * The implementation is in the header (interpolated_iv_solver.hpp) since
 * InterpolatedIVSolver is now a class template. This file provides explicit
 * instantiations for common surface types to improve compile times.
 */

#include "src/option/interpolated_iv_solver.hpp"
#include "src/option/table/american_price_surface.hpp"

namespace mango {

// Explicit instantiation for the standard AmericanPriceSurface case
template class InterpolatedIVSolver<AmericanPriceSurface>;

} // namespace mango
