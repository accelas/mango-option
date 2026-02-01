// SPDX-License-Identifier: MIT
/**
 * @file iv_solver_interpolated.cpp
 * @brief Explicit template instantiations for IVSolverInterpolated
 *
 * The implementation is in the header (iv_solver_interpolated.hpp) since
 * IVSolverInterpolated is now a class template. This file provides explicit
 * instantiations for common surface types to improve compile times.
 */

#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/american_price_surface.hpp"

namespace mango {

// Explicit instantiation for the standard AmericanPriceSurface case
template class IVSolverInterpolated<AmericanPriceSurface>;

} // namespace mango
