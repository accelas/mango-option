/**
 * @file bspline_utils.hpp
 * @brief B-spline utility functions (forwarding header)
 *
 * This header provides backward compatibility by forwarding to the
 * generic math implementations in src/math/bspline_basis.hpp.
 *
 * **Migration note:** New code should include src/math/bspline_basis.hpp directly.
 * This header exists only for compatibility with existing bspline code.
 */

#pragma once

// Forward all B-spline basis utilities from generic math module
#include "src/math/bspline_basis.hpp"

namespace mango {

// All utilities (clamped_knots_cubic, find_span_cubic, cubic_basis_nonuniform,
// cubic_basis_derivative_nonuniform, clamp_query) are now available via
// the bspline_basis.hpp include above.

}  // namespace mango
