// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/proof/bspline_cell.hpp"
namespace mango::detail::proof {
struct BSplineBoxEnclosure {
    Interval value{0};
    std::vector<Interval> partials;
    std::size_t cells = 0;
    StopReason reason = StopReason::Arithmetic;
};
/// Enclose a cubic tensor spline and selected first partials over a raw
/// coordinate box, including constant extension outside the stored knots.
/// The caller must verify actual evaluator grid clamps equal knot endpoints.
/// Partial outputs follow derivative_axes order; a clamped axis has slope0.
/// Only reason==None makes value/partials usable as rigorous enclosures.
BSplineBoxEnclosure enclose_cubic_bspline_box(std::span<const std::span<const double>> knots,
                                              std::span<const double> coefficients,
                                              std::span<const Interval> coordinates,
                                              std::span<const std::size_t> derivative_axes,
                                              std::size_t max_cells);
} // namespace mango::detail::proof
