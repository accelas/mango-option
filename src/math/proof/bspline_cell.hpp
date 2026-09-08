// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/proof/bernstein.hpp"
#include <optional>

namespace mango::detail::proof {
/// Extract one raw cubic B-spline knot cell (or its indicated partial) using
/// the exact stored binary64 knots and coefficients. No financial transform,
/// floor, EEP add-back, split, or public surface certification is performed.
/// This does not apply BSplineND's grid clamping; publication must bound
/// those coordinate branches around the extracted raw polynomials.
///
/// Each knot vector must be finite, sorted, clamped four times at either end,
/// with interior multiplicity at most three (a continuous raw function).
/// Cell spans index the original knots, and coefficients are row-major.
std::expected<BernsteinTensor, InputError>
extract_cubic_bspline_cell(std::span<const std::span<const double>> knots,
                           std::span<const double> coefficients,
                           std::span<const std::size_t> cell_spans,
                           std::optional<std::size_t> derivative_axis = std::nullopt);
} // namespace mango::detail::proof
