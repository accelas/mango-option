// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/proof/bernstein.hpp"
#include <optional>

namespace mango::detail::proof {
/// Extract one raw cubic B-spline knot cell (or its indicated partial) using
/// the exact stored binary64 knots and coefficients. No financial transform,
/// floor, EEP add-back, split, or public surface certification is performed.
/// In particular this does not certify BSplineND's grid clamping or the
/// evaluator's approximate right-endpoint snap; publication must cover them.
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
