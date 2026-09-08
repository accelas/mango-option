// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/chebyshev/chebyshev_polynomial.hpp"
#include "mango/math/proof/bernstein.hpp"
#include <optional>
namespace mango::detail::proof {
using UnitBox = std::vector<std::pair<double, double>>;
/// Bounds for the explicit modal polynomial (or its physical-axis partial),
/// restricted to normalized unit coordinates. This never accepts the old
/// barycentric storage/evaluator as if it were the same polynomial.
std::expected<Interval, InputError>
enclose_chebyshev(const ChebyshevPolynomial &polynomial,
                  std::span<const std::pair<double, double>> unit_box,
                  std::optional<std::size_t> derivative_axis = std::nullopt);
std::expected<BernsteinTensor, InputError>
chebyshev_to_bernstein(const ChebyshevPolynomial &polynomial,
                       std::optional<std::size_t> derivative_axis = std::nullopt);
std::expected<ProofResult, InputError>
prove_chebyshev_partial(const ChebyshevPolynomial &polynomial, std::size_t axis,
                        ProofBudget budget = {});
} // namespace mango::detail::proof
