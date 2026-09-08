// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

namespace mango::detail {
enum class PolynomialError { Shape, Domain, Nonfinite, Capacity };

/// Explicit modal polynomial representation, independent of barycentric
/// nodal interpolation. Coefficients multiply T_0,...,T_n without half factors.
/// The stored binary64 coefficients define the mathematical polynomial.
class ChebyshevPolynomial {
  public:
    using Bounds = std::pair<double, double>;
    static std::expected<ChebyshevPolynomial, PolynomialError>
    from_coefficients(std::vector<std::size_t> shape, std::vector<Bounds> domain,
                      std::vector<double> coefficients);
    /// DCT-I conversion of values ordered on ascending CGL tensor nodes.
    /// This approximation step defines a new stored polynomial; it does not
    /// preserve the old rounded-node barycentric rational expression exactly.
    static std::expected<ChebyshevPolynomial, PolynomialError>
    from_cgl_values(std::vector<std::size_t> shape, std::vector<Bounds> domain,
                    std::span<const double> values);
    [[nodiscard]] double eval(std::span<const double> query) const;
    [[nodiscard]] double partial(std::size_t axis, std::span<const double> query) const;
    [[nodiscard]] double eval_second_partial(std::size_t axis, std::span<const double> query) const;
    std::span<const std::size_t> shape() const { return shape_; }
    std::span<const Bounds> domain() const { return domain_; }
    std::span<const double> coefficients() const { return coefficients_; }

  private:
    ChebyshevPolynomial() = default;
    double evaluate(std::span<const double> query, std::size_t axis, unsigned order) const;
    std::vector<std::size_t> shape_;
    std::vector<Bounds> domain_;
    std::vector<double> coefficients_;
};
} // namespace mango::detail
