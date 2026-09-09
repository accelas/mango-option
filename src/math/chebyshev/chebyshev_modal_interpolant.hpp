// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/chebyshev/chebyshev_interpolant.hpp"
#include "mango/math/chebyshev/chebyshev_polynomial.hpp"
#include "mango/support/error_types.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

namespace mango {

/// Opt-in modal polynomial; existing financial aliases remain nodal.
/// The stored binary64 coefficients define the polynomial used for values,
/// derivatives and proof access. Converting nodal data is an explicit DCT-I
/// approximation, not an exact reinterpretation of barycentric storage.
/// All numeric storage is owned, so copying detaches the coefficient vector.
template <size_t N>
  requires(N >= 1 && N <= 4)
class ChebyshevModalInterpolant {
public:
  [[nodiscard]] static std::expected<ChebyshevModalInterpolant,
                                     InterpolationError>
  build_from_values(std::span<const double> values, const Domain<N> &domain,
                    const std::array<size_t, N> &num_pts) {
    auto bounds = validate(values, domain, num_pts);
    if (!bounds)
      return std::unexpected(bounds.error());
    auto polynomial = detail::ChebyshevPolynomial::from_cgl_values(
        std::vector<size_t>(num_pts.begin(), num_pts.end()), std::move(*bounds),
        values);
    if (!polynomial) {
      return std::unexpected(
          InterpolationError{InterpolationErrorCode::FittingFailed});
    }
    return ChebyshevModalInterpolant(domain, num_pts, std::move(*polynomial));
  }

  /// Restore actual modal coefficients verbatim; never apply a nodal DCT.
  [[nodiscard]] static std::expected<ChebyshevModalInterpolant,
                                     InterpolationError>
  build_from_coefficients(std::span<const double> coefficients,
                          const Domain<N> &domain,
                          const std::array<size_t, N> &num_pts) {
    auto bounds = validate(coefficients, domain, num_pts);
    if (!bounds)
      return std::unexpected(bounds.error());
    auto polynomial = detail::ChebyshevPolynomial::from_coefficients(
        std::vector<size_t>(num_pts.begin(), num_pts.end()), std::move(*bounds),
        std::vector<double>(coefficients.begin(), coefficients.end()));
    if (!polynomial) {
      return std::unexpected(
          InterpolationError{InterpolationErrorCode::FittingFailed});
    }
    return ChebyshevModalInterpolant(domain, num_pts, std::move(*polynomial));
  }

  [[nodiscard]] double eval(std::array<double, N> query) const {
    return polynomial_.eval(query);
  }
  [[nodiscard]] double partial(size_t axis, std::array<double, N> query) const {
    return polynomial_.partial(axis, query);
  }
  [[nodiscard]] double eval_second_partial(size_t axis,
                                           std::array<double, N> query) const {
    return polynomial_.eval_second_partial(axis, query);
  }
  [[nodiscard]] size_t compressed_size() const {
    return polynomial_.coefficients().size();
  }
  [[nodiscard]] const Domain<N> &domain() const { return domain_; }
  [[nodiscard]] const std::array<size_t, N> &num_pts() const {
    return num_pts_;
  }
  [[nodiscard]] const detail::ChebyshevPolynomial &polynomial() const {
    return polynomial_;
  }

private:
  using Bounds = detail::ChebyshevPolynomial::Bounds;
  static std::expected<std::vector<Bounds>, InterpolationError>
  validate(std::span<const double> values, const Domain<N> &domain,
           const std::array<size_t, N> &num_pts) {
    size_t total = 1;
    std::vector<Bounds> bounds;
    bounds.reserve(N);
    for (size_t d = 0; d < N; ++d) {
      if (num_pts[d] < 2) {
        return std::unexpected(InterpolationError{
            InterpolationErrorCode::InsufficientGridPoints, num_pts[d], d});
      }
      if (num_pts[d] > 257 || total > (size_t{1} << 20) / num_pts[d]) {
        return std::unexpected(InterpolationError{
            InterpolationErrorCode::ValueSizeMismatch, num_pts[d], d});
      }
      total *= num_pts[d];
      if (std::isnan(domain.lo[d]) || std::isnan(domain.hi[d])) {
        return std::unexpected(
            InterpolationError{InterpolationErrorCode::NaNInput, 0, d});
      }
      if (!std::isfinite(domain.lo[d]) || !std::isfinite(domain.hi[d])) {
        return std::unexpected(
            InterpolationError{InterpolationErrorCode::InfInput, 0, d});
      }
      if (domain.lo[d] == domain.hi[d]) {
        return std::unexpected(
            InterpolationError{InterpolationErrorCode::ZeroWidthGrid, 0, d});
      }
      if (domain.lo[d] > domain.hi[d]) {
        return std::unexpected(
            InterpolationError{InterpolationErrorCode::GridNotSorted, 0, d});
      }
      bounds.emplace_back(domain.lo[d], domain.hi[d]);
    }
    if (values.size() != total) {
      return std::unexpected(InterpolationError{
          InterpolationErrorCode::ValueSizeMismatch, values.size()});
    }
    for (size_t i = 0; i < values.size(); ++i) {
      if (!std::isfinite(values[i])) {
        return std::unexpected(InterpolationError{
            std::isnan(values[i]) ? InterpolationErrorCode::NaNInput
                                  : InterpolationErrorCode::InfInput,
            values.size(), i});
      }
    }
    return bounds;
  }

  ChebyshevModalInterpolant(Domain<N> domain, std::array<size_t, N> num_pts,
                            detail::ChebyshevPolynomial polynomial)
      : domain_(domain), num_pts_(num_pts), polynomial_(std::move(polynomial)) {
  }
  Domain<N> domain_;
  std::array<size_t, N> num_pts_;
  detail::ChebyshevPolynomial polynomial_;
};

} // namespace mango
