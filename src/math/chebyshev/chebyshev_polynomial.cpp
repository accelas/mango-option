// SPDX-License-Identifier: MIT
#include "mango/math/chebyshev/chebyshev_polynomial.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace mango::detail {
namespace {
template <std::size_t Axis, std::size_t Rank>
double contract(const double *coefficients, const std::array<std::vector<double>, 4> &weights,
                const std::array<std::size_t, 4> &strides) {
    double result = 0;
    for (std::size_t i = 0; i < weights[Axis].size(); ++i) {
        if constexpr (Axis + 1 == Rank)
            result += coefficients[i] * weights[Axis][i];
        else
            result += weights[Axis][i] *
                      contract<Axis + 1, Rank>(coefficients + i * strides[Axis], weights, strides);
    }
    return result;
}
} // namespace
std::expected<ChebyshevPolynomial, PolynomialError>
ChebyshevPolynomial::from_coefficients(std::vector<std::size_t> shape, std::vector<Bounds> domain,
                                       std::vector<double> coefficients) {
    if (shape.empty() || shape.size() > 4 || shape.size() != domain.size())
        return std::unexpected(PolynomialError::Shape);
    std::size_t total = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] == 0 || shape[axis] > 257 || total > (1u << 20) / shape[axis])
            return std::unexpected(PolynomialError::Capacity);
        total *= shape[axis];
        const auto [lo, hi] = domain[axis];
        const double width = hi - lo;
        if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi) || !std::isfinite(width) ||
            !std::isfinite(2 / width))
            return std::unexpected(PolynomialError::Domain);
    }
    if (coefficients.size() != total)
        return std::unexpected(PolynomialError::Shape);
    for (double value : coefficients)
        if (!std::isfinite(value))
            return std::unexpected(PolynomialError::Nonfinite);
    ChebyshevPolynomial result;
    result.shape_ = std::move(shape);
    result.domain_ = std::move(domain);
    result.coefficients_ = std::move(coefficients);
    return result;
}
std::expected<ChebyshevPolynomial, PolynomialError>
ChebyshevPolynomial::from_cgl_values(std::vector<std::size_t> shape, std::vector<Bounds> domain,
                                     std::span<const double> values) {
    if (values.size() > (1u << 20))
        return std::unexpected(PolynomialError::Capacity);
    auto result = from_coefficients(std::move(shape), std::move(domain),
                                    std::vector<double>(values.begin(), values.end()));
    if (!result)
        return result;
    std::size_t stride = result->coefficients_.size();
    for (const auto width : result->shape_) {
        stride /= width;
        if (width == 1)
            continue;
        const auto n = width - 1;
        std::vector<long double> matrix(width * width);
        for (std::size_t k = 0; k < width; ++k) {
            for (std::size_t j = 0; j < width; ++j) {
                long double basis;
                if (k == 0 || j == n)
                    basis = 1;
                else if (j == 0)
                    basis = (k % 2) ? -1 : 1;
                else if (k == n)
                    basis = ((k + j) % 2) ? -1 : 1;
                else
                    basis =
                        ((k % 2) ? -1 : 1) * std::cos(std::numbers::pi_v<long double> * k * j / n);
                const long double input_weight = (j == 0 || j == n) ? .5L : 1.L;
                const long double output_weight = (k == 0 || k == n) ? .5L : 1.L;
                matrix[k * width + j] = basis * input_weight * output_weight * 2 / n;
            }
        }
        auto transformed = result->coefficients_;
        for (std::size_t base = 0; base < transformed.size(); ++base) {
            if ((base / stride) % width != 0)
                continue;
            bool constant = true;
            for (std::size_t j = 1; j < width; ++j)
                constant = constant &&
                           result->coefficients_[base + j * stride] == result->coefficients_[base];
            for (std::size_t k = 0; k < width; ++k) {
                long double sum = 0;
                if (constant)
                    sum = k == 0 ? result->coefficients_[base] : 0;
                else
                    for (std::size_t j = 0; j < width; ++j)
                        sum += matrix[k * width + j] * result->coefficients_[base + j * stride];
                const double coefficient = static_cast<double>(sum);
                if (!std::isfinite(coefficient))
                    return std::unexpected(PolynomialError::Nonfinite);
                transformed[base + k * stride] = coefficient;
            }
        }
        result->coefficients_ = std::move(transformed);
    }
    return result;
}
double ChebyshevPolynomial::eval(std::span<const double> query) const {
    return evaluate(query, 0, 0);
}
double ChebyshevPolynomial::partial(std::size_t axis, std::span<const double> query) const {
    return evaluate(query, axis, 1);
}
double ChebyshevPolynomial::eval_second_partial(std::size_t axis,
                                                std::span<const double> query) const {
    return evaluate(query, axis, 2);
}
double ChebyshevPolynomial::evaluate(std::span<const double> query, std::size_t axis,
                                     unsigned order) const {
    constexpr double invalid = std::numeric_limits<double>::quiet_NaN();
    if (query.size() != shape_.size() || axis >= shape_.size() || domain_.size() != shape_.size() ||
        coefficients_.empty())
        return invalid;
    for (double x : query)
        if (!std::isfinite(x))
            return invalid;
    if (order && (query[axis] < domain_[axis].first || query[axis] > domain_[axis].second))
        return 0;
    std::array<std::vector<double>, 4> weights;
    std::array<std::size_t, 4> strides{};
    std::size_t stride = 1;
    for (std::size_t d = shape_.size(); d-- > 0;) {
        strides[d] = stride;
        stride *= shape_[d];
        const auto [lo, hi] = domain_[d];
        const double x = 2 * ((std::clamp(query[d], lo, hi) - lo) / (hi - lo)) - 1;
        const double scale = 2 / (hi - lo);
        auto &w = weights[d];
        w.resize(shape_[d]);
        const unsigned derivative = (d == axis) ? order : 0;
        double tm = 1, t = x, dm = 0, dt = 1, ddm = 0, ddt = 0;
        w[0] = derivative ? 0 : 1;
        if (w.size() > 1)
            w[1] = derivative == 0 ? t : derivative == 1 ? scale : 0;
        for (std::size_t k = 2; k < w.size(); ++k) {
            const double next = 2 * x * t - tm;
            const double next_d = 2 * t + 2 * x * dt - dm;
            const double next_dd = 4 * dt + 2 * x * ddt - ddm;
            w[k] = derivative == 0   ? next
                   : derivative == 1 ? next_d * scale
                                     : next_dd * scale * scale;
            tm = t;
            t = next;
            dm = dt;
            dt = next_d;
            ddm = ddt;
            ddt = next_dd;
        }
    }
    switch (shape_.size()) {
    case 1:
        return contract<0, 1>(coefficients_.data(), weights, strides);
    case 2:
        return contract<0, 2>(coefficients_.data(), weights, strides);
    case 3:
        return contract<0, 3>(coefficients_.data(), weights, strides);
    default:
        return contract<0, 4>(coefficients_.data(), weights, strides);
    }
}
} // namespace mango::detail
