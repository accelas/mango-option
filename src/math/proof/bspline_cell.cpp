// SPDX-License-Identifier: MIT
#include "mango/math/proof/bspline_cell.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mango::detail::proof {
namespace {
using Basis = std::vector<std::vector<Interval>>;

// Multiply a degree-(p-1) Bernstein polynomial by the linear polynomial
// with endpoint values a,b. Its degree-p coefficient j is
// (1-j/p)*a*c[j] + (j/p)*b*c[j-1], with absent terms equal to zero.
void add_linear_product(std::vector<Interval> &result, const std::vector<Interval> &coefficients,
                        const Interval &a, const Interval &b, std::size_t p) {
    for (std::size_t j = 0; j <= p; ++j) {
        if (j < p)
            result[j] = result[j] + coefficients[j] * a *
                                        (Interval(static_cast<double>(p - j)) /
                                         Interval(static_cast<double>(p)));
        if (j > 0)
            result[j] = result[j] +
                        coefficients[j - 1] * b *
                            (Interval(static_cast<double>(j)) / Interval(static_cast<double>(p)));
    }
}

// Cox-de Boor recurrence applied to polynomials on an entire knot cell,
// rather than to sampled point values. Only the p+1 active basis functions
// are materialized, in increasing global coefficient index order.
Basis cell_basis(std::span<const double> knots, std::size_t span, std::size_t degree) {
    // A fully clamped cell is already in the Bernstein basis. Preserve its
    // original stored coefficients exactly, including their signs and zeros.
    bool bezier = true;
    for (std::size_t i = 0; i <= degree; ++i)
        bezier = bezier && knots[span - i] == knots[span] && knots[span + 1 + i] == knots[span + 1];
    if (bezier) {
        Basis identity(degree + 1, std::vector<Interval>(degree + 1));
        for (std::size_t i = 0; i <= degree; ++i)
            identity[i][i] = Interval(1);
        return identity;
    }
    Basis previous{{Interval(1)}};
    const Interval a(knots[span]), b(knots[span + 1]);
    for (std::size_t p = 1; p <= degree; ++p) {
        Basis current(p + 1, std::vector<Interval>(p + 1));
        for (std::size_t i = 0; i <= p; ++i) {
            const auto index = span - p + i;
            if (i > 0) {
                const auto denominator = Interval(knots[index + p]) - Interval(knots[index]);
                if (!denominator.exact_zero()) {
                    add_linear_product(current[i], previous[i - 1],
                                       (a - Interval(knots[index])) / denominator,
                                       (b - Interval(knots[index])) / denominator, p);
                }
            }
            if (i < p) {
                const auto denominator =
                    Interval(knots[index + p + 1]) - Interval(knots[index + 1]);
                if (!denominator.exact_zero()) {
                    add_linear_product(current[i], previous[i],
                                       (Interval(knots[index + p + 1]) - a) / denominator,
                                       (Interval(knots[index + p + 1]) - b) / denominator, p);
                }
            }
        }
        previous = std::move(current);
    }
    return previous;
}
} // namespace

std::expected<BernsteinTensor, InputError> extract_cubic_bspline_cell(
    std::span<const std::span<const double>> knots, std::span<const double> coefficients,
    std::span<const std::size_t> cell_spans, std::optional<std::size_t> derivative_axis) {
    const auto rank = knots.size();
    if (rank == 0 || rank > 4 || cell_spans.size() != rank)
        return std::unexpected(InputError::Shape);
    if (derivative_axis && *derivative_axis >= rank)
        return std::unexpected(InputError::Axis);
    std::vector<std::size_t> shape(rank), degrees(rank, 3), strides(rank, 1);
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto k = knots[axis];
        if (k.size() < 8)
            return std::unexpected(InputError::Knots);
        for (std::size_t i = 0; i < k.size(); ++i)
            if (!std::isfinite(k[i]) || (i && k[i] < k[i - 1]))
                return std::unexpected(InputError::Knots);
        shape[axis] = k.size() - 4;
        const auto n = shape[axis];
        if (!(k.front() < k.back()))
            return std::unexpected(InputError::Knots);
        for (std::size_t i = 0; i < 4; ++i)
            if (k[i] != k.front() || k[n + i] != k.back())
                return std::unexpected(InputError::Knots);
        // More than three repeated interior knots would permit jumps. Their
        // direction needs a separate join proof, outside this continuous seam.
        for (std::size_t i = 4; i < n; ++i)
            if (!(k.front() < k[i] && k[i] < k.back()) || (i >= 7 && k[i] == k[i - 3]))
                return std::unexpected(InputError::Knots);
        if (cell_spans[axis] < 3 || cell_spans[axis] >= n ||
            !(k[cell_spans[axis]] < k[cell_spans[axis] + 1]))
            return std::unexpected(InputError::Cell);
        if (count > std::numeric_limits<std::size_t>::max() / n)
            return std::unexpected(InputError::Shape);
        count *= n;
    }
    if (count != coefficients.size())
        return std::unexpected(InputError::Shape);
    for (auto value : coefficients)
        if (!std::isfinite(value))
            return std::unexpected(InputError::Nonfinite);
    for (std::size_t axis = rank - 1; axis > 0; --axis)
        strides[axis - 1] = strides[axis] * shape[axis];
    if (derivative_axis)
        degrees[*derivative_axis] = 2;

    std::vector<Basis> bases;
    std::vector<std::size_t> local_strides(rank, 1);
    std::size_t local_count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        local_count *= degrees[axis] + 1;
        if (derivative_axis == axis) {
            const auto k = knots[axis];
            bases.push_back(cell_basis(k.subspan(1, k.size() - 2), cell_spans[axis] - 1, 2));
        } else {
            bases.push_back(cell_basis(knots[axis], cell_spans[axis], 3));
        }
    }
    for (std::size_t axis = rank - 1; axis > 0; --axis)
        local_strides[axis - 1] = local_strides[axis] * (degrees[axis] + 1);
    std::vector<Interval> local(local_count);
    for (std::size_t i = 0; i < local_count; ++i) {
        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < rank; ++axis)
            offset += (cell_spans[axis] - 3 + (i / local_strides[axis]) % (degrees[axis] + 1)) *
                      strides[axis];
        local[i] = Interval(coefficients[offset]);
        if (derivative_axis) {
            const auto axis = *derivative_axis;
            const auto index = cell_spans[axis] - 3 + (i / local_strides[axis]) % 3;
            // Differentiate original coefficients before polynomial conversion.
            // Identical stored coefficients then give exact structural zero,
            // even if other axes require inexact basis conversion.
            local[i] = Interval(3) * (Interval(coefficients[offset + strides[axis]]) - local[i]) /
                       (Interval(knots[axis][index + 4]) - Interval(knots[axis][index + 1]));
        }
    }
    for (std::size_t axis = 0; axis < rank; ++axis) {
        auto transformed = local;
        const auto width = degrees[axis] + 1, stride = local_strides[axis];
        for (std::size_t base = 0; base < local_count; ++base) {
            if ((base / stride) % width != 0)
                continue;
            for (std::size_t j = 0; j < width; ++j) {
                Interval sum;
                for (std::size_t i = 0; i < width; ++i)
                    sum = sum + local[base + i * stride] * bases[axis][i][j];
                transformed[base + j * stride] = std::move(sum);
            }
        }
        local = std::move(transformed);
    }
    return BernsteinTensor::create(std::move(degrees), std::move(local));
}
} // namespace mango::detail::proof
