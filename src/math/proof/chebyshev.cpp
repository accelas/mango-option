// SPDX-License-Identifier: MIT
#include "mango/math/proof/chebyshev.hpp"
#include <algorithm>
#include <array>
#include <cmath>
namespace mango::detail::proof {
namespace {
struct Prepared {
    std::span<const std::size_t> shape;
    std::vector<Interval> coefficients;
    std::array<std::vector<bool>, 4> active;
};
std::expected<Prepared, InputError> prepare(const ChebyshevPolynomial &polynomial,
                                            std::optional<std::size_t> axis) {
    if (polynomial.shape().empty() || polynomial.shape().size() > 4 ||
        polynomial.domain().size() != polynomial.shape().size())
        return std::unexpected(InputError::Shape);
    std::size_t count = 1;
    for (auto width : polynomial.shape()) {
        if (width == 0 || width > 257 || count > (1u << 20) / width)
            return std::unexpected(InputError::Shape);
        count *= width;
    }
    if (polynomial.coefficients().size() != count)
        return std::unexpected(InputError::Shape);
    if (axis && *axis >= polynomial.shape().size())
        return std::unexpected(InputError::Axis);
    Prepared result{polynomial.shape(), {}, {}};
    for (double c : polynomial.coefficients())
        result.coefficients.emplace_back(c);
    if (axis) {
        const auto width = result.shape[*axis];
        std::size_t stride = 1;
        for (std::size_t d = *axis + 1; d < result.shape.size(); ++d)
            stride *= result.shape[d];
        auto derivative = result.coefficients;
        const auto [lo, hi] = polynomial.domain()[*axis];
        const auto scale = Interval(2) / (Interval(hi) - Interval(lo));
        for (std::size_t base = 0; base < derivative.size(); ++base) {
            if ((base / stride) % width != 0)
                continue;
            for (std::size_t k = 0; k < width; ++k)
                derivative[base + k * stride] = Interval();
            for (std::size_t k = width - 1; k-- > 0;) {
                derivative[base + k * stride] = Interval(static_cast<double>(2 * (k + 1))) *
                                                result.coefficients[base + (k + 1) * stride];
                if (k + 2 < width)
                    derivative[base + k * stride] =
                        derivative[base + k * stride] + derivative[base + (k + 2) * stride];
            }
            derivative[base] = derivative[base] / Interval(2);
            for (std::size_t k = 0; k < width; ++k)
                derivative[base + k * stride] = derivative[base + k * stride] * scale;
        }
        result.coefficients = std::move(derivative);
    }
    for (std::size_t d = 0; d < result.shape.size(); ++d)
        result.active[d].resize(result.shape[d]);
    for (std::size_t i = 0; i < result.coefficients.size(); ++i) {
        if (!result.coefficients[i].finite())
            return std::unexpected(InputError::Nonfinite);
        if (result.coefficients[i].exact_zero())
            continue;
        auto index = i;
        for (std::size_t d = result.shape.size(); d-- > 0;) {
            result.active[d][index % result.shape[d]] = true;
            index /= result.shape[d];
        }
    }
    return result;
}
std::expected<Interval, InputError> bound(const Prepared &p, std::span<const Interval> box) {
    if (box.size() != p.shape.size())
        return std::unexpected(InputError::Shape);
    std::array<std::vector<Interval>, 4> basis;
    for (std::size_t d = 0; d < p.shape.size(); ++d) {
        const auto &unit = box[d];
        if (!unit.nonnegative() || !(Interval(1) - unit).nonnegative())
            return std::unexpected(InputError::Cell);
        const bool left = unit.exact_zero(), right = (unit - Interval(1)).exact_zero();
        const bool center = (unit - Interval(.5)).exact_zero();
        const bool whole = unit.lower_endpoint().exact_zero() &&
                           (unit.upper_endpoint() - Interval(1)).exact_zero();
        basis[d].resize(p.shape[d]);
        basis[d][0] = Interval(1);
        const auto x = intersection(unit * Interval(2) - Interval(1), Interval::hull(-1, 1));
        std::optional<Interval> angle;
        for (std::size_t k = 1; k < p.shape[d]; ++k) {
            if (!p.active[d][k])
                continue;
            if (whole)
                basis[d][k] = Interval::hull(-1, 1);
            else if (left || right)
                basis[d][k] = Interval(right || k % 2 == 0 ? 1 : -1);
            else if (center)
                basis[d][k] = Interval(k % 2 ? 0 : (k % 4 == 0 ? 1 : -1));
            else if (k == 1)
                basis[d][k] = x;
            else {
                if (!angle)
                    angle = acos(x);
                basis[d][k] = cos(Interval(static_cast<double>(k)) * (*angle));
            }
        }
    }
    Interval result;
    for (std::size_t i = 0; i < p.coefficients.size(); ++i) {
        if (p.coefficients[i].exact_zero())
            continue;
        auto index = i;
        Interval term = p.coefficients[i];
        for (std::size_t d = p.shape.size(); d-- > 0;) {
            term = term * basis[d][index % p.shape[d]];
            index /= p.shape[d];
        }
        result = result + term;
    }
    if (!result.finite())
        return std::unexpected(InputError::Nonfinite);
    return result;
}
std::expected<Interval, InputError> bound(const Prepared &p,
                                          std::span<const std::pair<double, double>> box) {
    std::vector<Interval> retained;
    for (const auto &[lo, hi] : box) {
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
            return std::unexpected(InputError::Cell);
        retained.push_back(Interval::hull(lo, hi));
    }
    return bound(p, std::span<const Interval>(retained));
}
Interval choose(std::size_t n, std::size_t k) {
    if (k > n)
        return Interval();
    k = std::min(k, n - k);
    Interval result(1);
    for (std::size_t i = 1; i <= k; ++i)
        result =
            result * Interval(static_cast<double>(n - k + i)) / Interval(static_cast<double>(i));
    return result;
}
std::expected<BernsteinTensor, InputError> as_bernstein(const Prepared &p) {
    std::vector<std::size_t> degrees(p.shape.size()), strides(p.shape.size(), 1);
    std::size_t total = 1;
    for (std::size_t d = 0; d < p.shape.size(); ++d) {
        for (std::size_t k = 0; k < p.shape[d]; ++k)
            if (p.active[d][k])
                degrees[d] = k;
        // Remove only identically zero modes in proof scratch. The explicit
        // polynomial payload, requested shape and physical domain do not change.
        if (degrees[d] > 64 || total > 4096 / (degrees[d] + 1))
            return std::unexpected(InputError::Shape);
        total *= degrees[d] + 1;
    }
    for (std::size_t d = p.shape.size() - 1; d > 0; --d)
        strides[d - 1] = strides[d] * (degrees[d] + 1);
    std::vector<Interval> coefficients(total);
    for (std::size_t i = 0; i < total; ++i) {
        std::size_t original = 0;
        for (std::size_t d = 0; d < p.shape.size(); ++d)
            original = original * p.shape[d] + (i / strides[d]) % (degrees[d] + 1);
        coefficients[i] = p.coefficients[original];
    }
    for (std::size_t d = 0; d < p.shape.size(); ++d) {
        const auto n = degrees[d], width = n + 1, stride = strides[d];
        std::vector<std::vector<Interval>> matrix(width, std::vector<Interval>(width));
        for (std::size_t k = 0; k <= n; ++k) {
            std::vector<Interval> row(k + 1);
            if (k == 0) {
                std::fill(matrix[0].begin(), matrix[0].end(), Interval(1));
                continue;
            }
            // T_k(2u-1) in degree-k Bernstein form. The binomial ratio is
            // evaluated with enclosures, never in binary64 monomial arithmetic.
            for (std::size_t j = 0; j <= k; ++j)
                row[j] = Interval((k - j) % 2 ? -1 : 1) * choose(2 * k, 2 * j) / choose(k, j);
            for (std::size_t degree = k + 1; degree <= n; ++degree) {
                std::vector<Interval> elevated(degree + 1);
                elevated.front() = row.front();
                elevated.back() = row.back();
                for (std::size_t j = 1; j < degree; ++j)
                    elevated[j] = (Interval(static_cast<double>(j)) * row[j - 1] +
                                   Interval(static_cast<double>(degree - j)) * row[j]) /
                                  Interval(static_cast<double>(degree));
                row = std::move(elevated);
            }
            matrix[k] = std::move(row);
        }
        auto transformed = coefficients;
        for (std::size_t base = 0; base < total; ++base) {
            if ((base / stride) % width)
                continue;
            for (std::size_t j = 0; j < width; ++j) {
                Interval value;
                for (std::size_t k = 0; k < width; ++k)
                    value = value + coefficients[base + k * stride] * matrix[k][j];
                transformed[base + j * stride] = std::move(value);
            }
        }
        coefficients = std::move(transformed);
    }
    return BernsteinTensor::create(std::move(degrees), std::move(coefficients));
}
} // namespace
std::expected<Interval, InputError>
enclose_chebyshev(const ChebyshevPolynomial &polynomial,
                  std::span<const std::pair<double, double>> box, std::optional<std::size_t> axis) {
    auto p = prepare(polynomial, axis);
    if (!p)
        return std::unexpected(p.error());
    return bound(*p, box);
}
std::expected<Interval, InputError>
enclose_chebyshev_physical(const ChebyshevPolynomial &polynomial,
                           std::span<const Interval> coordinates, std::optional<std::size_t> axis) {
    auto prepared = prepare(polynomial, axis);
    if (!prepared)
        return std::unexpected(prepared.error());
    if (coordinates.size() != polynomial.shape().size())
        return std::unexpected(InputError::Shape);
    for (const auto &coordinate : coordinates)
        if (!coordinate.finite())
            return std::unexpected(InputError::Cell);
    std::vector<Interval> units;
    bool partial_crosses_clamp = false;
    for (std::size_t d = 0; d < coordinates.size(); ++d) {
        const auto &x = coordinates[d];
        const auto [a, b] = polynomial.domain()[d];
        const Interval lo(a), hi(b);
        const bool below = (x.upper_endpoint() - lo).strictly_negative();
        const bool above = (x.lower_endpoint() - hi).strictly_positive();
        if (axis == d && (below || above))
            return Interval(0);
        if (axis == d && ((x.lower_endpoint() - lo).strictly_negative() ||
                          (x.upper_endpoint() - hi).strictly_positive()))
            partial_crosses_clamp = true;
        if ((x.upper_endpoint() - lo).nonpositive())
            units.emplace_back(0);
        else if ((x.lower_endpoint() - hi).nonnegative())
            units.emplace_back(1);
        else {
            auto clamped = lo + positive_part(x - lo);
            clamped = hi - positive_part(hi - clamped);
            units.push_back(intersection((clamped - lo) / (hi - lo), Interval::hull(0, 1)));
        }
    }
    auto result = bound(*prepared, std::span<const Interval>(units));
    if (result && partial_crosses_clamp)
        *result = hull(*result, Interval(0));
    return result;
}
std::expected<BernsteinTensor, InputError>
chebyshev_to_bernstein(const ChebyshevPolynomial &polynomial, std::optional<std::size_t> axis) {
    auto p = prepare(polynomial, axis);
    if (!p)
        return std::unexpected(p.error());
    return as_bernstein(*p);
}
std::expected<ProofResult, InputError>
prove_chebyshev_partial(const ChebyshevPolynomial &polynomial, std::size_t axis,
                        ProofBudget budget) {
    auto p = prepare(polynomial, axis);
    if (!p)
        return std::unexpected(p.error());
    ProofResult result;
    if (!budget.max_nodes) {
        result.reason = StopReason::NodeBudget;
        return result;
    }
    UnitBox whole(p->shape.size(), {0, 1});
    auto value = bound(*p, whole);
    result.nodes = 1;
    if (!value) {
        result.reason = StopReason::Arithmetic;
        return result;
    }
    if (value->nonnegative()) {
        result.status = ProofStatus::Certified;
        return result;
    }
    if (value->strictly_negative()) {
        result.status = ProofStatus::NegativeWitness;
        result.witness_box = whole;
        result.witness_bound = *value;
        return result;
    }
    if (budget.max_nodes == 1) {
        result.reason = StopReason::NodeBudget;
        return result;
    }
    if (auto bernstein = as_bernstein(*p)) {
        auto proof = prove_nonnegative(*bernstein, {budget.max_nodes - 1, budget.max_depth});
        ++proof.nodes;
        return proof;
    }
    struct Node {
        UnitBox box;
        std::size_t depth;
    };
    std::vector<Node> pending{{whole, 0}};
    result.nodes = 0;
    bool unresolved = false;
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    while (!pending.empty()) {
        if (result.nodes == budget.max_nodes) {
            result.reason = StopReason::NodeBudget;
            return result;
        }
        auto node = std::move(pending.back());
        pending.pop_back();
        auto enclosure = node.depth == 0 ? value : bound(*p, node.box);
        ++result.nodes;
        result.deepest = std::max(result.deepest, node.depth);
        if (!enclosure) {
            unresolved = true;
            result.reason = StopReason::Arithmetic;
            continue;
        }
        if (enclosure->nonnegative())
            continue;
        if (enclosure->strictly_negative()) {
            result.status = ProofStatus::NegativeWitness;
            result.reason = StopReason::None;
            result.witness_box = std::move(node.box);
            result.witness_bound = *enclosure;
            return result;
        }
        if (node.depth == depth_limit) {
            unresolved = true;
            result.reason = StopReason::DepthBudget;
            continue;
        }
        std::size_t split = node.depth % p->shape.size(), tried = 0;
        while (tried < p->shape.size() && std::none_of(
                                              p->active[split].begin() + 1, p->active[split].end(),
                                              [](bool x) { return x; })) {
            split = (split + 1) % p->shape.size();
            ++tried;
        }
        if (tried == p->shape.size()) {
            unresolved = true;
            result.reason = StopReason::Arithmetic;
            continue;
        }
        auto right = node.box;
        const double midpoint = (node.box[split].first + node.box[split].second) * .5;
        node.box[split].second = midpoint;
        right[split].first = midpoint;
        pending.push_back({std::move(right), node.depth + 1});
        pending.push_back({std::move(node.box), node.depth + 1});
    }
    if (!unresolved)
        result.status = ProofStatus::Certified;
    return result;
}
} // namespace mango::detail::proof
