// SPDX-License-Identifier: MIT
#include "mango/math/proof/bernstein.hpp"
#include <algorithm>
namespace mango::detail::proof {
std::expected<BernsteinTensor, InputError>
BernsteinTensor::create(std::vector<std::size_t> degrees, std::vector<Interval> coefficients) {
    if (degrees.empty() || degrees.size() > 4)
        return std::unexpected(InputError::Shape);
    std::size_t count = 1;
    for (auto degree : degrees) {
        if (degree > 64 || count > 4096 / (degree + 1))
            return std::unexpected(InputError::Shape);
        count *= degree + 1;
    }
    if (coefficients.size() != count)
        return std::unexpected(InputError::Shape);
    for (const auto &value : coefficients)
        if (!value.finite())
            return std::unexpected(InputError::Nonfinite);
    return BernsteinTensor(std::move(degrees), std::move(coefficients));
}

Interval BernsteinTensor::bounds() const {
    Interval result = coefficients_.front();
    for (const auto &value : coefficients_)
        result = hull(result, value);
    return result;
}
std::expected<std::pair<BernsteinTensor, BernsteinTensor>, InputError>
BernsteinTensor::split(std::size_t axis) const {
    if (axis >= degrees_.size())
        return std::unexpected(InputError::Axis);
    const auto degree = degrees_[axis];
    std::size_t stride = 1;
    for (std::size_t d = axis + 1; d < degrees_.size(); ++d)
        stride *= degrees_[d] + 1;
    auto left = coefficients_, right = coefficients_;
    std::vector<Interval> line(degree + 1);
    for (std::size_t base = 0; base < coefficients_.size(); ++base) {
        if ((base / stride) % (degree + 1) != 0)
            continue;
        for (std::size_t i = 0; i <= degree; ++i)
            line[i] = coefficients_[base + i * stride];
        for (std::size_t level = 1; level <= degree; ++level) {
            for (std::size_t i = 0; i <= degree - level; ++i)
                line[i] = (line[i] + line[i + 1]) / Interval(2);
            left[base + level * stride] = line[0];
            right[base + (degree - level) * stride] = line[degree - level];
        }
    }
    auto l = create(degrees_, std::move(left));
    auto r = create(degrees_, std::move(right));
    if (!l || !r)
        return std::unexpected(InputError::Nonfinite);
    return std::pair{std::move(*l), std::move(*r)};
}
ProofResult prove_nonnegative(const BernsteinTensor &polynomial, ProofBudget budget) {
    struct Node {
        BernsteinTensor polynomial;
        std::vector<std::pair<double, double>> box;
        std::size_t depth;
    };
    ProofResult result;
    std::vector<Node> pending;
    pending.push_back({polynomial,
                       std::vector<std::pair<double, double>>(polynomial.degrees().size(), {0, 1}),
                       0});
    bool unresolved = false;
    // Exact dyadic binary64 box labels through this depth. MPFR endpoints,
    // rather than these diagnostic labels, decide the coefficient signs.
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    while (!pending.empty()) {
        if (result.nodes == budget.max_nodes) {
            result.reason = StopReason::NodeBudget;
            return result;
        }
        Node node = std::move(pending.back());
        pending.pop_back();
        ++result.nodes;
        result.deepest = std::max(result.deepest, node.depth);
        const auto bound = node.polynomial.bounds();
        if (!bound.finite()) {
            unresolved = true;
            result.reason = StopReason::Arithmetic;
            continue;
        }
        if (bound.nonnegative())
            continue;
        if (bound.strictly_negative()) {
            result.status = ProofStatus::NegativeWitness;
            result.reason = StopReason::None;
            result.witness_box = std::move(node.box);
            result.witness_bound = bound;
            return result;
        }
        if (node.depth == depth_limit) {
            unresolved = true;
            result.reason = StopReason::DepthBudget;
            continue;
        }
        const auto rank = node.polynomial.degrees().size();
        std::size_t axis = node.depth % rank;
        for (std::size_t i = 0; i < rank && node.polynomial.degrees()[axis] == 0; ++i)
            axis = (axis + 1) % rank;
        if (node.polynomial.degrees()[axis] == 0) {
            unresolved = true;
            result.reason = StopReason::Arithmetic;
            continue;
        }
        auto children = node.polynomial.split(axis);
        if (!children) {
            unresolved = true;
            result.reason = StopReason::Arithmetic;
            continue;
        }
        auto right_box = node.box;
        const double midpoint = (node.box[axis].first + node.box[axis].second) * .5;
        node.box[axis].second = midpoint;
        right_box[axis].first = midpoint;
        pending.push_back({std::move(children->second), std::move(right_box), node.depth + 1});
        pending.push_back({std::move(children->first), std::move(node.box), node.depth + 1});
    }
    if (!unresolved)
        result.status = ProofStatus::Certified;
    return result;
}
} // namespace mango::detail::proof
