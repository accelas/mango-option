// SPDX-License-Identifier: MIT
#include "mango/option/table/certification/continuous_cell.hpp"
#include "mango/math/proof/bspline_cell.hpp"
#include "mango/option/table/certification/physical_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace mango::detail::certification {
namespace {
using Box = std::array<std::pair<double, double>, 4>;
struct Node {
    proof::BernsteinTensor value, derivative;
    Box unit;
    std::size_t depth = 0;
};
Interval coordinate(double a, double b, double unit) {
    if (unit == 0)
        return Interval(a);
    if (unit == 1)
        return Interval(b);
    return Interval(a) + (Interval(b) - Interval(a)) * Interval(unit);
}
// Binary64 is used only to choose work order, never for evidence/sign tests.
double variation(const proof::BernsteinTensor &p, std::size_t axis) {
    const auto width = p.degrees()[axis] + 1;
    std::size_t stride = 1;
    for (std::size_t d = axis + 1; d < 4; ++d)
        stride *= p.degrees()[d] + 1;
    double score = 0;
    const auto &c = p.coefficients();
    for (std::size_t i = 0; i < c.size(); ++i) {
        if ((i / stride) % width + 1 == width)
            continue;
        const auto delta = c[i + stride] - c[i];
        score = std::max({score, std::abs(delta.lower_bound()), std::abs(delta.upper_bound())});
    }
    return score;
}
std::optional<PricingParams> reachable_witness(const std::array<Interval, 4> &lower,
                                               const std::array<Interval, 4> &upper, double strike,
                                               OptionType type, double q,
                                               const SurfaceBounds &requested) {
    std::array<double, 4> p;
    for (std::size_t d = 0; d < 4; ++d)
        p[d] = std::midpoint(lower[d].lower_bound(), upper[d].upper_bound());
    const double spot = strike * std::exp(p[0]);
    MoneynessDomain ratio(requested.ratio_bounds.value_or(
        MoneynessBounds{std::exp(requested.m_min), std::exp(requested.m_max)}));
    if (!ratio.contains_quote(spot, strike) ||
        (requested.strike_bounds && !requested.strike_bounds->contains(strike)) ||
        !(p[1] > 0 && p[1] >= requested.tau_min && p[1] <= requested.tau_max) ||
        !(p[2] >= requested.sigma_min && p[2] <= requested.sigma_max) ||
        !(p[3] >= requested.rate_min && p[3] <= requested.rate_max))
        return std::nullopt;
    const std::array<Interval, 4> actual{log(Interval(spot) / Interval(strike)), Interval(p[1]),
                                         Interval(p[2]), Interval(p[3])};
    for (std::size_t d = 0; d < 4; ++d) {
        // The actual representable quote must lie strictly inside the proved
        // cell. Outward diagnostic conversion or exp/log roundtrip cannot
        // manufacture a reachable witness outside it.
        if (!(actual[d] - lower[d]).strictly_positive() ||
            !(upper[d] - actual[d]).strictly_positive())
            return std::nullopt;
    }
    return PricingParams(OptionSpec{.spot = spot,
                                    .strike = strike,
                                    .maturity = p[1],
                                    .rate = p[3],
                                    .dividend_yield = q,
                                    .option_type = type},
                         p[2]);
}
} // namespace
PhysicalCellProof prove_continuous_bspline_cell(const BSplineND<double, 4> &spline,
                                                const std::array<std::size_t, 4> &spans,
                                                double reference_strike, OptionType type,
                                                double dividend_yield,
                                                const SurfaceBounds &requested,
                                                proof::ProofBudget budget) {
    PhysicalCellProof result;
    std::array<std::span<const double>, 4> knots;
    for (std::size_t d = 0; d < 4; ++d)
        knots[d] = spline.knots(d);
    auto value = proof::extract_cubic_bspline_cell(knots, spline.coefficients(), spans);
    auto derivative = proof::extract_cubic_bspline_cell(knots, spline.coefficients(), spans, 2);
    if (!value || !derivative) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    Box physical;
    for (std::size_t d = 0; d < 4; ++d) {
        const auto &grid = spline.grid(d);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (!std::isfinite(grid[i]) || (i && !(grid[i - 1] < grid[i]))) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
        }
        const double a = knots[d][spans[d]], b = knots[d][spans[d] + 1];
        if (a < spline.grid(d).front() || b > spline.grid(d).back()) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
        physical[d] = {a, b};
    }
    std::vector<Node> pending;
    pending.push_back(
        {std::move(*value), std::move(*derivative), Box{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}});
    bool unresolved = false;
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    while (!pending.empty()) {
        if (result.nodes == budget.max_nodes) {
            result.reason = proof::StopReason::NodeBudget;
            return result;
        }
        Node node = std::move(pending.back());
        pending.pop_back();
        ++result.nodes;
        std::array<Interval, 4> lower, upper, bounds;
        for (std::size_t d = 0; d < 4; ++d) {
            lower[d] = coordinate(physical[d].first, physical[d].second, node.unit[d].first);
            upper[d] = coordinate(physical[d].first, physical[d].second, node.unit[d].second);
            bounds[d] = hull(lower[d], upper[d]);
        }
        const auto total = continuous_eep({node.value.bounds(), node.derivative.bounds()},
                                          {exp(bounds[0]), bounds[1], bounds[2], bounds[3]},
                                          reference_strike, type, dividend_yield);
        if (!total.finite()) {
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (total.sigma_partial.nonnegative())
            continue;
        if (total.sigma_partial.strictly_negative()) {
            auto witness =
                reachable_witness(lower, upper, reference_strike, type, dividend_yield, requested);
            if (witness) {
                result.status = PriceProofStatus::NegativeWitness;
                result.reason = proof::StopReason::None;
                result.witness = std::move(witness);
                result.witness_vega_per_strike = total.sigma_partial;
                return result;
            }
            // A negative enclosure outside the admitted quote set is not a
            // final witness. Nor does it prove this whole support cell valid.
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (node.depth == depth_limit) {
            unresolved = true;
            result.reason = proof::StopReason::DepthBudget;
            continue;
        }
        std::size_t axis = node.depth % 4;
        double largest = 0;
        for (std::size_t d = 0; d < 4; ++d) {
            const double score = variation(node.derivative, d);
            if (score > largest) {
                largest = score;
                axis = d;
            }
        }
        if (largest == 0) {
            for (std::size_t d = 0; d < 4; ++d) {
                const double score = variation(node.value, d);
                if (score > largest) {
                    largest = score;
                    axis = d;
                }
            }
        }
        const auto eu =
            european({exp(bounds[0]), bounds[1], bounds[2], bounds[3]}, type, dividend_yield);
        const auto raw_derivative = node.derivative.bounds();
        const double raw_width =
            (raw_derivative.upper_bound() - raw_derivative.lower_bound()) / reference_strike;
        const double eu_width = eu.sigma_partial.upper_bound() - eu.sigma_partial.lower_bound();
        // Once analytic add-back uncertainty dominates, repeatedly splitting
        // only the spline's varying axis cannot resolve its zero crossing.
        if (raw_width < eu_width)
            axis = node.depth % 4;
        auto values = node.value.split(axis), derivatives = node.derivative.split(axis);
        if (!values || !derivatives) {
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        auto right = node.unit;
        const double mid = std::midpoint(node.unit[axis].first, node.unit[axis].second);
        node.unit[axis].second = mid;
        right[axis].first = mid;
        pending.push_back(
            {std::move(values->second), std::move(derivatives->second), right, node.depth + 1});
        pending.push_back(
            {std::move(values->first), std::move(derivatives->first), node.unit, node.depth + 1});
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}
} // namespace mango::detail::certification
