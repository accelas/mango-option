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
using PhysicalBox = std::array<std::pair<Interval, Interval>, 4>;
struct Node {
    proof::BernsteinTensor value, derivative;
    Box unit;
    std::size_t depth = 0;
};
Interval coordinate(const Interval &a, const Interval &b, double unit) {
    if (unit == 0)
        return a;
    if (unit == 1)
        return b;
    return a + (b - a) * Interval(unit);
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
        // A negative derivative needs an interior sigma point; other axes
        // may be admitted singleton slices. Outward diagnostic conversion or
        // exp/log roundtrip cannot manufacture a quote outside the proved box.
        const auto from_lower = actual[d] - lower[d], to_upper = upper[d] - actual[d];
        if (d == 2 ? (!from_lower.strictly_positive() || !to_upper.strictly_positive())
                   : (!from_lower.nonnegative() || !to_upper.nonnegative()))
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
static PhysicalCellProof prove_patch(proof::BernsteinTensor value, proof::BernsteinTensor derivative,
                              const PhysicalBox &physical, double reference_strike, OptionType type,
                              double dividend_yield, const SurfaceBounds &requested,
                              proof::ProofBudget budget) {
    PhysicalCellProof result;
    std::vector<Node> pending;
    pending.push_back(
        {std::move(value), std::move(derivative), Box{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}});
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
    PhysicalBox physical;
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
        physical[d] = {Interval(a), Interval(b)};
    }
    return prove_patch(std::move(*value), std::move(*derivative), physical, reference_strike, type,
                       dividend_yield, requested, budget);
}

namespace {
struct AxisPiece {
    Interval lower, upper, unit_lower, unit_upper;
    std::size_t span;
    bool clamped;
};
Interval lesser(const Interval &a, const Interval &b) { return (a - b).nonpositive() ? a : b; }
Interval greater(const Interval &a, const Interval &b) { return (a - b).nonnegative() ? a : b; }
std::vector<AxisPiece> axis_pieces(std::span<const double> knots, const Interval &lower,
                                   const Interval &upper) {
    std::vector<AxisPiece> result;
    const auto n = knots.size() - 4;
    const Interval first(knots.front()), last(knots.back());
    if ((lower - first).strictly_negative()) {
        result.push_back({lower, lesser(upper, first), Interval(0), Interval(0), 3, true});
        if ((upper - first).nonpositive())
            return result;
    }
    for (std::size_t span = 3; span < n; ++span) {
        if (knots[span] == knots[span + 1])
            continue;
        const Interval a(knots[span]), b(knots[span + 1]);
        const auto lo = greater(lower, a), hi = lesser(upper, b);
        if ((hi - lo).strictly_negative())
            continue;
        if ((hi - lo).exact_zero() && !(upper - lower).exact_zero())
            continue;
        auto unit = [&](const Interval &x) {
            if ((x - a).exact_zero())
                return Interval(0);
            if ((x - b).exact_zero())
                return Interval(1);
            return (x - a) / (b - a);
        };
        result.push_back({lo, hi, unit(lo), unit(hi), span, false});
        if ((upper - lower).exact_zero())
            return result;
    }
    if ((upper - last).strictly_positive())
        result.push_back({greater(lower, last), upper, Interval(1), Interval(1), n - 1, true});
    return result;
}
} // namespace
PhysicalCellProof prove_continuous_bspline(const BSplineND<double, 4> &spline,
                                           double reference_strike, OptionType type,
                                           double dividend_yield, const SurfaceBounds &requested,
                                           proof::ProofBudget budget) {
    PhysicalCellProof result;
    MoneynessDomain moneyness(requested.ratio_bounds.value_or(
        MoneynessBounds{std::exp(requested.m_min), std::exp(requested.m_max)}));
    const auto ratio = moneyness.enclosure();
    if (!ratio.valid() || !std::isfinite(reference_strike) || reference_strike <= 0 ||
        !std::isfinite(requested.tau_min) || !std::isfinite(requested.tau_max) ||
        requested.tau_min < 0 || requested.tau_max <= 0 || requested.tau_max < requested.tau_min ||
        !std::isfinite(requested.sigma_min) || !std::isfinite(requested.sigma_max) ||
        requested.sigma_min <= 0 || requested.sigma_max <= requested.sigma_min ||
        !std::isfinite(requested.rate_min) || !std::isfinite(requested.rate_max) ||
        requested.rate_max < requested.rate_min) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    PhysicalBox domain{
        {{log(Interval(ratio.min)).lower_endpoint(), log(Interval(ratio.max)).upper_endpoint()},
         {Interval(requested.tau_min), Interval(requested.tau_max)},
         {Interval(requested.sigma_min), Interval(requested.sigma_max)},
         {Interval(requested.rate_min), Interval(requested.rate_max)}}};
    std::array<std::span<const double>, 4> knots;
    std::array<std::vector<AxisPiece>, 4> pieces;
    for (std::size_t d = 0; d < 4; ++d) {
        knots[d] = spline.knots(d);
        const auto &grid = spline.grid(d);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (!std::isfinite(grid[i]) || (i && !(grid[i - 1] < grid[i]))) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
        }
        // Financial spline payloads have matching clamped grid/knot domains.
        // Mismatched raw math metadata has no supported publication meaning.
        if (grid.front() != knots[d].front() || grid.back() != knots[d].back()) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
        // Validate stored knots before using their spans to partition.
        for (std::size_t i = 0; i < knots[d].size(); ++i) {
            if (!std::isfinite(knots[d][i]) || (i && knots[d][i] < knots[d][i - 1])) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
        }
        pieces[d] = axis_pieces(knots[d], domain[d].first, domain[d].second);
        if (pieces[d].empty()) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
    }
    bool unresolved = false, done = false;
    std::array<std::size_t, 4> index{};
    while (!done) {
        if (result.nodes == budget.max_nodes) {
            result.reason = proof::StopReason::NodeBudget;
            return result;
        }
        std::array<std::size_t, 4> spans;
        PhysicalBox physical;
        for (std::size_t d = 0; d < 4; ++d) {
            const auto &piece = pieces[d][index[d]];
            spans[d] = piece.span;
            physical[d] = {piece.lower, piece.upper};
        }
        auto value = proof::extract_cubic_bspline_cell(knots, spline.coefficients(), spans);
        auto derivative = proof::extract_cubic_bspline_cell(knots, spline.coefficients(), spans, 2);
        if (!value || !derivative) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
        for (std::size_t d = 0; d < 4; ++d) {
            const auto &piece = pieces[d][index[d]];
            value = value->restrict_axis(d, piece.unit_lower, piece.unit_upper);
            derivative = derivative->restrict_axis(d, piece.unit_lower, piece.unit_upper);
            if (!value || !derivative) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
        }
        if (pieces[2][index[2]].clamped)
            derivative = proof::BernsteinTensor::create({0, 0, 0, 0}, {Interval(0)});
        auto cell = prove_patch(std::move(*value), std::move(*derivative), physical,
                                reference_strike, type, dividend_yield, requested,
                                {budget.max_nodes - result.nodes, budget.max_depth});
        cell.nodes += result.nodes;
        if (cell.status == PriceProofStatus::NegativeWitness)
            return cell;
        result.nodes = cell.nodes;
        if (cell.status != PriceProofStatus::Certified) {
            unresolved = true;
            result.reason = cell.reason;
        }
        for (std::size_t d = 4; d > 0; --d) {
            if (++index[d - 1] < pieces[d - 1].size())
                break;
            index[d - 1] = 0;
            if (d == 1)
                done = true;
        }
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}
} // namespace mango::detail::certification
