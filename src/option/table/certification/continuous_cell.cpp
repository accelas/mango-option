// SPDX-License-Identifier: MIT
#include "mango/option/table/certification/continuous_cell.hpp"
#include "mango/math/proof/bspline_box.hpp"
#include "mango/math/proof/bspline_cell.hpp"
#include "mango/math/proof/chebyshev.hpp"
#include "mango/option/table/certification/physical_bounds.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <type_traits>
namespace mango::detail::certification {
namespace {
bool finite_shape_bounds(const PriceBounds &bounds, bool singleton_sigma) {
    return bounds.value.finite() && (singleton_sigma || bounds.sigma_partial.finite());
}
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
    // A point sigma domain has no pair of admitted volatilities on which
    // nondecrease can fail, regardless of an extension derivative's sign.
    if (!(requested.sigma_min < requested.sigma_max))
        return std::nullopt;
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
static PhysicalCellProof prove_patch(proof::BernsteinTensor value,
                                     proof::BernsteinTensor derivative, const PhysicalBox &physical,
                                     double reference_strike, OptionType type,
                                     double dividend_yield, const SurfaceBounds &requested,
                                     proof::ProofBudget budget) {
    PhysicalCellProof result;
    const bool singleton_sigma = (physical[2].second - physical[2].first).exact_zero();
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
        if (!finite_shape_bounds(total, singleton_sigma)) {
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (singleton_sigma || total.sigma_partial.nonnegative())
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

namespace {
std::optional<PhysicalBox> requested_domain(const SurfaceBounds &requested) {
    MoneynessDomain moneyness(requested.ratio_bounds.value_or(
        MoneynessBounds{std::exp(requested.m_min), std::exp(requested.m_max)}));
    const auto ratio = moneyness.enclosure();
    if (!ratio.valid() || !std::isfinite(requested.tau_min) || !std::isfinite(requested.tau_max) ||
        requested.tau_min < 0 || requested.tau_max <= 0 || requested.tau_max < requested.tau_min ||
        !std::isfinite(requested.sigma_min) || !std::isfinite(requested.sigma_max) ||
        requested.sigma_min <= 0 || requested.sigma_max < requested.sigma_min ||
        !std::isfinite(requested.rate_min) || !std::isfinite(requested.rate_max) ||
        requested.rate_max < requested.rate_min) {
        return std::nullopt;
    }
    PhysicalBox domain{
        {{log(Interval(ratio.min)).lower_endpoint(), log(Interval(ratio.max)).upper_endpoint()},
         {Interval(requested.tau_min), Interval(requested.tau_max)},
         {Interval(requested.sigma_min), Interval(requested.sigma_max)},
         {Interval(requested.rate_min), Interval(requested.rate_max)}}};
    return domain;
}
} // namespace
PhysicalCellProof prove_continuous_bspline(const BSplineND<double, 4> &spline,
                                           double reference_strike, OptionType type,
                                           double dividend_yield, const SurfaceBounds &requested,
                                           proof::ProofBudget budget) {
    PhysicalCellProof result;
    auto resolved = requested_domain(requested);
    if (!resolved || !std::isfinite(reference_strike) || reference_strike <= 0) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    const auto &domain = *resolved;
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

PhysicalCellProof prove_dimensionless_bspline(const BSplineND<double, 3> &spline,
                                              double reference_strike, OptionType type,
                                              const SurfaceBounds &requested,
                                              proof::ProofBudget budget) {
    PhysicalCellProof result;
    auto domain = requested_domain(requested);
    if (!domain || requested.rate_min <= 0 || !std::isfinite(reference_strike) ||
        reference_strike <= 0) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    const bool singleton_sigma = ((*domain)[2].second - (*domain)[2].first).exact_zero();
    std::array<std::span<const double>, 3> knots;
    for (std::size_t d = 0; d < 3; ++d) {
        knots[d] = spline.knots(d);
        const auto &grid = spline.grid(d);
        for (std::size_t i = 0; i < grid.size(); ++i)
            if (!std::isfinite(grid[i]) || (i && !(grid[i - 1] < grid[i]))) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
        if (grid.front() != knots[d].front() || grid.back() != knots[d].back()) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
    }

    struct QueryNode {
        Box unit;
        std::size_t depth = 0;
    };
    std::vector<QueryNode> pending{{Box{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}}};
    bool unresolved = false;
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    while (!pending.empty()) {
        if (result.nodes == budget.max_nodes) {
            result.reason = proof::StopReason::NodeBudget;
            return result;
        }
        auto node = pending.back();
        pending.pop_back();
        ++result.nodes;
        std::array<Interval, 4> lower, upper, bounds;
        for (std::size_t d = 0; d < 4; ++d) {
            lower[d] = coordinate((*domain)[d].first, (*domain)[d].second, node.unit[d].first);
            upper[d] = coordinate((*domain)[d].first, (*domain)[d].second, node.unit[d].second);
            bounds[d] = hull(lower[d], upper[d]);
        }
        const auto sigma_squared = square(bounds[2]);
        const std::array<Interval, 3> coordinates{bounds[0],
                                                  sigma_squared * bounds[1] / Interval(2),
                                                  log(Interval(2) * bounds[3] / sigma_squared)};
        const std::array<std::size_t, 2> axes{1, 2};
        auto raw = proof::enclose_cubic_bspline_box(knots, spline.coefficients(), coordinates, axes,
                                                    budget.max_nodes - result.nodes);
        result.nodes += raw.cells;
        if (raw.reason == proof::StopReason::NodeBudget) {
            result.reason = raw.reason;
            return result;
        }
        if (raw.reason != proof::StopReason::None) {
            unresolved = true;
            result.reason = raw.reason;
            continue;
        }
        const auto total = dimensionless_eep(raw.value, raw.partials[0], raw.partials[1],
                                             {exp(bounds[0]), bounds[1], bounds[2], bounds[3]},
                                             reference_strike, type);
        if (!finite_shape_bounds(total, singleton_sigma)) {
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (singleton_sigma || total.sigma_partial.nonnegative())
            continue;
        if (total.sigma_partial.strictly_negative()) {
            auto witness = reachable_witness(lower, upper, reference_strike, type, 0, requested);
            if (witness) {
                result.status = PriceProofStatus::NegativeWitness;
                result.reason = proof::StopReason::None;
                result.witness = std::move(witness);
                result.witness_vega_per_strike = total.sigma_partial;
                return result;
            }
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (node.depth == depth_limit) {
            unresolved = true;
            result.reason = proof::StopReason::DepthBudget;
            continue;
        }
        // Subdivide physical coordinates, not independent transformed u/z:
        // their sigma coupling is rebuilt on every child box.
        const std::array<std::size_t, 4> order{2, 1, 3, 0};
        std::size_t choice = node.depth % 4;
        while (((*domain)[order[choice]].second - (*domain)[order[choice]].first).exact_zero())
            choice = (choice + 1) % 4;
        const auto axis = order[choice];
        auto right = node.unit;
        const double middle = std::midpoint(node.unit[axis].first, node.unit[axis].second);
        node.unit[axis].second = middle;
        right[axis].first = middle;
        pending.push_back({right, node.depth + 1});
        pending.push_back({node.unit, node.depth + 1});
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}

namespace {
template <typename Inner>
bool valid_segmented_payload(const Inner &inner, const SurfaceBounds &requested) {
    const auto &refs = inner.split().k_refs();
    if (refs.empty() || refs.size() != inner.num_pieces() || !requested.strike_bounds ||
        !requested.strike_bounds->valid())
        return false;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        if (!std::isfinite(refs[i]) || refs[i] <= 0 || (i && !(refs[i - 1] < refs[i])))
            return false;
        const auto &member = inner.pieces()[i];
        const auto &split = member.split();
        const auto n = member.num_pieces();
        if (!n || split.K_ref() != refs[i] || split.tau_start().size() != n ||
            split.tau_end().size() != n || split.tau_min().size() != n ||
            split.tau_max().size() != n)
            return false;
        for (std::size_t j = 0; j < n; ++j) {
            const double a = split.tau_start()[j], b = split.tau_end()[j], lo = split.tau_min()[j],
                         hi = split.tau_max()[j];
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(lo) ||
                !std::isfinite(hi) || a < 0 || b <= a || lo < 0 || hi < lo ||
                (j && a < split.tau_end()[j - 1]))
                return false;
            const auto &leaf = member.pieces()[j];
            if (leaf.K_ref() != refs[i])
                return false;
            const auto &interp = leaf.interpolant();
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                         SharedBSplineInterp<4>>) {
                if (!interp.has_value())
                    return false;
                const auto &spline = interp.get();
                for (std::size_t d = 0; d < 4; ++d) {
                    const auto &grid = spline.grid(d);
                    for (std::size_t k = 0; k < grid.size(); ++k)
                        if (!std::isfinite(grid[k]) || (k && !(grid[k - 1] < grid[k])))
                            return false;
                    if (grid.front() != spline.knots(d).front() ||
                        grid.back() != spline.knots(d).back())
                        return false;
                }
            } else {
                static_assert(std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                             ChebyshevModalInterpolant<4>>);
                if (interp.polynomial().shape().size() != 4)
                    return false;
            }
        }
    }
    return requested.strike_bounds->min >= refs.front() &&
           requested.strike_bounds->max <= refs.back();
}
struct SegmentedBounds {
    PriceBounds price;
    std::size_t cells = 0;
    proof::StopReason reason = proof::StopReason::None;
    bool empty = false;
};
proof::BSplineBoxEnclosure enclose_modal(const ChebyshevPolynomial &polynomial,
                                         std::span<const Interval> coordinates,
                                         std::span<const std::size_t> axes, std::size_t max_work);
template <typename Inner>
SegmentedBounds segmented_bounds(const Inner &inner, double strike,
                                 const std::array<Interval, 4> &physical, std::size_t max_cells) {
    SegmentedBounds result;
    const auto bracket = inner.split().bracket(1, strike, 1, 1, 0);
    std::array<Interval, 2> weights{Interval(1), Interval(0)};
    if (bracket.count == 2) {
        const auto &refs = inner.split().k_refs();
        const Interval a(refs[bracket.entries[0].index]), b(refs[bracket.entries[1].index]);
        weights[1] = intersection((Interval(strike) - a) / (b - a), Interval::hull(0, 1));
        weights[0] = intersection(Interval(1) - weights[1], Interval::hull(0, 1));
    }
    std::vector<PriceBounds> members;
    for (std::size_t member_index = 0; member_index < bracket.count; ++member_index) {
        const auto &member = inner.pieces()[bracket.entries[member_index].index];
        const auto &split = member.split();
        std::optional<PriceBounds> combined;
        for (std::size_t j = 0; j < member.num_pieces(); ++j) {
            const auto time =
                intersection(physical[1], Interval::hull(split.tau_start()[j], split.tau_end()[j]));
            if (!time.finite())
                continue;
            auto local = time - Interval(split.tau_start()[j]);
            const Interval lo(split.tau_min()[j]), hi(split.tau_max()[j]);
            local = lo + positive_part(local - lo);
            local = hi - positive_part(hi - local);
            // Validated reference identities make both spot maps preserve
            // S/K and cancel all reference-price normalization factors.
            const std::array<Interval, 4> coordinates{physical[0], local, physical[2], physical[3]};
            const auto &interp = member.pieces()[j].interpolant();
            const std::array<std::size_t, 1> axis{2};
            proof::BSplineBoxEnclosure raw;
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                         SharedBSplineInterp<4>>) {
                const auto &spline = interp.get();
                std::array<std::span<const double>, 4> knots;
                for (std::size_t d = 0; d < 4; ++d)
                    knots[d] = spline.knots(d);
                raw = proof::enclose_cubic_bspline_box(knots, spline.coefficients(), coordinates,
                                                       axis, max_cells - result.cells);
            } else {
                static_assert(std::is_same_v<std::remove_cvref_t<decltype(interp)>,
                                             ChebyshevModalInterpolant<4>>);
                raw =
                    enclose_modal(interp.polynomial(), coordinates, axis, max_cells - result.cells);
            }
            result.cells += raw.cells;
            if (raw.reason != proof::StopReason::None) {
                result.reason = raw.reason;
                return result;
            }
            auto price = zero_floor({raw.value, raw.partials[0]});
            if (!price.finite()) {
                result.reason = proof::StopReason::Arithmetic;
                return result;
            }
            if (!combined)
                combined = price;
            else
                combined = PriceBounds{hull(combined->value, price.value),
                                       hull(combined->sigma_partial, price.sigma_partial)};
        }
        if (!combined) {
            result.empty = true;
            return result;
        }
        members.push_back(*combined);
    }
    result.price = weighted_sum(members, std::span<const Interval>(weights.data(), bracket.count));
    return result;
}
} // namespace
PhysicalCellProof prove_segmented_bspline(const BSplineMultiKRefInner &inner, OptionType type,
                                          double dividend_yield, const SurfaceBounds &requested,
                                          proof::ProofBudget budget) {
    PhysicalCellProof result;
    auto domain = requested_domain(requested);
    if (!domain || !valid_segmented_payload(inner, requested) || !std::isfinite(dividend_yield) ||
        dividend_yield < 0 || (type != OptionType::CALL && type != OptionType::PUT)) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    const bool singleton_sigma = ((*domain)[2].second - (*domain)[2].first).exact_zero();
    std::vector<double> strikes{requested.strike_bounds->min, requested.strike_bounds->max};
    for (double reference : inner.split().k_refs())
        if (reference > strikes.front() && reference < strikes[1])
            strikes.push_back(reference);
    std::sort(strikes.begin(), strikes.end());
    strikes.erase(std::unique(strikes.begin(), strikes.end()), strikes.end());
    bool unresolved = false;
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    struct QueryNode {
        Box unit;
        std::size_t depth = 0;
    };
    for (double strike : strikes) {
        // The normalized expression is affine in K within each reference
        // bracket. Proving its domain/bracket endpoints covers all K inside.
        std::vector<QueryNode> pending{{Box{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}}};
        while (!pending.empty()) {
            if (result.nodes == budget.max_nodes) {
                result.reason = proof::StopReason::NodeBudget;
                return result;
            }
            auto node = pending.back();
            pending.pop_back();
            ++result.nodes;
            std::array<Interval, 4> lower, upper, bounds;
            for (std::size_t d = 0; d < 4; ++d) {
                lower[d] = coordinate((*domain)[d].first, (*domain)[d].second, node.unit[d].first);
                upper[d] = coordinate((*domain)[d].first, (*domain)[d].second, node.unit[d].second);
                bounds[d] = hull(lower[d], upper[d]);
            }
            auto total = segmented_bounds(inner, strike, bounds, budget.max_nodes - result.nodes);
            result.nodes += total.cells;
            if (total.reason == proof::StopReason::NodeBudget) {
                result.reason = total.reason;
                return result;
            }
            if (total.reason != proof::StopReason::None ||
                !finite_shape_bounds(total.price, singleton_sigma)) {
                unresolved = true;
                result.reason = proof::StopReason::Arithmetic;
                continue;
            }
            if (total.empty || singleton_sigma || total.price.sigma_partial.nonnegative())
                continue;
            bool split_time = false;
            if (total.price.sigma_partial.strictly_negative()) {
                auto witness =
                    reachable_witness(lower, upper, strike, type, dividend_yield, requested);
                if (witness && inner.contains_maturity(witness->maturity)) {
                    result.status = PriceProofStatus::NegativeWitness;
                    result.reason = proof::StopReason::None;
                    result.witness = std::move(witness);
                    result.witness_vega_per_strike = total.price.sigma_partial;
                    return result;
                }
                split_time = witness && !inner.contains_maturity(witness->maturity);
            }
            if (node.depth == depth_limit) {
                unresolved = true;
                result.reason = proof::StopReason::DepthBudget;
                continue;
            }
            const std::array<std::size_t, 4> order{2, 1, 0, 3};
            std::size_t choice = node.depth % 4;
            while (((*domain)[order[choice]].second - (*domain)[order[choice]].first).exact_zero())
                choice = (choice + 1) % 4;
            const auto axis = split_time ? std::size_t{1} : order[choice];
            auto right = node.unit;
            const double mid = std::midpoint(node.unit[axis].first, node.unit[axis].second);
            node.unit[axis].second = mid;
            right[axis].first = mid;
            pending.push_back({right, node.depth + 1});
            pending.push_back({node.unit, node.depth + 1});
        }
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}

namespace {
struct ExpressionBounds {
    PriceBounds price;
    std::optional<std::size_t> axis_hint;
    std::size_t work = 0;
    proof::StopReason reason = proof::StopReason::None;
    bool empty = false;
};
using ExpressionBounder =
    std::function<ExpressionBounds(const std::array<Interval, 4> &, std::size_t)>;
using QueryAdmitter = std::function<bool(const PricingParams &)>;
PhysicalCellProof prove_expression_boxes(const PhysicalBox &domain, double strike, OptionType type,
                                         double q, const SurfaceBounds &requested,
                                         proof::ProofBudget budget,
                                         const ExpressionBounder &enclose,
                                         const QueryAdmitter &admit = {}) {
    PhysicalCellProof result;
    const bool singleton_sigma = (domain[2].second - domain[2].first).exact_zero();
    struct QueryNode {
        Box unit;
        std::size_t depth = 0;
    };
    std::vector<QueryNode> pending{{Box{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}}};
    const auto depth_limit = std::min<std::size_t>(budget.max_depth, 52);
    bool unresolved = false;
    while (!pending.empty()) {
        if (result.nodes == budget.max_nodes) {
            result.reason = proof::StopReason::NodeBudget;
            return result;
        }
        auto node = pending.back();
        pending.pop_back();
        ++result.nodes;
        std::array<Interval, 4> lower, upper, bounds;
        for (std::size_t d = 0; d < 4; ++d) {
            lower[d] = coordinate(domain[d].first, domain[d].second, node.unit[d].first);
            upper[d] = coordinate(domain[d].first, domain[d].second, node.unit[d].second);
            bounds[d] = hull(lower[d], upper[d]);
        }
        auto total = enclose(bounds, budget.max_nodes - result.nodes);
        if (total.work > budget.max_nodes - result.nodes) {
            result.reason = proof::StopReason::Arithmetic;
            return result;
        }
        result.nodes += total.work;
        if (total.reason == proof::StopReason::NodeBudget) {
            result.reason = total.reason;
            return result;
        }
        if (total.reason != proof::StopReason::None ||
            !finite_shape_bounds(total.price, singleton_sigma)) {
            unresolved = true;
            result.reason = proof::StopReason::Arithmetic;
            continue;
        }
        if (total.empty || singleton_sigma || total.price.sigma_partial.nonnegative())
            continue;
        bool split_time = false;
        if (total.price.sigma_partial.strictly_negative()) {
            auto witness = reachable_witness(lower, upper, strike, type, q, requested);
            if (witness && (!admit || admit(*witness))) {
                result.status = PriceProofStatus::NegativeWitness;
                result.reason = proof::StopReason::None;
                result.witness = std::move(witness);
                result.witness_vega_per_strike = total.price.sigma_partial;
                return result;
            }
            split_time = witness && admit && !admit(*witness);
        }
        if (node.depth == depth_limit) {
            unresolved = true;
            result.reason = proof::StopReason::DepthBudget;
            continue;
        }
        const std::array<std::size_t, 4> order{2, 1, 0, 3};
        std::size_t choice = node.depth % 4;
        while ((domain[order[choice]].second - domain[order[choice]].first).exact_zero())
            choice = (choice + 1) % 4;
        const auto axis = split_time ? std::size_t{1} : total.axis_hint.value_or(order[choice]);
        auto right = node.unit;
        const double mid = std::midpoint(node.unit[axis].first, node.unit[axis].second);
        node.unit[axis].second = mid;
        right[axis].first = mid;
        pending.push_back({right, node.depth + 1});
        pending.push_back({node.unit, node.depth + 1});
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}
proof::BSplineBoxEnclosure enclose_modal(const ChebyshevPolynomial &polynomial,
                                         std::span<const Interval> coordinates,
                                         std::span<const std::size_t> axes, std::size_t max_work) {
    proof::BSplineBoxEnclosure result;
    if (max_work < 1 + axes.size()) {
        result.reason = proof::StopReason::NodeBudget;
        return result;
    }
    auto value = proof::enclose_chebyshev_physical(polynomial, coordinates);
    ++result.cells;
    if (!value)
        return result;
    result.value = *value;
    for (auto axis : axes) {
        auto partial = proof::enclose_chebyshev_physical(polynomial, coordinates, axis);
        ++result.cells;
        if (!partial)
            return result;
        result.partials.push_back(*partial);
    }
    result.reason = proof::StopReason::None;
    return result;
}
} // namespace
PhysicalCellProof prove_continuous_chebyshev(const ChebyshevPolynomial &polynomial,
                                             double reference_strike, OptionType type,
                                             double dividend_yield, const SurfaceBounds &requested,
                                             proof::ProofBudget budget) {
    auto domain = requested_domain(requested);
    if (!domain || polynomial.shape().size() != 4 || !std::isfinite(reference_strike) ||
        reference_strike <= 0) {
        PhysicalCellProof result;
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    const bool sigma_only = [&] {
        for (std::size_t i = 0; i < polynomial.coefficients().size(); ++i) {
            if (polynomial.coefficients()[i] == 0)
                continue;
            auto index = i;
            for (std::size_t d = 4; d > 0; --d) {
                const auto mode = index % polynomial.shape()[d - 1];
                index /= polynomial.shape()[d - 1];
                if (d - 1 != 2 && mode != 0)
                    return false;
            }
        }
        return true;
    }();
    return prove_expression_boxes(
        *domain, reference_strike, type, dividend_yield, requested, budget,
        [&](const std::array<Interval, 4> &box, std::size_t remaining) {
            const std::array<std::size_t, 1> axes{2};
            auto raw = enclose_modal(polynomial, box, axes, remaining);
            ExpressionBounds result;
            result.work = raw.cells;
            result.reason = raw.reason;
            if (raw.reason == proof::StopReason::None) {
                const QuoteBox quote{exp(box[0]), box[1], box[2], box[3]};
                result.price = continuous_eep({raw.value, raw.partials[0]}, quote, reference_strike,
                                              type, dividend_yield);
                if (sigma_only && !result.price.sigma_partial.nonnegative() &&
                    !result.price.sigma_partial.strictly_negative()) {
                    const auto eu = european(quote, type, dividend_yield);
                    const double raw_width =
                        (raw.partials[0].upper_bound() - raw.partials[0].lower_bound()) /
                        reference_strike;
                    const double eu_width =
                        eu.sigma_partial.upper_bound() - eu.sigma_partial.lower_bound();
                    // Work-order hint only: exact zero modes establish that
                    // the raw polynomial varies only with sigma. Switch back
                    // to physical subdivision once analytic uncertainty dominates.
                    if (raw_width > eu_width || (raw.value.contains(0) && !raw.value.exact_zero()))
                        result.axis_hint = 2;
                }
            }
            return result;
        });
}

PhysicalCellProof prove_dimensionless_chebyshev(const ChebyshevPolynomial &polynomial,
                                                double reference_strike, OptionType type,
                                                const SurfaceBounds &requested,
                                                proof::ProofBudget budget) {
    auto domain = requested_domain(requested);
    if (!domain || polynomial.shape().size() != 3 || requested.rate_min <= 0 ||
        !std::isfinite(reference_strike) || reference_strike <= 0) {
        PhysicalCellProof result;
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    return prove_expression_boxes(
        *domain, reference_strike, type, 0, requested, budget,
        [&](const std::array<Interval, 4> &box, std::size_t remaining) {
            const auto sigma_squared = square(box[2]);
            const std::array<Interval, 3> coordinates{box[0], sigma_squared * box[1] / Interval(2),
                                                      log(Interval(2) * box[3] / sigma_squared)};
            const std::array<std::size_t, 2> axes{1, 2};
            auto raw = enclose_modal(polynomial, coordinates, axes, remaining);
            ExpressionBounds result;
            result.work = raw.cells;
            result.reason = raw.reason;
            if (raw.reason == proof::StopReason::None)
                result.price = dimensionless_eep(raw.value, raw.partials[0], raw.partials[1],
                                                 {exp(box[0]), box[1], box[2], box[3]},
                                                 reference_strike, type);
            return result;
        });
}

PhysicalCellProof prove_segmented_chebyshev(const ModalMultiKRefInner &inner, OptionType type,
                                            double dividend_yield, const SurfaceBounds &requested,
                                            proof::ProofBudget budget) {
    PhysicalCellProof result;
    auto domain = requested_domain(requested);
    if (!domain || !valid_segmented_payload(inner, requested) || !std::isfinite(dividend_yield) ||
        dividend_yield < 0 || (type != OptionType::CALL && type != OptionType::PUT)) {
        result.reason = proof::StopReason::Arithmetic;
        return result;
    }
    std::vector<double> strikes{requested.strike_bounds->min, requested.strike_bounds->max};
    for (double reference : inner.split().k_refs())
        if (reference > strikes.front() && reference < strikes[1])
            strikes.push_back(reference);
    std::sort(strikes.begin(), strikes.end());
    strikes.erase(std::unique(strikes.begin(), strikes.end()), strikes.end());
    bool unresolved = false;
    for (double strike : strikes) {
        auto endpoint = prove_expression_boxes(
            *domain, strike, type, dividend_yield, requested,
            {budget.max_nodes - result.nodes, budget.max_depth},
            [&](const std::array<Interval, 4> &box, std::size_t remaining) {
                auto bound = segmented_bounds(inner, strike, box, remaining);
                ExpressionBounds result;
                result.price = bound.price;
                result.work = bound.cells;
                result.reason = bound.reason;
                result.empty = bound.empty;
                return result;
            },
            [&](const PricingParams &p) { return inner.contains_maturity(p.maturity); });
        endpoint.nodes += result.nodes;
        if (endpoint.status == PriceProofStatus::NegativeWitness)
            return endpoint;
        result.nodes = endpoint.nodes;
        if (endpoint.status != PriceProofStatus::Certified) {
            unresolved = true;
            result.reason = endpoint.reason;
        }
    }
    if (!unresolved)
        result.status = PriceProofStatus::Certified;
    return result;
}
} // namespace mango::detail::certification
