// SPDX-License-Identifier: MIT
#include "mango/math/proof/bspline_box.hpp"
#include <cmath>
namespace mango::detail::proof {
namespace {
struct Piece {
    Interval lo, hi;
    std::size_t span;
    bool clamped;
};
Interval lesser(const Interval &a, const Interval &b) { return (a - b).nonpositive() ? a : b; }
Interval greater(const Interval &a, const Interval &b) { return (a - b).nonnegative() ? a : b; }
std::vector<Piece> pieces(std::span<const double> knots, const Interval &box) {
    std::vector<Piece> result;
    const auto n = knots.size() - 4;
    const auto lower = box.lower_endpoint(), upper = box.upper_endpoint();
    const Interval first(knots.front()), last(knots.back());
    if ((lower - first).strictly_negative()) {
        result.push_back({Interval(0), Interval(0), 3, true});
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
        result.push_back({unit(lo), unit(hi), span, false});
        if ((upper - lower).exact_zero())
            return result;
    }
    if ((upper - last).strictly_positive())
        result.push_back({Interval(1), Interval(1), n - 1, true});
    return result;
}
} // namespace
BSplineBoxEnclosure enclose_cubic_bspline_box(std::span<const std::span<const double>> knots,
                                              std::span<const double> coefficients,
                                              std::span<const Interval> coordinates,
                                              std::span<const std::size_t> derivative_axes,
                                              std::size_t max_cells) {
    BSplineBoxEnclosure result;
    const auto rank = knots.size();
    if (rank == 0 || rank > 4 || coordinates.size() != rank || derivative_axes.size() > rank)
        return result;
    for (auto axis : derivative_axes)
        if (axis >= rank)
            return result;
    std::vector<std::vector<Piece>> axes;
    for (std::size_t d = 0; d < rank; ++d) {
        if (knots[d].size() < 8 || !coordinates[d].finite())
            return result;
        for (std::size_t i = 0; i < knots[d].size(); ++i)
            if (!std::isfinite(knots[d][i]) || (i && knots[d][i] < knots[d][i - 1]))
                return result;
        axes.push_back(pieces(knots[d], coordinates[d]));
        if (axes.back().empty())
            return result;
    }
    std::vector<std::size_t> indices(rank), spans(rank);
    result.partials.resize(derivative_axes.size());
    bool done = false;
    while (!done) {
        if (result.cells == max_cells) {
            result.reason = StopReason::NodeBudget;
            return result;
        }
        for (std::size_t d = 0; d < rank; ++d)
            spans[d] = axes[d][indices[d]].span;
        auto bound = [&](std::optional<std::size_t> derivative) -> std::optional<Interval> {
            if (derivative && axes[*derivative][indices[*derivative]].clamped)
                return Interval(0);
            auto p = extract_cubic_bspline_cell(knots, coefficients, spans, derivative);
            if (!p)
                return std::nullopt;
            for (std::size_t d = 0; d < rank; ++d) {
                const auto &part = axes[d][indices[d]];
                p = p->restrict_axis(d, part.lo, part.hi);
                if (!p)
                    return std::nullopt;
            }
            return p->bounds();
        };
        auto value = bound(std::nullopt);
        if (!value || !value->finite())
            return result;
        result.value = result.cells ? hull(result.value, *value) : *value;
        for (std::size_t i = 0; i < derivative_axes.size(); ++i) {
            auto partial = bound(derivative_axes[i]);
            if (!partial || !partial->finite())
                return result;
            result.partials[i] = result.cells ? hull(result.partials[i], *partial) : *partial;
        }
        ++result.cells;
        for (std::size_t d = rank; d > 0; --d) {
            if (++indices[d - 1] < axes[d - 1].size())
                break;
            indices[d - 1] = 0;
            if (d == 1)
                done = true;
        }
    }
    result.reason = StopReason::None;
    return result;
}
} // namespace mango::detail::proof
