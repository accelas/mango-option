// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/math/latin_hypercube.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/dividend_utils.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <random>

namespace mango {

namespace {
constexpr double kMinPositive = 1e-6;

/// Number of sigma points in the monotonicity scan (spec D7).
constexpr size_t kMonotonicityPoints = 7;

/// One fixed holdout point with its cached references (spec D4).
/// Shared with the segmented builders' final validation (spec D9).
using HoldoutPoint = detail::ValidationPoint;

// Query construction is separate from the four fit axes. Segmented domains
// carry independent absolute K; continuous domains retain fixed-spot sampling.
bool valid_measurement_domain(const RefinementContext& ctx) {
    if (ctx.sample_bounds.strike_bounds && !ctx.sample_bounds.strike_bounds->valid()) return false;
    if (ctx.maturity_intervals) {
        if (ctx.maturity_intervals->empty()) return false;
        double previous = ctx.sample_bounds.tau_min;
        for (const auto& [lo, hi] : *ctx.maturity_intervals) {
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo < previous || !(hi > lo) ||
                hi > ctx.sample_bounds.tau_max) return false;
            previous = hi;
        }
    }
    return true;
}

double sample_tau(double tau, const RefinementContext& ctx) {
    if (!ctx.maturity_intervals) {
        return tau > 0.0 ? tau : std::lerp(ctx.sample_bounds.tau_min, ctx.sample_bounds.tau_max, 1e-4);
    }
    double width = 0.0;
    for (const auto& [lo, hi] : *ctx.maturity_intervals) width += hi - lo;
    const double unit = std::clamp((tau - ctx.sample_bounds.tau_min) /
        (ctx.sample_bounds.tau_max - ctx.sample_bounds.tau_min), 0.0, 1.0);
    double offset = unit * width;
    for (const auto& [lo, hi] : *ctx.maturity_intervals) {
        if (offset <= hi - lo) {
            const double result = std::clamp(lo + offset, lo, hi);
            return result > 0.0 ? result : std::lerp(lo, hi, 1e-4);
        }
        offset -= hi - lo;
    }
    return ctx.maturity_intervals->back().second;
}

std::expected<std::vector<HoldoutPoint>, PriceTableError> physical_samples(
    const std::vector<std::array<double, 4>>& input,
    const RefinementContext& ctx, uint64_t seed)
{
    const auto& requested = ctx.sample_bounds;
    const auto ratios = requested.ratio_bounds.value_or(MoneynessBounds{
        std::exp(requested.m_min), std::exp(requested.m_max)});
    const MoneynessDomain quote_domain(ratios);
    const auto ratio_at = [&](double x) {
        if (x == requested.m_min) return ratios.min;
        if (x == requested.m_max) return ratios.max;
        return std::exp(x);
    };
    std::vector<HoldoutPoint> samples;
    samples.reserve(input.size());
    std::vector<size_t> rank(input.size());
    std::iota(rank.begin(), rank.end(), size_t{0});
    std::mt19937_64 rng(seed ^ 0x4b524546ULL);
    std::shuffle(rank.begin(), rank.end(), rng);
    for (size_t i = 0; i < input.size(); ++i) {
        auto coords = input[i];
        double strike = ctx.spot * std::exp(-coords[0]);
        double spot = ctx.spot;
        if (ctx.sample_bounds.strike_bounds) {
            const auto& k = *ctx.sample_bounds.strike_bounds;
            const double u = (static_cast<double>(rank[i]) + .5) / static_cast<double>(input.size());
            strike = std::lerp(k.min, k.max, u);
            spot = strike * ratio_at(coords[0]);
        }
        coords[1] = sample_tau(coords[1], ctx);
        samples.push_back({.coords = coords, .strike = strike, .spot = spot, .refs = {}});
    }
    if (!ctx.sample_bounds.strike_bounds) return samples;

    // Fixed endpoint strata supplement the independent-K Latin hypercube
    // within the same declared sample count. Include interior moneyness so
    // a small budget does not consist solely of flat exercise/OTM tails.
    const auto& b = ctx.sample_bounds;
    const auto& k = *b.strike_bounds;
    const double mid_m = std::midpoint(b.m_min, b.m_max);
    const double mid_t = std::midpoint(b.tau_min, b.tau_max);
    const double mid_v = std::midpoint(b.sigma_min, b.sigma_max);
    const double mid_r = std::midpoint(b.rate_min, b.rate_max);
    std::vector<HoldoutPoint> mandatory;
    auto add = [&](double m, double strike, double tau, double sigma, double rate) {
        std::array<double, 4> coords{m, sample_tau(tau, ctx), sigma, rate};
        if (std::any_of(mandatory.begin(), mandatory.end(), [&](const auto& p) {
                return p.coords == coords && p.strike == strike;
            })) return;
        mandatory.push_back({.coords = coords, .strike = strike,
            .spot = strike * ratio_at(m), .refs = {}});
    };
    for (double m : {b.m_min, b.m_max}) {
        for (double strike : {k.min, k.max}) add(m, strike, mid_t, mid_v, mid_r);
    }
    add(mid_m, k.min, b.tau_min, b.sigma_min, b.rate_min);
    add(mid_m, k.max, b.tau_max, b.sigma_max, b.rate_max);
    for (size_t i = 0; i < std::min(mandatory.size(), samples.size()); ++i) samples[i] = mandatory[i];
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!quote_domain.contains_quote(samples[i].spot, samples[i].strike)) {
            // A refused physical query is not a filtered IV observation.
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 0, i});
        }
    }
    return samples;
}

/// Outcome of scoring one candidate surface over a set of samples.
struct SampleEval {
    double max_error = 0.0;
    double avg_error = 0.0;
    /// Samples whose score engaged and produced a usable error.  Points the
    /// score fn deliberately skipped (TV/K or vega floor) are not counted
    /// here: they are not measurements of this surface.
    size_t measured = 0;
    size_t pde_solves_validation = 0;
    ErrorBins error_bins;
    /// False when any evaluation produced a non-finite price/score or a
    /// negative score (spec D5 viability).
    bool all_finite = true;
};

/// One recorded candidate surface (spec D5).
struct Candidate {
    std::vector<double> moneyness, tau, vol, rate;
    std::shared_ptr<const void> state;
    double holdout_max = std::numeric_limits<double>::quiet_NaN();
    double holdout_avg = std::numeric_limits<double>::quiet_NaN();
    size_t holdout_measured = 0;  ///< holdout points that actually measured it
    ErrorBins bins;
    size_t iteration = 0;
    bool viable = false;
    bool fresh_converged = false;
};

}  // namespace

void expand_domain_bounds(double& lo, double& hi, double min_spread,
                          double lo_clamp) {
    if (hi - lo < min_spread) {
        double mid = (lo + hi) / 2.0;
        lo = mid - min_spread / 2.0;
        hi = mid + min_spread / 2.0;
    }
    if (lo < lo_clamp) {
        hi += (lo_clamp - lo);
        lo = lo_clamp;
    }
}

double spline_support_headroom(double domain_width, size_t n_knots) {
    size_t n = std::max(n_knots, size_t{4});
    return 3.0 * domain_width / static_cast<double>(n - 1);
}

std::vector<double> select_probes(const std::vector<double>& items,
                                  double reference_value) {
    if (items.size() <= 3) return items;
    std::vector<double> probes;
    probes.push_back(items.front());
    probes.push_back(items.back());
    auto atm_it = std::min_element(items.begin(), items.end(),
        [&](double a, double b) {
            return std::abs(a - reference_value) < std::abs(b - reference_value);
        });
    if (*atm_it != items.front() && *atm_it != items.back()) {
        probes.push_back(*atm_it);
    }
    return probes;
}

double total_discrete_dividends(const std::vector<Dividend>& dividends,
                                double maturity) {
    double total = 0.0;
    for (const auto& div : dividends) {
        if (div.calendar_time > 0.0 && div.calendar_time < maturity &&
            div.amount > 0.0) {
            total += div.amount;
        }
    }
    return total;
}

SegmentBoundaries compute_segment_boundaries(
    const std::vector<Dividend>& dividends, double maturity,
    double tau_min, double tau_max)
{
    constexpr double kInset = 5e-4;  // gap half-width around dividend in tau-space

    // Filter and merge same-date dividends (shared with legacy builder)
    auto merged = filter_and_merge_dividends(dividends, maturity);

    // Collect tau-space split points
    std::vector<double> splits;
    for (const auto& div : merged) {
        double tau_split = maturity - div.calendar_time;
        if (tau_split > tau_min + 2 * kInset && tau_split < tau_max - 2 * kInset) {
            splits.push_back(tau_split);
        }
    }
    std::sort(splits.begin(), splits.end());

    // Deduplicate splits that are too close (would create overlapping gaps)
    std::vector<double> unique_splits;
    for (double sp : splits) {
        if (!unique_splits.empty() &&
            sp - unique_splits.back() < 4 * kInset) {
            // Merge: keep midpoint of the cluster
            unique_splits.back() = (unique_splits.back() + sp) * 0.5;
        } else {
            unique_splits.push_back(sp);
        }
    }

    // Build boundaries and gap flags.
    // Pattern per dividend: real, GAP, real, GAP, real, ...
    // Odd-indexed segments (1, 3, 5, ...) are gaps.
    std::vector<double> bounds;
    std::vector<bool> is_gap;
    bounds.push_back(tau_min);
    for (double sp : unique_splits) {
        is_gap.push_back(false);  // real segment before this gap
        bounds.push_back(sp - kInset);
        is_gap.push_back(true);   // gap segment around dividend
        bounds.push_back(sp + kInset);
    }
    is_gap.push_back(false);  // final real segment after last gap
    bounds.push_back(tau_max);

    return {std::move(bounds), std::move(is_gap)};
}

TauSegmentSplit make_tau_split_from_segments(
    const std::vector<double>& bounds,
    const std::vector<bool>& is_gap,
    double K_ref)
{
    const size_t n_seg = is_gap.size();
    std::vector<double> tau_start, tau_end, tau_min, tau_max;

    for (size_t s = 0; s < n_seg; ++s) {
        if (is_gap[s]) continue;

        double start = bounds[s];
        double end = bounds[s + 1];

        // Absorb gap to the left
        if (s > 0 && is_gap[s - 1]) {
            double gap_lo = bounds[s - 1];
            double gap_hi = bounds[s];
            start = (gap_lo + gap_hi) * 0.5;
        }

        // Absorb gap to the right
        if (s + 1 < n_seg && is_gap[s + 1]) {
            double gap_lo = bounds[s + 1];
            double gap_hi = bounds[s + 2];
            end = (gap_lo + gap_hi) * 0.5;
        }

        tau_start.push_back(start);
        tau_end.push_back(end);
        // Routing remains contiguous, but the fitted support excludes gaps.
        tau_min.push_back(bounds[s] - start);
        tau_max.push_back(bounds[s + 1] - start);
    }

    return TauSegmentSplit(
        std::move(tau_start), std::move(tau_end),
        std::move(tau_min), std::move(tau_max), K_ref);
}

std::vector<std::pair<double, double>> admitted_maturity_intervals(
    const TauSegmentSplit& split, double tau_min, double tau_max)
{
    std::vector<std::pair<double, double>> intervals;
    for (size_t i = 0; i < split.tau_start().size(); ++i) {
        double lo = std::max(tau_min, split.tau_start()[i] + split.tau_min()[i]);
        double hi = std::min(tau_max, split.tau_start()[i] + split.tau_max()[i]);
        // Addition/subtraction round trips at a physical endpoint can need
        // one representable step inward to satisfy the split's own predicate.
        for (int j = 0; j < 4 && lo < hi && !split.contains_maturity(lo); ++j) {
            lo = std::nextafter(lo, hi);
        }
        for (int j = 0; j < 4 && lo < hi && !split.contains_maturity(hi); ++j) {
            hi = std::nextafter(hi, lo);
        }
        if (!(hi > lo) || !split.contains_maturity(lo) || !split.contains_maturity(hi)) continue;
        if (!intervals.empty() && lo <= intervals.back().second) {
            intervals.back().second = std::max(intervals.back().second, hi);
        } else {
            intervals.emplace_back(lo, hi);
        }
    }
    return intervals;
}

MaxGridSizes aggregate_max_sizes(const std::vector<RefinementResult>& probe_results) {
    MaxGridSizes s;
    for (const auto& pr : probe_results) {
        s.moneyness = std::max(s.moneyness, pr.moneyness.size());
        s.vol = std::max(s.vol, pr.vol.size());
        s.rate = std::max(s.rate, pr.rate.size());
        s.tau_points = std::max(s.tau_points, pr.tau_points);
    }
    return s;
}

namespace {

template <typename Grids>
auto& grid_axis(Grids& grids, size_t axis) {
    const std::array axes{&grids.moneyness, &grids.tau, &grids.vol, &grids.rate};
    return *axes[axis];
}

bool ordered_finite_nodes(const std::vector<double>& nodes) {
    return nodes.size() >= 4 &&
        std::ranges::all_of(nodes, [](double x) { return std::isfinite(x); }) &&
        std::adjacent_find(nodes.begin(), nodes.end(), std::greater_equal<double>{}) == nodes.end();
}

template <typename Grids>
auto grid_coordinates(const Grids& grids) {
    return std::tie(grids.moneyness, grids.tau, grids.vol, grids.rate);
}

std::vector<double> nodes_in_interval(const std::vector<double>& nodes, double lo, double hi) {
    return {std::lower_bound(nodes.begin(), nodes.end(), lo),
            std::upper_bound(nodes.begin(), nodes.end(), hi)};
}

// Add only existing candidate positions. Maximum distance from the retained
// set provides deterministic coverage; the sorted input breaks ties low-first.
void fill_from_positions(std::vector<double>& selected,
                         const std::vector<double>& available, size_t cap) {
    while (selected.size() < cap) {
        std::optional<double> next;
        double greatest = -1.0;
        for (double x : available) {
            auto right = std::lower_bound(selected.begin(), selected.end(), x);
            if (right != selected.end() && *right == x) continue;
            double distance = std::numeric_limits<double>::infinity();
            if (right != selected.end()) distance = *right - x;
            if (right != selected.begin()) distance = std::min(distance, x - *(right - 1));
            if (distance > greatest) { greatest = distance; next = x; }
        }
        if (!next) break;
        selected.insert(std::lower_bound(selected.begin(), selected.end(), *next), *next);
    }
}

}  // namespace

std::expected<std::vector<SeededGrids>, PriceTableError>
aggregate_refinement_grids(
    std::span<const RefinementResult> probes, const SeededGrids& required,
    size_t max_points_per_dim,
    std::span<const std::pair<double, double>> tau_intervals) {
    if (probes.empty() || max_points_per_dim < 4) return std::unexpected(
        PriceTableError{PriceTableErrorCode::InvalidConfig});
    SeededGrids united;
    std::array<std::vector<std::pair<double, double>>, 4> axis_intervals;
    bool overflow = false;
    for (size_t axis = 0; axis < 4; ++axis) {
        const auto& seeds = grid_axis(required, axis);
        if (!ordered_finite_nodes(seeds)) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig, axis});
        auto& intervals = axis_intervals[axis];
        if (axis == 1 && !tau_intervals.empty()) {
            intervals.assign(tau_intervals.begin(), tau_intervals.end());
        } else {
            intervals.emplace_back(seeds.front(), seeds.back());
        }
        for (size_t i = 0; i < intervals.size(); ++i) {
            auto [lo, hi] = intervals[i];
            if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(hi - lo) || !(lo < hi) ||
                (i > 0 && !(intervals[i - 1].second < lo))) return std::unexpected(
                    PriceTableError{PriceTableErrorCode::InvalidConfig, axis});
        }
        const auto covered = [&intervals](double x) {
            return std::ranges::any_of(intervals, [x](auto interval) {
                return x >= interval.first && x <= interval.second;
            });
        };
        auto& nodes = grid_axis(united, axis);
        nodes = seeds;
        for (const auto& probe : probes) {
            const auto& selected = grid_axis(probe, axis);
            if (!ordered_finite_nodes(selected)) return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig, axis});
            nodes.insert(nodes.end(), selected.begin(), selected.end());
        }
        if (!std::ranges::all_of(nodes, covered)) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig, axis});
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        for (auto [lo, hi] : intervals) {
            const auto first = std::lower_bound(seeds.begin(), seeds.end(), lo);
            const auto last = std::upper_bound(seeds.begin(), seeds.end(), hi);
            const size_t count = last - first;
            if (count < 4 || count > max_points_per_dim ||
                *first != lo || *(last - 1) != hi) return std::unexpected(
                    PriceTableError{PriceTableErrorCode::InvalidConfig, axis, count});
            const size_t total = std::upper_bound(nodes.begin(), nodes.end(), hi) -
                std::lower_bound(nodes.begin(), nodes.end(), lo);
            overflow |= total > max_points_per_dim;
        }
    }
    if (!overflow) return std::vector<SeededGrids>{std::move(united)};

    // One coverage candidate and at most three probe-priority candidates.
    // The financial caller evaluates all of them on identical references.
    std::vector<const RefinementResult*> ordered;
    for (const auto& probe : probes) ordered.push_back(&probe);
    std::sort(ordered.begin(), ordered.end(), [](auto* a, auto* b) {
        return grid_coordinates(*a) < grid_coordinates(*b);
    });
    const size_t priorities = std::min(size_t{3}, ordered.size());
    std::vector<SeededGrids> candidates;
    for (size_t priority = 0; priority <= priorities; ++priority) {
        SeededGrids candidate;
        for (size_t axis = 0; axis < 4; ++axis) {
            for (auto [lo, hi] : axis_intervals[axis]) {
                auto available = nodes_in_interval(grid_axis(united, axis), lo, hi);
                std::vector<double> selected;
                if (available.size() <= max_points_per_dim) {
                    selected = std::move(available);
                } else {
                    selected = nodes_in_interval(grid_axis(required, axis), lo, hi);
                    if (priority > 0) fill_from_positions(selected,
                        nodes_in_interval(grid_axis(*ordered[priority - 1], axis), lo, hi),
                        max_points_per_dim);
                    fill_from_positions(selected, available, max_points_per_dim);
                }
                auto& result = grid_axis(candidate, axis);
                result.insert(result.end(), selected.begin(), selected.end());
            }
        }
        candidates.push_back(std::move(candidate));
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return grid_coordinates(a) < grid_coordinates(b);
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return grid_coordinates(a) == grid_coordinates(b);
    }), candidates.end());
    return candidates;
}

std::expected<SeededGrids, PriceTableError>
refine_aggregate_grids(
    const SeededGrids& retained, size_t max_points_per_dim,
    std::span<const std::pair<double, double>> tau_intervals) {
    const RefinementResult probe{.moneyness = retained.moneyness, .tau = retained.tau,
        .vol = retained.vol, .rate = retained.rate};
    auto valid = aggregate_refinement_grids(std::span{&probe, 1}, retained,
                                           max_points_per_dim, tau_intervals);
    if (!valid) return std::unexpected(valid.error());
    SeededGrids result;
    constexpr std::array<size_t, 4> additions{2, 2, 1, 1};
    for (size_t axis = 0; axis < 4; ++axis) {
        const auto& nodes = grid_axis(retained, axis);
        std::vector<std::pair<double, double>> intervals;
        if (axis == 1 && !tau_intervals.empty()) intervals.assign(tau_intervals.begin(), tau_intervals.end());
        else intervals.emplace_back(nodes.front(), nodes.back());
        for (auto [lo, hi] : intervals) {
            auto part = nodes_in_interval(nodes, lo, hi);
            const size_t target = part.size() + std::min(additions[axis], max_points_per_dim - part.size());
            while (part.size() < target) {
                size_t gap = 0;
                for (size_t j = 1; j + 1 < part.size(); ++j) {
                    if (part[j + 1] - part[j] > part[gap + 1] - part[gap]) gap = j;
                }
                const double midpoint = std::midpoint(part[gap], part[gap + 1]);
                if (!(midpoint > part[gap] && midpoint < part[gap + 1])) break;
                part.insert(part.begin() + gap + 1, midpoint);
            }
            auto& out = grid_axis(result, axis);
            out.insert(out.end(), part.begin(), part.end());
        }
    }
    return result;
}

std::vector<double> linspace(double lo, double hi, size_t n) {
    if (n < 2) {
        return {lo, hi};  // Minimum valid grid
    }
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + (hi - lo) * i / (n - 1);
    }
    return v;
}

std::vector<double> seed_grid(const std::vector<double>& user_knots,
                               double lo, double hi, size_t fallback_n) {
    std::vector<double> grid;

    if (!user_knots.empty()) {
        // Filter knots to domain bounds
        for (double v : user_knots) {
            if (v >= lo && v <= hi) {
                grid.push_back(v);
            }
        }
        // Ensure domain endpoints are included
        if (grid.empty() || grid.front() > lo + 1e-12) {
            grid.insert(grid.begin(), lo);
        }
        if (grid.back() < hi - 1e-12) {
            grid.push_back(hi);
        }
        std::sort(grid.begin(), grid.end());
        grid.erase(std::unique(grid.begin(), grid.end()), grid.end());

        // Need minimum 4 points for cubic B-spline
        while (grid.size() < 4) {
            // Insert midpoint in largest gap
            double max_gap = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i + 1 < grid.size(); ++i) {
                double gap = grid[i + 1] - grid[i];
                if (gap > max_gap) { max_gap = gap; max_idx = i; }
            }
            grid.push_back((grid[max_idx] + grid[max_idx + 1]) / 2.0);
            std::sort(grid.begin(), grid.end());
        }
    } else {
        grid = linspace(lo, hi, fallback_n);
    }

    return grid;
}

static std::vector<std::array<double, 4>> generate_validation_samples(
    const AdaptiveGridParams& params,
    size_t iteration,
    const std::array<std::pair<double, double>, 4>& bounds,
    const std::array<std::vector<size_t>, 4>& focus_bins,
    bool focus_active) {
    const size_t total_samples = params.validation_samples;
    bool has_focus_bins = focus_active;
    if (focus_active) {
        has_focus_bins = std::any_of(focus_bins.begin(), focus_bins.end(),
                                     [](const std::vector<size_t>& bins) { return !bins.empty(); });
    }

    size_t base_samples = (has_focus_bins && total_samples > 1)
        ? std::max<size_t>(total_samples / 2, 1)
        : total_samples;
    size_t targeted_samples = has_focus_bins ? total_samples - base_samples : 0;

    auto base_unit_samples = latin_hypercube_4d(base_samples,
                                                params.lhs_seed + iteration);

    std::vector<std::array<double, 4>> samples = scale_lhs_samples(base_unit_samples, bounds);

    if (targeted_samples > 0 && has_focus_bins) {
        std::mt19937_64 targeted_rng(params.lhs_seed ^ (iteration * 1315423911ULL + 0x9e3779b97f4a7c15ULL));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        std::vector<std::array<double, 4>> targeted_unit;
        targeted_unit.reserve(targeted_samples);

        for (size_t i = 0; i < targeted_samples; ++i) {
            std::array<double, 4> point{};
            for (size_t d = 0; d < 4; ++d) {
                double u = uniform(targeted_rng);
                if (!focus_bins[d].empty()) {
                    const auto& dim_bins = focus_bins[d];
                    size_t bin = dim_bins[i % dim_bins.size()];
                    double bin_lo = static_cast<double>(bin) / ErrorBins::N_BINS;
                    double bin_hi = static_cast<double>(bin + 1) / ErrorBins::N_BINS;
                    double span = bin_hi - bin_lo;
                    point[d] = bin_lo + u * span;
                } else {
                    point[d] = u;
                }
            }
            targeted_unit.push_back(point);
        }

        auto targeted_scaled = scale_lhs_samples(targeted_unit, bounds);
        samples.insert(samples.end(), targeted_scaled.begin(), targeted_scaled.end());
    }

    return samples;
}

/// Score a candidate surface over freshly drawn samples (spec D4).
///
/// Failed reference solves are skipped (they carry no evidence about the
/// surface), as are points the score fn filters out (`std::nullopt`).
/// Non-finite prices or scores do not contribute to the error statistics but
/// do clear `all_finite`, which disqualifies the candidate.
static SampleEval evaluate_fresh_samples(
    const std::vector<HoldoutPoint>& samples,
    const SurfaceHandle& handle,
    const PrepareRefsFn& prepare_refs,
    const ScoreErrorFn& score,
    const RefinementContext& ctx,
    double target_iv_error) {
    SampleEval ev;
    double sum_error = 0.0;

    for (const auto& sample : samples) {
        double m = sample.coords[0];
        double tau = sample.coords[1];
        double sigma = sample.coords[2];
        double rate = sample.coords[3];

        // Interpolated price from surface via callback
        double strike = sample.strike;
        double interp_price = handle.price(sample.spot, strike, tau, sigma, rate);

        // Fresh FD refs (price + vega) for reference via callback.
        // Note: prepare_refs performs 3 PDE solves internally (base + two
        // sigma-bump solves), but we count 1 per successful sample here to
        // preserve pre-existing pde_solves_validation accounting -- callers
        // (e.g. bspline_adaptive.cpp) multiply this count by 3 downstream.
        auto refs_result = prepare_refs(sample.spot, strike, tau, sigma, rate);

        if (!refs_result.has_value()) {
            continue;  // Skip failed solves
        }

        ev.pde_solves_validation++;

        auto scored = score(
            interp_price, refs_result.value(),
            sample.spot, strike, tau, sigma, rate);

        // A NaN price on an in-domain fresh sample disqualifies the candidate
        // even if the fixed holdout missed that location (spec D5) -- and
        // even where the error metric is filtered out: the filter says the
        // *IV error* is undefined there, not that garbage prices are fine.
        if (!std::isfinite(interp_price)) {
            ev.all_finite = false;
            continue;
        }
        if (!scored.has_value()) {
            continue;  // filtered: the metric is undefined here (D4)
        }
        const double iv_error = *scored;
        if (!std::isfinite(iv_error) || iv_error < 0.0) {
            ev.all_finite = false;
            continue;
        }

        ev.max_error = std::max(ev.max_error, iv_error);
        sum_error += iv_error;
        ev.measured++;

        // Normalize position for error bins over the SAMPLE domain (spec
        // D2) -- bins must line up with the domain the samples came from.
        const auto& sb = ctx.sample_bounds;
        const auto normalized = [](double value, double lo, double hi) {
            // A singleton requested axis has one physical coordinate, not
            // a zero-denominator interval. It is never selected for refinement.
            return hi > lo ? (value - lo) / (hi - lo) : 0.5;
        };
        std::array<double, 4> norm_pos = {{
            normalized(m, sb.m_min, sb.m_max),
            normalized(tau, sb.tau_min, sb.tau_max),
            normalized(sigma, sb.sigma_min, sb.sigma_max),
            normalized(rate, sb.rate_min, sb.rate_max)
        }};
        ev.error_bins.record_error(norm_pos, iv_error, target_iv_error);
    }

    ev.avg_error = ev.measured > 0
        ? sum_error / static_cast<double>(ev.measured)
        : 0.0;
    return ev;
}

/// Score a candidate surface over the fixed holdout using cached references
/// (spec D4): interpolations plus arithmetic, no FD solves.
///
/// Any non-finite price or score makes the whole holdout score NaN, which
/// both disqualifies the candidate (D5 viability) and removes it from the
/// exploration-base ranking (which requires finite scores).
///
/// The arithmetic is `detail::score_final_surface`'s -- the segmented
/// builders' final gate scores its surfaces exactly the way the loop scores
/// its candidates, and the two must not drift apart.  This wrapper only
/// re-shapes the result into `SampleEval`; the holdout leaves
/// `pde_solves_validation` and `error_bins` at their defaults (it performs no
/// solves, and refinement bins come from the fresh samples).
static SampleEval evaluate_holdout(
    const std::vector<HoldoutPoint>& holdout,
    const SurfaceHandle& handle,
    const ScoreErrorFn& score,
    const RefinementContext& ctx) {
    const auto scored = detail::score_final_surface(holdout, handle, score, ctx);
    SampleEval ev;
    ev.max_error = scored.max_error;
    ev.avg_error = scored.avg_error;
    ev.measured = scored.measured;
    ev.all_finite = scored.all_finite;
    return ev;
}

/// Pick the highest-scoring untried axis (spec D6 step 1-2).
///
/// score[d] = concentration = max bin count / total bin count, defined as 0
/// when the bin total is zero.  Ties -- including the all-zero case -- break
/// by dimension order (moneyness, tau, sigma, rate).  Returns -1 when every
/// axis has been tried.
static int pick_refinement_axis(const ErrorBins& bins,
                                const std::array<bool, 4>& tried) {
    int best_dim = -1;
    double best_score = -1.0;
    for (size_t d = 0; d < ErrorBins::N_DIMS; ++d) {
        if (tried[d]) continue;
        size_t max_count = std::ranges::max(bins.bin_counts[d]);
        size_t total = std::reduce(bins.bin_counts[d].begin(),
                                   bins.bin_counts[d].end());
        double concentration = total == 0
            ? 0.0
            : static_cast<double>(max_count) / static_cast<double>(total);
        if (concentration > best_score) {  // strict: ties keep dim order
            best_score = concentration;
            best_dim = static_cast<int>(d);
        }
    }
    return best_dim;
}

/// Convert an axis's problematic bins into physical focus intervals (D2).
static std::vector<std::pair<double, double>> bins_to_intervals(
    const ErrorBins& bins, size_t dim,
    const std::pair<double, double>& axis_bounds) {
    auto problematic = bins.problematic_bins(dim);
    std::vector<std::pair<double, double>> intervals;
    intervals.reserve(problematic.size());
    constexpr double kNBins = static_cast<double>(ErrorBins::N_BINS);
    const double span = axis_bounds.second - axis_bounds.first;
    if (span == 0.0) return intervals;
    for (size_t bin : problematic) {
        intervals.push_back({
            axis_bounds.first + span * static_cast<double>(bin) / kNBins,
            axis_bounds.first + span * static_cast<double>(bin + 1) / kNBins});
    }
    return intervals;
}

/// Monotonicity statistics for the returned candidate (spec D7).
///
/// Diagnostics only, never a gate: at each valid holdout (m, tau, r), scan 7
/// equally spaced sigma across the user sigma-range and count steps where the
/// price falls by more than the noise floor.
void detail::scan_monotonicity(const std::vector<HoldoutPoint>& holdout,
                               const SurfaceHandle& handle,
                               const RefinementContext& ctx,
                               double target_iv_error,
                               double vega_floor,
                               BuildDiagnostics& diag) {
    const double sigma_lo = ctx.sample_bounds.sigma_min;
    const double sigma_hi = ctx.sample_bounds.sigma_max;
    if (!(sigma_hi > sigma_lo)) {
        return;  // degenerate sigma range: scan skipped
    }
    const auto sigmas = linspace(sigma_lo, sigma_hi, kMonotonicityPoints);
    for (const auto& pt : holdout) {
        const double tol = std::max(1e-8 * pt.spot, target_iv_error * vega_floor);
        double prev_price = std::numeric_limits<double>::quiet_NaN();
        double prev_sigma = std::numeric_limits<double>::quiet_NaN();
        for (double sigma : sigmas) {
            double price = handle.price(pt.spot, pt.strike, pt.coords[1],
                                        sigma, pt.coords[3]);
            if (!std::isfinite(price)) {
                diag.monotonicity_points_invalid++;
                prev_price = std::numeric_limits<double>::quiet_NaN();
                prev_sigma = sigma;
                continue;
            }
            if (std::isfinite(prev_price) && price < prev_price - tol) {
                diag.monotonicity_violations++;
                double slope = (price - prev_price) / (sigma - prev_sigma);
                diag.worst_vega_slope = std::min(diag.worst_vega_slope, slope);
            }
            prev_price = price;
            prev_sigma = sigma;
        }
    }
}

// ============================================================================
// Final-surface validation for the segmented builders (spec D9)
// ============================================================================

std::expected<detail::FinalValidationSet, PriceTableError>
detail::prepare_final_validation(const AdaptiveGridParams& params,
                                 const RefinementContext& ctx,
                                 const PrepareRefsFn& prepare_refs,
                                 uint64_t seed) {
    if (!valid_measurement_domain(ctx)) return std::unexpected(
        PriceTableError{PriceTableErrorCode::InvalidConfig});
    const std::array<std::pair<double, double>, 4> axis_bounds = {{
        {ctx.sample_bounds.m_min, ctx.sample_bounds.m_max},
        {ctx.sample_bounds.tau_min, ctx.sample_bounds.tau_max},
        {ctx.sample_bounds.sigma_min, ctx.sample_bounds.sigma_max},
        {ctx.sample_bounds.rate_min, ctx.sample_bounds.rate_max}
    }};

    auto unit = latin_hypercube_4d(params.validation_samples, seed);
    auto scaled = scale_lhs_samples(unit, axis_bounds);

    FinalValidationSet set;
    set.points.reserve(scaled.size());
    auto physical = physical_samples(scaled, ctx, seed);
    if (!physical) return std::unexpected(physical.error());
    for (auto pt : *physical) {
        ++set.ref_attempts;
        auto refs = prepare_refs(pt.spot, pt.strike, pt.coords[1], pt.coords[2], pt.coords[3]);
        if (!refs.has_value() || !std::isfinite(refs->ref_price) || !std::isfinite(refs->vega)) {
            ++set.invalid;
            continue;
        }
        pt.refs = *refs;
        set.points.push_back(std::move(pt));
    }

    const size_t min_valid =
        std::max<size_t>(4, params.validation_samples / 4);
    if (set.points.size() < min_valid) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ValidationFailed});
    }
    return set;
}

detail::FinalScore detail::score_final_surface(
    const std::vector<ValidationPoint>& points,
    const SurfaceHandle& handle,
    const ScoreErrorFn& score,
    const RefinementContext& ctx) {
    FinalScore ev;
    double sum_error = 0.0;

    for (const auto& pt : points) {
        const double tau = pt.coords[1];
        const double sigma = pt.coords[2];
        const double rate = pt.coords[3];
        const double interp =
            handle.price(pt.spot, pt.strike, tau, sigma, rate);
        const auto err = score(interp, pt.refs, pt.spot, pt.strike,
                               tau, sigma, rate);
        // A NaN price is garbage whether or not the metric is defined here.
        if (!std::isfinite(interp)) {
            ev.all_finite = false;
            ++ev.skipped;
            continue;
        }
        if (!err.has_value()) {
            // Deliberately filtered (TV/K or vega floor): no evidence either
            // way, so it enters no statistic and cannot certify the surface.
            ++ev.filtered;
            continue;
        }
        if (!std::isfinite(*err) || *err < 0.0) {
            ev.all_finite = false;
            ++ev.skipped;
            continue;
        }
        ev.max_error = std::max(ev.max_error, *err);
        sum_error += *err;
        // Every measured point counts -- a zero error is a measurement, not a
        // missing one, and using it as the avg denominator's gate produced a
        // spurious "target met" for a surface nobody had measured.
        ++ev.measured;
    }

    if (!ev.all_finite) {
        ev.max_error = std::numeric_limits<double>::quiet_NaN();
        ev.avg_error = std::numeric_limits<double>::quiet_NaN();
        return ev;
    }
    ev.avg_error = ev.measured > 0
        ? sum_error / static_cast<double>(ev.measured)
        : 0.0;
    return ev;
}

bool detail::needs_final_retry(const FinalScore& original,
                               double target_iv_error) {
    return original.max_error > target_iv_error || !original.viable();
}

detail::FinalPick detail::select_final_surface(
    const FinalScore& original,
    const std::optional<FinalScore>& retry) {
    const bool orig_ok = original.viable();
    const bool retry_ok = retry.has_value() && retry->viable();

    if (orig_ok && retry_ok) {
        return retry->max_error < original.max_error ? FinalPick::Retry
                                                     : FinalPick::Original;
    }
    if (orig_ok) return FinalPick::Original;
    if (retry_ok) return FinalPick::Retry;
    return FinalPick::None;
}

SeededGrids seed_refinement_grids(const AdaptiveGridParams& params,
                                  const RefinementContext& ctx,
                                  const InitialGrids& initial_grids) {
    SeededGrids g;
    if (initial_grids.exact) {
        // Use grids exactly as provided (Chebyshev CGL/CC nodes)
        g.moneyness = initial_grids.moneyness;
        g.tau = initial_grids.tau;
        g.vol = initial_grids.vol;
        g.rate = initial_grids.rate;
        return g;
    }

    // Seed grids from user-provided knots (or linspace fallback) over the
    // FIT domain.  This ensures user-specified knots (e.g. benchmark vols)
    // are always grid points.
    g.moneyness = seed_grid(initial_grids.moneyness, ctx.bounds.m_min,
                            ctx.bounds.m_max, params.min_moneyness_points);
    g.tau = seed_grid(initial_grids.tau, ctx.bounds.tau_min,
                      ctx.bounds.tau_max, 5);
    g.vol = seed_grid(initial_grids.vol, ctx.bounds.sigma_min,
                      ctx.bounds.sigma_max, 5);
    g.rate = seed_grid(initial_grids.rate, ctx.bounds.rate_min,
                       ctx.bounds.rate_max, 4);

    // Moneyness needs higher density than the other axes (exercise boundary
    // curvature): insert midpoints in the largest gaps until the minimum.
    while (g.moneyness.size() < params.min_moneyness_points) {
        double max_gap = 0.0;
        size_t max_idx = 0;
        for (size_t i = 0; i + 1 < g.moneyness.size(); ++i) {
            double gap = g.moneyness[i + 1] - g.moneyness[i];
            if (gap > max_gap) { max_gap = gap; max_idx = i; }
        }
        g.moneyness.push_back(
            (g.moneyness[max_idx] + g.moneyness[max_idx + 1]) / 2.0);
        std::sort(g.moneyness.begin(), g.moneyness.end());
    }
    return g;
}

std::expected<RefinementResult, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    RefineFn refine_fn,
    const RefinementContext& ctx,
    const PrepareRefsFn& prepare_refs,
    const ScoreErrorFn& score,
    const InitialGrids& initial_grids,
    const RefineStateHooks& hooks)
{
    // ---------------------------------------------------------------------
    // 1. Parameter validation (spec D3)
    // ---------------------------------------------------------------------
    const auto invalid_config = [] {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig});
    };

    if (!valid_measurement_domain(ctx)) return invalid_config();
    if (!std::isfinite(params.target_iv_error) || params.target_iv_error <= 0.0) {
        return invalid_config();
    }
    if (!std::isfinite(params.vega_floor) || params.vega_floor <= 0.0) {
        return invalid_config();
    }
    if (!std::isfinite(params.refinement_factor) ||
        params.refinement_factor <= 1.0) {
        return invalid_config();
    }
    if (params.max_iter < 1) {
        return invalid_config();
    }
    if (params.validation_samples < 8) {
        return invalid_config();
    }
    // B-spline requires minimum 4 control points per dimension
    if (params.min_moneyness_points < 4) {
        return invalid_config();
    }

    // Grids are seeded over and span the FIT domain, while all measurement (validation sampling, bin normalization,
    // focus intervals) happens over the user-facing SAMPLE domain (spec D2).
    const std::array<std::pair<double, double>, 4> sample_axis_bounds = {{
        {ctx.sample_bounds.m_min, ctx.sample_bounds.m_max},
        {ctx.sample_bounds.tau_min, ctx.sample_bounds.tau_max},
        {ctx.sample_bounds.sigma_min, ctx.sample_bounds.sigma_max},
        {ctx.sample_bounds.rate_min, ctx.sample_bounds.rate_max}
    }};

    // Singleton axes are sampled at their one requested coordinate. They
    // need numerical fit support, but have no requested interval to refine.
    std::array<bool, 4> fixed_axes{};
    for (size_t d = 0; d < sample_axis_bounds.size(); ++d) {
        const auto [lo, hi] = sample_axis_bounds[d];
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi < lo) {
            return invalid_config();
        }
        fixed_axes[d] = hi == lo;
    }

    auto seeded = seed_refinement_grids(params, ctx, initial_grids);
    std::vector<double> moneyness_grid = std::move(seeded.moneyness);
    std::vector<double> maturity_grid = std::move(seeded.tau);
    std::vector<double> vol_grid = std::move(seeded.vol);
    std::vector<double> rate_grid = std::move(seeded.rate);

    // ---------------------------------------------------------------------
    // 3. Fixed holdout with cached references (spec D4)
    // ---------------------------------------------------------------------
    constexpr uint64_t kHoldoutSeedMix = 0x484F4C44ULL;  // "HOLD"
    auto holdout_unit = latin_hypercube_4d(params.validation_samples,
                                           params.lhs_seed ^ kHoldoutSeedMix);
    auto holdout_scaled = scale_lhs_samples(holdout_unit, sample_axis_bounds);

    std::vector<HoldoutPoint> holdout;
    holdout.reserve(holdout_scaled.size());
    size_t holdout_invalid = 0;
    auto physical_holdout = physical_samples(holdout_scaled, ctx, params.lhs_seed ^ kHoldoutSeedMix);
    if (!physical_holdout) return std::unexpected(physical_holdout.error());
    for (auto pt : *physical_holdout) {
        auto refs = prepare_refs(pt.spot, pt.strike, pt.coords[1], pt.coords[2], pt.coords[3]);
        if (!refs.has_value() || !std::isfinite(refs->ref_price) || !std::isfinite(refs->vega)) {
            ++holdout_invalid;
            continue;
        }
        pt.refs = *refs;
        holdout.push_back(std::move(pt));
    }

    // A holdout that cannot measure cannot certify retention.
    const size_t min_valid_holdout =
        std::max<size_t>(4, params.validation_samples / 4);
    if (holdout.size() < min_valid_holdout) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ValidationFailed});
    }

    // ---------------------------------------------------------------------
    // 4. Refinement loop with candidate retention (spec D5/D6)
    // ---------------------------------------------------------------------
    const auto snapshot_state = [&hooks]() -> std::shared_ptr<const void> {
        return hooks.snapshot ? hooks.snapshot()
                              : std::shared_ptr<const void>{};
    };
    const auto restore_state =
        [&hooks](const std::shared_ptr<const void>& snap) {
        if (hooks.restore) hooks.restore(snap);
    };

    RefinementResult result;
    BuildDiagnostics& diag = result.diagnostics;
    diag.iterations.reserve(params.max_iter + 1);
    // `holdout_points` is the size of the *usable reference set*: points whose
    // FD refs were produced and finite.  How many of them actually measured
    // the returned surface is `holdout_points_measured`, filled in from the
    // picked candidate below -- the two differ by the points the score fn
    // filters out (TV/K, vega floor).
    diag.holdout_points = holdout.size();
    diag.holdout_points_invalid = holdout_invalid;

    std::vector<Candidate> candidates;
    candidates.reserve(params.max_iter);

    Candidate base;  ///< exploration base (spec D5 roles)
    bool have_base = false;
    bool have_finite_base = false;
    double prev_best_holdout = std::numeric_limits<double>::infinity();

    std::array<bool, 4> tried = fixed_axes;
    std::array<std::vector<size_t>, 4> focus_bins;
    bool focus_active = false;

    size_t iteration = 0;            ///< built iterations (budget consumed)
    int pending_refined_dim = -1;    ///< axis that produced the current grids
    std::optional<SurfaceHandle> last_handle;
    size_t last_built_iteration = 0;
    bool last_attempt_failed = false;

    while (true) {
        auto iter_start = std::chrono::steady_clock::now();

        IterationStats stats;
        stats.iteration = iteration;
        stats.refined_dim = pending_refined_dim;
        stats.grid_sizes = {
            moneyness_grid.size(),
            maturity_grid.size(),
            vol_grid.size(),
            rate_grid.size()
        };

        // a. BUILD via callback
        auto surface_result =
            build_fn(moneyness_grid, maturity_grid, vol_grid, rate_grid);

        if (!surface_result.has_value()) {
            // Seed build failure is terminal (spec D5).
            if (iteration == 0) {
                return std::unexpected(surface_result.error());
            }
            // A failed refinement trial must not strand exploration: mark the
            // axis tried, roll back to the exploration base, and continue.
            if (pending_refined_dim >= 0 && pending_refined_dim < 4) {
                tried[static_cast<size_t>(pending_refined_dim)] = true;
            }
            moneyness_grid = base.moneyness;
            maturity_grid = base.tau;
            vol_grid = base.vol;
            rate_grid = base.rate;
            restore_state(base.state);

            stats.build_failed = true;
            stats.elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - iter_start).count();
            diag.iterations.push_back(stats);
            diag.build_failure_fallback = true;
            last_attempt_failed = true;
            ++iteration;
        } else {
            auto& handle = surface_result.value();
            stats.pde_solves_table = handle.pde_solves;

            // b. FRESH SAMPLES (from the sample domain, spec D2)
            auto samples = generate_validation_samples(
                params, iteration, sample_axis_bounds, focus_bins,
                focus_active);
            auto physical = physical_samples(samples, ctx, params.lhs_seed + iteration);
            if (!physical) return std::unexpected(physical.error());
            auto fresh = evaluate_fresh_samples(
                *physical, handle, prepare_refs, score, ctx, params.target_iv_error);
            stats.pde_solves_validation = fresh.pde_solves_validation;
            stats.max_error = fresh.max_error;
            stats.avg_error = fresh.avg_error;

            // c. HOLDOUT (cached refs, no FD solves)
            auto hold = evaluate_holdout(holdout, handle, score, ctx);

            // d. RECORD THE CANDIDATE
            Candidate cand;
            cand.moneyness = moneyness_grid;
            cand.tau = maturity_grid;
            cand.vol = vol_grid;
            cand.rate = rate_grid;
            cand.state = snapshot_state();
            cand.holdout_max = hold.max_error;
            cand.holdout_avg = hold.avg_error;
            cand.holdout_measured = hold.measured;
            cand.bins = fresh.error_bins;
            cand.iteration = iteration;
            cand.fresh_converged =
                fresh.measured > 0 &&
                fresh.max_error <= params.target_iv_error;
            // `hold.measured > 0`: a candidate whose every holdout point was
            // filtered out has been measured nowhere, and a max of 0 over an
            // empty set must not certify it (spec D5, final-review amendment
            // 2026-08-29).
            cand.viable = hold.all_finite && fresh.all_finite &&
                          hold.measured > 0 &&
                          std::isfinite(hold.max_error) &&
                          hold.max_error <= kViabilityBound;

            stats.elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - iter_start).count();
            diag.iterations.push_back(stats);

            last_handle = handle;
            last_built_iteration = iteration;
            last_attempt_failed = false;
            ++iteration;

            // e. WALK BOOKKEEPING (spec D6 step 5)
            if (pending_refined_dim >= 0 && pending_refined_dim < 4) {
                if (std::isfinite(cand.holdout_max) &&
                    cand.holdout_max <
                        prev_best_holdout * (1.0 - kMinRelImprovement)) {
                    tried = fixed_axes;  // restart only nonconstant requested axes
                } else {
                    tried[static_cast<size_t>(pending_refined_dim)] = true;
                }
            }

            // Any improvement (even sub-threshold) advances the base.  A
            // candidate measured nowhere (holdout_measured == 0) reports
            // holdout_max = 0 vacuously and must not seize the base.
            if (!have_base ||
                (cand.holdout_measured > 0 &&
                 std::isfinite(cand.holdout_max) &&
                 (!have_finite_base || cand.holdout_max < prev_best_holdout))) {
                base = cand;
                have_base = true;
                if (std::isfinite(cand.holdout_max)) {
                    have_finite_base = true;
                    prev_best_holdout = cand.holdout_max;
                }
            }

            candidates.push_back(std::move(cand));

            // f. CONVERGENCE requires both sample sets under target (D4) on a
            //    candidate that could actually be returned: a non-viable one
            //    (a NaN fresh sample is simply skipped in the statistics)
            //    would otherwise read as converged and strand untried axes.
            if (candidates.back().viable &&
                candidates.back().fresh_converged &&
                candidates.back().holdout_max <= params.target_iv_error) {
                break;
            }
        }

        if (iteration >= params.max_iter) break;

        // g. AXIS SELECTION over the exploration base's bins (spec D6)
        bool have_next = false;
        while (true) {
            int axis = pick_refinement_axis(base.bins, tried);
            if (axis < 0) break;  // all axes exhausted

            // Reset grids AND backend state to the exploration base.
            moneyness_grid = base.moneyness;
            maturity_grid = base.tau;
            vol_grid = base.vol;
            rate_grid = base.rate;
            restore_state(base.state);

            auto focus_intervals = bins_to_intervals(
                base.bins, static_cast<size_t>(axis),
                sample_axis_bounds[static_cast<size_t>(axis)]);

            RefineOutcome outcome = refine_fn(
                static_cast<size_t>(axis), focus_intervals,
                moneyness_grid, maturity_grid, vol_grid, rate_grid);

            if (!outcome.changed) {
                // No build consumed (spec D6 step 4).
                tried[static_cast<size_t>(axis)] = true;
                continue;
            }

            pending_refined_dim =
                (outcome.changed_dim >= 0 && outcome.changed_dim < 4)
                    ? outcome.changed_dim
                    : axis;
            have_next = true;
            break;
        }
        if (!have_next) break;

        focus_active = false;
        for (size_t d = 0; d < focus_bins.size(); ++d) {
            focus_bins[d] = fixed_axes[d] ? std::vector<size_t>{} : base.bins.problematic_bins(d);
            if (!focus_bins[d].empty()) focus_active = true;
        }
    }

    // ---------------------------------------------------------------------
    // 5. Retention: return the best viable candidate (spec D5)
    // ---------------------------------------------------------------------
    const Candidate* picked = nullptr;
    for (const auto& cand : candidates) {
        if (!cand.viable) continue;
        if (picked == nullptr ||
            cand.holdout_max < picked->holdout_max ||
            (cand.holdout_max == picked->holdout_max &&
             cand.holdout_avg < picked->holdout_avg)) {
            picked = &cand;  // earliest iteration wins remaining ties
        }
    }
    if (picked == nullptr) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::NoViableSurface});
    }

    // The loop must never return grids that do not describe the caller's
    // captured surface: rebuild once when the pick is not the last build.
    //
    // `last_attempt_failed` is part of that test because a failed build_fn can
    // still have overwritten the caller's captured state: the B-spline path's
    // build_cached_surface assigns `last_spline`/`last_axes` *before* wrapping
    // the surface, so a wrapper failure returns an error while leaving the
    // caller pointing at the failed trial's spline.  Rebuilding is the cheap,
    // backend-independent way to guarantee the capture matches the pick.
    if (picked->iteration != last_built_iteration || last_attempt_failed ||
        !last_handle.has_value()) {
        auto iter_start = std::chrono::steady_clock::now();
        auto rebuilt = build_fn(picked->moneyness, picked->tau,
                                picked->vol, picked->rate);
        if (!rebuilt.has_value()) {
            return std::unexpected(rebuilt.error());
        }
        IterationStats stats;
        stats.iteration = iteration;  // next slot in the build sequence
        stats.refined_dim = -2;  // final rebuild marker (spec D7)
        stats.grid_sizes = {picked->moneyness.size(), picked->tau.size(),
                            picked->vol.size(), picked->rate.size()};
        stats.pde_solves_table = rebuilt->pde_solves;
        stats.max_error = picked->holdout_max;
        stats.avg_error = picked->holdout_avg;
        stats.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iter_start).count();
        diag.iterations.push_back(stats);
        diag.final_rebuild = true;
        last_handle = std::move(rebuilt.value());
    }

    // ---------------------------------------------------------------------
    // 6. Result + diagnostics (spec D7)
    // ---------------------------------------------------------------------
    result.moneyness = picked->moneyness;
    result.tau = picked->tau;
    result.vol = picked->vol;
    result.rate = picked->rate;
    result.tau_points = static_cast<int>(picked->tau.size());
    result.achieved_max_error = picked->holdout_max;
    result.achieved_avg_error = picked->holdout_avg;
    result.target_met = picked->holdout_max <= params.target_iv_error &&
                        picked->fresh_converged;

    diag.target_met = result.target_met;
    diag.achieved_max_error = result.achieved_max_error;
    diag.achieved_avg_error = result.achieved_avg_error;
    diag.picked_iteration = picked->iteration;
    diag.holdout_points_measured = picked->holdout_measured;
    diag.total_iterations = iteration;  // excludes the final rebuild

    detail::scan_monotonicity(holdout, *last_handle, ctx,
                              params.target_iv_error, params.vega_floor, diag);

    result.iterations = diag.iterations;
    return result;
}

std::expected<std::vector<double>, PriceTableError>
validate_k_ref_values(std::span<const double> values, size_t maximum_count) {
    if (values.empty() || values.size() > maximum_count ||
        !std::all_of(values.begin(), values.end(),
            [](double k) { return std::isfinite(k) && k > 0.0; })) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 4, values.size()});
    }
    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 4, sorted.size()});
    }
    return sorted;
}

std::expected<std::vector<double>, PriceTableError>
resolve_k_refs(const MultiKRefConfig& config, const StrikeBounds& bounds) {
    if (!bounds.valid() || config.max_references == 0 || config.max_selection_rounds == 0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 4});
    }
    if (!config.K_refs.empty()) {
        auto refs = validate_k_ref_values(config.K_refs, config.max_references);
        if (!refs) return std::unexpected(refs.error());
        if (refs->front() > bounds.min || refs->back() < bounds.max) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 4, refs->size()});
        }
        return refs;
    }
    if (bounds.min == bounds.max) return std::vector{bounds.min};
    if (config.max_references < 2) return std::unexpected(
        PriceTableError{PriceTableErrorCode::InvalidConfig, 4, config.max_references});
    const double middle = std::midpoint(bounds.min, bounds.max);
    if (config.max_references >= 3 && middle > bounds.min && middle < bounds.max) {
        return std::vector{bounds.min, middle, bounds.max};
    }
    return std::vector{bounds.min, bounds.max};
}

std::expected<SurfaceBounds, PriceTableError>
requested_segmented_domain(const IVGrid& domain, double maturity,
                           std::optional<MoneynessBounds> ratios) {
    const auto valid_axis = [](const std::vector<double>& axis) {
        return !axis.empty() && std::ranges::all_of(axis, [](double value) {
            return std::isfinite(value);
        }) && std::ranges::is_sorted(axis);
    };
    if (!std::isfinite(maturity) || maturity <= 0.0 ||
        !valid_axis(domain.moneyness) || !valid_axis(domain.vol) ||
        !valid_axis(domain.rate) || domain.vol.front() <= 0.0 ||
        (ratios && (!ratios->valid() || std::log(ratios->min) != domain.moneyness.front() ||
                    std::log(ratios->max) != domain.moneyness.back()))) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    return SurfaceBounds{
        .m_min = domain.moneyness.front(), .m_max = domain.moneyness.back(),
        .tau_min = 0.0, .tau_max = maturity,
        .sigma_min = domain.vol.front(), .sigma_max = domain.vol.back(),
        .rate_min = domain.rate.front(), .rate_max = domain.rate.back(),
        .ratio_bounds = ratios,
    };
}

std::expected<SurfaceBounds, PriceTableError>
expand_segmented_domain(const IVGrid& domain,
                        double maturity,
                        double /*dividend_yield*/,
                        const std::vector<Dividend>& discrete_dividends,
                        double min_K_ref) {
    // Input admission is separate from optional numerical padding.
    auto requested = requested_segmented_domain(domain, maturity);
    if (!requested) return std::unexpected(requested.error());

    // domain.moneyness is already log(S/K) — take min/max directly
    double min_m = domain.moneyness.front();
    double max_m = domain.moneyness.back();

    // Expand lower bound for cumulative discrete-dividend spot shifts
    double total_div = total_discrete_dividends(discrete_dividends, maturity);
    double expansion = (min_K_ref > 0.0) ? total_div / min_K_ref : 0.0;
    if (expansion > 0.0) {
        double m_min_money = std::exp(min_m);
        double expanded = std::max(m_min_money - expansion, 0.01);
        // Padding may add support, but its floor cannot remove a requested tail.
        min_m = std::min(min_m, std::log(expanded));
    }

    double min_vol = domain.vol.front();
    double max_vol = domain.vol.back();
    double min_rate = domain.rate.front();
    double max_rate = domain.rate.back();

    // Apply standard minimum spreads
    expand_domain_bounds(min_m, max_m, 0.10);
    expand_domain_bounds(min_vol, max_vol, 0.10, std::min(kMinPositive, min_vol));
    expand_domain_bounds(min_rate, max_rate, 0.04);

    // The first segment includes an exact analytical payoff construction row.
    const double min_tau = 0.0;
    const double max_tau = maturity;

    return SurfaceBounds{
        .m_min = min_m, .m_max = max_m,
        .tau_min = min_tau, .tau_max = max_tau,
        .sigma_min = min_vol, .sigma_max = max_vol,
        .rate_min = min_rate, .rate_max = max_rate,
    };
}

std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain, size_t expected_m_knots) {
    if (chain.strikes.empty() || chain.maturities.empty() ||
        chain.implied_vols.empty() || chain.rates.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    const auto positive_finite = [](double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (!std::ranges::all_of(chain.maturities, positive_finite) ||
        !std::ranges::all_of(chain.implied_vols, positive_finite)) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double min_m = std::numeric_limits<double>::max();
    double max_m = std::numeric_limits<double>::lowest();
    for (double strike : chain.strikes) {
        double m = std::log(chain.spot / strike);
        min_m = std::min(min_m, m);
        max_m = std::max(max_m, m);
    }

    auto [min_tau, max_tau] = std::minmax_element(chain.maturities.begin(), chain.maturities.end());
    auto [min_vol, max_vol] = std::minmax_element(chain.implied_vols.begin(), chain.implied_vols.end());
    auto [min_rate, max_rate] = std::minmax_element(chain.rates.begin(), chain.rates.end());

    // The requested physical domain is independent of numerical support.
    // In particular, a nondegenerate 1--7 day request must not acquire a
    // six-month validation/publication range through minimum-spread padding.
    SurfaceBounds sample{
        .m_min = min_m, .m_max = max_m,
        .tau_min = *min_tau, .tau_max = *max_tau,
        .sigma_min = *min_vol, .sigma_max = *max_vol,
        .rate_min = *min_rate, .rate_max = *max_rate,
    };

    // Preserve the existing numerical support policy and supplied seeds.
    // A singleton request remains a singleton in sample_bounds: this padding
    // does not grant callers a wider physical query domain.
    SurfaceBounds fit = sample;
    expand_domain_bounds(fit.m_min, fit.m_max, 0.10);
    expand_domain_bounds(fit.tau_min, fit.tau_max, 0.5,
                         std::min(kMinPositive, fit.tau_min));
    expand_domain_bounds(fit.sigma_min, fit.sigma_max, 0.10,
                         std::min(kMinPositive, fit.sigma_min));
    expand_domain_bounds(fit.rate_min, fit.rate_max, 0.04);
    // B-spline support headroom uses the expected seeded moneyness density,
    // not the number of user strikes, and never changes sample_bounds.
    double h = spline_support_headroom(fit.m_max - fit.m_min,
                                       expected_m_knots);
    fit.m_min -= h;
    fit.m_max += h;

    return RefinementContext{
        .spot = chain.spot,
        .dividend_yield = chain.dividend_yield,
        .option_type = {},  // caller sets this
        .bounds = fit,
        .sample_bounds = sample,
    };
}

InitialGrids extract_initial_grids(const OptionGrid& chain) {
    InitialGrids grids;
    grids.moneyness.reserve(chain.strikes.size());
    for (double strike : chain.strikes) {
        grids.moneyness.push_back(std::log(chain.spot / strike));
    }
    grids.tau = chain.maturities;
    grids.vol = chain.implied_vols;
    grids.rate = chain.rates;
    return grids;
}

}  // namespace mango
