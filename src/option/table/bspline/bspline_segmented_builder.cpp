// SPDX-License-Identifier: MIT
#include "mango/option/table/bspline/bspline_segmented_builder.hpp"
#include "mango/option/dividend_utils.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/table/adaptive_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mango {

namespace {

constexpr int kCubicSplineDegree = 3;

/// Generate a τ grid for a segment [tau_start, tau_end].
/// When tau_target_dt > 0, scales points proportionally to segment width.
/// Otherwise falls back to constant min_points.
std::vector<double> make_segment_tau_grid(
    double tau_start, double tau_end, int min_points,
    double tau_target_dt, int tau_points_min, int tau_points_max,
    size_t& cap_hits)
{
    double seg_width = tau_end - tau_start;

    int n;
    if (tau_target_dt > 0.0) {
        // Width-proportional: wider segments get more points
        const double requested = std::ceil(seg_width / tau_target_dt) + 1.0;
        if (requested > tau_points_max) ++cap_hits;
        n = static_cast<int>(std::clamp(
            requested, static_cast<double>(tau_points_min),
            static_cast<double>(tau_points_max)));
    } else {
        // Legacy constant mode
        n = std::max(min_points, 4);
    }

    std::vector<double> grid;
    grid.reserve(static_cast<size_t>(n));

    double step = (tau_end - tau_start) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        grid.push_back(tau_start + step * static_cast<double>(i));
    }

    return grid;
}

/// Filter dividends: delegates to shared filter_and_merge_dividends().
std::vector<Dividend> filter_dividends(
    const std::vector<Dividend>& divs, double T)
{
    return filter_and_merge_dividends(divs, T);
}

/// Append an upper guard band sized by cubic-spline support in log-moneyness.
///
/// The interpolation axis is log-moneyness. Appending enough local knot
/// intervals keeps the original upper domain away from clamped endpoint basis
/// effects and adds one diffusion-length guard for the widest segment.
void append_upper_tail_log_moneyness(std::vector<double>& log_grid,
                                     double sigma_max,
                                     double max_segment_width) {
    if (log_grid.size() < 2) return;

    std::sort(log_grid.begin(), log_grid.end());
    log_grid.erase(std::unique(log_grid.begin(), log_grid.end()), log_grid.end());
    if (log_grid.size() < 2) return;

    const size_t n = log_grid.size();
    double h_upper = 0.0;
    const size_t first = (n >= static_cast<size_t>(kCubicSplineDegree + 1))
        ? (n - static_cast<size_t>(kCubicSplineDegree + 1))
        : 0;
    for (size_t i = first; i + 1 < n; ++i) {
        h_upper = std::max(h_upper, log_grid[i + 1] - log_grid[i]);
    }
    if (!(h_upper > 0.0)) return;

    const double support_headroom =
        static_cast<double>(kCubicSplineDegree) * h_upper;
    const double diffusion_headroom =
        (sigma_max > 0.0 && max_segment_width > 0.0)
        ? (sigma_max * std::sqrt(max_segment_width))
        : 0.0;
    const double required_headroom =
        std::max(support_headroom, diffusion_headroom);
    const int tail_points = std::max(
        kCubicSplineDegree,
        static_cast<int>(std::ceil(required_headroom / h_upper)));

    const double x_max = log_grid.back();
    for (int i = 1; i <= tail_points; ++i) {
        log_grid.push_back(x_max + h_upper * static_cast<double>(i));
    }
}

/// Expand a log-moneyness grid to cover dividend-induced downward shifts and
/// add upper-tail headroom for B-spline support and diffusion.
///
/// Returns the expanded, sorted, deduplicated log-moneyness grid, or an error
/// if the resulting grid is too small or contains non-finite values.
static std::expected<std::vector<double>, PriceTableError>
expand_log_moneyness_grid(
    const std::vector<double>& input_grid,
    const std::vector<Dividend>& dividends,
    double K_ref,
    double sigma_max,
    double max_segment_width) {

    std::vector<double> expanded_log_m_grid = input_grid;
    std::sort(expanded_log_m_grid.begin(), expanded_log_m_grid.end());
    expanded_log_m_grid.erase(
        std::unique(expanded_log_m_grid.begin(), expanded_log_m_grid.end()),
        expanded_log_m_grid.end());

    if (expanded_log_m_grid.size() < 2) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InsufficientGridPoints, 0});
    }
    if (!std::isfinite(expanded_log_m_grid.front()) ||
        !std::isfinite(expanded_log_m_grid.back())) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    double total_div = 0.0;
    for (const auto& div : dividends) {
        total_div += div.amount;
    }

    // Expand lower side in moneyness-space, then map back to log-moneyness.
    const double x_min = expanded_log_m_grid.front();
    const double m_min = std::exp(x_min);
    double m_min_expanded = std::max(m_min - total_div / K_ref, 0.01);
    double x_min_expanded = std::log(m_min_expanded);

    // Only expand when the shift is materially below the current lower edge.
    // With no dividends (total_div == 0) the exp/log round trip can land one
    // ULP under x_min, and the unguarded comparison then inserts three knots
    // separated by ~1e-17 -- which the cubic collocation solver rejects as an
    // unsorted (near-duplicate) grid.
    constexpr double kMinExpansion = 1e-9;
    if (x_min_expanded < expanded_log_m_grid.front() - kMinExpansion) {
        double step = (expanded_log_m_grid.front() - x_min_expanded) / 3.0;
        for (int i = 2; i >= 0; --i) {
            double x = x_min_expanded + step * static_cast<double>(i);
            if (x < expanded_log_m_grid.front()) {
                expanded_log_m_grid.insert(expanded_log_m_grid.begin(), x);
            }
        }
    }

    // Add right-tail headroom. Scale by one diffusion length in log-space
    // (sigma_max * sqrt(max segment width)) and at least one cubic-support
    // band so the original upper domain stays away from endpoint effects.
    append_upper_tail_log_moneyness(
        expanded_log_m_grid, sigma_max, max_segment_width);

    std::sort(expanded_log_m_grid.begin(), expanded_log_m_grid.end());
    expanded_log_m_grid.erase(
        std::unique(expanded_log_m_grid.begin(), expanded_log_m_grid.end()),
        expanded_log_m_grid.end());

    // Ensure at least 4 log-moneyness points
    if (expanded_log_m_grid.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0});
    }

    for (double x : expanded_log_m_grid) {
        if (!std::isfinite(x)) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InvalidConfig});
        }
    }

    return expanded_log_m_grid;
}

}  // namespace

std::expected<BSplineSegmentedSurface, PriceTableError>
SegmentedPriceTableBuilder::build(const Config& config) {
    auto result = build_with_diagnostics(config);
    if (!result) return std::unexpected(result.error());
    return std::move(result->surface);
}

std::expected<SegmentedPriceTableBuilder::BuildResult, PriceTableError>
SegmentedPriceTableBuilder::build_with_diagnostics(const Config& config) {
    // =====================================================================
    // Validate inputs
    // =====================================================================
    if (!(std::isfinite(config.K_ref) && config.K_ref > 0.0)) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (!(std::isfinite(config.maturity) && config.maturity > 0.0)) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (!validate_grid_accuracy(config.pde_accuracy)) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    if (config.grid.moneyness.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0});
    }
    if (config.grid.vol.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 2});
    }
    if (config.grid.rate.size() < 4) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 3});
    }
    for (const auto& [axis, values] : {
             std::pair{size_t{0}, &config.grid.moneyness},
             std::pair{size_t{2}, &config.grid.vol},
             std::pair{size_t{3}, &config.grid.rate}}) {
        for (double value : *values) {
            if (!std::isfinite(value)) return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig, axis});
        }
        if (std::adjacent_find(values->begin(), values->end(),
                              std::greater_equal<double>{}) != values->end()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::GridNotSorted, axis});
        }
    }
    if (config.grid.vol.front() <= 0.0 ||
        !std::isfinite(config.dividends.dividend_yield) ||
        !std::isfinite(config.tau_target_dt) || config.tau_target_dt < 0.0 ||
        config.tau_points_per_segment < 4 || config.tau_points_min < 4 ||
        config.tau_points_max < config.tau_points_min) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    const double T = config.maturity;
    const double K_ref = config.K_ref;

    // =====================================================================
    // Step 1: Filter and sort dividends
    // =====================================================================
    auto dividends = filter_dividends(config.dividends.discrete_dividends, T);

    // =====================================================================
    // Step 2: Compute segment boundaries in τ-space
    // =====================================================================
    // With N dividends at calendar times t_1 < ... < t_N the segment
    // boundaries in τ are:
    //   {0, T - t_N, T - t_{N-1}, ..., T - t_1, T}
    // Segment 0 (closest to expiry) covers [0, T - t_N].
    // Segment k covers (boundary[k], boundary[k+1]].
    std::vector<double> boundaries;
    boundaries.push_back(0.0);
    for (auto it = dividends.rbegin(); it != dividends.rend(); ++it) {
        boundaries.push_back(T - it->calendar_time);
    }
    boundaries.push_back(T);

    // =====================================================================
    // Step 3: Expand log-moneyness grid downward
    // =====================================================================
    double sigma_max = *std::max_element(
        config.grid.vol.begin(), config.grid.vol.end());
    double max_segment_width = 0.0;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        max_segment_width =
            std::max(max_segment_width, boundaries[i + 1] - boundaries[i]);
    }

    auto grid_result = expand_log_moneyness_grid(
        config.grid.moneyness, dividends, K_ref,
        sigma_max, max_segment_width);
    if (!grid_result.has_value()) {
        return std::unexpected(grid_result.error());
    }
    auto expanded_log_m_grid = std::move(*grid_result);

    // =====================================================================
    // Each sample belongs to one temporal regime. The existing inset topology
    // excludes event sides not represented by the solver's single snapshot.
    auto [sample_bounds, gaps] = compute_segment_boundaries(dividends, T, 0.0, T);
    for (const auto& dividend : dividends) {
        const double event_tau = T - dividend.calendar_time;
        bool excluded = false;
        for (size_t s = 0; s < gaps.size(); ++s) {
            excluded |= gaps[s] && event_tau > sample_bounds[s]
                && event_tau < sample_bounds[s + 1];
        }
        if (!excluded) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig});
    }
    auto split = make_tau_split_from_segments(sample_bounds, gaps, K_ref);

    std::vector<std::vector<double>> segment_times;
    std::vector<double> requested_times;
    size_t tau_cap_hits = 0;
    for (size_t s = 0; s < split.tau_start().size(); ++s) {
        auto local = make_segment_tau_grid(
            0.0, split.tau_end()[s] - split.tau_start()[s],
            config.tau_points_per_segment, config.tau_target_dt,
            config.tau_points_min, config.tau_points_max, tau_cap_hits);
        local.front() = split.tau_min()[s];
        local.back() = split.tau_max()[s];
        std::vector<double> global;
        for (double t : local) global.push_back(split.tau_start()[s] + t);
        const auto ordered = [&] {
            return std::adjacent_find(global.begin(), global.end(),
                                     std::greater_equal<double>{}) == global.end();
        };
        if (!ordered()) {
            // Endpoint insets can overtake generated interior nodes in a
            // short regime. Keep the requested count on its valid support.
            const double lo = global.front(), hi = global.back();
            for (size_t j = 0; j < global.size(); ++j) {
                global[j] = std::lerp(lo, hi,
                    static_cast<double>(j) / static_cast<double>(global.size() - 1));
            }
        }
        if (!ordered()) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig});
        requested_times.insert(requested_times.end(), global.begin(), global.end());
        segment_times.push_back(std::move(global));
    }
    std::sort(requested_times.begin(), requested_times.end());
    requested_times.erase(std::unique(requested_times.begin(), requested_times.end()),
                          requested_times.end());

    // One fixed-expiry PDE per (sigma, rate), including the actual events.
    // No fitted surface supplies an initial condition for another solve.
    std::vector<PricingParams> batch_params;
    for (double sigma : config.grid.vol) {
        for (double rate : config.grid.rate) {
            PricingParams p(OptionSpec{.spot = K_ref, .strike = K_ref,
                .maturity = T, .rate = rate,
                .dividend_yield = config.dividends.dividend_yield,
                .option_type = config.option_type}, sigma);
            p.discrete_dividends = dividends;
            batch_params.push_back(std::move(p));
        }
    }
    auto accuracy = config.pde_accuracy;
    accuracy.log_moneyness_coverage = LogMoneynessRange::of(expanded_log_m_grid);
    auto grid = estimate_batch_pde_grid_config(batch_params, accuracy);
    grid.mandatory_times = requested_times;
    BatchAmericanOptionSolver solver;
    solver.set_snapshot_times(requested_times);
    auto batch = solver.solve_batch(batch_params, true, nullptr, PDEGridSpec{grid});

    // Missing rows are counted before fitting. This path is strict, without
    // the previous implicit 50% repair allowance on chained segments.
    size_t missing_rows = 0;
    for (const auto& result : batch.results) {
        if (!result) {
            missing_rows += requested_times.size();
            continue;
        }
        const auto actual = result->snapshot_times();
        for (double time : requested_times) {
            if (!std::binary_search(actual.begin(), actual.end(), time)) ++missing_rows;
        }
    }
    if (batch.results.size() != batch_params.size() || missing_rows != 0) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ExtractionFailed, 1, missing_rows});
    }

    std::vector<BSplineSegmentedLeaf> leaves;
    const size_t nm = expanded_log_m_grid.size();
    const size_t nv = config.grid.vol.size(), nr = config.grid.rate.size();
    for (size_t s = 0; s < segment_times.size(); ++s) {
        std::vector<double> local_tau;
        for (double time : segment_times[s]) local_tau.push_back(time - split.tau_start()[s]);
        auto setup = PriceTableBuilder::from_vectors(
            expanded_log_m_grid, local_tau, config.grid.vol, config.grid.rate,
            K_ref, config.pde_accuracy, config.option_type,
            config.dividends.dividend_yield);
        if (!setup) return std::unexpected(setup.error());
        auto& [builder, axes] = *setup;
        auto tensor = PriceTensorND<4>::create(axes.shape());
        if (!tensor) return std::unexpected(
            PriceTableError{PriceTableErrorCode::TensorCreationFailed});
        for (size_t vi = 0; vi < nv; ++vi) for (size_t ri = 0; ri < nr; ++ri) {
            const auto& result = *batch.results[vi * nr + ri];
            auto x = result.grid()->x();
            if (x.front() > expanded_log_m_grid.front() || x.back() < expanded_log_m_grid.back()) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::ExtractionFailed});
            }
            const auto actual_times = result.snapshot_times();
            for (size_t j = 0; j < segment_times[s].size(); ++j) {
                const double time = segment_times[s][j];
                const size_t row = std::lower_bound(actual_times.begin(), actual_times.end(), time)
                    - actual_times.begin();
                auto values = result.at_time(row);
                CubicSpline<double> spatial;
                if (spatial.build(x, values)) return std::unexpected(
                    PriceTableError{PriceTableErrorCode::ExtractionFailed, 1, 1});
                for (size_t i = 0; i < nm; ++i) {
                    // Fill the exact expiry payoff without spatially smoothing
                    // its strike kink; all positive times use raw PDE states.
                    const double value = time == 0.0
                        ? intrinsic_value(K_ref * std::exp(expanded_log_m_grid[i]),
                                          K_ref, config.option_type) / K_ref
                        : spatial.eval(expanded_log_m_grid[i]);
                    if (!std::isfinite(value)) return std::unexpected(
                        PriceTableError{PriceTableErrorCode::ExtractionFailed, 1, 1});
                    tensor->view[i, j, vi, ri] = value;
                }
            }
        }
        auto fit = builder.fit_coeffs(*tensor, axes);
        if (!fit) return std::unexpected(fit.error());
        BSplineND<double, 4>::KnotArray knots;
        for (size_t d = 0; d < 4; ++d) knots[d] = clamped_knots_cubic(axes.grids[d]);
        auto spline = BSplineND<double, 4>::create(
            axes.grids, std::move(knots), std::move(fit->coefficients));
        if (!spline) return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
        leaves.emplace_back(SharedBSplineInterp<4>(
            std::make_shared<const BSplineND<double, 4>>(std::move(*spline))),
            StandardTransform4D{}, K_ref);
    }
    const size_t rows = requested_times.size() * batch_params.size();
    return BuildResult{
        .surface = BSplineSegmentedSurface(std::move(leaves), std::move(split)),
        .pde_solves = batch.results.size(),
        .sample_rows = rows,
        .sample_points = rows * nm,
        .tau_point_cap_hits = tau_cap_hits,
    };
}

// ===========================================================================
// Segmented surface assembly
// ===========================================================================

std::expected<BSplineSegmentedSurface, PriceTableError>
build_segmented_surface(BSplineSegmentedConfig config) {
    if (config.segments.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    std::vector<double> tau_start;
    std::vector<double> tau_end;
    std::vector<double> tau_min;
    std::vector<double> tau_max;
    std::vector<BSplineSegmentedLeaf> leaves;

    tau_start.reserve(config.segments.size());
    tau_end.reserve(config.segments.size());
    tau_min.reserve(config.segments.size());
    tau_max.reserve(config.segments.size());
    leaves.reserve(config.segments.size());

    for (auto& seg : config.segments) {
        tau_start.push_back(seg.tau_start);
        tau_end.push_back(seg.tau_end);
        tau_min.push_back(seg.spline->grid(1).front());
        tau_max.push_back(seg.spline->grid(1).back());

        SharedBSplineInterp<4> interp(seg.spline);
        StandardTransform4D xform;
        leaves.emplace_back(std::move(interp), xform, config.K_ref);
    }

    TauSegmentSplit split(
        std::move(tau_start), std::move(tau_end),
        std::move(tau_min), std::move(tau_max),
        config.K_ref);

    return BSplineSegmentedSurface(std::move(leaves), std::move(split));
}

// ===========================================================================
// Multi-K_ref surface assembly
// ===========================================================================

std::expected<BSplineMultiKRefInner, PriceTableError>
build_multi_kref_surface(std::vector<BSplineMultiKRefEntry> entries) {
    if (entries.empty()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig, 0, 0});
    }

    std::sort(entries.begin(), entries.end(),
              [](const BSplineMultiKRefEntry& a, const BSplineMultiKRefEntry& b) {
                  return a.K_ref < b.K_ref;
              });

    std::vector<double> k_refs;
    std::vector<BSplineSegmentedSurface> slices;
    k_refs.reserve(entries.size());
    slices.reserve(entries.size());

    for (auto& entry : entries) {
        k_refs.push_back(entry.K_ref);
        slices.push_back(std::move(entry.surface));
    }

    MultiKRefSplit split(std::move(k_refs));

    return BSplineMultiKRefInner(std::move(slices), std::move(split));
}

}  // namespace mango
