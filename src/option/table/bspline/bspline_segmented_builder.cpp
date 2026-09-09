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

struct ResolvedTemporalGrid {
    TauSegmentSplit split;
    std::vector<std::vector<double>> segments;
    std::vector<double> nodes;
    size_t cap_hits;
};

std::expected<ResolvedTemporalGrid, PriceTableError>
resolve_temporal_grid(const SegmentedPriceTableBuilder::Config& config) {
    if (!std::isfinite(config.maturity) || config.maturity <= 0.0 ||
        !std::isfinite(config.K_ref) || config.K_ref <= 0.0) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig});
    if (config.tau_grid.empty() &&
        (!std::isfinite(config.tau_target_dt) || config.tau_target_dt < 0.0 ||
         config.tau_points_per_segment < 4 || config.tau_points_min < 4 ||
         config.tau_points_max < config.tau_points_min)) return std::unexpected(
            PriceTableError{PriceTableErrorCode::InvalidConfig, 1});
    const double T = config.maturity, K_ref = config.K_ref;
    const auto dividends = filter_dividends(config.dividends.discrete_dividends, T);
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
    if (!config.tau_grid.empty()) {
        for (double time : config.tau_grid) {
            if (!std::isfinite(time)) return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig, 1});
        }
        if (std::adjacent_find(config.tau_grid.begin(), config.tau_grid.end(),
                              std::greater_equal<double>{}) != config.tau_grid.end()) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::GridNotSorted, 1});
        }
        segment_times.resize(split.tau_start().size());
        for (double time : config.tau_grid) {
            bool assigned = false;
            for (size_t s = 0; s < segment_times.size(); ++s) {
                const double lo = split.tau_start()[s] + split.tau_min()[s];
                const double hi = split.tau_start()[s] + split.tau_max()[s];
                if (time >= lo && time <= hi) {
                    segment_times[s].push_back(time);
                    assigned = true;
                    break;
                }
            }
            if (!assigned) return std::unexpected(
                PriceTableError{PriceTableErrorCode::InvalidConfig, 1});
        }
        for (size_t s = 0; s < segment_times.size(); ++s) {
            const auto& times = segment_times[s];
            if (times.size() < 4) return std::unexpected(
                PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 1, times.size()});
            if (times.front() != split.tau_start()[s] + split.tau_min()[s] ||
                times.back() != split.tau_start()[s] + split.tau_max()[s]) {
                return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig, 1});
            }
        }
        requested_times = config.tau_grid;
    } else {
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
    }
    std::sort(requested_times.begin(), requested_times.end());
    requested_times.erase(std::unique(requested_times.begin(), requested_times.end()),
                          requested_times.end());

    return ResolvedTemporalGrid{std::move(split), std::move(segment_times),
                                std::move(requested_times), tau_cap_hits};
}

}  // namespace

std::expected<std::vector<double>, PriceTableError>
SegmentedPriceTableBuilder::make_tau_grid(const Config& config) {
    auto grid = resolve_temporal_grid(config);
    if (!grid) return std::unexpected(grid.error());
    return std::move(grid->nodes);
}

std::expected<BSplineSegmentedSurface, PriceTableError>
SegmentedPriceTableBuilder::build(const Config& config) {
    auto result = build_with_diagnostics(config);
    if (!result) return std::unexpected(result.error());
    return std::move(result->surface);
}

std::expected<void, PriceTableError>
SegmentedPriceTableBuilder::validate_config(const Config& config) {
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
        !std::isfinite(config.dividends.dividend_yield)) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    return {};
}

std::expected<SegmentedPriceTableBuilder::BuildResult, PriceTableError>
SegmentedPriceTableBuilder::build_with_diagnostics(const Config& config) {
    // =====================================================================
    // Validate inputs
    // =====================================================================
    auto valid = validate_config(config);
    if (!valid) return std::unexpected(valid.error());

    const double T = config.maturity;
    const double K_ref = config.K_ref;

    // =====================================================================
    // Step 1: Filter and sort dividends
    // =====================================================================
    auto dividends = filter_dividends(config.dividends.discrete_dividends, T);

    // These are the selected interpolation sites. Numerical spatial
    // coverage and cash-jump support belong to the PDE grid estimator.
    const auto& m_grid = config.grid.moneyness;

    auto temporal = resolve_temporal_grid(config);
    if (!temporal) return std::unexpected(temporal.error());
    auto& split = temporal->split;
    const auto& segment_times = temporal->segments;
    const auto& requested_times = temporal->nodes;
    const size_t tau_cap_hits = temporal->cap_hits;

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
    accuracy.log_moneyness_coverage = LogMoneynessRange::of(m_grid);
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
    const size_t nm = m_grid.size();
    const size_t nv = config.grid.vol.size(), nr = config.grid.rate.size();
    for (size_t s = 0; s < segment_times.size(); ++s) {
        std::vector<double> local_tau;
        for (double time : segment_times[s]) local_tau.push_back(time - split.tau_start()[s]);
        auto setup = PriceTableBuilder::from_vectors(
            m_grid, local_tau, config.grid.vol, config.grid.rate,
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
            if (x.front() > m_grid.front() || x.back() < m_grid.back()) {
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
                        ? intrinsic_value(K_ref * std::exp(m_grid[i]),
                                          K_ref, config.option_type) / K_ref
                        : spatial.eval(m_grid[i]);
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

    std::vector<double> requested_refs;
    requested_refs.reserve(entries.size());
    for (const auto& entry : entries) requested_refs.push_back(entry.K_ref);
    auto valid_refs = validate_k_ref_values(requested_refs, entries.size());
    if (!valid_refs) return std::unexpected(valid_refs.error());

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
