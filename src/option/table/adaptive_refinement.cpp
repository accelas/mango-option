// SPDX-License-Identifier: MIT
#include "mango/option/table/adaptive_refinement.hpp"
#include "mango/math/latin_hypercube.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/dividend_utils.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <random>

namespace mango {

namespace {
constexpr double kMinPositive = 1e-6;
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
        tau_min.push_back(0.0);
        tau_max.push_back(bounds[s + 1] - bounds[s]);
    }

    return TauSegmentSplit(
        std::move(tau_start), std::move(tau_end),
        std::move(tau_min), std::move(tau_max), K_ref);
}

double compute_iv_error(double price_error, double vega,
                        double vega_floor, double target_iv_error) {
    double vega_clamped = std::max(std::abs(vega), vega_floor);
    double iv_error = price_error / vega_clamped;
    double price_tol = target_iv_error * vega_floor;
    if (price_error <= price_tol) {
        iv_error = std::min(iv_error, target_iv_error);
    }
    return iv_error;
}

ComputeErrorFn make_fd_vega_error_fn(const AdaptiveGridParams& params,
                                      const ValidateFn& validate_fn,
                                      OptionType option_type) {
    double vega_floor = params.vega_floor;
    double target = params.target_iv_error;
    // Copy validate_fn by value so the returned lambda is self-contained.
    return [vega_floor, target, validate_fn, option_type](
        double interp, double ref_price,
        double spot, double strike, double tau,
        double sigma, double rate, double /*div_yield*/) -> double
    {
        // TV/K filter: skip points where IV is undefined
        constexpr double kTVKThreshold = 1e-4;
        double intrinsic = intrinsic_value(spot, strike, option_type);
        if ((ref_price - intrinsic) / strike < kTVKThreshold) {
            return 0.0;
        }

        double price_error = std::abs(interp - ref_price);

        // FD American vega via central difference
        double eps = std::max(1e-4, 0.01 * sigma);
        double sigma_dn = std::max(1e-4, sigma - eps);
        double sigma_up = sigma + eps;
        double effective_eps = (sigma_up - sigma_dn) / 2.0;

        auto fd_up = validate_fn(spot, strike, tau, sigma_up, rate);
        auto fd_dn = validate_fn(spot, strike, tau, sigma_dn, rate);

        double vega = 0.0;
        if (fd_up.has_value() && fd_dn.has_value() && effective_eps > 1e-6) {
            vega = (fd_up.value() - fd_dn.value()) / (2.0 * effective_eps);
        }
        return compute_iv_error(price_error, vega, vega_floor, target);
    };
}

ValidateFn make_validate_fn(double dividend_yield,
                            OptionType option_type,
                            const std::vector<Dividend>& discrete_dividends) {
    return [dividend_yield, option_type, discrete_dividends](
        double spot, double strike, double tau,
        double sigma, double rate) -> std::expected<double, SolverError>
    {
        PricingParams p;
        p.spot = spot;
        p.strike = strike;
        p.maturity = tau;
        p.rate = rate;
        p.dividend_yield = dividend_yield;
        p.option_type = option_type;
        p.volatility = sigma;
        p.discrete_dividends = discrete_dividends;
        auto fd = solve_american_option(p);
        if (!fd.has_value()) return std::unexpected(fd.error());
        return fd->value();
    };
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

std::expected<RefinementResult, PriceTableError> run_refinement(
    const AdaptiveGridParams& params,
    BuildFn build_fn,
    ValidateFn validate_fn,
    RefineFn refine_fn,
    const RefinementContext& ctx,
    const ComputeErrorFn& compute_error,
    const InitialGrids& initial_grids)
{
    // Validation requires at least one sample per iteration
    if (params.validation_samples == 0) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig
        });
    }

    // B-spline requires minimum 4 control points per dimension
    if (params.min_moneyness_points < 4) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::InvalidConfig
        });
    }

    const double min_moneyness = ctx.min_moneyness;
    const double max_moneyness = ctx.max_moneyness;
    const double min_tau = ctx.min_tau;
    const double max_tau = ctx.max_tau;
    const double min_vol = ctx.min_vol;
    const double max_vol = ctx.max_vol;
    const double min_rate = ctx.min_rate;
    const double max_rate = ctx.max_rate;

    std::vector<double> moneyness_grid, maturity_grid, vol_grid, rate_grid;

    if (initial_grids.exact) {
        // Use grids exactly as provided (Chebyshev CGL/CC nodes)
        moneyness_grid = initial_grids.moneyness;
        maturity_grid = initial_grids.tau;
        vol_grid = initial_grids.vol;
        rate_grid = initial_grids.rate;
    } else {
        // Seed grids from user-provided knots (or linspace fallback)
        // This ensures user-specified knots (e.g., benchmark vols) are always grid points
        moneyness_grid = seed_grid(initial_grids.moneyness, min_moneyness, max_moneyness,
                                   params.min_moneyness_points);
        maturity_grid = seed_grid(initial_grids.tau, min_tau, max_tau, 5);
        vol_grid = seed_grid(initial_grids.vol, min_vol, max_vol, 5);
        rate_grid = seed_grid(initial_grids.rate, min_rate, max_rate, 4);

        // Ensure moneyness grid meets minimum density requirement
        // Moneyness needs higher density due to exercise boundary curvature
        while (moneyness_grid.size() < params.min_moneyness_points) {
            // Insert midpoints in largest gaps until we reach minimum
            double max_gap = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i + 1 < moneyness_grid.size(); ++i) {
                double gap = moneyness_grid[i + 1] - moneyness_grid[i];
                if (gap > max_gap) { max_gap = gap; max_idx = i; }
            }
            moneyness_grid.push_back(
                (moneyness_grid[max_idx] + moneyness_grid[max_idx + 1]) / 2.0);
            std::sort(moneyness_grid.begin(), moneyness_grid.end());
        }
    }

    RefinementResult result;
    result.iterations.reserve(params.max_iter);

    std::array<std::vector<size_t>, 4> focus_bins;
    bool focus_active = false;

    for (size_t iteration = 0; iteration < params.max_iter; ++iteration) {
        auto iter_start = std::chrono::steady_clock::now();

        IterationStats stats;
        stats.iteration = iteration;
        stats.grid_sizes = {
            moneyness_grid.size(),
            maturity_grid.size(),
            vol_grid.size(),
            rate_grid.size()
        };

        // a. BUILD/UPDATE TABLE via callback
        auto surface_result = build_fn(moneyness_grid, maturity_grid, vol_grid, rate_grid);
        if (!surface_result.has_value()) {
            return std::unexpected(surface_result.error());
        }
        auto& handle = surface_result.value();

        stats.pde_solves_table = handle.pde_solves;

        // b. GENERATE VALIDATION SAMPLE
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

        std::array<std::pair<double, double>, 4> bounds = {{
            {min_moneyness, max_moneyness},
            {min_tau, max_tau},
            {min_vol, max_vol},
            {min_rate, max_rate}
        }};
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

        // c. VALIDATE AGAINST FRESH FD SOLVES
        double max_error = 0.0;
        double sum_error = 0.0;
        size_t valid_samples = 0;
        ErrorBins error_bins;

        for (const auto& sample : samples) {
            double m = sample[0];
            double tau = sample[1];
            double sigma = sample[2];
            double rate = sample[3];

            // Interpolated price from surface via callback
            double strike = ctx.spot * std::exp(-m);
            double interp_price = handle.price(ctx.spot, strike, tau, sigma, rate);

            // Fresh FD solve for reference via callback
            auto fd_result = validate_fn(ctx.spot, strike, tau, sigma, rate);

            if (!fd_result.has_value()) {
                continue;  // Skip failed solves
            }

            stats.pde_solves_validation++;

            double ref_price = fd_result.value();
            double iv_error = compute_error(
                interp_price, ref_price,
                ctx.spot, strike, tau, sigma, rate,
                ctx.dividend_yield);
            max_error = std::max(max_error, iv_error);
            sum_error += iv_error;
            valid_samples++;

            // Normalize position for error bins
            std::array<double, 4> norm_pos = {{
                (m - min_moneyness) / (max_moneyness - min_moneyness),
                (tau - min_tau) / (max_tau - min_tau),
                (sigma - min_vol) / (max_vol - min_vol),
                (rate - min_rate) / (max_rate - min_rate)
            }};
            error_bins.record_error(norm_pos, iv_error, params.target_iv_error);
        }

        if (valid_samples == 0) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::ExtractionFailed, /*axis=*/0, /*detail=*/0
            });
        }

        double avg_error = sum_error / valid_samples;

        stats.max_error = max_error;
        stats.avg_error = avg_error;

        auto iter_end = std::chrono::steady_clock::now();
        stats.elapsed_seconds = std::chrono::duration<double>(iter_end - iter_start).count();

        // d. CHECK CONVERGENCE
        bool converged = (max_error <= params.target_iv_error);

        if (converged || iteration == params.max_iter - 1) {
            // Final iteration - save results
            stats.refined_dim = -1;  // No refinement on final iteration
            result.iterations.push_back(stats);
            result.moneyness = moneyness_grid;
            result.tau = maturity_grid;
            result.vol = vol_grid;
            result.rate = rate_grid;
            result.tau_points = static_cast<int>(maturity_grid.size());
            result.achieved_max_error = max_error;
            result.achieved_avg_error = avg_error;
            result.target_met = converged;
            break;
        }

        // e. DIAGNOSE & REFINE
        size_t worst_dim = error_bins.worst_dimension();

        bool refined = refine_fn(worst_dim, error_bins,
                                 moneyness_grid, maturity_grid,
                                 vol_grid, rate_grid);
        if (!refined) {
            // Maxed out — treat as final iteration
            stats.refined_dim = -1;
            result.iterations.push_back(stats);
            result.moneyness = moneyness_grid;
            result.tau = maturity_grid;
            result.vol = vol_grid;
            result.rate = rate_grid;
            result.tau_points = static_cast<int>(maturity_grid.size());
            result.achieved_max_error = max_error;
            result.achieved_avg_error = avg_error;
            result.target_met = false;
            break;
        }

        focus_active = false;
        for (size_t d = 0; d < focus_bins.size(); ++d) {
            focus_bins[d] = error_bins.problematic_bins(d);
            if (!focus_bins[d].empty()) {
                focus_active = true;
            }
        }

        stats.refined_dim = static_cast<int>(worst_dim);
        result.iterations.push_back(stats);
    }

    return result;
}

std::expected<std::vector<double>, PriceTableError>
resolve_k_refs(const MultiKRefConfig& config, double spot) {
    // If K_refs explicitly provided, sort and return
    if (!config.K_refs.empty()) {
        std::vector<double> sorted = config.K_refs;
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }

    // Generate from count/span
    if (config.K_ref_count < 1 || config.K_ref_span <= 0.0
        || config.K_ref_span >= 1.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    const int count = config.K_ref_count;
    const double span = config.K_ref_span;
    std::vector<double> K_refs;
    K_refs.reserve(static_cast<size_t>(count));

    if (count == 1) {
        K_refs.push_back(spot);
    } else {
        const double log_lo = std::log(1.0 - span);
        const double log_hi = std::log(1.0 + span);
        for (int i = 0; i < count; ++i) {
            double t = static_cast<double>(i)
                     / static_cast<double>(count - 1);
            K_refs.push_back(spot * std::exp(log_lo + t * (log_hi - log_lo)));
        }
    }

    std::sort(K_refs.begin(), K_refs.end());
    return K_refs;
}

std::expected<RefinementContext, PriceTableError>
extract_chain_domain(const OptionGrid& chain) {
    if (chain.strikes.empty() || chain.maturities.empty() ||
        chain.implied_vols.empty() || chain.rates.empty()) {
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

    expand_domain_bounds(min_m, max_m, 0.10);
    double h = spline_support_headroom(max_m - min_m, chain.strikes.size());
    min_m -= h;
    max_m += h;

    double lo_tau = *min_tau, hi_tau = *max_tau;
    double lo_vol = *min_vol, hi_vol = *max_vol;
    double lo_rate = *min_rate, hi_rate = *max_rate;

    expand_domain_bounds(lo_tau, hi_tau, 0.5, kMinPositive);
    expand_domain_bounds(lo_vol, hi_vol, 0.10, kMinPositive);
    expand_domain_bounds(lo_rate, hi_rate, 0.04);

    return RefinementContext{
        .spot = chain.spot,
        .dividend_yield = chain.dividend_yield,
        .option_type = {},  // caller sets this
        .min_moneyness = min_m, .max_moneyness = max_m,
        .min_tau = lo_tau, .max_tau = hi_tau,
        .min_vol = lo_vol, .max_vol = hi_vol,
        .min_rate = lo_rate, .max_rate = hi_rate,
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
