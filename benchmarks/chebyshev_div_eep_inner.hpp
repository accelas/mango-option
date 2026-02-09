// SPDX-License-Identifier: MIT
//
// Multi-K_ref segmented Chebyshev 4D for discrete dividends.
// Axes: (x = ln(S/K_ref), tau, sigma, rate).
// Splits τ-axis at dividend dates (per-segment PDE solves).
// Stores V/K_ref directly (no EEP decomposition).
// Multiple K_refs with linear strike-blending for portability.
// Benchmark experiment only.
#pragma once

#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace mango {

// ============================================================================
// Config and result types
// ============================================================================

struct SegCheb4DDivConfig {
    size_t num_x = 40;
    size_t num_tau_per_seg = 9;   // CGL nodes per τ-segment
    size_t num_sigma = 15;
    size_t num_rate = 9;
    double epsilon = 1e-8;
    bool use_tucker = false;

    double x_min = -0.50;
    double x_max = 0.40;
    double tau_min = 0.019;       // ~7 days
    double tau_max = 2.0;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.02;
    std::vector<Dividend> discrete_dividends;
    OptionType option_type = OptionType::PUT;

    // Multi-K_ref: separate PDE solves per K_ref, blended at query time.
    // Empty = single K_ref (passed to builder).
    std::vector<double> K_refs;
};

struct SegCheb4DDivSegment {
    ChebyshevTucker4D interp;
    double tau_lo, tau_hi;
};

struct SegCheb4DDivKrefEntry {
    double K_ref;
    std::vector<SegCheb4DDivSegment> segments;
};

struct SegCheb4DDivResult {
    std::vector<SegCheb4DDivKrefEntry> entries;  // per K_ref
    int n_pde_solves;
    double build_seconds;
};

// ============================================================================
// Query-time adapter with multi-K_ref blending
// ============================================================================

class SegCheb4DDivInner {
public:
    SegCheb4DDivInner(std::vector<SegCheb4DDivKrefEntry> entries,
                      OptionType type)
        : entries_(std::move(entries)), type_(type) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        auto [lo, hi, w] = bracket(q.strike);
        double p_lo = eval_kref(lo, q);
        if (lo == hi) return p_lo;
        double p_hi = eval_kref(hi, q);
        return (1.0 - w) * p_lo + w * p_hi;
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        auto [lo, hi, w] = bracket(q.strike);
        double v_lo = vega_kref(lo, q);
        if (lo == hi) return v_lo;
        double v_hi = vega_kref(hi, q);
        return (1.0 - w) * v_lo + w * v_hi;
    }

    [[nodiscard]] size_t num_krefs() const { return entries_.size(); }
    [[nodiscard]] size_t num_segments() const {
        return entries_.empty() ? 0 : entries_[0].segments.size();
    }

private:
    [[nodiscard]] double eval_kref(size_t idx, const PriceQuery& q) const {
        double K_ref = entries_[idx].K_ref;
        double x = std::log(q.spot / K_ref);  // ln(S/K_ref), not ln(S/K)
        size_t seg = find_segment(entries_[idx].segments, q.tau);
        double v_norm = entries_[idx].segments[seg].interp.eval(
            {x, q.tau, q.sigma, q.rate});
        // V/K_ref * K: dimensionless scaling from K_ref to query strike
        return v_norm * q.strike;
    }

    [[nodiscard]] double vega_kref(size_t idx, const PriceQuery& q) const {
        double K_ref = entries_[idx].K_ref;
        double x = std::log(q.spot / K_ref);
        size_t seg = find_segment(entries_[idx].segments, q.tau);
        // sigma is axis 2
        double dvds = entries_[idx].segments[seg].interp.partial(
            2, {x, q.tau, q.sigma, q.rate});
        return dvds * q.strike;
    }

    struct BracketResult { size_t lo, hi; double w; };

    [[nodiscard]] BracketResult bracket(double K) const {
        const size_t n = entries_.size();
        if (n == 1 || K <= entries_.front().K_ref)
            return {0, 0, 0.0};
        if (K >= entries_.back().K_ref)
            return {n - 1, n - 1, 0.0};

        size_t hi = 1;
        while (hi < n && entries_[hi].K_ref < K) ++hi;
        size_t lo = hi - 1;
        double t = (K - entries_[lo].K_ref) /
                   (entries_[hi].K_ref - entries_[lo].K_ref);
        return {lo, hi, t};
    }

    static size_t find_segment(const std::vector<SegCheb4DDivSegment>& segs,
                               double tau) {
        for (int i = static_cast<int>(segs.size()) - 1; i >= 0; --i) {
            if (tau >= segs[static_cast<size_t>(i)].tau_lo)
                return static_cast<size_t>(i);
        }
        return 0;
    }

    std::vector<SegCheb4DDivKrefEntry> entries_;
    OptionType type_;
};

// ============================================================================
// Builder: multi-K_ref segmented batch-PDE Chebyshev 4D
// ============================================================================

namespace detail {

// Segment boundary computation (shared across K_refs).
struct SegBounds { double lo, hi; };

inline std::vector<SegBounds> compute_seg_bounds(
    const std::vector<Dividend>& divs,
    double tau_min, double tau_max)
{
    constexpr double kInset = 5e-4;

    std::vector<double> div_taus;
    for (const auto& d : divs) {
        if (d.calendar_time > tau_min + kInset &&
            d.calendar_time < tau_max - kInset) {
            div_taus.push_back(d.calendar_time);
        }
    }
    std::sort(div_taus.begin(), div_taus.end());

    std::vector<SegBounds> bounds;
    double prev = tau_min;
    for (double dt : div_taus) {
        bounds.push_back({prev, dt - kInset});
        prev = dt + kInset;
    }
    bounds.push_back({prev, tau_max});
    return bounds;
}

// Build segmented tensors for a single K_ref.
inline std::vector<SegCheb4DDivSegment> build_segments_for_kref(
    const SegCheb4DDivConfig& cfg,
    double K_ref,
    const std::vector<SegBounds>& seg_bounds,
    const std::vector<std::vector<double>>& seg_tau_nodes,
    const std::vector<double>& x_nodes,
    const std::vector<double>& sigma_nodes,
    const std::vector<double>& rate_nodes,
    double x_lo, double x_hi,
    double sigma_lo, double sigma_hi,
    double rate_lo, double rate_hi,
    int& total_pde_solves)
{
    const size_t Nx = cfg.num_x;
    const size_t Ns = cfg.num_sigma;
    const size_t Nr = cfg.num_rate;
    const size_t n_segs = seg_bounds.size();

    std::vector<SegCheb4DDivSegment> segments;
    segments.reserve(n_segs);

    for (size_t seg = 0; seg < n_segs; ++seg) {
        const size_t Nt = cfg.num_tau_per_seg;
        const auto& tau_nodes = seg_tau_nodes[seg];
        double seg_maturity = tau_nodes.back() * 1.01;

        // Filter dividends to those before segment boundary
        std::vector<Dividend> seg_divs;
        for (const auto& d : cfg.discrete_dividends) {
            if (d.calendar_time < seg_bounds[seg].hi)
                seg_divs.push_back(d);
        }

        // Batch-solve at this K_ref
        std::vector<PricingParams> batch;
        batch.reserve(Ns * Nr);
        for (size_t s = 0; s < Ns; ++s) {
            for (size_t r = 0; r < Nr; ++r) {
                PricingParams p(
                    OptionSpec{.spot = K_ref, .strike = K_ref,
                               .maturity = seg_maturity,
                               .rate = rate_nodes[r],
                               .dividend_yield = cfg.dividend_yield,
                               .option_type = cfg.option_type},
                    sigma_nodes[s]);
                p.discrete_dividends = seg_divs;
                batch.push_back(std::move(p));
            }
        }

        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
        solver.set_snapshot_times(std::span<const double>{tau_nodes});
        auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);
        total_pde_solves += static_cast<int>(Ns * Nr);

        // Build tensor
        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

        for (size_t s = 0; s < Ns; ++s) {
            for (size_t r = 0; r < Nr; ++r) {
                size_t batch_idx = s * Nr + r;
                if (!batch_result.results[batch_idx].has_value()) continue;

                const auto& result = batch_result.results[batch_idx].value();
                auto x_grid = result.grid()->x();

                for (size_t j = 0; j < Nt; ++j) {
                    auto spatial = result.at_time(j);

                    CubicSpline<double> spline;
                    if (spline.build(x_grid, spatial).has_value()) continue;

                    for (size_t i = 0; i < Nx; ++i) {
                        tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] =
                            spline.eval(x_nodes[i]);
                    }
                }
            }
        }

        ChebyshevTucker4DDomain dom{
            .bounds = {{{x_lo, x_hi},
                        {seg_bounds[seg].lo, seg_bounds[seg].hi},
                        {sigma_lo, sigma_hi},
                        {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon,
            .use_tucker = cfg.use_tucker};

        segments.push_back({
            .interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg),
            .tau_lo = seg_bounds[seg].lo,
            .tau_hi = seg_bounds[seg].hi,
        });
    }

    return segments;
}

}  // namespace detail

inline SegCheb4DDivResult build_segmented_cheb_4d_div(
    const SegCheb4DDivConfig& cfg,
    double default_K_ref)
{
    auto t0 = std::chrono::steady_clock::now();

    // K_refs: use config list or fall back to single default
    std::vector<double> K_refs = cfg.K_refs;
    if (K_refs.empty())
        K_refs.push_back(default_K_ref);
    std::sort(K_refs.begin(), K_refs.end());

    // ---- 1. Segment boundaries (shared across K_refs) ----
    auto seg_bounds = detail::compute_seg_bounds(
        cfg.discrete_dividends, cfg.tau_min, cfg.tau_max);
    size_t n_segs = seg_bounds.size();

    // ---- 2. Per-axis setup (shared across K_refs) ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hx     = headroom_fn(cfg.x_min,     cfg.x_max,     cfg.num_x);
    double hsigma = headroom_fn(cfg.sigma_min,  cfg.sigma_max, cfg.num_sigma);
    double hrate  = headroom_fn(cfg.rate_min,   cfg.rate_max,  cfg.num_rate);

    double x_lo     = cfg.x_min - hx;
    double x_hi     = cfg.x_max + hx;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    auto x_nodes     = chebyshev_nodes(cfg.num_x,     x_lo,     x_hi);
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate,  rate_lo,  rate_hi);

    // ---- 3. Per-segment CGL tau nodes (shared) ----
    std::vector<std::vector<double>> seg_tau_nodes(n_segs);
    for (size_t seg = 0; seg < n_segs; ++seg) {
        seg_tau_nodes[seg] = chebyshev_nodes(
            cfg.num_tau_per_seg, seg_bounds[seg].lo, seg_bounds[seg].hi);
    }

    // ---- 4. Build per-K_ref segmented tensors ----
    int total_pde_solves = 0;
    std::vector<SegCheb4DDivKrefEntry> entries;
    entries.reserve(K_refs.size());

    for (double K_ref : K_refs) {
        std::fprintf(stderr, "    K_ref=%.0f: %zu segs × %zu PDE...\n",
                     K_ref, n_segs,
                     cfg.num_sigma * cfg.num_rate);

        auto segments = detail::build_segments_for_kref(
            cfg, K_ref, seg_bounds, seg_tau_nodes,
            x_nodes, sigma_nodes, rate_nodes,
            x_lo, x_hi, sigma_lo, sigma_hi, rate_lo, rate_hi,
            total_pde_solves);

        entries.push_back({
            .K_ref = K_ref,
            .segments = std::move(segments),
        });
    }

    auto t1 = std::chrono::steady_clock::now();

    return {
        .entries = std::move(entries),
        .n_pde_solves = total_pde_solves,
        .build_seconds = std::chrono::duration<double>(t1 - t0).count(),
    };
}

}  // namespace mango
