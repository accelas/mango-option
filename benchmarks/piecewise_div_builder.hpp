// SPDX-License-Identifier: MIT
//
// Piecewise Chebyshev 4D for discrete dividends:
//   multi-K_ref × τ-segmented × 3 x-elements (ITM/boundary/OTM).
// Stores V/K_ref directly (no EEP decomposition).
// Benchmark experiment only.
#pragma once

#include "bump_blend.hpp"
#include "pde_slice_cache.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/european_option.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace mango {

// ============================================================================
// Config types
// ============================================================================

struct PiecewiseDivConfig {
    size_t num_x_coarse = 20;      // ITM/OTM elements
    size_t num_x_dense = 30;       // boundary element
    size_t num_tau_per_seg = 9;    // CGL nodes per τ-segment
    size_t num_sigma = 9;
    size_t num_rate = 5;
    double epsilon = 1e-8;
    bool use_tucker = false;

    double x_min = -0.50;
    double x_max = 0.40;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;
    double tau_min = 0.019;        // ~7 days
    double tau_max = 2.0;

    double dividend_yield = 0.02;
    std::vector<Dividend> discrete_dividends;
    OptionType option_type = OptionType::PUT;

    std::vector<double> K_refs;

    // Boundary detection
    size_t n_scan_points = 200;
    double delta_min = 0.10;
    double delta_margin = 0.05;

    // Headroom reference counts (fixed across elements)
    size_t sigma_headroom_ref = 15;
    size_t rate_headroom_ref = 9;
};

// ============================================================================
// Internal data structures
// ============================================================================

struct PwDivElement {
    ChebyshevTucker4D interp;     // axes: (x, tau, sigma, rate)
    double x_lo, x_hi;
};

struct PwDivOverlap {
    size_t left_idx, right_idx;   // indices into segment's elements
    double x_lo, x_hi;
};

struct PwDivBoundary {
    double x_star;
    double delta;
    size_t n_valid;
    size_t n_sampled;
};

struct PwDivSegment {
    std::vector<PwDivElement> elements;   // 3: ITM, boundary, OTM
    std::vector<PwDivOverlap> overlaps;   // 2: ITM-boundary, boundary-OTM
    double tau_lo, tau_hi;
    PwDivBoundary boundary;
};

struct PwDivKrefEntry {
    double K_ref;
    std::vector<PwDivSegment> segments;
};

struct PwDivBuildResult {
    std::vector<PwDivKrefEntry> entries;
    int n_pde_solves;
    double build_seconds;
};

// ============================================================================
// Evaluator: multi-K_ref + segment routing + piecewise x-blending
// ============================================================================

class PiecewiseDivEvaluator {
public:
    PiecewiseDivEvaluator(std::vector<PwDivKrefEntry> entries, OptionType type)
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
    [[nodiscard]] size_t total_elements() const {
        size_t count = 0;
        for (const auto& e : entries_)
            for (const auto& s : e.segments)
                count += s.elements.size();
        return count;
    }

private:
    [[nodiscard]] double eval_kref(size_t idx, const PriceQuery& q) const {
        double K_ref = entries_[idx].K_ref;
        double x = std::log(q.spot / K_ref);  // ln(S/K_ref)
        size_t seg = find_segment(entries_[idx].segments, q.tau);
        double v_norm = eval_segment(entries_[idx].segments[seg],
                                      x, q.tau, q.sigma, q.rate);
        return v_norm * q.strike;
    }

    [[nodiscard]] double vega_kref(size_t idx, const PriceQuery& q) const {
        double K_ref = entries_[idx].K_ref;
        double x = std::log(q.spot / K_ref);
        size_t seg = find_segment(entries_[idx].segments, q.tau);
        double dvds = vega_segment(entries_[idx].segments[seg],
                                    x, q.tau, q.sigma, q.rate);
        return dvds * q.strike;
    }

    /// Evaluate V/K_ref within a segment using piecewise x-blending.
    [[nodiscard]] double eval_segment(const PwDivSegment& seg,
                                       double x, double tau,
                                       double sigma, double rate) const {
        auto [elems, weights] = find_x_elements(seg, x);
        double result = 0.0;
        for (size_t i = 0; i < elems.size(); ++i) {
            result += weights[i] * seg.elements[elems[i]].interp.eval(
                {x, tau, sigma, rate});
        }
        return result;
    }

    /// Evaluate ∂(V/K_ref)/∂σ within a segment using piecewise x-blending.
    [[nodiscard]] double vega_segment(const PwDivSegment& seg,
                                       double x, double tau,
                                       double sigma, double rate) const {
        auto [elems, weights] = find_x_elements(seg, x);
        double result = 0.0;
        for (size_t i = 0; i < elems.size(); ++i) {
            // sigma is axis 2
            result += weights[i] * seg.elements[elems[i]].interp.partial(
                2, {x, tau, sigma, rate});
        }
        return result;
    }

    /// Find x-element(s) with C∞ bump blending weights.
    /// Returns (element_indices, weights) — 1 or 2 elements.
    struct XElementResult {
        std::vector<size_t> indices;
        std::vector<double> weights;
    };

    [[nodiscard]] XElementResult find_x_elements(
        const PwDivSegment& seg, double x) const {
        // Check overlap zones
        for (const auto& oz : seg.overlaps) {
            if (x >= oz.x_lo && x <= oz.x_hi) {
                double w_right = overlap_weight_right(x, oz.x_lo, oz.x_hi);
                return {{oz.left_idx, oz.right_idx},
                        {1.0 - w_right, w_right}};
            }
        }
        // Single element
        for (size_t i = 0; i < seg.elements.size(); ++i) {
            if (x >= seg.elements[i].x_lo && x <= seg.elements[i].x_hi) {
                return {{i}, {1.0}};
            }
        }
        // Fallback: closest
        size_t closest = 0;
        double best_dist = 1e99;
        for (size_t i = 0; i < seg.elements.size(); ++i) {
            double mid = (seg.elements[i].x_lo + seg.elements[i].x_hi) / 2.0;
            double dist = std::abs(x - mid);
            if (dist < best_dist) { best_dist = dist; closest = i; }
        }
        return {{closest}, {1.0}};
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

    static size_t find_segment(const std::vector<PwDivSegment>& segs,
                               double tau) {
        for (int i = static_cast<int>(segs.size()) - 1; i >= 0; --i) {
            if (tau >= segs[static_cast<size_t>(i)].tau_lo)
                return static_cast<size_t>(i);
        }
        return 0;
    }

    std::vector<PwDivKrefEntry> entries_;
    OptionType type_;
};

// ============================================================================
// Builder internals
// ============================================================================

namespace pw_div_detail {

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

/// Detect exercise boundary from cached PDE slices for a single τ-segment.
/// Uses EEP = Am - Eu(continuous-dividend) as proxy for boundary detection.
inline PwDivBoundary detect_boundary_from_cache(
    const PDESliceCache& cache,
    const std::vector<double>& tau_nodes,
    const std::vector<double>& sigma_nodes,
    const std::vector<double>& rate_nodes,
    double K_ref, OptionType option_type,
    double dividend_yield,
    double x_min, double x_max,
    size_t n_scan_points, double delta_min, double delta_margin)
{
    std::vector<double> all_x_stars;
    size_t n_sampled = 0;

    for (size_t si = 0; si < sigma_nodes.size(); ++si) {
        double sigma = sigma_nodes[si];
        for (size_t ri = 0; ri < rate_nodes.size(); ++ri) {
            double rate = rate_nodes[ri];
            for (size_t ti = 0; ti < tau_nodes.size(); ++ti) {
                auto* spline = cache.get_slice(sigma, rate, ti);
                if (!spline) continue;
                n_sampled++;

                double tau = tau_nodes[ti];

                // Scan EEP = Am - Eu at scan points
                double max_grad = 0.0;
                int max_grad_idx = -1;
                std::vector<double> eep(n_scan_points);
                std::vector<double> xs(n_scan_points);

                for (size_t k = 0; k < n_scan_points; ++k) {
                    double x = x_min + (x_max - x_min) * k /
                               (n_scan_points - 1);
                    xs[k] = x;

                    double am = spline->eval(x) * K_ref;
                    double spot = std::exp(x) * K_ref;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot, .strike = K_ref,
                                   .maturity = tau, .rate = rate,
                                   .dividend_yield = dividend_yield,
                                   .option_type = option_type},
                        sigma).solve();
                    if (!eu) continue;
                    eep[k] = am - eu->value();
                }

                // Find steepest gradient
                for (size_t k = 1; k < n_scan_points; ++k) {
                    double dx = xs[k] - xs[k - 1];
                    if (dx < 1e-15) continue;
                    double grad = std::abs(eep[k] - eep[k - 1]) / dx;
                    if (grad > max_grad) {
                        max_grad = grad;
                        max_grad_idx = static_cast<int>(k);
                    }
                }

                double eep_range = *std::max_element(eep.begin(), eep.end()) -
                                   *std::min_element(eep.begin(), eep.end());
                if (max_grad_idx < 0 || eep_range < 1e-6) continue;

                double x_star = (xs[max_grad_idx - 1] + xs[max_grad_idx]) / 2.0;
                all_x_stars.push_back(x_star);
            }
        }
    }

    if (n_sampled == 0 || all_x_stars.size() < 0.3 * n_sampled) {
        return {0.0, delta_min, all_x_stars.size(), n_sampled};
    }

    std::sort(all_x_stars.begin(), all_x_stars.end());
    size_t n = all_x_stars.size();
    double median = all_x_stars[n / 2];
    size_t p10_idx = n / 10;
    size_t p90_idx = std::min(n * 9 / 10, n - 1);
    double spread = all_x_stars[p90_idx] - all_x_stars[p10_idx];
    double delta = std::max(delta_min, spread / 2.0 + delta_margin);

    return {median, delta, n, n_sampled};
}

/// Build 3 x-elements for a single segment from cached PDE slices.
/// Stores V/K_ref directly (no EEP decomposition).
inline PwDivSegment build_segment_elements(
    const PiecewiseDivConfig& cfg,
    const PDESliceCache& cache,
    const std::vector<double>& tau_nodes,
    const std::vector<double>& sigma_nodes,
    const std::vector<double>& rate_nodes,
    double /*K_ref*/,
    double seg_tau_lo, double seg_tau_hi,
    double sigma_lo, double sigma_hi,
    double rate_lo, double rate_hi,
    const PwDivBoundary& boundary)
{
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };

    double x_star = boundary.x_star;
    double delta = boundary.delta;
    double h_boundary = 2.0 * delta / (cfg.num_x_dense - 1);
    double w_overlap = 2.0 * h_boundary;

    // No τ headroom: element tau domain = segment domain exactly.
    // PDE snapshots are at CGL nodes on [seg_tau_lo, seg_tau_hi], so
    // element CGL nodes must match. Extending tau into adjacent dividend
    // regimes would require PDE data we don't have.
    double tau_lo_ext = seg_tau_lo;
    double tau_hi_ext = seg_tau_hi;

    // x-element boundaries
    double itm_hi = std::max(x_star - delta, cfg.x_min + 0.05);
    double otm_lo = std::min(x_star + delta, cfg.x_max - 0.05);

    if (otm_lo <= itm_hi + 0.02) {
        double mid = (itm_hi + otm_lo) / 2.0;
        itm_hi = mid - 0.05;
        otm_lo = mid + 0.05;
    }

    // Element specs: ITM, Boundary, OTM
    struct ElemSpec {
        double x_lo, x_hi;
        size_t num_x;
    };

    double hx_itm = headroom_fn(cfg.x_min, itm_hi, cfg.num_x_coarse);
    double hx_otm = headroom_fn(otm_lo, cfg.x_max, cfg.num_x_coarse);

    ElemSpec specs[3] = {
        {cfg.x_min - hx_itm, itm_hi + w_overlap, cfg.num_x_coarse},
        {itm_hi - w_overlap, otm_lo + w_overlap, cfg.num_x_dense},
        {otm_lo - w_overlap, cfg.x_max + hx_otm, cfg.num_x_coarse},
    };

    // Overlap zones
    std::vector<PwDivOverlap> overlaps = {
        {0, 1, itm_hi - w_overlap, itm_hi + w_overlap},
        {1, 2, otm_lo - w_overlap, otm_lo + w_overlap},
    };

    // Build each element's tensor from cache
    const size_t Nt = cfg.num_tau_per_seg;
    const size_t Ns = sigma_nodes.size();
    const size_t Nr = rate_nodes.size();

    std::vector<PwDivElement> elements;
    elements.reserve(3);

    for (int ei = 0; ei < 3; ++ei) {
        const auto& spec = specs[ei];
        const size_t Nx = spec.num_x;
        auto x_nodes = chebyshev_nodes(Nx, spec.x_lo, spec.x_hi);
        auto tau_nodes_elem = chebyshev_nodes(Nt, tau_lo_ext, tau_hi_ext);

        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

        for (size_t s = 0; s < Ns; ++s) {
            double sigma = sigma_nodes[s];
            for (size_t r = 0; r < Nr; ++r) {
                double rate = rate_nodes[r];
                for (size_t j = 0; j < Nt; ++j) {
                    double tau = tau_nodes_elem[j];

                    // Map element tau to cache tau index (interpolate)
                    double frac_idx = 0.0;
                    bool exact = false;
                    for (size_t k = 0; k < tau_nodes.size(); ++k) {
                        if (std::abs(tau - tau_nodes[k]) < 1e-12) {
                            frac_idx = static_cast<double>(k);
                            exact = true;
                            break;
                        }
                    }
                    if (!exact) {
                        for (size_t k = 0; k + 1 < tau_nodes.size(); ++k) {
                            if (tau >= tau_nodes[k] && tau <= tau_nodes[k + 1]) {
                                double alpha = (tau - tau_nodes[k]) /
                                               (tau_nodes[k + 1] - tau_nodes[k]);
                                frac_idx = k + alpha;
                                break;
                            }
                        }
                        if (frac_idx == 0.0 && tau > tau_nodes.back()) {
                            frac_idx = static_cast<double>(tau_nodes.size() - 1);
                        }
                    }

                    size_t idx_lo = static_cast<size_t>(frac_idx);
                    size_t idx_hi = std::min(idx_lo + 1, tau_nodes.size() - 1);
                    double alpha = frac_idx - idx_lo;

                    auto* slice_lo = cache.get_slice(sigma, rate, idx_lo);
                    auto* slice_hi = (idx_lo != idx_hi)
                        ? cache.get_slice(sigma, rate, idx_hi)
                        : slice_lo;

                    if (!slice_lo) continue;

                    for (size_t i = 0; i < Nx; ++i) {
                        double x = x_nodes[i];
                        double v_lo = slice_lo->eval(x);
                        double v = v_lo;
                        if (alpha > 1e-12 && slice_hi && slice_hi != slice_lo) {
                            double v_hi = slice_hi->eval(x);
                            v = (1.0 - alpha) * v_lo + alpha * v_hi;
                        }
                        // Store V/K_ref directly
                        tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] = v;
                    }
                }
            }
        }

        ChebyshevTucker4DDomain dom{
            .bounds = {{{spec.x_lo, spec.x_hi},
                        {tau_lo_ext, tau_hi_ext},
                        {sigma_lo, sigma_hi},
                        {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon,
            .use_tucker = cfg.use_tucker};

        elements.push_back({
            .interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg),
            .x_lo = spec.x_lo,
            .x_hi = spec.x_hi,
        });
    }

    return {
        .elements = std::move(elements),
        .overlaps = std::move(overlaps),
        .tau_lo = seg_tau_lo,
        .tau_hi = seg_tau_hi,
        .boundary = boundary,
    };
}

}  // namespace pw_div_detail

// ============================================================================
// Top-level builder
// ============================================================================

inline PwDivBuildResult build_piecewise_div(
    const PiecewiseDivConfig& cfg,
    double default_K_ref)
{
    auto t0 = std::chrono::steady_clock::now();

    // K_refs
    std::vector<double> K_refs = cfg.K_refs;
    if (K_refs.empty()) K_refs.push_back(default_K_ref);
    std::sort(K_refs.begin(), K_refs.end());

    // 1. Segment boundaries (shared)
    auto seg_bounds = pw_div_detail::compute_seg_bounds(
        cfg.discrete_dividends, cfg.tau_min, cfg.tau_max);
    size_t n_segs = seg_bounds.size();

    // 2. Shared axis setup
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.sigma_headroom_ref);
    double hrate  = headroom_fn(cfg.rate_min, cfg.rate_max, cfg.rate_headroom_ref);

    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate, rate_lo, rate_hi);

    // 3. Per-segment CGL tau nodes (shared)
    std::vector<std::vector<double>> seg_tau_nodes(n_segs);
    for (size_t seg = 0; seg < n_segs; ++seg) {
        seg_tau_nodes[seg] = chebyshev_nodes(
            cfg.num_tau_per_seg, seg_bounds[seg].lo, seg_bounds[seg].hi);
    }

    // 4. Build per-K_ref
    int total_pde_solves = 0;
    std::vector<PwDivKrefEntry> entries;
    entries.reserve(K_refs.size());

    for (double K_ref : K_refs) {
        std::fprintf(stderr, "    K_ref=%.0f: %zu segs...\n", K_ref, n_segs);

        // Per-segment: PDE solve → cache → boundary detect → piecewise elements
        std::vector<PwDivSegment> kref_segments;
        kref_segments.reserve(n_segs);

        for (size_t seg = 0; seg < n_segs; ++seg) {
            const auto& tau_nodes = seg_tau_nodes[seg];
            double seg_maturity = tau_nodes.back() * 1.01;

            // Filter dividends to those before segment boundary
            std::vector<Dividend> seg_divs;
            for (const auto& d : cfg.discrete_dividends) {
                if (d.calendar_time < seg_bounds[seg].hi)
                    seg_divs.push_back(d);
            }

            // Batch PDE solve at this K_ref
            std::vector<PricingParams> batch;
            batch.reserve(cfg.num_sigma * cfg.num_rate);
            for (size_t s = 0; s < cfg.num_sigma; ++s) {
                for (size_t r = 0; r < cfg.num_rate; ++r) {
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
            total_pde_solves += static_cast<int>(cfg.num_sigma * cfg.num_rate);

            // Store in per-segment cache
            PDESliceCache cache;
            for (size_t s = 0; s < cfg.num_sigma; ++s) {
                for (size_t r = 0; r < cfg.num_rate; ++r) {
                    size_t batch_idx = s * cfg.num_rate + r;
                    if (!batch_result.results[batch_idx].has_value()) continue;

                    const auto& result = batch_result.results[batch_idx].value();
                    auto x_grid = result.grid()->x();

                    for (size_t j = 0; j < cfg.num_tau_per_seg; ++j) {
                        auto spatial = result.at_time(j);
                        cache.store_slice(sigma_nodes[s], rate_nodes[r], j,
                                          x_grid, spatial);
                    }
                }
            }

            // Detect exercise boundary for this segment
            auto boundary = pw_div_detail::detect_boundary_from_cache(
                cache, tau_nodes, sigma_nodes, rate_nodes,
                K_ref, cfg.option_type, cfg.dividend_yield,
                cfg.x_min, cfg.x_max,
                cfg.n_scan_points, cfg.delta_min, cfg.delta_margin);

            // Build 3 piecewise elements from cache
            auto pw_seg = pw_div_detail::build_segment_elements(
                cfg, cache, tau_nodes, sigma_nodes, rate_nodes, K_ref,
                seg_bounds[seg].lo, seg_bounds[seg].hi,
                sigma_lo, sigma_hi, rate_lo, rate_hi,
                boundary);

            kref_segments.push_back(std::move(pw_seg));
        }

        entries.push_back({
            .K_ref = K_ref,
            .segments = std::move(kref_segments),
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
