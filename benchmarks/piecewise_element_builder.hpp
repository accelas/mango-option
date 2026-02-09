// SPDX-License-Identifier: MIT
//
// Build 6 piecewise Chebyshev 4D elements: 2 τ-bands × 3 x-elements.
// Reads from PDESliceCache populated by Phase A incremental builder.
#pragma once

#include "pde_slice_cache.hpp"
#include "boundary_detector.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mango {

struct ElementSpec {
    double x_lo, x_hi;       // element x-bounds (with headroom on outer edges)
    double tau_lo, tau_hi;    // element τ-bounds
    size_t num_x;             // CGL nodes in x (15 or 25)
    size_t num_tau;           // CGL nodes in τ
    size_t tau_band;          // 0=short, 1=long
};

struct OverlapZone {
    size_t left_idx, right_idx;
    double x_lo, x_hi;
};

struct PiecewiseElementSet {
    std::vector<ChebyshevTucker4D> elements;   // 6 elements
    std::vector<ElementSpec> specs;             // per-element config
    std::vector<OverlapZone> x_overlaps;       // 2 per τ-band = 4 total

    double tau_blend_lo, tau_blend_hi;         // τ-band overlap [55d, 65d]

    // Shared axes bounds (extended)
    double sigma_lo, sigma_hi;
    double rate_lo, rate_hi;

    size_t total_pde_solves;
    double build_seconds;

    // Boundary detection results
    BoundaryResult short_boundary, long_boundary;
};

struct PiecewiseElementBuildConfig {
    // CC levels for σ and r (reuse Phase A cache)
    size_t sigma_level = 4;
    size_t rate_level = 3;

    // Node counts per element type
    size_t num_x_coarse = 15;      // ITM and OTM elements
    size_t num_x_dense = 25;       // boundary element
    size_t num_tau = 9;            // τ nodes per band

    // τ-band boundaries (in years)
    double tau_short_lo = 0.019;   // ~7d
    double tau_short_hi = 60.0 / 365.0;
    double tau_long_lo = 60.0 / 365.0;
    double tau_long_hi = 2.0;

    // τ-band overlap zone (in years)
    double tau_blend_lo = 55.0 / 365.0;
    double tau_blend_hi = 65.0 / 365.0;

    // x-domain bounds (physical)
    double x_min = -0.50;
    double x_max = 0.40;

    // σ, r physical bounds
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    // Fixed headroom references (must match Phase A IncrementalBuildConfig)
    size_t sigma_headroom_ref = 15;
    size_t rate_headroom_ref = 9;

    double epsilon = 1e-8;
    bool use_tucker = false;
    double dividend_yield = 0.0;
    bool use_hard_max = true;
    double K_ref = 100.0;
    OptionType option_type = OptionType::PUT;
};

inline PiecewiseElementSet build_piecewise_elements(
    const PiecewiseElementBuildConfig& cfg,
    const PDESliceCache& cache,
    std::span<const double> cache_tau_nodes)
{
    auto t0 = std::chrono::steady_clock::now();

    // ---- 1. Shared axis setup ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    size_t n_sigma = (1u << cfg.sigma_level) + 1;
    size_t n_rate = (1u << cfg.rate_level) + 1;

    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.sigma_headroom_ref);
    double hrate = headroom_fn(cfg.rate_min, cfg.rate_max, cfg.rate_headroom_ref);

    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi = cfg.rate_max + hrate;

    auto sigma_nodes = cc_level_nodes(cfg.sigma_level, sigma_lo, sigma_hi);
    auto rate_nodes = cc_level_nodes(cfg.rate_level, rate_lo, rate_hi);

    // ---- 2. Detect exercise boundary per τ-band ----
    // Map τ ranges to cache tau_idx ranges
    auto find_tau_idx_range = [&](double tau_lo_phys, double tau_hi_phys)
        -> std::pair<size_t, size_t> {
        size_t lo = 0, hi = cache_tau_nodes.size() - 1;
        for (size_t i = 0; i < cache_tau_nodes.size(); ++i) {
            if (cache_tau_nodes[i] >= tau_lo_phys) { lo = i; break; }
        }
        for (size_t i = cache_tau_nodes.size(); i > 0; --i) {
            if (cache_tau_nodes[i - 1] <= tau_hi_phys) { hi = i - 1; break; }
        }
        return {lo, hi};
    };

    auto [short_tau_lo_idx, short_tau_hi_idx] = find_tau_idx_range(
        cfg.tau_short_lo, cfg.tau_short_hi);
    auto [long_tau_lo_idx, long_tau_hi_idx] = find_tau_idx_range(
        cfg.tau_long_lo, cfg.tau_long_hi);

    // Detect long-τ boundary first (more reliable, deeper ITM)
    BoundaryDetectorConfig det_cfg;
    det_cfg.x_min = cfg.x_min;
    det_cfg.x_max = cfg.x_max;

    auto long_boundary = detect_exercise_boundary(
        cache, cache_tau_nodes, sigma_nodes, rate_nodes,
        long_tau_lo_idx, long_tau_hi_idx,
        cfg.K_ref, cfg.option_type, det_cfg, cfg.dividend_yield);

    auto short_boundary = detect_exercise_boundary(
        cache, cache_tau_nodes, sigma_nodes, rate_nodes,
        short_tau_lo_idx, short_tau_hi_idx,
        cfg.K_ref, cfg.option_type, det_cfg, cfg.dividend_yield,
        long_boundary.x_star);  // fallback to long-τ boundary

    // ---- 3. Build element specs: 3 x-elements per τ-band ----
    struct BandInfo {
        double tau_lo, tau_hi;
        BoundaryResult boundary;
    };
    BandInfo bands[] = {
        {cfg.tau_short_lo, cfg.tau_short_hi, short_boundary},
        {cfg.tau_long_lo, cfg.tau_long_hi, long_boundary},
    };

    std::vector<ElementSpec> specs;
    std::vector<OverlapZone> x_overlaps;
    size_t elem_idx = 0;

    for (size_t band = 0; band < 2; ++band) {
        auto& b = bands[band];
        double x_star = b.boundary.x_star;
        double delta = b.boundary.delta;
        double h_boundary = 2.0 * delta / (cfg.num_x_dense - 1);
        double w_overlap = 2.0 * h_boundary;

        // τ headroom
        double htau = headroom_fn(b.tau_lo, b.tau_hi, cfg.num_tau);
        double tau_lo_ext = std::max(b.tau_lo - htau, 1e-4);
        double tau_hi_ext = b.tau_hi + htau;

        // x-element boundaries (physical), clamped to domain
        double itm_hi = std::max(x_star - delta, cfg.x_min + 0.05);
        double otm_lo = std::min(x_star + delta, cfg.x_max - 0.05);

        // Ensure boundary element has positive width
        if (otm_lo <= itm_hi + 0.02) {
            double mid = (itm_hi + otm_lo) / 2.0;
            itm_hi = mid - 0.05;
            otm_lo = mid + 0.05;
        }

        // ITM element: [x_min, itm_hi] with outer headroom
        double hx_itm = headroom_fn(cfg.x_min, itm_hi, cfg.num_x_coarse);
        specs.push_back({
            .x_lo = cfg.x_min - hx_itm, .x_hi = itm_hi + w_overlap,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_coarse, .num_tau = cfg.num_tau,
            .tau_band = band});

        // Boundary element: [itm_hi, otm_lo] — dense, with overlap on both sides
        specs.push_back({
            .x_lo = itm_hi - w_overlap, .x_hi = otm_lo + w_overlap,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_dense, .num_tau = cfg.num_tau,
            .tau_band = band});

        // OTM element: [otm_lo, x_max] with outer headroom
        double hx_otm = headroom_fn(otm_lo, cfg.x_max, cfg.num_x_coarse);
        specs.push_back({
            .x_lo = otm_lo - w_overlap, .x_hi = cfg.x_max + hx_otm,
            .tau_lo = tau_lo_ext, .tau_hi = tau_hi_ext,
            .num_x = cfg.num_x_coarse, .num_tau = cfg.num_tau,
            .tau_band = band});

        // Overlap zones: ITM-Boundary and Boundary-OTM
        x_overlaps.push_back({
            .left_idx = elem_idx, .right_idx = elem_idx + 1,
            .x_lo = itm_hi - w_overlap, .x_hi = itm_hi + w_overlap});
        x_overlaps.push_back({
            .left_idx = elem_idx + 1, .right_idx = elem_idx + 2,
            .x_lo = otm_lo - w_overlap, .x_hi = otm_lo + w_overlap});

        elem_idx += 3;
    }

    // ---- 4. Build per-element tensors from cache ----
    // Map an element's τ-CGL node to a cached snapshot.
    // Returns fractional index: integer part = tau_idx_lo, fractional part = alpha.
    auto map_tau_to_cache = [&](double tau) -> double {
        // Exact match first
        for (size_t i = 0; i < cache_tau_nodes.size(); ++i) {
            if (std::abs(tau - cache_tau_nodes[i]) < 1e-12)
                return static_cast<double>(i);
        }
        // Bracket and interpolate
        for (size_t i = 0; i + 1 < cache_tau_nodes.size(); ++i) {
            if (tau >= cache_tau_nodes[i] && tau <= cache_tau_nodes[i + 1]) {
                double alpha = (tau - cache_tau_nodes[i]) /
                               (cache_tau_nodes[i + 1] - cache_tau_nodes[i]);
                return i + alpha;
            }
        }
        // Clamp
        return tau < cache_tau_nodes[0] ? 0.0
             : static_cast<double>(cache_tau_nodes.size() - 1);
    };

    std::vector<ChebyshevTucker4D> elements;
    elements.reserve(specs.size());

    for (const auto& spec : specs) {
        auto x_nodes = chebyshev_nodes(spec.num_x, spec.x_lo, spec.x_hi);
        auto tau_nodes_elem = chebyshev_nodes(spec.num_tau, spec.tau_lo, spec.tau_hi);

        size_t Nx = spec.num_x;
        size_t Nt = spec.num_tau;
        size_t Ns = n_sigma;
        size_t Nr = n_rate;
        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

        for (size_t s = 0; s < Ns; ++s) {
            double sigma = sigma_nodes[s];
            for (size_t r = 0; r < Nr; ++r) {
                double rate = rate_nodes[r];
                for (size_t j = 0; j < Nt; ++j) {
                    double tau = tau_nodes_elem[j];
                    double frac_idx = map_tau_to_cache(tau);
                    size_t idx_lo = static_cast<size_t>(frac_idx);
                    size_t idx_hi = std::min(idx_lo + 1, cache_tau_nodes.size() - 1);
                    double alpha = frac_idx - idx_lo;

                    auto* slice_lo = cache.get_slice(sigma, rate, idx_lo);
                    auto* slice_hi = (idx_lo != idx_hi)
                        ? cache.get_slice(sigma, rate, idx_hi)
                        : slice_lo;

                    if (!slice_lo) continue;

                    for (size_t i = 0; i < Nx; ++i) {
                        double x = x_nodes[i];

                        // Interpolate American price from cache
                        double am_lo = slice_lo->eval(x) * cfg.K_ref;
                        double am = am_lo;
                        if (alpha > 1e-12 && slice_hi && slice_hi != slice_lo) {
                            double am_hi = slice_hi->eval(x) * cfg.K_ref;
                            am = (1.0 - alpha) * am_lo + alpha * am_hi;
                        }

                        // European price
                        double spot = std::exp(x) * cfg.K_ref;
                        auto eu = EuropeanOptionSolver(
                            OptionSpec{.spot = spot, .strike = cfg.K_ref,
                                       .maturity = tau, .rate = rate,
                                       .dividend_yield = cfg.dividend_yield,
                                       .option_type = cfg.option_type},
                            sigma).solve();
                        if (!eu) continue;

                        double eep_raw = am - eu->value();

                        // Softplus floor (same as incremental builder)
                        constexpr double kSharpness = 100.0;
                        double eep;
                        if (kSharpness * eep_raw > 500.0) {
                            eep = eep_raw;
                        } else {
                            double softplus = std::log1p(std::exp(
                                kSharpness * eep_raw)) / kSharpness;
                            double bias = std::log(2.0) / kSharpness;
                            eep = cfg.use_hard_max
                                ? std::max(0.0, softplus - bias)
                                : (softplus - bias);
                        }

                        tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] = eep;
                    }
                }
            }
        }

        ChebyshevTucker4DDomain dom{
            .bounds = {{{spec.x_lo, spec.x_hi}, {spec.tau_lo, spec.tau_hi},
                        {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon,
            .use_tucker = cfg.use_tucker};

        elements.push_back(
            ChebyshevTucker4D::build_from_values(tensor, dom, tcfg));
    }

    auto t1 = std::chrono::steady_clock::now();

    return {
        .elements = std::move(elements),
        .specs = std::move(specs),
        .x_overlaps = std::move(x_overlaps),
        .tau_blend_lo = cfg.tau_blend_lo,
        .tau_blend_hi = cfg.tau_blend_hi,
        .sigma_lo = sigma_lo, .sigma_hi = sigma_hi,
        .rate_lo = rate_lo, .rate_hi = rate_hi,
        .total_pde_solves = cache.total_pde_solves(),
        .build_seconds = std::chrono::duration<double>(t1 - t0).count(),
        .short_boundary = short_boundary,
        .long_boundary = long_boundary,
    };
}

}  // namespace mango
