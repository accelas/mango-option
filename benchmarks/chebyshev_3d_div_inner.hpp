// SPDX-License-Identifier: MIT
//
// Per-maturity piecewise Chebyshev 3D for discrete dividends:
//   multi-K_ref × per-maturity × 3 piecewise x-elements × ChebyshevTucker3D.
// Stores V/K_ref directly (no EEP) — same as B-spline segmented surface.
// Piecewise x-splitting handles the exercise boundary kink that causes
// Gibbs oscillations in single-element Chebyshev.
// Benchmark experiment only.
#pragma once

#include "bump_blend.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
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
#include <vector>

namespace mango {

// ============================================================================
// Config
// ============================================================================

struct Cheb3DDivConfig {
    size_t num_x_coarse = 20;   // ITM/OTM elements
    size_t num_x_dense = 30;    // boundary element
    size_t num_sigma = 9;
    size_t num_rate = 5;
    double epsilon = 1e-8;

    double x_min = -0.50, x_max = 0.40;
    double sigma_min = 0.05, sigma_max = 0.50;
    double rate_min = 0.01, rate_max = 0.10;

    double dividend_yield = 0.02;
    OptionType option_type = OptionType::PUT;

    std::vector<double> K_refs;

    // Boundary detection
    size_t n_scan_points = 200;
    double delta_min = 0.10;
    double delta_margin = 0.05;

    // Headroom reference counts
    size_t sigma_headroom_ref = 15;
    size_t rate_headroom_ref = 9;
};

// ============================================================================
// Data structures
// ============================================================================

struct Cheb3DElement {
    ChebyshevTucker3D interp;  // axes: (x, sigma, rate)
    double x_lo, x_hi;
};

struct Cheb3DOverlap {
    size_t left_idx, right_idx;
    double x_lo, x_hi;
};

struct Cheb3DBoundary {
    double x_star;
    double delta;
    size_t n_valid, n_sampled;
};

struct Cheb3DKrefEntry {
    double K_ref;
    std::vector<Cheb3DElement> elements;
    std::vector<Cheb3DOverlap> overlaps;
    Cheb3DBoundary boundary;
};

struct Cheb3DMaturityEntry {
    double maturity;
    std::vector<Cheb3DKrefEntry> krefs;
};

struct Cheb3DDivBuildResult {
    std::vector<Cheb3DMaturityEntry> maturities;
    int n_pde_solves;
    double build_seconds;
};

// ============================================================================
// Evaluator
// ============================================================================

class Cheb3DDivEvaluator {
public:
    Cheb3DDivEvaluator() = default;

    Cheb3DDivEvaluator(std::vector<Cheb3DMaturityEntry> maturities,
                       OptionType type)
        : maturities_(std::move(maturities)), type_(type) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        const auto& mat = find_maturity(q.tau);
        auto [lo, hi, w] = bracket(mat.krefs, q.strike);
        double p_lo = eval_kref(mat.krefs[lo], q);
        if (lo == hi) return p_lo;
        double p_hi = eval_kref(mat.krefs[hi], q);
        return (1.0 - w) * p_lo + w * p_hi;
    }

    [[nodiscard]] size_t num_maturities() const { return maturities_.size(); }
    [[nodiscard]] size_t num_krefs() const {
        return maturities_.empty() ? 0 : maturities_[0].krefs.size();
    }
    [[nodiscard]] size_t total_elements() const {
        size_t count = 0;
        for (const auto& m : maturities_)
            for (const auto& k : m.krefs)
                count += k.elements.size();
        return count;
    }

private:
    struct BracketResult { size_t lo, hi; double w; };

    [[nodiscard]] double eval_kref(const Cheb3DKrefEntry& entry,
                                   const PriceQuery& q) const {
        double x = std::log(q.spot / entry.K_ref);
        double v_norm = eval_piecewise(entry, x, q.sigma, q.rate);
        return v_norm * q.strike;
    }

    [[nodiscard]] double eval_piecewise(const Cheb3DKrefEntry& entry,
                                        double x, double sigma,
                                        double rate) const {
        // Check overlap zones for blended eval
        for (const auto& oz : entry.overlaps) {
            if (x >= oz.x_lo && x <= oz.x_hi) {
                double w_right = overlap_weight_right(x, oz.x_lo, oz.x_hi);
                double v_left = entry.elements[oz.left_idx].interp.eval(
                    {x, sigma, rate});
                double v_right = entry.elements[oz.right_idx].interp.eval(
                    {x, sigma, rate});
                return (1.0 - w_right) * v_left + w_right * v_right;
            }
        }
        // Single element
        for (const auto& elem : entry.elements) {
            if (x >= elem.x_lo && x <= elem.x_hi) {
                return elem.interp.eval({x, sigma, rate});
            }
        }
        // Fallback: closest element
        size_t closest = 0;
        double best_dist = 1e99;
        for (size_t i = 0; i < entry.elements.size(); ++i) {
            double mid = (entry.elements[i].x_lo + entry.elements[i].x_hi) / 2.0;
            if (std::abs(x - mid) < best_dist) {
                best_dist = std::abs(x - mid);
                closest = i;
            }
        }
        return entry.elements[closest].interp.eval({x, sigma, rate});
    }

    [[nodiscard]] const Cheb3DMaturityEntry& find_maturity(double tau) const {
        size_t best = 0;
        double best_dist = std::abs(maturities_[0].maturity - tau);
        for (size_t i = 1; i < maturities_.size(); ++i) {
            double d = std::abs(maturities_[i].maturity - tau);
            if (d < best_dist) { best = i; best_dist = d; }
        }
        return maturities_[best];
    }

    [[nodiscard]] static BracketResult bracket(
        const std::vector<Cheb3DKrefEntry>& krefs, double K) {
        const size_t n = krefs.size();
        if (n == 1 || K <= krefs.front().K_ref) return {0, 0, 0.0};
        if (K >= krefs.back().K_ref) return {n - 1, n - 1, 0.0};
        size_t hi = 1;
        while (hi < n && krefs[hi].K_ref < K) ++hi;
        size_t lo = hi - 1;
        double t = (K - krefs[lo].K_ref) /
                   (krefs[hi].K_ref - krefs[lo].K_ref);
        return {lo, hi, t};
    }

    std::vector<Cheb3DMaturityEntry> maturities_;
    OptionType type_;
};

// ============================================================================
// Boundary detection (adapted from piecewise_div_builder.hpp)
// ============================================================================

namespace cheb3d_detail {

inline Cheb3DBoundary detect_boundary(
    const std::vector<CubicSpline<double>>& splines,
    const std::vector<double>& sigma_nodes,
    const std::vector<double>& rate_nodes,
    double K_ref, double maturity,
    OptionType option_type, double dividend_yield,
    double x_min, double x_max,
    size_t n_scan_points, double delta_min, double delta_margin)
{
    std::vector<double> all_x_stars;
    size_t n_sampled = 0;

    for (size_t si = 0; si < sigma_nodes.size(); ++si) {
        double sigma = sigma_nodes[si];
        for (size_t ri = 0; ri < rate_nodes.size(); ++ri) {
            double rate = rate_nodes[ri];
            size_t idx = si * rate_nodes.size() + ri;
            n_sampled++;

            std::vector<double> eep(n_scan_points);
            std::vector<double> xs(n_scan_points);

            for (size_t k = 0; k < n_scan_points; ++k) {
                double x = x_min + (x_max - x_min) * k /
                           (n_scan_points - 1);
                xs[k] = x;

                double am = splines[idx].eval(x) * K_ref;
                double spot = std::exp(x) * K_ref;
                auto eu = EuropeanOptionSolver(
                    OptionSpec{.spot = spot, .strike = K_ref,
                               .maturity = maturity, .rate = rate,
                               .dividend_yield = dividend_yield,
                               .option_type = option_type},
                    sigma).solve();
                if (!eu) continue;
                eep[k] = am - eu->value();
            }

            double max_grad = 0.0;
            int max_grad_idx = -1;
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

}  // namespace cheb3d_detail

// ============================================================================
// Builder
// ============================================================================

inline Cheb3DDivBuildResult build_cheb3d_div(
    const Cheb3DDivConfig& cfg,
    double default_K_ref,
    const std::vector<double>& maturities,
    const std::vector<Dividend>& all_dividends)
{
    auto t0 = std::chrono::steady_clock::now();

    std::vector<double> K_refs = cfg.K_refs;
    if (K_refs.empty()) K_refs.push_back(default_K_ref);
    std::sort(K_refs.begin(), K_refs.end());

    // Shared axis setup with headroom
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) /
               static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max,
                                cfg.sigma_headroom_ref);
    double hrate  = headroom_fn(cfg.rate_min, cfg.rate_max,
                                cfg.rate_headroom_ref);

    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate, rate_lo, rate_hi);

    const size_t Ns = cfg.num_sigma;
    const size_t Nr = cfg.num_rate;

    int total_pde = 0;
    std::vector<Cheb3DMaturityEntry> mat_entries;

    for (double maturity : maturities) {
        // Filter dividends to those before this maturity
        std::vector<Dividend> divs;
        for (const auto& d : all_dividends) {
            if (d.calendar_time < maturity)
                divs.push_back(d);
        }

        std::vector<Cheb3DKrefEntry> kref_entries;

        for (double K_ref : K_refs) {
            // Batch PDE: Ns × Nr solves at this (K_ref, maturity)
            double tau_solve = maturity * 1.01 + 1e-4;
            std::array<double, 1> snap_times = {maturity};

            std::vector<PricingParams> batch;
            batch.reserve(Ns * Nr);
            for (size_t s = 0; s < Ns; ++s) {
                for (size_t r = 0; r < Nr; ++r) {
                    PricingParams p(
                        OptionSpec{
                            .spot = K_ref, .strike = K_ref,
                            .maturity = tau_solve,
                            .rate = rate_nodes[r],
                            .dividend_yield = cfg.dividend_yield,
                            .option_type = cfg.option_type},
                        sigma_nodes[s]);
                    p.discrete_dividends = divs;
                    batch.push_back(std::move(p));
                }
            }

            BatchAmericanOptionSolver solver;
            solver.set_grid_accuracy(
                make_grid_accuracy(GridAccuracyProfile::Ultra));
            solver.set_snapshot_times(
                std::span<const double>(snap_times));
            auto batch_result =
                solver.solve_batch(batch, /*use_shared_grid=*/true);
            total_pde += static_cast<int>(Ns * Nr);

            // Build splines from PDE solutions (one per sigma×rate)
            std::vector<CubicSpline<double>> splines(Ns * Nr);
            for (size_t s = 0; s < Ns; ++s) {
                for (size_t r = 0; r < Nr; ++r) {
                    size_t idx = s * Nr + r;
                    if (!batch_result.results[idx].has_value()) continue;

                    const auto& result =
                        batch_result.results[idx].value();
                    auto x_grid = result.grid()->x();
                    auto spatial = result.at_time(0);
                    splines[idx].build(x_grid, spatial);
                }
            }

            // Detect exercise boundary
            auto boundary = cheb3d_detail::detect_boundary(
                splines, sigma_nodes, rate_nodes,
                K_ref, maturity, cfg.option_type, cfg.dividend_yield,
                cfg.x_min, cfg.x_max,
                cfg.n_scan_points, cfg.delta_min, cfg.delta_margin);

            // Compute x-element boundaries
            double x_star = boundary.x_star;
            double delta = boundary.delta;
            double h_boundary = 2.0 * delta / (cfg.num_x_dense - 1);
            double w_overlap = 2.0 * h_boundary;

            double itm_hi = std::max(x_star - delta, cfg.x_min + 0.05);
            double otm_lo = std::min(x_star + delta, cfg.x_max - 0.05);

            if (otm_lo <= itm_hi + 0.02) {
                double mid = (itm_hi + otm_lo) / 2.0;
                itm_hi = mid - 0.05;
                otm_lo = mid + 0.05;
            }

            struct ElemSpec { double x_lo, x_hi; size_t num_x; };

            double hx_itm = headroom_fn(cfg.x_min, itm_hi, cfg.num_x_coarse);
            double hx_otm = headroom_fn(otm_lo, cfg.x_max, cfg.num_x_coarse);

            ElemSpec specs[3] = {
                {cfg.x_min - hx_itm, itm_hi + w_overlap, cfg.num_x_coarse},
                {itm_hi - w_overlap, otm_lo + w_overlap, cfg.num_x_dense},
                {otm_lo - w_overlap, cfg.x_max + hx_otm, cfg.num_x_coarse},
            };

            std::vector<Cheb3DOverlap> overlaps = {
                {0, 1, itm_hi - w_overlap, itm_hi + w_overlap},
                {1, 2, otm_lo - w_overlap, otm_lo + w_overlap},
            };

            // Build 3 elements
            std::vector<Cheb3DElement> elements;
            elements.reserve(3);

            for (int ei = 0; ei < 3; ++ei) {
                const auto& spec = specs[ei];
                const size_t Nx = spec.num_x;
                auto x_nodes = chebyshev_nodes(Nx, spec.x_lo, spec.x_hi);

                std::vector<double> tensor(Nx * Ns * Nr, 0.0);

                for (size_t s = 0; s < Ns; ++s) {
                    for (size_t r = 0; r < Nr; ++r) {
                        size_t idx = s * Nr + r;
                        for (size_t i = 0; i < Nx; ++i) {
                            tensor[i * Ns * Nr + s * Nr + r] =
                                splines[idx].eval(x_nodes[i]);
                        }
                    }
                }

                ChebyshevTuckerDomain dom{{{
                    {spec.x_lo, spec.x_hi},
                    {sigma_lo, sigma_hi},
                    {rate_lo, rate_hi}
                }}};
                ChebyshevTuckerConfig tcfg{
                    .num_pts = {Nx, Ns, Nr},
                    .epsilon = cfg.epsilon
                };

                elements.push_back({
                    .interp = ChebyshevTucker3D::build_from_values(
                        std::span<const double>(tensor), dom, tcfg),
                    .x_lo = spec.x_lo,
                    .x_hi = spec.x_hi,
                });
            }

            kref_entries.push_back({
                .K_ref = K_ref,
                .elements = std::move(elements),
                .overlaps = std::move(overlaps),
                .boundary = boundary,
            });
        }

        size_t n_krefs = kref_entries.size();
        mat_entries.push_back({maturity, std::move(kref_entries)});
        std::fprintf(stderr, "  T=%.4f: %zu K_refs, x*=%.3f delta=%.3f\n",
                     maturity, n_krefs,
                     mat_entries.back().krefs.empty() ? 0.0 :
                         mat_entries.back().krefs[0].boundary.x_star,
                     mat_entries.back().krefs.empty() ? 0.0 :
                         mat_entries.back().krefs[0].boundary.delta);
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    return {std::move(mat_entries), total_pde, secs};
}

}  // namespace mango
