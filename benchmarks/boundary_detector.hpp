// SPDX-License-Identifier: MIT
//
// Exercise boundary detection from cached PDE slices.
// Scans EEP = Am - Eu to locate the zero-crossing x* per τ-band.
#pragma once

#include "pde_slice_cache.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/option_spec.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mango {

struct BoundaryResult {
    double x_star;     // median boundary location
    double delta;      // half-width for boundary element
    size_t n_valid;    // number of valid triples
    size_t n_sampled;  // total triples sampled
};

struct BoundaryDetectorConfig {
    double x_min = -0.50;            // scan domain
    double x_max = 0.40;
    size_t n_scan_points = 200;      // x-points for EEP scan
    double eps_scale = 1e-6;         // ε = max(eps_scale * K_ref, 1e-8)
    double delta_min = 0.10;         // minimum half-width
    double delta_margin = 0.05;      // added to (p90 - p10) / 2
    double valid_fraction = 0.30;    // min valid triples before fallback
};

/// Detect exercise boundary for a τ-band.
///
/// @param cache        Populated PDE slice cache
/// @param tau_nodes    Physical τ values matching cache tau_idx 0..N-1
/// @param sigma_nodes  σ nodes in the cache
/// @param rate_nodes   Rate nodes in the cache
/// @param tau_idx_lo   First tau_idx in this band (inclusive)
/// @param tau_idx_hi   Last tau_idx in this band (inclusive)
/// @param K_ref        Reference strike
/// @param option_type  PUT or CALL
/// @param cfg          Detection parameters
/// @param dividend_yield Continuous dividend yield
/// @param fallback_x_star Optional fallback from other τ-band (NaN if none)
inline BoundaryResult detect_exercise_boundary(
    const PDESliceCache& cache,
    std::span<const double> tau_nodes,
    std::span<const double> sigma_nodes,
    std::span<const double> rate_nodes,
    size_t tau_idx_lo, size_t tau_idx_hi,
    double K_ref, OptionType option_type,
    const BoundaryDetectorConfig& cfg = {},
    double dividend_yield = 0.0,
    double fallback_x_star = std::numeric_limits<double>::quiet_NaN())
{
    double eps = std::max(cfg.eps_scale * K_ref, 1e-8);

    std::vector<double> all_x_stars;
    size_t n_sampled = 0;

    for (size_t si = 0; si < sigma_nodes.size(); ++si) {
        double sigma = sigma_nodes[si];
        for (size_t ri = 0; ri < rate_nodes.size(); ++ri) {
            double rate = rate_nodes[ri];
            for (size_t ti = tau_idx_lo; ti <= tau_idx_hi; ++ti) {
                auto* spline = cache.get_slice(sigma, rate, ti);
                if (!spline) continue;
                n_sampled++;

                double tau = tau_nodes[ti];

                // 1. Evaluate EEP at scan points
                std::vector<double> eep(cfg.n_scan_points);
                std::vector<double> xs(cfg.n_scan_points);
                for (size_t k = 0; k < cfg.n_scan_points; ++k) {
                    double x = cfg.x_min + (cfg.x_max - cfg.x_min) * k /
                               (cfg.n_scan_points - 1);
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

                // 2. Find the steepest EEP gradient (max |dEEP/dx|) as the
                //    boundary location. This is robust even when EEP > 0
                //    across the entire domain (long-dated puts).
                //    For puts, the boundary is where EEP transitions from
                //    large (ITM) to small (OTM), producing a large |gradient|.
                double max_grad = 0.0;
                int max_grad_idx = -1;
                for (size_t k = 1; k < cfg.n_scan_points; ++k) {
                    double dx = xs[k] - xs[k - 1];
                    if (dx < 1e-15) continue;
                    double grad = std::abs(eep[k] - eep[k - 1]) / dx;
                    if (grad > max_grad) {
                        max_grad = grad;
                        max_grad_idx = static_cast<int>(k);
                    }
                }

                // Skip if no meaningful gradient (flat EEP, or EEP ≈ 0 everywhere)
                double eep_range = *std::max_element(eep.begin(), eep.end()) -
                                   *std::min_element(eep.begin(), eep.end());
                if (max_grad_idx < 0 || eep_range < eps) continue;

                // The boundary is between xs[max_grad_idx-1] and xs[max_grad_idx]
                double x_star = (xs[max_grad_idx - 1] + xs[max_grad_idx]) / 2.0;
                all_x_stars.push_back(x_star);
            }
        }
    }

    // 5. Compute center and half-width
    if (n_sampled == 0 || all_x_stars.size() < cfg.valid_fraction * n_sampled) {
        double x_star = std::isfinite(fallback_x_star) ? fallback_x_star : 0.0;
        return {x_star, cfg.delta_min, all_x_stars.size(), n_sampled};
    }

    std::sort(all_x_stars.begin(), all_x_stars.end());
    size_t n = all_x_stars.size();
    double median = all_x_stars[n / 2];

    size_t p10_idx = n / 10;
    size_t p90_idx = std::min(n * 9 / 10, n - 1);
    double spread = all_x_stars[p90_idx] - all_x_stars[p10_idx];
    double delta = std::max(cfg.delta_min, spread / 2.0 + cfg.delta_margin);

    return {median, delta, n, n_sampled};
}

}  // namespace mango
