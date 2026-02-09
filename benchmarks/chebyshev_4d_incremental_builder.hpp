// SPDX-License-Identifier: MIT
//
// Incremental Chebyshev 4D EEP builder using nested CC levels + PDE slice cache.
// Phase A of the true adaptive Chebyshev design.
#pragma once

#include "pde_slice_cache.hpp"
#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

namespace mango {

struct IncrementalBuildConfig {
    size_t num_x = 40;
    size_t num_tau = 15;
    size_t sigma_level = 3;   // CC level -> 2^l + 1 nodes
    size_t rate_level = 2;    // CC level -> 2^l + 1 nodes

    double epsilon = 1e-8;
    bool use_tucker = false;

    // Fixed physical domain bounds (no headroom coupling during refinement)
    double x_min = -0.50, x_max = 0.40;
    double tau_min = 0.019, tau_max = 2.0;
    double sigma_min = 0.05, sigma_max = 0.50;
    double rate_min = 0.01, rate_max = 0.10;

    double dividend_yield = 0.0;
    bool use_hard_max = true;

    // Fixed reference counts for sigma/rate headroom computation.
    // These MUST NOT change across refinement levels so the extended domain
    // stays fixed, enabling cross-level PDE cache reuse.
    size_t sigma_headroom_ref = 15;
    size_t rate_headroom_ref = 9;

    // PDE solver accuracy (default: Ultra)
    GridAccuracyParams grid_accuracy = make_grid_accuracy(GridAccuracyProfile::Ultra);
};

struct IncrementalBuildResult {
    ChebyshevTucker4D interp;
    size_t new_pde_solves;
    double build_seconds;
};

inline IncrementalBuildResult build_chebyshev_4d_eep_incremental(
    const IncrementalBuildConfig& cfg,
    PDESliceCache& cache,
    double K_ref,
    OptionType option_type)
{
    auto t0 = std::chrono::steady_clock::now();

    // 1. Compute extended domains with headroom.
    //    sigma/rate headroom uses FIXED reference counts so the extended domain
    //    stays invariant across CC levels, enabling cross-level cache reuse.
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    size_t n_sigma = (1u << cfg.sigma_level) + 1;
    size_t n_rate = (1u << cfg.rate_level) + 1;

    double hx     = headroom_fn(cfg.x_min, cfg.x_max, cfg.num_x);
    double htau   = headroom_fn(cfg.tau_min, cfg.tau_max, cfg.num_tau);
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.sigma_headroom_ref);
    double hrate  = headroom_fn(cfg.rate_min, cfg.rate_max, cfg.rate_headroom_ref);

    double x_lo     = cfg.x_min - hx;
    double x_hi     = cfg.x_max + hx;
    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    // 2. Generate nodes
    auto x_nodes     = chebyshev_nodes(cfg.num_x, x_lo, x_hi);
    auto tau_nodes   = chebyshev_nodes(cfg.num_tau, tau_lo, tau_hi);
    auto sigma_nodes = cc_level_nodes(cfg.sigma_level, sigma_lo, sigma_hi);
    auto rate_nodes  = cc_level_nodes(cfg.rate_level, rate_lo, rate_hi);

    // 3. Find missing (sigma, rate) pairs using physical node values
    auto missing = cache.missing_pairs(
        std::span<const double>{sigma_nodes},
        std::span<const double>{rate_nodes});

    // 4. Batch-solve only missing pairs
    size_t new_solves = 0;
    if (!missing.empty()) {
        const double tau_solve = tau_nodes.back() * 1.01;

        std::vector<PricingParams> batch;
        batch.reserve(missing.size());
        for (auto [si, ri] : missing) {
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[ri],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[si]);
        }

        BatchAmericanOptionSolver solver;
        solver.set_grid_accuracy(cfg.grid_accuracy);
        solver.set_snapshot_times(std::span<const double>{tau_nodes});
        auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);
        new_solves = missing.size();

        // Store results in cache using physical node values
        for (size_t bi = 0; bi < missing.size(); ++bi) {
            auto [si, ri] = missing[bi];
            if (!batch_result.results[bi].has_value()) continue;

            const auto& result = batch_result.results[bi].value();
            auto x_grid = result.grid()->x();

            for (size_t j = 0; j < cfg.num_tau; ++j) {
                auto spatial = result.at_time(j);
                cache.store_slice(sigma_nodes[si], rate_nodes[ri], j,
                                  x_grid, spatial);
            }
        }
        cache.record_pde_solves(new_solves);
    }

    // 5. Extract tensor from cache
    const size_t Nx = cfg.num_x;
    const size_t Nt = cfg.num_tau;
    const size_t Ns = n_sigma;
    const size_t Nr = n_rate;
    std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

    for (size_t s = 0; s < Ns; ++s) {
        double sigma = sigma_nodes[s];
        for (size_t r = 0; r < Nr; ++r) {
            double rate = rate_nodes[r];
            for (size_t j = 0; j < Nt; ++j) {
                auto* spline = cache.get_slice(sigma, rate, j);
                if (!spline) continue;

                double tau = tau_nodes[j];
                for (size_t i = 0; i < Nx; ++i) {
                    double am = spline->eval(x_nodes[i]) * K_ref;

                    double spot_local = std::exp(x_nodes[i]) * K_ref;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot_local, .strike = K_ref,
                                   .maturity = tau, .rate = rate,
                                   .dividend_yield = cfg.dividend_yield,
                                   .option_type = option_type},
                        sigma).solve().value();

                    double eep_raw = am - eu.value();

                    constexpr double kSharpness = 100.0;
                    double eep;
                    if (kSharpness * eep_raw > 500.0) {
                        eep = eep_raw;
                    } else {
                        double softplus =
                            std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
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

    // 6. Build ChebyshevTucker4D
    ChebyshevTucker4DDomain dom{
        .bounds = {{{x_lo, x_hi}, {tau_lo, tau_hi},
                    {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
    ChebyshevTucker4DConfig tcfg{
        .num_pts = {Nx, Nt, Ns, Nr},
        .epsilon = cfg.epsilon,
        .use_tucker = cfg.use_tucker};

    auto interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), new_solves,
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
