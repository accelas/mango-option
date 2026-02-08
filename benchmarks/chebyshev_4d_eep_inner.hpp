// SPDX-License-Identifier: MIT
//
// Benchmark-local adapter and builder for 4D Chebyshev-Tucker EEP surfaces.
// Uses standard parameterization (ln(S/K), tau, sigma, rate) instead of
// the 3D dimensionless (x, tau', ln_kappa) coordinates.
// Benchmark experiment only.
#pragma once

#include "mango/option/table/dimensionless/chebyshev_tucker_4d.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/price_query.hpp"
#include "mango/option/european_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace mango {

// ============================================================================
// Chebyshev4DEEPInner: price/vega adapter for ChebyshevTucker4D EEP surface
// ============================================================================

class Chebyshev4DEEPInner {
public:
    Chebyshev4DEEPInner(ChebyshevTucker4D interp, OptionType type,
                        double K_ref, double dividend_yield)
        : interp_(std::move(interp)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double eep = interp_.eval({x, q.tau, q.sigma, q.rate});

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * (q.strike / K_ref_) + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        std::array<double, 4> coords = {x, q.tau, q.sigma, q.rate};

        // sigma is axis 2 — partial(2, ...) gives dEEP/dsigma directly.
        // No chain rule needed (unlike 3D dimensionless where sigma maps
        // to both tau' and ln_kappa).
        double eep_vega = (q.strike / K_ref_) * interp_.partial(2, coords);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep_vega + eu.vega();
    }

    [[nodiscard]] const ChebyshevTucker4D& interp() const { return interp_; }

private:
    ChebyshevTucker4D interp_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

// ============================================================================
// Config and result types
// ============================================================================

struct Chebyshev4DEEPConfig {
    size_t num_x = 10;
    size_t num_tau = 10;
    size_t num_sigma = 15;
    size_t num_rate = 6;
    double epsilon = 1e-8;

    double x_min = -0.50;     // ln(0.60)
    double x_max = 0.40;      // ln(1.50)
    double tau_min = 0.019;   // ~7 days
    double tau_max = 2.0;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.0;
};

struct Chebyshev4DEEPResult {
    ChebyshevTucker4D interp;
    int n_pde_solves;
    double build_seconds;
};

// ============================================================================
// Builder: batch-PDE Chebyshev 4D EEP surface
// ============================================================================

inline Chebyshev4DEEPResult build_chebyshev_4d_eep(
    const Chebyshev4DEEPConfig& cfg,
    double K_ref,
    OptionType option_type)
{
    // ---- 1. Compute per-axis headroom: 3 * domain_width / (n-1) per side ----
    auto headroom = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hx     = headroom(cfg.x_min,     cfg.x_max,     cfg.num_x);
    double htau   = headroom(cfg.tau_min,    cfg.tau_max,   cfg.num_tau);
    double hsigma = headroom(cfg.sigma_min,  cfg.sigma_max, cfg.num_sigma);
    double hrate  = headroom(cfg.rate_min,   cfg.rate_max,  cfg.num_rate);

    // ---- 2. Extended bounds with clamping ----
    double x_lo     = cfg.x_min - hx;
    double x_hi     = cfg.x_max + hx;
    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    ChebyshevTucker4DDomain dom{
        .bounds = {{{x_lo, x_hi}, {tau_lo, tau_hi},
                    {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
    ChebyshevTucker4DConfig tcfg{
        .num_pts = {cfg.num_x, cfg.num_tau, cfg.num_sigma, cfg.num_rate},
        .epsilon = cfg.epsilon};

    // ---- 3. Generate Chebyshev nodes per axis on extended domain ----
    auto x_nodes     = chebyshev_nodes(cfg.num_x,     x_lo,     x_hi);
    auto tau_nodes   = chebyshev_nodes(cfg.num_tau,    tau_lo,   tau_hi);
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma,  sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate,   rate_lo,  rate_hi);

    auto t0 = std::chrono::steady_clock::now();

    // ---- 4. Batch-solve: N_sigma x N_rate PDEs ----
    // For each (sigma, rate) pair, solve one PDE with spot=strike=K_ref,
    // maturity slightly beyond tau_max, with snapshots at tau Chebyshev nodes.
    const double tau_solve = tau_nodes.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(cfg.num_sigma * cfg.num_rate);
    for (size_t s = 0; s < cfg.num_sigma; ++s) {
        for (size_t r = 0; r < cfg.num_rate; ++r) {
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[r],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[s]);
        }
    }

    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{tau_nodes});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // ---- 5-7. Extract EEP tensor: resample PDE at Chebyshev x nodes ----
    // Tensor layout: row-major [x, tau, sigma, rate]
    const size_t Nx = cfg.num_x;
    const size_t Nt = cfg.num_tau;
    const size_t Ns = cfg.num_sigma;
    const size_t Nr = cfg.num_rate;
    std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);

    for (size_t s = 0; s < Ns; ++s) {
        for (size_t r = 0; r < Nr; ++r) {
            size_t batch_idx = s * Nr + r;
            if (!batch_result.results[batch_idx].has_value()) continue;

            const auto& result = batch_result.results[batch_idx].value();
            auto x_grid = result.grid()->x();
            double sigma = sigma_nodes[s];
            double rate  = rate_nodes[r];

            for (size_t j = 0; j < Nt; ++j) {
                auto spatial = result.at_time(j);
                CubicSpline<double> spline;
                if (spline.build(x_grid, spatial).has_value()) continue;

                double tau = tau_nodes[j];

                for (size_t i = 0; i < Nx; ++i) {
                    double am = spline.eval(x_nodes[i]);

                    // European price for this (x, tau, sigma, rate) point
                    double spot_local = std::exp(x_nodes[i]) * K_ref;
                    auto eu = EuropeanOptionSolver(
                        OptionSpec{.spot = spot_local, .strike = K_ref,
                                   .maturity = tau, .rate = rate,
                                   .dividend_yield = cfg.dividend_yield,
                                   .option_type = option_type},
                        sigma).solve().value();

                    double eep_raw = am - eu.value();

                    // Debiased softplus floor (from eep_transform.cpp:41-49)
                    constexpr double kSharpness = 100.0;
                    double eep;
                    if (kSharpness * eep_raw > 500.0) {
                        eep = eep_raw;  // overflow protection
                    } else {
                        double softplus = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
                        double bias = std::log(2.0) / kSharpness;
                        eep = std::max(0.0, softplus - bias);
                    }

                    tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] = eep;
                }
            }
        }
    }

    // ---- 8. Build ChebyshevTucker4D from values ----
    auto interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp),
            static_cast<int>(cfg.num_sigma * cfg.num_rate),
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
