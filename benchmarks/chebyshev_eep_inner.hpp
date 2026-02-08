// SPDX-License-Identifier: MIT
//
// Benchmark-local adapter and builder for Chebyshev-Tucker EEP surfaces.
// Mirrors DimensionlessEEPInner but wraps ChebyshevTucker3D.
#pragma once

#include "mango/option/table/dimensionless/chebyshev_tucker.hpp"
#include "mango/option/table/dimensionless/chebyshev_nodes.hpp"
#include "mango/option/table/dimensionless/dimensionless_european.hpp"
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
// ChebyshevEEPInner: price/vega adapter for ChebyshevTucker3D EEP surface
// ============================================================================

class ChebyshevEEPInner {
public:
    ChebyshevEEPInner(ChebyshevTucker3D interp, OptionType type,
                       double K_ref, double dividend_yield)
        : interp_(std::move(interp)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double tau_prime = q.sigma * q.sigma * q.tau / 2.0;
        double ln_kappa = std::log(2.0 * q.rate / (q.sigma * q.sigma));

        double eep = interp_.eval({x, tau_prime, ln_kappa});

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * q.strike + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        double tau_prime = q.sigma * q.sigma * q.tau / 2.0;
        double ln_kappa = std::log(2.0 * q.rate / (q.sigma * q.sigma));
        std::array<double, 3> coords = {x, tau_prime, ln_kappa};

        double dEEP_dtau_prime = interp_.partial(1, coords);
        double dEEP_dln_kappa  = interp_.partial(2, coords);
        double eep_vega = q.sigma * q.tau * dEEP_dtau_prime
                        - (2.0 / q.sigma) * dEEP_dln_kappa;

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return q.strike * eep_vega + eu.vega();
    }

    [[nodiscard]] const ChebyshevTucker3D& interp() const { return interp_; }

private:
    ChebyshevTucker3D interp_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

// ============================================================================
// Builder: batch-PDE Chebyshev EEP surface (same cost as B-spline builder)
// ============================================================================

struct ChebyshevEEPConfig {
    size_t num_x = 25;
    size_t num_tp = 25;
    size_t num_lk = 25;
    double epsilon = 1e-8;

    // Wide domain covering σ ∈ [0.05, 0.80], r ∈ [0.005, 0.10],
    // τ ∈ [7d, 2y], S/K ∈ [0.65, 1.50]
    double x_min = -0.50;   // ln(0.60)
    double x_max = 0.40;    // ln(1.50)
    double tp_min = 0.001;  // σ=0.10, τ=7d → 0.00010; need margin for Newton
    double tp_max = 0.64;   // σ=0.80, τ=2y → 0.64
    double lk_min = -3.5;   // σ=0.80, r=0.005 → ln(2*0.005/0.64) = -4.16
    double lk_max = 4.0;    // σ=0.05, r=0.10 → ln(2*0.10/0.0025) = 4.38
};

struct ChebyshevEEPResult {
    ChebyshevTucker3D interp;
    int n_pde_solves;
    double build_seconds;
};

inline ChebyshevEEPResult build_chebyshev_eep(
    const ChebyshevEEPConfig& cfg,
    double K_ref,
    OptionType option_type)
{
    // Add headroom: 3 * domain_width / (n-1) per side (same as B-spline convention)
    auto headroom = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hx = headroom(cfg.x_min, cfg.x_max, cfg.num_x);
    double htp = headroom(cfg.tp_min, cfg.tp_max, cfg.num_tp);
    double hlk = headroom(cfg.lk_min, cfg.lk_max, cfg.num_lk);

    double x_lo = cfg.x_min - hx, x_hi = cfg.x_max + hx;
    double tp_lo = std::max(cfg.tp_min - htp, 1e-4), tp_hi = cfg.tp_max + htp;
    double lk_lo = cfg.lk_min - hlk, lk_hi = cfg.lk_max + hlk;

    ChebyshevTuckerDomain dom{
        .bounds = {{{x_lo, x_hi}, {tp_lo, tp_hi}, {lk_lo, lk_hi}}}};
    ChebyshevTuckerConfig tcfg{
        .num_pts = {cfg.num_x, cfg.num_tp, cfg.num_lk},
        .epsilon = cfg.epsilon};

    auto x_nodes = chebyshev_nodes(cfg.num_x, x_lo, x_hi);
    auto tp_nodes = chebyshev_nodes(cfg.num_tp, tp_lo, tp_hi);
    auto lk_nodes = chebyshev_nodes(cfg.num_lk, lk_lo, lk_hi);

    auto t0 = std::chrono::steady_clock::now();

    // One PDE per ln κ node (same cost pattern as dimensionless_builder)
    const double sigma_eff = std::sqrt(2.0);
    const double tp_max = tp_nodes.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(cfg.num_lk);
    for (size_t k = 0; k < cfg.num_lk; ++k) {
        double kappa = std::exp(lk_nodes[k]);
        batch.emplace_back(
            OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tp_max,
                       .rate = kappa, .dividend_yield = 0.0,
                       .option_type = option_type},
            sigma_eff);
    }

    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{tp_nodes});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // Extract EEP tensor: resample PDE at Chebyshev x nodes
    std::vector<double> tensor(cfg.num_x * cfg.num_tp * cfg.num_lk, 0.0);

    for (size_t k = 0; k < cfg.num_lk; ++k) {
        if (!batch_result.results[k].has_value()) continue;
        const auto& result = batch_result.results[k].value();
        auto x_grid = result.grid()->x();
        double kappa = std::exp(lk_nodes[k]);

        for (size_t j = 0; j < cfg.num_tp; ++j) {
            auto spatial = result.at_time(j);
            CubicSpline<double> spline;
            if (spline.build(x_grid, spatial).has_value()) continue;

            for (size_t i = 0; i < cfg.num_x; ++i) {
                double am = spline.eval(x_nodes[i]);
                double eu = dimensionless_european(
                    x_nodes[i], tp_nodes[j], kappa, option_type);
                tensor[i * cfg.num_tp * cfg.num_lk + j * cfg.num_lk + k] =
                    std::max(am - eu, 0.0);
            }
        }
    }

    auto interp = ChebyshevTucker3D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), static_cast<int>(cfg.num_lk),
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
