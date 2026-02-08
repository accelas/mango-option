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
// Coordinate transform parameters (stored with interpolant for query-time use)
// ============================================================================

struct Chebyshev4DTransforms {
    bool use_sinh_x = false;
    double sinh_alpha = 3.0;
    double x_mid = 0.0;
    double x_half = 1.0;

    bool use_sqrt_tau = false;

    bool use_log_eep = false;
    double log_eep_eps = 1e-10;
};

// ============================================================================
// Chebyshev4DEEPInner: price/vega adapter for ChebyshevTucker4D EEP surface
// ============================================================================

class Chebyshev4DEEPInner {
public:
    Chebyshev4DEEPInner(ChebyshevTucker4D interp, OptionType type,
                        double K_ref, double dividend_yield,
                        Chebyshev4DTransforms transforms = {})
        : interp_(std::move(interp)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield),
          transforms_(transforms) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        auto [c0, c1] = map_to_computational(x, q.tau);

        double raw = interp_.eval({c0, c1, q.sigma, q.rate});

        double eep = transforms_.use_log_eep
            ? std::max(0.0, std::exp(raw) - transforms_.log_eep_eps)
            : raw;

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * (q.strike / K_ref_) + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        auto [c0, c1] = map_to_computational(x, q.tau);
        std::array<double, 4> coords = {c0, c1, q.sigma, q.rate};

        // sigma is axis 2 — partial(2, ...) gives dEEP/dsigma directly.
        double eep_vega = (q.strike / K_ref_) * interp_.partial(2, coords);

        // With log_eep, partial gives d(log(eep+eps))/dsigma; chain rule needed.
        if (transforms_.use_log_eep) {
            double raw = interp_.eval(coords);
            eep_vega *= std::exp(raw);  // (eep+eps) * d(log)/dsigma
        }

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep_vega + eu.vega();
    }

    [[nodiscard]] const ChebyshevTucker4D& interp() const { return interp_; }

private:
    [[nodiscard]] std::pair<double, double>
    map_to_computational(double x, double tau) const {
        double c0 = x;
        if (transforms_.use_sinh_x) {
            c0 = std::asinh(std::sinh(transforms_.sinh_alpha) *
                 (x - transforms_.x_mid) / transforms_.x_half) /
                 transforms_.sinh_alpha;
        }
        double c1 = transforms_.use_sqrt_tau ? std::sqrt(tau) : tau;
        return {c0, c1};
    }

    ChebyshevTucker4D interp_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
    Chebyshev4DTransforms transforms_;
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

    // EEP floor control
    bool use_hard_max = true;     // max(0, softplus-bias) — set false for smooth-only

    // Coordinate transforms (individually toggleable)
    bool use_sinh_x = false;      // sinh mapping on x axis (clusters near ATM)
    double sinh_alpha = 3.0;      // clustering intensity

    bool use_sqrt_tau = false;    // sqrt(tau) coordinate (clusters short maturities)

    bool use_log_eep = false;     // interpolate log(EEP + eps) instead of EEP
    double log_eep_eps = 1e-10;   // floor for log transform
};

struct Chebyshev4DEEPResult {
    ChebyshevTucker4D interp;
    Chebyshev4DTransforms transforms;
    int n_pde_solves;
    double build_seconds;
};

// ============================================================================
// Piecewise Chebyshev 4D: spectral elements along x-axis
// ============================================================================

struct PiecewiseChebyshev4DConfig {
    // Segment boundaries for x-axis (N+1 values for N segments, ascending)
    std::vector<double> x_breaks = {-0.50, -0.10, 0.15, 0.40};
    size_t num_x_per_seg = 15;    // CGL nodes per segment

    // Shared axes (same as Chebyshev4DEEPConfig)
    size_t num_tau = 15;
    size_t num_sigma = 15;
    size_t num_rate = 6;
    double epsilon = 1e-8;

    double tau_min = 0.019;
    double tau_max = 2.0;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.0;
    bool use_hard_max = true;
};

struct PiecewiseChebyshev4DResult {
    std::vector<ChebyshevTucker4D> segments;
    std::vector<double> x_bounds;   // N+1 boundaries (with outer headroom)
    int n_pde_solves;
    double build_seconds;
};

class PiecewiseChebyshev4DEEPInner {
public:
    PiecewiseChebyshev4DEEPInner(std::vector<ChebyshevTucker4D> segments,
                                  std::vector<double> x_bounds,
                                  OptionType type, double K_ref,
                                  double dividend_yield)
        : segments_(std::move(segments)), x_bounds_(std::move(x_bounds)),
          type_(type), K_ref_(K_ref), dividend_yield_(dividend_yield) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        size_t seg = find_segment(x);
        double eep = segments_[seg].eval({x, q.tau, q.sigma, q.rate});

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep * (q.strike / K_ref_) + eu.value();
    }

    [[nodiscard]] double vega(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        size_t seg = find_segment(x);
        std::array<double, 4> coords = {x, q.tau, q.sigma, q.rate};

        double eep_vega = (q.strike / K_ref_) * segments_[seg].partial(2, coords);

        auto eu = EuropeanOptionSolver(
            OptionSpec{.spot = q.spot, .strike = q.strike, .maturity = q.tau,
                       .rate = q.rate, .dividend_yield = dividend_yield_,
                       .option_type = type_},
            q.sigma).solve().value();

        return eep_vega + eu.vega();
    }

private:
    [[nodiscard]] size_t find_segment(double x) const {
        size_t n = segments_.size();
        for (size_t i = 0; i + 1 < n; ++i) {
            if (x < x_bounds_[i + 1]) return i;
        }
        return n - 1;
    }

    std::vector<ChebyshevTucker4D> segments_;
    std::vector<double> x_bounds_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
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
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double hx     = headroom_fn(cfg.x_min,     cfg.x_max,     cfg.num_x);
    double htau   = headroom_fn(cfg.tau_min,    cfg.tau_max,   cfg.num_tau);
    double hsigma = headroom_fn(cfg.sigma_min,  cfg.sigma_max, cfg.num_sigma);
    double hrate  = headroom_fn(cfg.rate_min,   cfg.rate_max,  cfg.num_rate);

    // ---- 2. Extended bounds with clamping ----
    double x_lo     = cfg.x_min - hx;
    double x_hi     = cfg.x_max + hx;
    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    // ---- 3. Set up transforms and computational coordinates ----
    Chebyshev4DTransforms transforms;
    transforms.use_sinh_x = cfg.use_sinh_x;
    transforms.sinh_alpha = cfg.sinh_alpha;
    transforms.use_sqrt_tau = cfg.use_sqrt_tau;
    transforms.use_log_eep = cfg.use_log_eep;
    transforms.log_eep_eps = cfg.log_eep_eps;

    // Axis 0: x or sinh-mapped u
    double axis0_lo, axis0_hi;
    std::vector<double> x_physical;

    if (cfg.use_sinh_x) {
        axis0_lo = -1.0;
        axis0_hi = 1.0;
        transforms.x_mid = (x_lo + x_hi) / 2.0;
        transforms.x_half = (x_hi - x_lo) / 2.0;
        double sinh_a = std::sinh(cfg.sinh_alpha);

        auto u_nodes = chebyshev_nodes(cfg.num_x, -1.0, 1.0);
        x_physical.resize(cfg.num_x);
        for (size_t i = 0; i < cfg.num_x; ++i) {
            x_physical[i] = transforms.x_mid +
                transforms.x_half * std::sinh(cfg.sinh_alpha * u_nodes[i]) / sinh_a;
        }
    } else {
        axis0_lo = x_lo;
        axis0_hi = x_hi;
        x_physical = chebyshev_nodes(cfg.num_x, x_lo, x_hi);
    }

    // Axis 1: tau or sqrt(tau)
    double axis1_lo, axis1_hi;
    std::vector<double> tau_physical;

    if (cfg.use_sqrt_tau) {
        double s_lo = std::sqrt(tau_lo);
        double s_hi = std::sqrt(tau_hi);
        axis1_lo = s_lo;
        axis1_hi = s_hi;

        auto s_nodes = chebyshev_nodes(cfg.num_tau, s_lo, s_hi);
        tau_physical.resize(cfg.num_tau);
        for (size_t i = 0; i < cfg.num_tau; ++i) {
            tau_physical[i] = s_nodes[i] * s_nodes[i];
        }
    } else {
        axis1_lo = tau_lo;
        axis1_hi = tau_hi;
        tau_physical = chebyshev_nodes(cfg.num_tau, tau_lo, tau_hi);
    }

    // Axes 2, 3: sigma, rate (no transforms)
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate,  rate_lo,  rate_hi);

    ChebyshevTucker4DDomain dom{
        .bounds = {{{axis0_lo, axis0_hi}, {axis1_lo, axis1_hi},
                    {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
    ChebyshevTucker4DConfig tcfg{
        .num_pts = {cfg.num_x, cfg.num_tau, cfg.num_sigma, cfg.num_rate},
        .epsilon = cfg.epsilon};

    auto t0 = std::chrono::steady_clock::now();

    // ---- 4. Batch-solve: N_sigma x N_rate PDEs ----
    const double tau_solve = tau_physical.back() * 1.01;

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
    solver.set_snapshot_times(std::span<const double>{tau_physical});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // ---- 5-7. Extract EEP tensor: resample PDE at node points ----
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

                double tau = tau_physical[j];

                for (size_t i = 0; i < Nx; ++i) {
                    // PDE solution is normalized V/K_ref; convert to dollars
                    double am = spline.eval(x_physical[i]) * K_ref;

                    // European price for this (x, tau, sigma, rate) point
                    double spot_local = std::exp(x_physical[i]) * K_ref;
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
                        eep = cfg.use_hard_max
                            ? std::max(0.0, softplus - bias)
                            : (softplus - bias);
                    }

                    // Optional log transform for smoother interpolation
                    double value = cfg.use_log_eep
                        ? std::log(eep + cfg.log_eep_eps)
                        : eep;

                    tensor[i * Nt * Ns * Nr + j * Ns * Nr + s * Nr + r] = value;
                }
            }
        }
    }

    // ---- 8. Build ChebyshevTucker4D from values ----
    auto interp = ChebyshevTucker4D::build_from_values(tensor, dom, tcfg);
    auto t1 = std::chrono::steady_clock::now();

    return {std::move(interp), transforms,
            static_cast<int>(cfg.num_sigma * cfg.num_rate),
            std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace mango
