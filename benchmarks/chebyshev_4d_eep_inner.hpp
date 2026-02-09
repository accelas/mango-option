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
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/math/cubic_spline_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
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
                        Chebyshev4DTransforms transforms = {},
                        std::vector<Dividend> dividends = {},
                        double maturity = 0.0)
        : interp_(std::move(interp)), type_(type),
          K_ref_(K_ref), dividend_yield_(dividend_yield),
          transforms_(transforms), dividends_(std::move(dividends)),
          maturity_(maturity) {}

    [[nodiscard]] double price(const PriceQuery& q) const {
        double x = std::log(q.spot / q.strike);
        auto [c0, c1] = map_to_computational(x, q.tau);

        double raw = interp_.eval({c0, c1, q.sigma, q.rate});

        if (!dividends_.empty()) {
            // No-EEP mode: tensor stores V/K_ref directly.
            // Discrete dividends break the European decomposition because
            // only the last segment knows the European value analytically.
            return raw * q.strike;
        }

        // EEP mode: tensor stores EEP = Am - Eu.
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

        if (!dividends_.empty()) {
            // No-EEP mode: ∂V/∂σ = (∂(V/K_ref)/∂σ) * K
            return interp_.partial(2, coords) * q.strike;
        }

        // EEP mode
        double eep_vega = (q.strike / K_ref_) * interp_.partial(2, coords);

        if (transforms_.use_log_eep) {
            double raw = interp_.eval(coords);
            eep_vega *= std::exp(raw);
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

    /// Escrowed spot: S - PV(future dividends).
    /// Future dividends are those with calendar_time > maturity - tau.
    /// When no dividends, returns spot unchanged.
    [[nodiscard]] double escrowed_spot(double spot, double tau, double rate) const {
        if (dividends_.empty()) return spot;
        double t_now = maturity_ - tau;
        double pv = 0.0;
        for (const auto& d : dividends_) {
            if (d.calendar_time > t_now) {
                pv += d.amount * std::exp(-rate * (d.calendar_time - t_now));
            }
        }
        return std::max(spot - pv, 1e-8);
    }

    ChebyshevTucker4D interp_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
    Chebyshev4DTransforms transforms_;
    std::vector<Dividend> dividends_;
    double maturity_;
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
    bool use_tucker = true;   // false = skip HOSVD, store raw tensor

    double x_min = -0.50;     // ln(0.60)
    double x_max = 0.40;      // ln(1.50)
    double tau_min = 0.019;   // ~7 days
    double tau_max = 2.0;
    double sigma_min = 0.05;
    double sigma_max = 0.50;
    double rate_min = 0.01;
    double rate_max = 0.10;

    double dividend_yield = 0.0;
    std::vector<Dividend> discrete_dividends = {};

    // EEP floor control
    bool use_hard_max = true;     // max(0, softplus-bias) — set false for smooth-only

    // Coordinate transforms (individually toggleable)
    bool use_sinh_x = false;      // sinh mapping on x axis (clusters near ATM)
    double sinh_alpha = 3.0;      // clustering intensity

    bool use_sqrt_tau = false;    // sqrt(tau) coordinate (clusters short maturities)

    bool use_log_eep = false;     // interpolate log(EEP + eps) instead of EEP
    double log_eep_eps = 1e-10;   // floor for log transform

    // Override extended bounds per axis (bypass headroom formula).
    // NaN = use headroom as normal. Set both lo and hi to override.
    double rate_ext_lo = std::numeric_limits<double>::quiet_NaN();
    double rate_ext_hi = std::numeric_limits<double>::quiet_NaN();
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
    bool use_tucker = true;

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
// Builder: piecewise Chebyshev 4D EEP surface (spectral elements along x)
// ============================================================================

inline PiecewiseChebyshev4DResult build_piecewise_chebyshev_4d_eep(
    const PiecewiseChebyshev4DConfig& cfg,
    double K_ref,
    OptionType option_type)
{
    const size_t n_seg = cfg.x_breaks.size() - 1;

    // ---- 1. Shared axes: headroom + CGL nodes ----
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) / static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    double htau   = headroom_fn(cfg.tau_min,   cfg.tau_max,   cfg.num_tau);
    double hsigma = headroom_fn(cfg.sigma_min, cfg.sigma_max, cfg.num_sigma);
    double hrate  = headroom_fn(cfg.rate_min,  cfg.rate_max,  cfg.num_rate);

    double tau_lo   = std::max(cfg.tau_min - htau, 1e-4);
    double tau_hi   = cfg.tau_max + htau;
    double sigma_lo = std::max(cfg.sigma_min - hsigma, 0.01);
    double sigma_hi = cfg.sigma_max + hsigma;
    double rate_lo  = std::max(cfg.rate_min - hrate, -0.05);
    double rate_hi  = cfg.rate_max + hrate;

    auto tau_nodes   = chebyshev_nodes(cfg.num_tau,   tau_lo,   tau_hi);
    auto sigma_nodes = chebyshev_nodes(cfg.num_sigma, sigma_lo, sigma_hi);
    auto rate_nodes  = chebyshev_nodes(cfg.num_rate,  rate_lo,  rate_hi);

    // ---- 2. Per-segment x-domains (headroom on outer edges only) ----
    std::vector<double> x_bounds(n_seg + 1);
    std::vector<std::vector<double>> seg_x_nodes(n_seg);

    for (size_t s = 0; s < n_seg; ++s) {
        double lo = cfg.x_breaks[s];
        double hi = cfg.x_breaks[s + 1];

        if (s == 0) {
            double h = headroom_fn(lo, hi, cfg.num_x_per_seg);
            lo -= h;
        }
        if (s == n_seg - 1) {
            double h = headroom_fn(cfg.x_breaks[s], cfg.x_breaks[s + 1],
                                    cfg.num_x_per_seg);
            hi += h;
        }

        x_bounds[s] = lo;
        if (s == n_seg - 1) x_bounds[s + 1] = hi;

        seg_x_nodes[s] = chebyshev_nodes(cfg.num_x_per_seg, lo, hi);
    }
    // Interior boundaries use the break values directly
    for (size_t s = 1; s < n_seg; ++s) {
        x_bounds[s] = cfg.x_breaks[s];
    }

    auto t0 = std::chrono::steady_clock::now();

    // ---- 3. Shared PDE batch: N_sigma x N_rate ----
    const double tau_solve = tau_nodes.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(cfg.num_sigma * cfg.num_rate);
    for (size_t sv = 0; sv < cfg.num_sigma; ++sv) {
        for (size_t rv = 0; rv < cfg.num_rate; ++rv) {
            batch.emplace_back(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[rv],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[sv]);
        }
    }

    BatchAmericanOptionSolver solver;
    solver.set_grid_accuracy(make_grid_accuracy(GridAccuracyProfile::Ultra));
    solver.set_snapshot_times(std::span<const double>{tau_nodes});
    auto batch_result = solver.solve_batch(batch, /*use_shared_grid=*/true);

    // ---- 4. Phase 1: cache splines per (sigma, rate, tau) ----
    // Layout: spline_cache[batch_idx][tau_idx] — one CubicSpline per triple
    const size_t Ns = cfg.num_sigma;
    const size_t Nr = cfg.num_rate;
    const size_t Nt = cfg.num_tau;
    const size_t Nx = cfg.num_x_per_seg;

    struct SplineEntry {
        CubicSpline<double> spline;
        bool valid = false;
    };
    std::vector<std::vector<SplineEntry>> spline_cache(Ns * Nr);

    for (size_t sv = 0; sv < Ns; ++sv) {
        for (size_t rv = 0; rv < Nr; ++rv) {
            size_t batch_idx = sv * Nr + rv;
            auto& cache = spline_cache[batch_idx];
            cache.resize(Nt);

            if (!batch_result.results[batch_idx].has_value()) continue;

            const auto& result = batch_result.results[batch_idx].value();
            auto x_grid = result.grid()->x();

            for (size_t j = 0; j < Nt; ++j) {
                auto spatial = result.at_time(j);
                if (!cache[j].spline.build(x_grid, spatial).has_value()) {
                    cache[j].valid = true;
                }
            }
        }
    }

    // ---- 5. Phase 2: fill per-segment tensors from cached splines ----
    std::vector<ChebyshevTucker4D> segments;
    segments.reserve(n_seg);

    for (size_t seg = 0; seg < n_seg; ++seg) {
        std::vector<double> tensor(Nx * Nt * Ns * Nr, 0.0);
        const auto& x_nodes = seg_x_nodes[seg];

        for (size_t sv = 0; sv < Ns; ++sv) {
            double sigma = sigma_nodes[sv];
            for (size_t rv = 0; rv < Nr; ++rv) {
                double rate = rate_nodes[rv];
                size_t batch_idx = sv * Nr + rv;
                const auto& cache = spline_cache[batch_idx];

                for (size_t j = 0; j < Nt; ++j) {
                    if (!cache[j].valid) continue;
                    double tau = tau_nodes[j];

                    for (size_t i = 0; i < Nx; ++i) {
                        double am = cache[j].spline.eval(x_nodes[i]) * K_ref;

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

                        tensor[i * Nt * Ns * Nr + j * Ns * Nr + sv * Nr + rv] = eep;
                    }
                }
            }
        }

        double seg_x_lo = x_bounds[seg];
        double seg_x_hi = x_bounds[seg + 1];

        ChebyshevTucker4DDomain dom{
            .bounds = {{{seg_x_lo, seg_x_hi}, {tau_lo, tau_hi},
                        {sigma_lo, sigma_hi}, {rate_lo, rate_hi}}}};
        ChebyshevTucker4DConfig tcfg{
            .num_pts = {Nx, Nt, Ns, Nr},
            .epsilon = cfg.epsilon,
            .use_tucker = cfg.use_tucker};

        segments.push_back(
            ChebyshevTucker4D::build_from_values(tensor, dom, tcfg));
    }

    auto t1 = std::chrono::steady_clock::now();

    return {std::move(segments), x_bounds,
            static_cast<int>(cfg.num_sigma * cfg.num_rate),
            std::chrono::duration<double>(t1 - t0).count()};
}

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
    double rate_lo  = std::isnan(cfg.rate_ext_lo)
        ? std::max(cfg.rate_min - hrate, -0.05) : cfg.rate_ext_lo;
    double rate_hi  = std::isnan(cfg.rate_ext_hi)
        ? cfg.rate_max + hrate : cfg.rate_ext_hi;

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
        .epsilon = cfg.epsilon,
        .use_tucker = cfg.use_tucker};

    auto t0 = std::chrono::steady_clock::now();

    // ---- 4. Batch-solve: N_sigma x N_rate PDEs ----
    const double tau_solve = tau_physical.back() * 1.01;

    std::vector<PricingParams> batch;
    batch.reserve(cfg.num_sigma * cfg.num_rate);
    for (size_t s = 0; s < cfg.num_sigma; ++s) {
        for (size_t r = 0; r < cfg.num_rate; ++r) {
            PricingParams p(
                OptionSpec{.spot = K_ref, .strike = K_ref, .maturity = tau_solve,
                           .rate = rate_nodes[r],
                           .dividend_yield = cfg.dividend_yield,
                           .option_type = option_type},
                sigma_nodes[s]);
            p.discrete_dividends = cfg.discrete_dividends;
            batch.push_back(std::move(p));
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
                    // PDE solution is normalized V/K_ref
                    double v_norm = spline.eval(x_physical[i]);

                    double value;
                    if (!cfg.discrete_dividends.empty()) {
                        // No-EEP mode: store V/K_ref directly.
                        // Discrete dividends break EEP because only the last
                        // segment knows the European value analytically.
                        value = v_norm;
                    } else {
                        // EEP mode: subtract European, apply softplus floor
                        double am = v_norm * K_ref;
                        double spot_local = std::exp(x_physical[i]) * K_ref;
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
                            double softplus = std::log1p(std::exp(
                                kSharpness * eep_raw)) / kSharpness;
                            double bias = std::log(2.0) / kSharpness;
                            eep = cfg.use_hard_max
                                ? std::max(0.0, softplus - bias)
                                : (softplus - bias);
                        }

                        value = cfg.use_log_eep
                            ? std::log(eep + cfg.log_eep_eps) : eep;
                    }

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

// ============================================================================
// Anisotropic sweep: fixed (Nx, Ntau, domains), sweep (Nsigma, Nrate)
// ============================================================================

struct SweepConfig {
    size_t num_x = 40;
    size_t num_tau = 15;

    // Fixed extended domain bounds (frozen — no headroom coupling)
    double x_min = -0.50, x_max = 0.40;
    double tau_min = 0.019, tau_max = 2.0;
    double sigma_min = 0.05, sigma_max = 0.50;
    double rate_min = 0.01, rate_max = 0.10;

    double dividend_yield = 0.0;
};

struct SweepEntry {
    size_t num_sigma, num_rate;
    size_t pde_solves;
    double build_seconds;
    double max_error;    // IV error (fractional), T>=60d probes
    double avg_error;
    Chebyshev4DEEPInner inner;
};

/// Run anisotropic sweep over (Nsigma, Nrate) candidates.
/// Uses chain solver for validation.  Returns sorted by PDE cost.
inline std::vector<SweepEntry> run_chebyshev_4d_sweep(
    const SweepConfig& cfg,
    std::span<const size_t> sigma_levels,
    std::span<const size_t> rate_levels,
    double K_ref,
    OptionType option_type)
{
    // Pre-build validation chain solves (shared across all configs).
    // Fixed probes: 5 sigma × 4 rate × 6 tau × multiple x = deterministic.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    constexpr size_t kSigmaVal = 5, kRateVal = 4;
    constexpr size_t kTauSnaps = 6;
    constexpr size_t kXProbes = 4;
    constexpr double kVegaFloor = 0.01;
    constexpr double kVegaBumpFrac = 0.01;
    constexpr double kMinValTau = 60.0 / 365.0;

    std::vector<double> val_sigmas(kSigmaVal), val_rates(kRateVal);
    for (size_t i = 0; i < kSigmaVal; ++i)
        val_sigmas[i] = cfg.sigma_min +
            (cfg.sigma_max - cfg.sigma_min) * (i + 0.5) / kSigmaVal;
    for (size_t i = 0; i < kRateVal; ++i)
        val_rates[i] = cfg.rate_min +
            (cfg.rate_max - cfg.rate_min) * (i + 0.5) / kRateVal;

    double val_tau_lo = std::max(cfg.tau_min, kMinValTau);
    std::vector<double> val_taus(kTauSnaps);
    for (size_t i = 0; i < kTauSnaps; ++i)
        val_taus[i] = val_tau_lo +
            (cfg.tau_max - val_tau_lo) * (i + 0.5) / kTauSnaps;
    std::sort(val_taus.begin(), val_taus.end());

    // Random x probes (fixed across all configs)
    std::vector<double> val_xs(kXProbes);
    for (size_t i = 0; i < kXProbes; ++i)
        val_xs[i] = cfg.x_min + u01(rng) * (cfg.x_max - cfg.x_min);

    // Build validation chain batch: (sigma, rate) pairs × 3 (base + vega bumps)
    struct ValPair { double sigma; double rate; size_t base_idx; };
    std::vector<ValPair> val_pairs;
    std::vector<PricingParams> val_batch;

    double val_tau_max = val_taus.back() * 1.01;
    for (size_t si = 0; si < kSigmaVal; ++si) {
        for (size_t ri = 0; ri < kRateVal; ++ri) {
            double sigma = val_sigmas[si];
            double rate  = val_rates[ri];
            double eps = std::max(1e-4, kVegaBumpFrac * sigma);

            size_t base_idx = val_batch.size();
            val_pairs.push_back({sigma, rate, base_idx});

            for (double sig : {sigma, sigma + eps,
                               std::max(1e-4, sigma - eps)}) {
                val_batch.emplace_back(
                    OptionSpec{.spot = K_ref, .strike = K_ref,
                               .maturity = val_tau_max, .rate = rate,
                               .dividend_yield = cfg.dividend_yield,
                               .option_type = option_type},
                    sig);
            }
        }
    }

    std::fprintf(stderr, "  [sweep] solving %zu validation chains...\n",
                 val_batch.size());
    BatchAmericanOptionSolver val_solver;
    val_solver.set_grid_accuracy(
        make_grid_accuracy(GridAccuracyProfile::Ultra));
    val_solver.set_snapshot_times(std::span<const double>{val_taus});
    auto val_result = val_solver.solve_batch(val_batch, true);

    // Compute headroom once using a reference config (e.g. Ns=15, Nr=10)
    // then freeze the extended domains for all configs.
    auto headroom_fn = [](double lo, double hi, size_t n) {
        return 3.0 * (hi - lo) /
               static_cast<double>(std::max(n, size_t{4}) - 1);
    };
    // Use moderate reference counts for headroom computation
    constexpr size_t kRefNs = 15, kRefNr = 10;
    double hx   = headroom_fn(cfg.x_min, cfg.x_max, cfg.num_x);
    double htau  = headroom_fn(cfg.tau_min, cfg.tau_max, cfg.num_tau);
    double hsig  = headroom_fn(cfg.sigma_min, cfg.sigma_max, kRefNs);
    double hrate = headroom_fn(cfg.rate_min, cfg.rate_max, kRefNr);

    double ext_x_lo   = cfg.x_min - hx;
    double ext_x_hi   = cfg.x_max + hx;
    double ext_tau_lo  = std::max(cfg.tau_min - htau, 1e-4);
    double ext_tau_hi  = cfg.tau_max + htau;
    double ext_sig_lo  = std::max(cfg.sigma_min - hsig, 0.01);
    double ext_sig_hi  = cfg.sigma_max + hsig;
    double ext_rate_lo = std::max(cfg.rate_min - hrate, -0.05);
    double ext_rate_hi = cfg.rate_max + hrate;

    // Sweep
    std::vector<SweepEntry> entries;
    for (size_t ns : sigma_levels) {
        for (size_t nr : rate_levels) {
            std::fprintf(stderr, "  [sweep] Ns=%zu Nr=%zu (%zu PDE)...\n",
                         ns, nr, ns * nr);

            Chebyshev4DEEPConfig build_cfg;
            build_cfg.num_x = cfg.num_x;
            build_cfg.num_tau = cfg.num_tau;
            build_cfg.num_sigma = ns;
            build_cfg.num_rate = nr;
            build_cfg.x_min = cfg.x_min;     build_cfg.x_max = cfg.x_max;
            build_cfg.tau_min = cfg.tau_min;  build_cfg.tau_max = cfg.tau_max;
            build_cfg.sigma_min = cfg.sigma_min;
            build_cfg.sigma_max = cfg.sigma_max;
            build_cfg.rate_min = cfg.rate_min;
            build_cfg.rate_max = cfg.rate_max;
            // Freeze extended domains
            build_cfg.rate_ext_lo = ext_rate_lo;
            build_cfg.rate_ext_hi = ext_rate_hi;
            build_cfg.dividend_yield = cfg.dividend_yield;
            build_cfg.use_tucker = false;

            auto result = build_chebyshev_4d_eep(build_cfg, K_ref, option_type);

            Chebyshev4DEEPInner inner(
                std::move(result.interp), option_type, K_ref,
                cfg.dividend_yield, result.transforms);

            // Validate against pre-computed chain solves
            double max_err = 0.0, sum_err = 0.0;
            size_t n_valid = 0;

            for (const auto& vp : val_pairs) {
                auto& base_res = val_result.results[vp.base_idx];
                auto& up_res   = val_result.results[vp.base_idx + 1];
                auto& dn_res   = val_result.results[vp.base_idx + 2];
                if (!base_res.has_value()) continue;

                auto x_grid = base_res->grid()->x();

                for (size_t ti = 0; ti < kTauSnaps; ++ti) {
                    auto spatial = base_res->at_time(ti);
                    CubicSpline<double> spline;
                    if (spline.build(x_grid, spatial).has_value()) continue;

                    for (double x : val_xs) {
                        double spot = std::exp(x) * K_ref;
                        double ref_price = spline.eval(x) * K_ref;

                        PriceQuery pq{.spot = spot, .strike = K_ref,
                                      .tau = val_taus[ti],
                                      .sigma = vp.sigma, .rate = vp.rate};
                        double interp_price = inner.price(pq);
                        double price_err = std::abs(interp_price - ref_price);

                        // Vega
                        double vega = kVegaFloor;
                        if (up_res && dn_res) {
                            auto sp_up = up_res->at_time(ti);
                            auto sp_dn = dn_res->at_time(ti);
                            CubicSpline<double> su, sd;
                            if (!su.build(x_grid, sp_up).has_value() &&
                                !sd.build(x_grid, sp_dn).has_value()) {
                                double p_up = su.eval(x) * K_ref;
                                double p_dn = sd.eval(x) * K_ref;
                                double eps = std::max(1e-4,
                                    kVegaBumpFrac * vp.sigma);
                                double s_up = vp.sigma + eps;
                                double s_dn = std::max(1e-4, vp.sigma - eps);
                                double eff = (s_up - s_dn) / 2.0;
                                if (eff > 1e-6)
                                    vega = std::max(kVegaFloor,
                                        std::abs((p_up - p_dn) / (2 * eff)));
                            }
                        }

                        double iv_err = price_err / vega;
                        max_err = std::max(max_err, iv_err);
                        sum_err += iv_err;
                        n_valid++;
                    }
                }
            }

            double avg_err = n_valid > 0 ? sum_err / n_valid : 1.0;

            entries.push_back({
                .num_sigma = ns, .num_rate = nr,
                .pde_solves = ns * nr,
                .build_seconds = result.build_seconds,
                .max_error = max_err, .avg_error = avg_err,
                .inner = std::move(inner)
            });
        }
    }

    // Sort by PDE cost
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  return a.pde_solves < b.pde_solves;
              });
    return entries;
}

}  // namespace mango
