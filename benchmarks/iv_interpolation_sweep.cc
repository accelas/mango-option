// SPDX-License-Identifier: MIT
/**
 * @file iv_interpolation_sweep.cc
 * @brief Interpolated IV accuracy: adaptive grid scaling
 *
 * Measures interpolation error by comparing interpolated IV against
 * mango's own high-accuracy FDM solver. Adaptive grid builder finds
 * the base grid, then uniform midpoint insertion scales resolution.
 *
 * Standard path: continuous dividends (AdaptiveGridBuilder::build)
 * Segmented path: discrete dividends (AdaptiveGridBuilder::build_segmented)
 *
 * Run with: bazel run //benchmarks:iv_interpolation_sweep
 */

#include "iv_benchmark_common.hpp"
#include "iv_benchmark_ql.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/adaptive_grid_builder.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/option/table/segmented_price_table_builder.hpp"
#include "mango/option/table/spliced_surface_builder.hpp"

#include <benchmark/benchmark.h>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <vector>

using namespace mango;
using namespace mango::bench;

// ============================================================================
// Test parameters
// ============================================================================

static constexpr double kTrueVol = 0.20;
static constexpr double kMaturity = 1.0;

// Strikes for IV recovery (OTM/ATM/ITM for puts)
// moneyness = spot/strike: 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
static constexpr std::array<double, 7> kStrikes = {
    142.86, 125.0, 111.11, 100.0, 90.91, 83.33, 76.92
};

// Scales: 1x = adaptive base, 2x = midpoint refinement
// (4x+ causes B-spline FittingFailed on the 309-point moneyness axis)
static constexpr std::array<int, 2> kScales = {1, 2};

// ============================================================================
// Reference prices
// ============================================================================

/// Solve American put at given accuracy for all kStrikes
static std::vector<double> solve_reference_prices(const GridAccuracyParams& accuracy) {
    std::vector<double> p;
    p.reserve(kStrikes.size());

    for (double K : kStrikes) {
        PricingParams params(
            OptionSpec{.spot = kSpot, .strike = K, .maturity = kMaturity,
                .rate = kRate, .dividend_yield = kDivYield,
                .option_type = OptionType::PUT},
            kTrueVol);

        auto [grid_spec, time_domain] = estimate_pde_grid(params, accuracy);
        size_t n = grid_spec.n_points();
        std::pmr::vector<double> buffer(
            PDEWorkspace::required_size(n), std::pmr::get_default_resource());
        auto workspace = PDEWorkspace::from_buffer(buffer, n);
        if (!workspace) {
            p.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        auto solver = AmericanOptionSolver::create(
            params, *workspace,
            PDEGridConfig{.grid_spec = grid_spec, .n_time = time_domain.n_steps()});
        if (!solver) {
            p.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        auto result = solver->solve();
        p.push_back(result ? result->value_at(kSpot)
                           : std::numeric_limits<double>::quiet_NaN());
    }
    return p;
}

/// High-accuracy reference (401-501 spatial pts) — the "true" price
static const std::vector<double>& get_reference_prices() {
    static std::vector<double> prices = [] {
        GridAccuracyParams high_accuracy;
        high_accuracy.min_spatial_points = 401;
        high_accuracy.max_spatial_points = 501;
        return solve_reference_prices(high_accuracy);
    }();
    return prices;
}

/// Table-accuracy reference (201-301 spatial pts) — same PDE accuracy as the
/// interpolation table.  Recovering IV from these prices via the interpolated
/// surface isolates B-spline interpolation error from PDE discretisation error.
static const std::vector<double>& get_pde_baseline_prices() {
    static std::vector<double> prices = [] {
        GridAccuracyParams table_accuracy;
        table_accuracy.min_spatial_points = 201;
        table_accuracy.max_spatial_points = 301;
        return solve_reference_prices(table_accuracy);
    }();
    return prices;
}

// ============================================================================
// Grid axis refinement: insert uniform midpoints
// ============================================================================

/// For N base points, scale S produces (N-1)*S + 1 points
static std::vector<double> refine_axis(const std::vector<double>& base, int scale) {
    if (base.size() < 2 || scale <= 1) return base;
    size_t n_intervals = base.size() - 1;
    size_t n_out = n_intervals * static_cast<size_t>(scale) + 1;
    std::vector<double> refined;
    refined.reserve(n_out);
    for (size_t i = 0; i < n_intervals; ++i) {
        double lo = base[i], hi = base[i + 1];
        for (int j = 0; j < scale; ++j) {
            double t = static_cast<double>(j) / static_cast<double>(scale);
            refined.push_back(lo + t * (hi - lo));
        }
    }
    refined.push_back(base.back());
    return refined;
}

// ============================================================================
// Cached adaptive-scaled IV solvers
// ============================================================================

struct AdaptiveSolverEntry {
    std::unique_ptr<DefaultInterpolatedIVSolver> solver;
    double build_time_ms = 0.0;
    size_t n_pde_solves = 0;
    std::array<size_t, 4> base_grid_sizes = {};  // [m, tau, sigma, r]
    bool target_met = false;
};

static const AdaptiveSolverEntry& get_adaptive_solver(int scale) {
    static std::map<int, AdaptiveSolverEntry> cache;
    auto it = cache.find(scale);
    if (it != cache.end()) return it->second;

    auto t0 = std::chrono::steady_clock::now();

    // 1. Build adaptive base (only for scale=1, reuse axes for larger scales)
    static AdaptiveResult* base_result = nullptr;
    static PriceTableAxes<4> base_axes;
    static double base_K_ref = 0.0;

    if (!base_result) {
        // Domain for adaptive calibration
        OptionGrid chain;
        chain.spot = kSpot;
        chain.strikes = {kStrikes.begin(), kStrikes.end()};
        chain.maturities = {0.25, 0.5, 1.0, 1.5, 2.0};
        chain.implied_vols = {0.05, 0.10, 0.20, 0.30, 0.50};
        chain.rates = {0.01, 0.03, 0.05, 0.10};
        chain.dividend_yield = kDivYield;

        AdaptiveGridParams params;
        params.target_iv_error = 2e-5;  // 2 bps

        AdaptiveGridBuilder builder(params);
        // High PDE accuracy for table building
        GridAccuracyParams pde_accuracy;
        pde_accuracy.min_spatial_points = 201;
        pde_accuracy.max_spatial_points = 301;
        auto result = builder.build(chain, PDEGridSpec{pde_accuracy}, OptionType::PUT);
        if (!result.has_value()) {
            std::fprintf(stderr, "AdaptiveGridBuilder::build failed\n");
            std::abort();
        }
        static AdaptiveResult stored = std::move(*result);
        base_result = &stored;
        base_axes = stored.axes;
        base_K_ref = stored.surface->metadata().K_ref;
    }

    // 2. For scale=1, use adaptive result directly
    std::shared_ptr<const PriceTableSurface<4>> surface;
    size_t total_pde = base_result->total_pde_solves;
    std::array<size_t, 4> grid_sizes = {};

    if (scale == 1) {
        surface = base_result->surface;
        for (int d = 0; d < 4; ++d) {
            grid_sizes[d] = base_axes.grids[d].size();
        }
    } else {
        // 3. Refine base axes by inserting midpoints
        auto m_refined = refine_axis(base_axes.grids[0], scale);
        auto tau_refined = refine_axis(base_axes.grids[1], scale);
        auto sig_refined = refine_axis(base_axes.grids[2], scale);
        auto r_refined = refine_axis(base_axes.grids[3], scale);

        grid_sizes = {m_refined.size(), tau_refined.size(),
                      sig_refined.size(), r_refined.size()};

        // Rebuild price table with refined axes
        GridAccuracyParams pde_accuracy;
        pde_accuracy.min_spatial_points = 201;
        pde_accuracy.max_spatial_points = 301;

        auto setup = PriceTableBuilder<4>::from_vectors(
            m_refined, tau_refined, sig_refined, r_refined,
            base_K_ref, PDEGridSpec{pde_accuracy}, OptionType::PUT, kDivYield);
        if (!setup) {
            std::fprintf(stderr, "PriceTableBuilder::from_vectors failed (scale=%d)\n", scale);
            std::abort();
        }
        auto& [ptb, axes] = *setup;
        auto result = ptb.build(axes);
        if (!result) {
            std::fprintf(stderr, "PriceTableBuilder::build failed (scale=%d)\n", scale);
            std::abort();
        }
        surface = result->surface;
        total_pde = result->n_pde_solves;
    }

    // 4. Create InterpolatedIVSolver
    auto aps = AmericanPriceSurface::create(surface, OptionType::PUT);
    if (!aps) {
        std::fprintf(stderr, "AmericanPriceSurface::create failed (scale=%d)\n", scale);
        std::abort();
    }
    auto solver = DefaultInterpolatedIVSolver::create(std::move(*aps));
    if (!solver) {
        std::fprintf(stderr, "InterpolatedIVSolver::create failed (scale=%d)\n", scale);
        std::abort();
    }

    auto t1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto [pos, _] = cache.emplace(scale, AdaptiveSolverEntry{
        std::make_unique<DefaultInterpolatedIVSolver>(std::move(*solver)),
        build_ms,
        total_pde,
        grid_sizes,
        base_result->target_met,
    });
    return pos->second;
}

// ============================================================================
// BM_Adaptive_IV_Scaled: parametrized by (strike_idx, scale_idx)
// ============================================================================

static void BM_Adaptive_IV_Scaled(benchmark::State& state) {
    size_t si = static_cast<size_t>(state.range(0));
    double K = kStrikes[si];
    int scale = kScales[static_cast<size_t>(state.range(1))];

    double ref_price = get_reference_prices()[si];
    double pde_price = get_pde_baseline_prices()[si];
    if (!std::isfinite(ref_price) || !std::isfinite(pde_price)) {
        state.SkipWithError("Reference price not available");
        return;
    }

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kMaturity,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    const auto& entry = get_adaptive_solver(scale);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = entry.solver->solve(query);
        if (!last_result) {
            state.SkipWithError("Interp IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kTrueVol) * 10000.0;

    // Interpolation-only error: recover IV from PDE-baseline price
    // (same PDE accuracy as table → isolates B-spline error)
    double interp_err_bps = std::numeric_limits<double>::quiet_NaN();
    auto interp_result = entry.solver->solve(IVQuery(spec, pde_price));
    if (interp_result) {
        interp_err_bps = std::abs(interp_result->implied_vol - kTrueVol) * 10000.0;
    }

    state.SetLabel(std::format("K={:.0f} {}x adaptive", K, scale));
    state.counters["strike"] = K;
    state.counters["scale"] = scale;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["interp_err_bps"] = interp_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
    state.counters["build_ms"] = entry.build_time_ms;
    state.counters["n_pde_solves"] = static_cast<double>(entry.n_pde_solves);
    state.counters["base_grid_m"] = static_cast<double>(entry.base_grid_sizes[0]);
    state.counters["base_grid_tau"] = static_cast<double>(entry.base_grid_sizes[1]);
    state.counters["base_grid_sig"] = static_cast<double>(entry.base_grid_sizes[2]);
    state.counters["base_grid_r"] = static_cast<double>(entry.base_grid_sizes[3]);
    state.counters["adaptive_target_met"] = entry.target_met ? 1.0 : 0.0;
}

BENCHMARK(BM_Adaptive_IV_Scaled)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kStrikes.size()) - 1, 1),
        benchmark::CreateDenseRange(0, static_cast<int>(kScales.size()) - 1, 1),
    })
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Discrete dividend scenario
// ============================================================================

// Per-strike surface: one segmented surface per test strike
static const std::vector<double> kStrikeGrid = {
    kStrikes.begin(), kStrikes.end()
};

// ============================================================================
// Reference prices: QuantLib FD with discrete dividends (2001x20000)
// ============================================================================

static const std::vector<double>& get_div_reference_prices() {
    static std::vector<double> prices = [] {
        std::vector<double> p;
        p.reserve(kStrikes.size());
        auto divs = make_div_schedule(kMaturity);
        for (double K : kStrikes) {
            p.push_back(price_ql_div(kSpot, K, kTrueVol, kMaturity,
                                      kRate, kDivYield, divs, 2001, 20000));
        }
        return p;
    }();
    return prices;
}

/// Table-accuracy reference for segmented path (mango FDM with discrete divs,
/// default grid accuracy — same PDE resolution the segmented table builder uses).
static const std::vector<double>& get_div_pde_baseline_prices() {
    static std::vector<double> prices = [] {
        std::vector<double> p;
        p.reserve(kStrikes.size());
        auto divs = make_div_schedule(kMaturity);
        for (double K : kStrikes) {
            PricingParams params(
                OptionSpec{.spot = kSpot, .strike = K, .maturity = kMaturity,
                    .rate = kRate, .dividend_yield = kDivYield,
                    .option_type = OptionType::PUT},
                kTrueVol);
            params.discrete_dividends = divs;
            auto result = solve_american_option(params);
            p.push_back(result ? result->value()
                               : std::numeric_limits<double>::quiet_NaN());
        }
        return p;
    }();
    return prices;
}

// ============================================================================
// Cached segmented adaptive-scaled IV solvers (discrete dividends)
// ============================================================================

struct SegmentedSolverEntry {
    std::unique_ptr<InterpolatedIVSolver<StrikeSurfaceWrapper<>>> solver;
    double build_time_ms = 0.0;
    size_t n_pde_solves = 0;
    std::array<size_t, 3> base_grid_sizes = {};  // [m, sigma, r]
    int base_tau_points = 0;
    bool target_met = false;
};

static const SegmentedSolverEntry& get_segmented_solver(int scale) {
    static std::map<int, SegmentedSolverEntry> cache;
    auto it = cache.find(scale);
    if (it != cache.end()) return it->second;

    auto t0 = std::chrono::steady_clock::now();

    // 1. Build adaptive base (only once, reuse grid for larger scales)
    static StrikeAdaptiveResult* base_result = nullptr;

    if (!base_result) {
        AdaptiveGridParams params;
        params.target_iv_error = 2e-5;  // 2 bps

        SegmentedAdaptiveConfig seg_config{
            .spot = kSpot,
            .option_type = OptionType::PUT,
            .dividend_yield = kDivYield,
            .discrete_dividends = make_div_schedule(kMaturity),
            .maturity = kMaturity,
            .kref_config = {.K_refs = kStrikeGrid},
        };

        IVGrid domain{
            .moneyness = {0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30},
            .vol = {0.05, 0.10, 0.20, 0.30, 0.50},
            .rate = {0.01, 0.03, 0.05, 0.10},
        };

        AdaptiveGridBuilder builder(params);
        auto result = builder.build_segmented_strike(seg_config, kStrikeGrid, domain);
        if (!result.has_value()) {
            std::fprintf(stderr, "AdaptiveGridBuilder::build_segmented_strike failed\n");
            std::abort();
        }
        static StrikeAdaptiveResult stored = std::move(*result);
        base_result = &stored;
    }

    // 2. Build solver for this scale
    std::array<size_t, 3> grid_sizes = {};
    int tau_pts = 0;

    // For scale > 1, refine base grid; for scale == 1, rebuild from base grid
    // (StrikeSurface is move-only, so we always rebuild from grid specs)
    auto m_refined = (scale <= 1) ? base_result->grid.moneyness
                                  : refine_axis(base_result->grid.moneyness, scale);
    auto v_refined = (scale <= 1) ? base_result->grid.vol
                                  : refine_axis(base_result->grid.vol, scale);
    auto r_refined = (scale <= 1) ? base_result->grid.rate
                                  : refine_axis(base_result->grid.rate, scale);
    int tau_refined = (scale <= 1) ? base_result->tau_points_per_segment
                                   : base_result->tau_points_per_segment * scale;

    grid_sizes = {m_refined.size(), v_refined.size(), r_refined.size()};
    tau_pts = tau_refined;

    // Rebuild per-strike segmented surfaces and assemble
    auto divs = make_div_schedule(kMaturity);
    DividendSpec div_spec{.dividend_yield = kDivYield, .discrete_dividends = divs};

    std::vector<StrikeEntry> entries;
    for (double strike : kStrikeGrid) {
        SegmentedPriceTableBuilder::Config seg_cfg{
            .K_ref = strike,
            .option_type = OptionType::PUT,
            .dividends = div_spec,
            .grid = {.moneyness = m_refined, .vol = v_refined, .rate = r_refined},
            .maturity = kMaturity,
            .tau_points_per_segment = tau_refined,
        };
        auto seg = SegmentedPriceTableBuilder::build(seg_cfg);
        if (!seg.has_value()) {
            std::fprintf(stderr, "SegmentedPriceTableBuilder::build failed (K=%.0f, scale=%d)\n",
                strike, scale);
            std::abort();
        }
        entries.push_back({.strike = strike, .surface = std::move(*seg)});
    }
    auto surface = build_strike_surface(std::move(entries), /*use_nearest=*/true);
    if (!surface.has_value()) {
        std::fprintf(stderr, "build_strike_surface failed (scale=%d)\n", scale);
        std::abort();
    }

    // 3. Wrap in StrikeSurfaceWrapper and create IV solver
    auto minmax_m = std::minmax_element(base_result->grid.moneyness.begin(),
                                         base_result->grid.moneyness.end());
    auto minmax_v = std::minmax_element(base_result->grid.vol.begin(),
                                         base_result->grid.vol.end());
    auto minmax_r = std::minmax_element(base_result->grid.rate.begin(),
                                         base_result->grid.rate.end());

    StrikeSurfaceWrapper<>::Bounds bounds{
        .m_min = *minmax_m.first,
        .m_max = *minmax_m.second,
        .tau_min = 0.0,
        .tau_max = kMaturity,
        .sigma_min = *minmax_v.first,
        .sigma_max = *minmax_v.second,
        .rate_min = *minmax_r.first,
        .rate_max = *minmax_r.second,
    };

    auto wrapper = StrikeSurfaceWrapper<>(
        std::move(*surface), bounds, OptionType::PUT, kDivYield);

    auto solver = InterpolatedIVSolver<StrikeSurfaceWrapper<>>::create(
        std::move(wrapper));
    if (!solver.has_value()) {
        std::fprintf(stderr, "InterpolatedIVSolver create failed (scale=%d)\n", scale);
        std::abort();
    }

    auto t1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto [pos, ins] = cache.emplace(scale, SegmentedSolverEntry{
        .solver = std::make_unique<InterpolatedIVSolver<StrikeSurfaceWrapper<>>>(std::move(*solver)),
        .build_time_ms = build_ms,
        .n_pde_solves = 0,
        .base_grid_sizes = grid_sizes,
        .base_tau_points = tau_pts,
        .target_met = true,
    });
    return pos->second;
}

// ============================================================================
// BM_Adaptive_IV_Div_Scaled: segmented (discrete dividends)
// ============================================================================

static void BM_Adaptive_IV_Div_Scaled(benchmark::State& state) {
    size_t si = static_cast<size_t>(state.range(0));
    double K = kStrikes[si];
    int scale = kScales[static_cast<size_t>(state.range(1))];

    double ref_price = get_div_reference_prices()[si];
    double pde_price = get_div_pde_baseline_prices()[si];
    if (!std::isfinite(ref_price) || !std::isfinite(pde_price)) {
        state.SkipWithError("Reference price not available");
        return;
    }

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kMaturity,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    const auto& entry = get_segmented_solver(scale);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = entry.solver->solve(query);
        if (!last_result) {
            state.SkipWithError("Segmented interp IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kTrueVol) * 10000.0;

    // Interpolation-only error: recover IV from PDE-baseline price
    double interp_err_bps = std::numeric_limits<double>::quiet_NaN();
    auto interp_result = entry.solver->solve(IVQuery(spec, pde_price));
    if (interp_result) {
        interp_err_bps = std::abs(interp_result->implied_vol - kTrueVol) * 10000.0;
    }

    state.SetLabel(std::format("K={:.0f} {}x segmented", K, scale));
    state.counters["strike"] = K;
    state.counters["scale"] = scale;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["interp_err_bps"] = interp_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
    state.counters["build_ms"] = entry.build_time_ms;
    state.counters["n_pde_solves"] = static_cast<double>(entry.n_pde_solves);
    state.counters["base_grid_m"] = static_cast<double>(entry.base_grid_sizes[0]);
    state.counters["base_grid_sig"] = static_cast<double>(entry.base_grid_sizes[1]);
    state.counters["base_grid_r"] = static_cast<double>(entry.base_grid_sizes[2]);
    state.counters["base_tau_pts"] = static_cast<double>(entry.base_tau_points);
}

BENCHMARK(BM_Adaptive_IV_Div_Scaled)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kStrikes.size()) - 1, 1),
        benchmark::CreateDenseRange(0, static_cast<int>(kScales.size()) - 1, 1),
    })
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
