// SPDX-License-Identifier: MIT
/**
 * @file iv_strike_sweep.cc
 * @brief IV accuracy across strikes: mango vs QuantLib
 *
 * Recovery test: price American puts at known σ=0.20 across strikes,
 * then invert to IV and compare recovered vol accuracy.
 *
 * Reference prices from QuantLib high-res (2001×20000).
 * Both solvers use their default grid (~101 spatial points) and
 * Brent root-finding to recover IV.
 *
 * Run with: bazel run //benchmarks:iv_strike_sweep
 */

#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/table/price_table_builder.hpp"
#include "mango/option/table/price_table_surface.hpp"
#include "mango/option/table/american_price_surface.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <benchmark/benchmark.h>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <memory_resource>
#include <vector>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Test parameters
// ============================================================================

static constexpr double kSpot = 100.0;
static constexpr double kMaturity = 1.0;
static constexpr double kTrueVol = 0.20;
static constexpr double kRate = 0.05;
static constexpr double kDivYield = 0.02;

static constexpr std::array<double, 9> kStrikes = {
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0
};

// Fixed evaluation date for reproducible results
static const ql::Date kEvalDate(1, ql::January, 2024);

// ============================================================================
// QuantLib pricing helper (parameterized by strike and vol)
// ============================================================================

static double price_ql(double strike, double vol, size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(kMaturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, strike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(kSpot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kRate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivYield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), vol, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_h, div_h, rate_h, vol_h);

    option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(process, time_steps, grid_steps));

    return option.NPV();
}

// ============================================================================
// QuantLib reference prices (2001×20000, computed once)
// ============================================================================

static const std::vector<double>& get_ql_reference_prices() {
    static std::vector<double> prices = [] {
        std::vector<double> p;
        p.reserve(kStrikes.size());
        for (double K : kStrikes) {
            p.push_back(price_ql(K, kTrueVol, 2001, 20000));
        }
        return p;
    }();
    return prices;
}

// ============================================================================
// QuantLib base grid dimensions (to match mango's auto-estimated grid)
// ============================================================================

struct BaseGrid {
    size_t nx;
    size_t nt;
};

static BaseGrid get_mango_base_grid(double strike) {
    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = kMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kTrueVol);
    auto [gs, td] = estimate_pde_grid(params);
    return {gs.n_points(), td.n_steps()};
}

// ============================================================================
// Simple Brent solver for QuantLib IV
// ============================================================================

static double ql_solve_iv(double strike, double target_price, size_t nx, size_t nt) {
    // Brent's method: find vol in [0.01, 3.0] such that price_ql(K, vol) = target
    double a = 0.01, b = 3.0;
    double fa = price_ql(strike, a, nx, nt) - target_price;
    double fb = price_ql(strike, b, nx, nt) - target_price;

    if (fa * fb > 0) return -1.0;  // not bracketed

    // Ensure |f(b)| < |f(a)|
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a, fc = fa;
    bool mflag = true;
    double d = 0.0;
    constexpr double tol = 1e-6;
    constexpr size_t max_iter = 100;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fb) < tol || std::abs(b - a) < tol) {
            return b;
        }

        double s;
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }

        double bisect = (3.0 * a + b) / 4.0;
        bool cond1 = !((s > bisect && s < b) || (s < bisect && s > b));
        bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        bool cond4 = mflag && std::abs(b - c) < tol;
        bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = price_ql(strike, s, nx, nt) - target_price;
        if (!std::isfinite(fs)) return -1.0;

        d = c; c = b; fc = fb;
        if (fa * fs < 0.0) { b = s; fb = fs; }
        else { a = s; fa = fs; }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b); std::swap(fa, fb);
        }
    }
    return b;  // max iter
}

// ============================================================================
// Mango IV: default grid
// ============================================================================

static void BM_Mango_IV(benchmark::State& state) {
    int idx = static_cast<int>(state.range(0));
    double K = kStrikes[idx];
    double ref_price = get_ql_reference_prices()[idx];

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kMaturity,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    IVSolverConfig config;
    IVSolver solver(config);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = solver.solve(query);
        if (!last_result) {
            state.SkipWithError("IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kTrueVol) * 10000.0;

    state.SetLabel(std::format("K={:.0f} S/K={:.2f}", K, kSpot / K));
    state.counters["strike"] = K;
    state.counters["ref_price"] = ref_price;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
}

BENCHMARK(BM_Mango_IV)
    ->DenseRange(0, static_cast<int>(kStrikes.size()) - 1)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// QuantLib IV: matching grid dimensions
// ============================================================================

static void BM_QuantLib_IV(benchmark::State& state) {
    int idx = static_cast<int>(state.range(0));
    double K = kStrikes[idx];
    double ref_price = get_ql_reference_prices()[idx];

    // Use mango's auto-estimated grid size for fair comparison
    auto base = get_mango_base_grid(K);

    double iv = 0;
    for (auto _ : state) {
        iv = ql_solve_iv(K, ref_price, base.nx, base.nt);
        benchmark::DoNotOptimize(iv);
    }

    double iv_err_bps = (iv > 0) ? std::abs(iv - kTrueVol) * 10000.0 : -1.0;

    state.SetLabel(std::format("K={:.0f} QL", K));
    state.counters["strike"] = K;
    state.counters["ref_price"] = ref_price;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["nx"] = static_cast<double>(base.nx);
    state.counters["nt"] = static_cast<double>(base.nt);
}

BENCHMARK(BM_QuantLib_IV)
    ->DenseRange(0, static_cast<int>(kStrikes.size()) - 1)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Grid-scaled IV: mango (explicit PDEGridConfig)
// ============================================================================

// Scales to test: 1x, 2x, 4x of auto-estimated base grid
static constexpr std::array<int, 3> kScales = {1, 2, 4};

// Representative strikes for scaled tests: deep ITM, ATM, deep OTM
static constexpr std::array<double, 3> kScaledStrikes = {80.0, 100.0, 120.0};

// Map kScaledStrikes to their index in kStrikes (80→0, 100→4, 120→8)
static constexpr int scaled_strike_to_idx(double K) {
    if (K == 80.0) return 0;
    if (K == 100.0) return 4;
    return 8; // 120.0
}

// Build a scaled PDEGridConfig: same clusters/domain as auto-estimated, but Nx*scale, Nt*scale
static PDEGridConfig make_scaled_grid(double strike, int scale) {
    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = kMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kTrueVol);
    auto [gs, td] = estimate_pde_grid(params);

    size_t nx = gs.n_points() * static_cast<size_t>(scale);
    // Ensure odd for centered stencils
    if (nx % 2 == 0) nx++;

    size_t nt = td.n_steps() * static_cast<size_t>(scale);

    // Rebuild multi-sinh grid at scaled resolution, reusing clusters from base
    std::vector<MultiSinhCluster<double>> clusters(gs.clusters().begin(), gs.clusters().end());
    auto scaled_gs = GridSpec<double>::multi_sinh_spaced(
        gs.x_min(), gs.x_max(), nx, std::move(clusters));

    return PDEGridConfig{
        .grid_spec = scaled_gs.value(),
        .n_time = nt,
    };
}

static void BM_Mango_IV_Scaled(benchmark::State& state) {
    double K = kScaledStrikes[static_cast<size_t>(state.range(0))];
    int scale = kScales[static_cast<size_t>(state.range(1))];
    int ref_idx = scaled_strike_to_idx(K);
    double ref_price = get_ql_reference_prices()[ref_idx];

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kMaturity,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    auto grid_config = make_scaled_grid(K, scale);
    size_t nx = grid_config.grid_spec.n_points();
    size_t nt = grid_config.n_time;

    IVSolverConfig config;
    config.grid = grid_config;
    IVSolver solver(config);

    std::expected<IVSuccess, IVError> last_result;
    for (auto _ : state) {
        last_result = solver.solve(query);
        if (!last_result) {
            state.SkipWithError("IV solve failed");
            return;
        }
        benchmark::DoNotOptimize(last_result);
    }

    double iv = last_result->implied_vol;
    double iv_err_bps = std::abs(iv - kTrueVol) * 10000.0;

    state.SetLabel(std::format("K={:.0f} {}x", K, scale));
    state.counters["strike"] = K;
    state.counters["scale"] = scale;
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
}

BENCHMARK(BM_Mango_IV_Scaled)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kScaledStrikes.size()) - 1, 1),
        benchmark::CreateDenseRange(0, static_cast<int>(kScales.size()) - 1, 1),
    })
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Grid-scaled IV: QuantLib (matching dimensions)
// ============================================================================

static void BM_QuantLib_IV_Scaled(benchmark::State& state) {
    double K = kScaledStrikes[static_cast<size_t>(state.range(0))];
    int scale = kScales[static_cast<size_t>(state.range(1))];
    int ref_idx = scaled_strike_to_idx(K);
    double ref_price = get_ql_reference_prices()[ref_idx];

    auto base = get_mango_base_grid(K);
    size_t nx = base.nx * static_cast<size_t>(scale);
    size_t nt = base.nt * static_cast<size_t>(scale);

    double iv = 0;
    for (auto _ : state) {
        iv = ql_solve_iv(K, ref_price, nx, nt);
        benchmark::DoNotOptimize(iv);
    }

    double iv_err_bps = (iv > 0) ? std::abs(iv - kTrueVol) * 10000.0 : -1.0;

    state.SetLabel(std::format("K={:.0f} {}x QL", K, scale));
    state.counters["strike"] = K;
    state.counters["scale"] = scale;
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
}

BENCHMARK(BM_QuantLib_IV_Scaled)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kScaledStrikes.size()) - 1, 1),
        benchmark::CreateDenseRange(0, static_cast<int>(kScales.size()) - 1, 1),
    })
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Grid-scaled IV: interpolated (B-spline price table)
// ============================================================================

// Cached interpolated IV solvers keyed by scale (build is expensive)
struct InterpSolverEntry {
    std::unique_ptr<DefaultInterpolatedIVSolver> solver;
    double build_time_ms = 0.0;
    size_t n_pde_solves = 0;
};

static const InterpSolverEntry& get_interp_solver(int scale) {
    static std::map<int, InterpSolverEntry> cache;
    auto it = cache.find(scale);
    if (it != cache.end()) return it->second;

    auto t0 = std::chrono::steady_clock::now();

    // Scale FDM resolution via GridAccuracyParams (auto-sizes domain per option)
    size_t base_points = 101;
    GridAccuracyParams accuracy;
    accuracy.min_spatial_points = base_points * static_cast<size_t>(scale);
    accuracy.max_spatial_points = accuracy.min_spatial_points + 100;

    // Interpolation axes: dense enough that interpolation error < FDM error
    auto setup = PriceTableBuilder<4>::from_vectors(
        {0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95, 0.97, 1.00,
         1.03, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30},
        {0.25, 0.5, 0.75, 1.0, 1.5},              // maturity
        {0.08, 0.12, 0.16, 0.20, 0.25, 0.30},     // volatility
        {0.02, 0.03, 0.05, 0.07},                  // rate
        100.0,                                     // K_ref
        accuracy,
        OptionType::PUT,
        kDivYield);

    if (!setup) {
        std::fprintf(stderr, "PriceTableBuilder::from_vectors failed (scale=%d)\n", scale);
        std::abort();
    }
    auto [builder, axes] = std::move(setup.value());
    auto result = builder.build(axes);
    if (!result) {
        std::fprintf(stderr, "PriceTableBuilder::build failed (scale=%d, code=%d, axis=%zu, count=%zu)\n",
            scale, static_cast<int>(result.error().code),
            result.error().axis_index, result.error().count);
        std::abort();
    }

    auto aps = AmericanPriceSurface::create(
        result.value().surface, OptionType::PUT);
    if (!aps) {
        std::fprintf(stderr, "AmericanPriceSurface::create failed (scale=%d)\n", scale);
        std::abort();
    }
    auto solver = DefaultInterpolatedIVSolver::create(std::move(aps.value()));
    if (!solver) {
        std::fprintf(stderr, "InterpolatedIVSolver::create failed (scale=%d)\n", scale);
        std::abort();
    }

    auto t1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto [pos, _] = cache.emplace(scale, InterpSolverEntry{
        std::make_unique<DefaultInterpolatedIVSolver>(std::move(solver.value())),
        build_ms,
        result.value().n_pde_solves,
    });
    return pos->second;
}

static void BM_Interp_IV_Scaled(benchmark::State& state) {
    double K = kScaledStrikes[static_cast<size_t>(state.range(0))];
    int scale = kScales[static_cast<size_t>(state.range(1))];
    int ref_idx = scaled_strike_to_idx(K);
    double ref_price = get_ql_reference_prices()[ref_idx];

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kMaturity,
        .rate = kRate, .dividend_yield = kDivYield,
        .option_type = OptionType::PUT
    };
    IVQuery query(spec, ref_price);

    const auto& entry = get_interp_solver(scale);

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

    state.SetLabel(std::format("K={:.0f} {}x interp", K, scale));
    state.counters["strike"] = K;
    state.counters["scale"] = scale;
    state.counters["iv"] = iv;
    state.counters["iv_err_bps"] = iv_err_bps;
    state.counters["iters"] = static_cast<double>(last_result->iterations);
    state.counters["build_ms"] = entry.build_time_ms;
    state.counters["n_pde_solves"] = static_cast<double>(entry.n_pde_solves);
}

BENCHMARK(BM_Interp_IV_Scaled)
    ->ArgsProduct({
        benchmark::CreateDenseRange(0, static_cast<int>(kScaledStrikes.size()) - 1, 1),
        benchmark::CreateDenseRange(0, static_cast<int>(kScales.size()) - 1, 1),
    })
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Info
// ============================================================================

static void BM_Info(benchmark::State& state) {
    auto& prices = get_ql_reference_prices();
    for (auto _ : state) {}

    state.SetLabel(std::format("S={:.0f} σ={:.2f} T={:.1f} r={:.2f} q={:.2f}",
        kSpot, kTrueVol, kMaturity, kRate, kDivYield));
    state.counters["true_vol"] = kTrueVol;
    state.counters["n_strikes"] = static_cast<double>(kStrikes.size());
    state.counters["ref_K80"] = prices[0];
    state.counters["ref_K100"] = prices[4];
    state.counters["ref_K120"] = prices[8];
}
BENCHMARK(BM_Info)->Iterations(1);

BENCHMARK_MAIN();
