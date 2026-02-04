// SPDX-License-Identifier: MIT
/**
 * @file iv_strike_sweep.cc
 * @brief IV accuracy across strikes: mango vs QuantLib
 *
 * Recovery test: price American puts at known σ across strikes,
 * then invert to IV and compare recovered vol accuracy.
 *
 * Multi-scenario averaging: 4 scenarios (2 vols × 2 maturities) × 9 strikes
 * to eliminate per-case error cancellation artifacts. Reports per-bucket
 * RMS (OTM/ATM/ITM) and overall RMS in basis points.
 *
 * Reference prices from QuantLib high-res (2001×20000).
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
#include <numeric>
#include <vector>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Test parameters
// ============================================================================

static constexpr double kSpot = 100.0;
static constexpr double kRate = 0.05;
static constexpr double kDivYield = 0.02;

static constexpr std::array<double, 9> kStrikes = {
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0
};

// Fixed evaluation date for reproducible results
static const ql::Date kEvalDate(1, ql::January, 2024);

// For scaled benchmarks (single fixed scenario)
static constexpr double kScaledVol = 0.20;
static constexpr double kScaledMaturity = 1.0;

// ============================================================================
// Scenario definitions
// ============================================================================

struct IVScenario {
    double true_vol;
    double maturity;
    const char* label;
};

static constexpr size_t kNStrikes = 9;
static constexpr size_t kNScenarios = 4;

static constexpr std::array<IVScenario, kNScenarios> kVanillaScenarios = {{
    {0.15, 1.0, "lo/med"},
    {0.15, 2.0, "lo/long"},
    {0.30, 1.0, "hi/med"},
    {0.30, 2.0, "hi/long"},
}};

// Dividend: quarterly $0.50 scaled to maturity
static std::vector<Dividend> make_div_schedule(double maturity) {
    return {
        Dividend{.calendar_time = maturity * 0.25, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.50, .amount = 0.50},
        Dividend{.calendar_time = maturity * 0.75, .amount = 0.50},
    };
}

// ============================================================================
// Moneyness bucket RMS computation
// ============================================================================
// OTM put: K ∈ {80, 85, 90}  (idx 0,1,2) — S/K > 1.05
// ATM:     K ∈ {95, 100, 105} (idx 3,4,5)
// ITM put: K ∈ {110, 115, 120} (idx 6,7,8) — S/K < 0.95

static double compute_bucket_rms(const std::array<double, kNScenarios * kNStrikes>& errors_bps,
                                  size_t strike_begin, size_t strike_count) {
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t s = 0; s < kNScenarios; ++s) {
        for (size_t k = strike_begin; k < strike_begin + strike_count; ++k) {
            double e = errors_bps[s * kNStrikes + k];
            if (std::isfinite(e)) {
                sum_sq += e * e;
                ++count;
            }
        }
    }
    return count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : -1.0;
}

static void report_iv_metrics(benchmark::State& state,
                               const std::array<double, kNScenarios * kNStrikes>& errors_bps,
                               const std::array<IVScenario, kNScenarios>& scenarios) {
    state.counters["overall_rms_bps"] = compute_bucket_rms(errors_bps, 0, kNStrikes);
    state.counters["otm_rms_bps"] = compute_bucket_rms(errors_bps, 0, 3);
    state.counters["atm_rms_bps"] = compute_bucket_rms(errors_bps, 3, 3);
    state.counters["itm_rms_bps"] = compute_bucket_rms(errors_bps, 6, 3);

    size_t n_failed = 0;
    for (size_t s = 0; s < kNScenarios; ++s) {
        double sum_sq = 0.0;
        size_t count = 0;
        for (size_t k = 0; k < kNStrikes; ++k) {
            double e = errors_bps[s * kNStrikes + k];
            if (std::isfinite(e)) {
                sum_sq += e * e;
                ++count;
            } else {
                ++n_failed;
            }
        }
        state.counters[std::format("rms_{}", scenarios[s].label)] =
            count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : -1.0;
    }
    state.counters["n_failed"] = static_cast<double>(n_failed);
}

// ============================================================================
// QuantLib pricing helper (parameterized by strike and vol)
// ============================================================================

static double price_ql(double strike, double vol, double maturity,
                       size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

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
// QuantLib pricing with discrete dividends
// ============================================================================

static double price_ql_div(double strike, double vol, double maturity,
                            const std::vector<Dividend>& divs,
                            size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(maturity * 365), ql::Days);

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

    std::vector<ql::Date> div_dates;
    std::vector<ql::Real> div_amounts;
    for (const auto& d : divs) {
        div_dates.push_back(today + ql::Period(static_cast<int>(d.calendar_time * 365), ql::Days));
        div_amounts.push_back(d.amount);
    }

    option.setPricingEngine(
        ql::MakeFdBlackScholesVanillaEngine(process)
            .withTGrid(time_steps)
            .withXGrid(grid_steps)
            .withCashDividends(div_dates, div_amounts));

    return option.NPV();
}

// ============================================================================
// Reference prices (QuantLib 2001×20000, computed once per scenario)
// ============================================================================

struct ScenarioData {
    std::array<double, kNStrikes> ref_prices;
};

static const std::array<ScenarioData, kNScenarios>& get_vanilla_scenario_data() {
    static auto data = [] {
        std::array<ScenarioData, kNScenarios> d;
        for (size_t s = 0; s < kNScenarios; ++s) {
            for (size_t k = 0; k < kNStrikes; ++k) {
                d[s].ref_prices[k] = price_ql(
                    kStrikes[k], kVanillaScenarios[s].true_vol,
                    kVanillaScenarios[s].maturity, 2001, 20000);
            }
        }
        return d;
    }();
    return data;
}

static const std::array<ScenarioData, kNScenarios>& get_div_scenario_data() {
    static auto data = [] {
        std::array<ScenarioData, kNScenarios> d;
        for (size_t s = 0; s < kNScenarios; ++s) {
            auto divs = make_div_schedule(kVanillaScenarios[s].maturity);
            for (size_t k = 0; k < kNStrikes; ++k) {
                d[s].ref_prices[k] = price_ql_div(
                    kStrikes[k], kVanillaScenarios[s].true_vol,
                    kVanillaScenarios[s].maturity, divs, 2001, 20000);
            }
        }
        return d;
    }();
    return data;
}

// Legacy reference for scaled benchmarks (σ=0.20, T=1.0)
static const std::vector<double>& get_scaled_reference_prices() {
    static std::vector<double> prices = [] {
        std::vector<double> p;
        p.reserve(kStrikes.size());
        for (double K : kStrikes) {
            p.push_back(price_ql(K, kScaledVol, kScaledMaturity, 2001, 20000));
        }
        return p;
    }();
    return prices;
}

// ============================================================================
// Base grid dimensions (mango auto-estimated)
// ============================================================================

struct BaseGrid {
    size_t nx;
    size_t nt;
};

static BaseGrid get_mango_base_grid(double strike, double maturity, double vol) {
    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = maturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        vol);
    auto [gs, td] = estimate_pde_grid(params);
    return {gs.n_points(), td.n_steps()};
}

static BaseGrid get_mango_base_grid_div(double strike, double maturity, double vol,
                                         const std::vector<Dividend>& divs) {
    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = maturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        vol, divs);
    auto [gs, td] = estimate_pde_grid(params);
    return {gs.n_points(), td.n_steps()};
}

// ============================================================================
// Generic Brent solver for IV recovery
// ============================================================================

template <typename PriceFn>
static double brent_solve_iv(PriceFn&& price_fn, double target_price) {
    double a = 0.01, b = 3.0;
    double fa = price_fn(a) - target_price;
    double fb = price_fn(b) - target_price;

    if (!std::isfinite(fa) || !std::isfinite(fb) || fa * fb > 0) return -1.0;

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

        double fs = price_fn(s) - target_price;
        if (!std::isfinite(fs)) return -1.0;

        d = c; c = b; fc = fb;
        if (fa * fs < 0.0) { b = s; fb = fs; }
        else { a = s; fa = fs; }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b); std::swap(fa, fb);
        }
    }
    return b;
}

// Convenience wrappers
static double ql_solve_iv(double strike, double maturity, double target_price,
                           size_t nx, size_t nt) {
    return brent_solve_iv(
        [&](double vol) { return price_ql(strike, vol, maturity, nx, nt); },
        target_price);
}

static double ql_solve_iv_div(double strike, double maturity, double target_price,
                               const std::vector<Dividend>& divs, size_t nx, size_t nt) {
    return brent_solve_iv(
        [&](double vol) { return price_ql_div(strike, vol, maturity, divs, nx, nt); },
        target_price);
}

static double mango_solve_iv_div(double strike, double maturity, double target_price,
                                  const std::vector<Dividend>& divs) {
    return brent_solve_iv(
        [&](double vol) -> double {
            PricingParams params(
                OptionSpec{.spot = kSpot, .strike = strike, .maturity = maturity,
                    .rate = kRate, .dividend_yield = kDivYield,
                    .option_type = OptionType::PUT},
                vol, divs);
            auto result = solve_american_option(params);
            if (!result) return std::numeric_limits<double>::quiet_NaN();
            return result->value_at(kSpot);
        },
        target_price);
}

// ============================================================================
// Multi-scenario vanilla IV benchmarks (4 scenarios × 9 strikes)
// ============================================================================

static void BM_Mango_IV(benchmark::State& state) {
    auto& scenario_data = get_vanilla_scenario_data();
    IVSolverConfig config;
    IVSolver solver(config);

    constexpr size_t N = kNScenarios * kNStrikes;
    std::array<double, N> cached_ivs{};

    for (auto _ : state) {
        for (size_t s = 0; s < kNScenarios; ++s) {
            const auto& sc = kVanillaScenarios[s];
            for (size_t k = 0; k < kNStrikes; ++k) {
                OptionSpec spec{
                    .spot = kSpot, .strike = kStrikes[k], .maturity = sc.maturity,
                    .rate = kRate, .dividend_yield = kDivYield,
                    .option_type = OptionType::PUT
                };
                IVQuery query(spec, scenario_data[s].ref_prices[k]);
                auto result = solver.solve(query);
                cached_ivs[s * kNStrikes + k] = result
                    ? result->implied_vol
                    : std::numeric_limits<double>::quiet_NaN();
            }
        }
        benchmark::DoNotOptimize(cached_ivs);
    }

    std::array<double, N> errors_bps{};
    for (size_t s = 0; s < kNScenarios; ++s) {
        for (size_t k = 0; k < kNStrikes; ++k) {
            double iv = cached_ivs[s * kNStrikes + k];
            errors_bps[s * kNStrikes + k] = std::isfinite(iv)
                ? std::abs(iv - kVanillaScenarios[s].true_vol) * 10000.0
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    report_iv_metrics(state, errors_bps, kVanillaScenarios);
}

BENCHMARK(BM_Mango_IV)->Unit(benchmark::kMillisecond);

static void BM_QuantLib_IV(benchmark::State& state) {
    auto& scenario_data = get_vanilla_scenario_data();

    // Pre-compute base grids outside timing loop
    std::array<BaseGrid, kNScenarios * kNStrikes> grids;
    for (size_t s = 0; s < kNScenarios; ++s) {
        const auto& sc = kVanillaScenarios[s];
        for (size_t k = 0; k < kNStrikes; ++k) {
            grids[s * kNStrikes + k] = get_mango_base_grid(kStrikes[k], sc.maturity, sc.true_vol);
        }
    }

    constexpr size_t N = kNScenarios * kNStrikes;
    std::array<double, N> cached_ivs{};

    for (auto _ : state) {
        for (size_t s = 0; s < kNScenarios; ++s) {
            const auto& sc = kVanillaScenarios[s];
            for (size_t k = 0; k < kNStrikes; ++k) {
                auto& base = grids[s * kNStrikes + k];
                double iv = ql_solve_iv(
                    kStrikes[k], sc.maturity, scenario_data[s].ref_prices[k], base.nx, base.nt);
                cached_ivs[s * kNStrikes + k] = (iv > 0) ? iv
                    : std::numeric_limits<double>::quiet_NaN();
            }
        }
        benchmark::DoNotOptimize(cached_ivs);
    }

    std::array<double, N> errors_bps{};
    for (size_t s = 0; s < kNScenarios; ++s) {
        for (size_t k = 0; k < kNStrikes; ++k) {
            double iv = cached_ivs[s * kNStrikes + k];
            errors_bps[s * kNStrikes + k] = std::isfinite(iv)
                ? std::abs(iv - kVanillaScenarios[s].true_vol) * 10000.0
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    report_iv_metrics(state, errors_bps, kVanillaScenarios);
}

BENCHMARK(BM_QuantLib_IV)->Unit(benchmark::kMillisecond);

// ============================================================================
// Multi-scenario dividend IV benchmarks (4 scenarios × 9 strikes)
// ============================================================================

static void BM_Mango_IV_Div(benchmark::State& state) {
    auto& scenario_data = get_div_scenario_data();

    constexpr size_t N = kNScenarios * kNStrikes;
    std::array<double, N> cached_ivs{};

    for (auto _ : state) {
        for (size_t s = 0; s < kNScenarios; ++s) {
            const auto& sc = kVanillaScenarios[s];
            auto divs = make_div_schedule(sc.maturity);
            for (size_t k = 0; k < kNStrikes; ++k) {
                double iv = mango_solve_iv_div(
                    kStrikes[k], sc.maturity, scenario_data[s].ref_prices[k], divs);
                cached_ivs[s * kNStrikes + k] = (iv > 0) ? iv
                    : std::numeric_limits<double>::quiet_NaN();
            }
        }
        benchmark::DoNotOptimize(cached_ivs);
    }

    std::array<double, N> errors_bps{};
    for (size_t s = 0; s < kNScenarios; ++s) {
        for (size_t k = 0; k < kNStrikes; ++k) {
            double iv = cached_ivs[s * kNStrikes + k];
            errors_bps[s * kNStrikes + k] = std::isfinite(iv)
                ? std::abs(iv - kVanillaScenarios[s].true_vol) * 10000.0
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    report_iv_metrics(state, errors_bps, kVanillaScenarios);
}

BENCHMARK(BM_Mango_IV_Div)->Unit(benchmark::kMillisecond);

static void BM_QuantLib_IV_Div(benchmark::State& state) {
    auto& scenario_data = get_div_scenario_data();

    // Pre-compute base grids outside timing loop
    std::array<BaseGrid, kNScenarios * kNStrikes> grids;
    for (size_t s = 0; s < kNScenarios; ++s) {
        const auto& sc = kVanillaScenarios[s];
        auto divs = make_div_schedule(sc.maturity);
        for (size_t k = 0; k < kNStrikes; ++k) {
            grids[s * kNStrikes + k] = get_mango_base_grid_div(
                kStrikes[k], sc.maturity, sc.true_vol, divs);
        }
    }

    constexpr size_t N = kNScenarios * kNStrikes;
    std::array<double, N> cached_ivs{};

    for (auto _ : state) {
        for (size_t s = 0; s < kNScenarios; ++s) {
            const auto& sc = kVanillaScenarios[s];
            auto divs = make_div_schedule(sc.maturity);
            for (size_t k = 0; k < kNStrikes; ++k) {
                auto& base = grids[s * kNStrikes + k];
                double iv = ql_solve_iv_div(
                    kStrikes[k], sc.maturity, scenario_data[s].ref_prices[k],
                    divs, base.nx, base.nt);
                cached_ivs[s * kNStrikes + k] = (iv > 0) ? iv
                    : std::numeric_limits<double>::quiet_NaN();
            }
        }
        benchmark::DoNotOptimize(cached_ivs);
    }

    std::array<double, N> errors_bps{};
    for (size_t s = 0; s < kNScenarios; ++s) {
        for (size_t k = 0; k < kNStrikes; ++k) {
            double iv = cached_ivs[s * kNStrikes + k];
            errors_bps[s * kNStrikes + k] = std::isfinite(iv)
                ? std::abs(iv - kVanillaScenarios[s].true_vol) * 10000.0
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    report_iv_metrics(state, errors_bps, kVanillaScenarios);
}

BENCHMARK(BM_QuantLib_IV_Div)->Unit(benchmark::kMillisecond);

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
        OptionSpec{.spot = kSpot, .strike = strike, .maturity = kScaledMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kScaledVol);
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
    double ref_price = get_scaled_reference_prices()[ref_idx];

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kScaledMaturity,
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
    double iv_err_bps = std::abs(iv - kScaledVol) * 10000.0;

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
    double ref_price = get_scaled_reference_prices()[ref_idx];

    auto base = get_mango_base_grid(K, kScaledMaturity, kScaledVol);
    size_t nx = base.nx * static_cast<size_t>(scale);
    size_t nt = base.nt * static_cast<size_t>(scale);

    double iv = 0;
    for (auto _ : state) {
        iv = ql_solve_iv(K, kScaledMaturity, ref_price, nx, nt);
        benchmark::DoNotOptimize(iv);
    }

    double iv_err_bps = (iv > 0) ? std::abs(iv - kScaledVol) * 10000.0 : -1.0;

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
    double ref_price = get_scaled_reference_prices()[ref_idx];

    OptionSpec spec{
        .spot = kSpot, .strike = K, .maturity = kScaledMaturity,
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
    double iv_err_bps = std::abs(iv - kScaledVol) * 10000.0;

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
    auto& vanilla = get_vanilla_scenario_data();
    auto& div = get_div_scenario_data();
    for (auto _ : state) {}

    state.SetLabel(std::format("S={:.0f} r={:.2f} q={:.2f} 4scen×9K", kSpot, kRate, kDivYield));
    state.counters["n_scenarios"] = kNScenarios;
    state.counters["n_strikes"] = kNStrikes;

    for (size_t s = 0; s < kNScenarios; ++s) {
        state.counters[std::format("van_{}_K100", kVanillaScenarios[s].label)] =
            vanilla[s].ref_prices[4];
        state.counters[std::format("div_{}_K100", kVanillaScenarios[s].label)] =
            div[s].ref_prices[4];
    }
}
BENCHMARK(BM_Info)->Iterations(1);

BENCHMARK_MAIN();
