// SPDX-License-Identifier: MIT
/**
 * @file iv_fdm_sweep.cc
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
 * Run with: bazel run //benchmarks:iv_fdm_sweep
 */

#include "iv_benchmark_common.hpp"
#include "iv_benchmark_ql.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <benchmark/benchmark.h>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <vector>

using namespace mango;
using namespace mango::bench;

// ============================================================================
// Test parameters
// ============================================================================

static constexpr std::array<double, 9> kStrikes = {
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0
};

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
                    kSpot, kStrikes[k], kVanillaScenarios[s].true_vol,
                    kVanillaScenarios[s].maturity, kRate, kDivYield, 2001, 20000);
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
                    kSpot, kStrikes[k], kVanillaScenarios[s].true_vol,
                    kVanillaScenarios[s].maturity, kRate, kDivYield, divs, 2001, 20000);
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
            p.push_back(price_ql(kSpot, K, kScaledVol, kScaledMaturity,
                                  kRate, kDivYield, 2001, 20000));
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
// Convenience IV solver wrappers
// ============================================================================

static double ql_solve_iv(double strike, double maturity, double target_price,
                           size_t nx, size_t nt) {
    return brent_solve_iv(
        [&](double vol) { return price_ql(kSpot, strike, vol, maturity, kRate, kDivYield, nx, nt); },
        target_price);
}

static double ql_solve_iv_div(double strike, double maturity, double target_price,
                               const std::vector<Dividend>& divs, size_t nx, size_t nt) {
    return brent_solve_iv(
        [&](double vol) { return price_ql_div(kSpot, strike, vol, maturity, kRate, kDivYield, divs, nx, nt); },
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
                cached_ivs[s * kNStrikes + k] = std::isfinite(iv)
                    ? iv : std::numeric_limits<double>::quiet_NaN();
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
                cached_ivs[s * kNStrikes + k] = std::isfinite(iv)
                    ? iv : std::numeric_limits<double>::quiet_NaN();
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
                cached_ivs[s * kNStrikes + k] = std::isfinite(iv)
                    ? iv : std::numeric_limits<double>::quiet_NaN();
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

    double iv_err_bps = std::isfinite(iv) ? std::abs(iv - kScaledVol) * 10000.0 : -1.0;

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
