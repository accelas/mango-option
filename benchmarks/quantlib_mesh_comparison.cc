// SPDX-License-Identifier: MIT
/**
 * @file quantlib_mesh_comparison.cc
 * @brief Mesh-size convergence comparison between mango-option and QuantLib
 *
 * Prices an ATM American put at multiple grid resolutions:
 *   1x (auto-estimated), 2x, 4x, 8x
 * for both mango-option and QuantLib FdBlackScholesVanillaEngine.
 *
 * Reports both timing (Google Benchmark) and accuracy (vs high-res reference).
 *
 * Run with: bazel run //benchmarks:quantlib_mesh_comparison
 * Requires: libquantlib-dev (apt-get install libquantlib-dev)
 */

#include "mango/option/american_option.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include <benchmark/benchmark.h>
#include <array>
#include <cmath>
#include <format>
#include <memory_resource>
#include <vector>

// QuantLib includes
#include <ql/quantlib.hpp>

using namespace mango;
namespace ql = QuantLib;

// ============================================================================
// Shared test parameters
// ============================================================================

static constexpr double kSpot = 100.0;
static constexpr double kStrike = 100.0;
static constexpr double kMaturity = 1.0;
static constexpr double kVol = 0.20;
static constexpr double kRate = 0.05;
static constexpr double kDivYield = 0.02;

// ============================================================================
// QuantLib helper
// ============================================================================

// Fixed evaluation date for reproducible results across calendar days
static const ql::Date kEvalDate(1, ql::January, 2024);

static double price_ql(size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(kMaturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, kStrike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(kSpot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kRate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivYield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), kVol, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_h, div_h, rate_h, vol_h);

    option.setPricingEngine(
        ql::ext::make_shared<ql::FdBlackScholesVanillaEngine>(process, time_steps, grid_steps));

    return option.NPV();
}

// ============================================================================
// High-resolution reference price (computed once)
// ============================================================================

static double get_reference_price() {
    static double ref = price_ql(2001, 20000);
    return ref;
}

// ============================================================================
// Auto-estimated base grid dimensions (computed once)
// ============================================================================

struct BaseGrid {
    size_t nx;
    size_t nt;
    double x_min;
    double x_max;
    double alpha;
};

static BaseGrid get_base_grid() {
    static BaseGrid base = [] {
        PricingParams params(
            OptionSpec{.spot = kSpot, .strike = kStrike, .maturity = kMaturity,
                .rate = kRate, .dividend_yield = kDivYield,
                .option_type = OptionType::PUT},
            kVol);
        auto [gs, td] = estimate_pde_grid(params);
        return BaseGrid{gs.n_points(), td.n_steps(), gs.x_min(), gs.x_max(), 2.0};
    }();
    return base;
}

// ============================================================================
// Mango benchmark: parameterized by scale factor
// ============================================================================

static void BM_Mango_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    auto base = get_base_grid();

    size_t nx = base.nx * scale;
    // Ensure odd for centered stencils
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = kStrike, .maturity = kMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kVol);

    auto grid_spec = GridSpec<double>::sinh_spaced(base.x_min, base.x_max, nx, base.alpha).value();
    PDEGridConfig grid_config{.grid_spec = grid_spec, .n_time = nt};

    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(nx), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, nx).value();

    double price = 0.0;
    for (auto _ : state) {
        auto solver = AmericanOptionSolver::create(params, workspace, grid_config).value();
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error");
        }
        price = result->value_at(kSpot);
        benchmark::DoNotOptimize(price);
    }

    double ref = get_reference_price();
    double error = std::abs(price - ref);

    state.SetLabel(std::format("mango {}x{}", nx, nt));
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["abs_err"] = error;
    state.counters["ref"] = ref;
}

BENCHMARK(BM_Mango_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Mango with equidistribution-optimal sinh concentration
// ============================================================================

static void BM_Mango_Alpha4_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    auto base = get_base_grid();

    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    PricingParams params(
        OptionSpec{.spot = kSpot, .strike = kStrike, .maturity = kMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        kVol);

    // Equidistribution-optimal: alpha = 2*arcsinh(n_sigma/sqrt(2))
    double n_sigma = 5.0;
    double alpha_opt = 2.0 * std::asinh(n_sigma / std::sqrt(2.0));  // ≈ 3.95
    auto grid_spec = GridSpec<double>::sinh_spaced(base.x_min, base.x_max, nx, alpha_opt).value();
    PDEGridConfig grid_config{.grid_spec = grid_spec, .n_time = nt};

    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(nx), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, nx).value();

    double price = 0.0;
    for (auto _ : state) {
        auto solver = AmericanOptionSolver::create(params, workspace, grid_config).value();
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error");
        }
        price = result->value_at(kSpot);
        benchmark::DoNotOptimize(price);
    }

    double ref = get_reference_price();
    double error = std::abs(price - ref);

    state.SetLabel(std::format("mango-a4 {}x{}", nx, nt));
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["abs_err"] = error;
    state.counters["ref"] = ref;
}

BENCHMARK(BM_Mango_Alpha4_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// QuantLib benchmark: parameterized by scale factor
// ============================================================================

static void BM_QuantLib_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    auto base = get_base_grid();

    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    double price = 0.0;
    for (auto _ : state) {
        price = price_ql(nx, nt);
        benchmark::DoNotOptimize(price);
    }

    double ref = get_reference_price();
    double error = std::abs(price - ref);

    state.SetLabel(std::format("QuantLib {}x{}", nx, nt));
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["abs_err"] = error;
    state.counters["ref"] = ref;
}

BENCHMARK(BM_QuantLib_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Discrete dividend parameters
// ============================================================================

static constexpr double kDivSpot = 100.0;
static constexpr double kDivStrike = 100.0;
static constexpr double kDivMaturity = 1.0;
static constexpr double kDivVol = 0.25;
static constexpr double kDivRate = 0.05;
static constexpr double kDivContYield = 0.01;

struct DiscreteDividend {
    double calendar_time;
    double amount;
};

static constexpr std::array<DiscreteDividend, 3> kDividends = {{
    {0.25, 0.50},   // $0.50 at 3 months
    {0.50, 0.50},   // $0.50 at 6 months
    {0.75, 0.50},   // $0.50 at 9 months
}};

// ============================================================================
// QuantLib helper with discrete dividends
// ============================================================================

static double price_ql_div(size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(kDivMaturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, kDivStrike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(kDivSpot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivRate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivContYield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), kDivVol, ql::Actual365Fixed()));

    auto process = ql::ext::make_shared<ql::BlackScholesMertonProcess>(spot_h, div_h, rate_h, vol_h);

    // Discrete dividend dates and amounts
    std::vector<ql::Date> div_dates;
    std::vector<ql::Real> div_amounts;
    for (const auto& d : kDividends) {
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
// Discrete dividend reference price (computed once)
// ============================================================================

static double get_div_reference_price() {
    static double ref = price_ql_div(2001, 20000);
    return ref;
}

// ============================================================================
// Discrete dividend base grid (computed once)
// ============================================================================

static BaseGrid get_div_base_grid() {
    static BaseGrid base = [] {
        std::vector<Dividend> divs;
        for (const auto& d : kDividends) {
            divs.push_back({.calendar_time = d.calendar_time, .amount = d.amount});
        }
        PricingParams params(
            OptionSpec{.spot = kDivSpot, .strike = kDivStrike, .maturity = kDivMaturity,
                .rate = kDivRate, .dividend_yield = kDivContYield,
                .option_type = OptionType::PUT},
            kDivVol, divs);
        auto [gs, td] = estimate_pde_grid(params);
        return BaseGrid{gs.n_points(), td.n_steps(), gs.x_min(), gs.x_max(), 2.0};
    }();
    return base;
}

// ============================================================================
// Mango discrete dividend benchmark
// ============================================================================

static void BM_Mango_Div_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    auto base = get_div_base_grid();

    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    std::vector<Dividend> divs;
    for (const auto& d : kDividends) {
        divs.push_back({.calendar_time = d.calendar_time, .amount = d.amount});
    }

    PricingParams params(
        OptionSpec{.spot = kDivSpot, .strike = kDivStrike, .maturity = kDivMaturity,
            .rate = kDivRate, .dividend_yield = kDivContYield,
            .option_type = OptionType::PUT},
        kDivVol, divs);

    // Convert dividend calendar times to mandatory tau points
    std::vector<double> mandatory_tau;
    for (const auto& d : kDividends) {
        double tau = kDivMaturity - d.calendar_time;
        if (tau > 0.0 && tau < kDivMaturity) {
            mandatory_tau.push_back(tau);
        }
    }

    auto grid_spec = GridSpec<double>::sinh_spaced(base.x_min, base.x_max, nx, base.alpha).value();
    PDEGridConfig grid_config{
        .grid_spec = grid_spec,
        .n_time = nt,
        .mandatory_times = mandatory_tau,
    };

    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(nx), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, nx).value();

    double price = 0.0;
    for (auto _ : state) {
        auto solver = AmericanOptionSolver::create(params, workspace, grid_config).value();
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error");
        }
        price = result->value_at(kDivSpot);
        benchmark::DoNotOptimize(price);
    }

    double ref = get_div_reference_price();
    double error = std::abs(price - ref);

    state.SetLabel(std::format("mango-div {}x{}", nx, nt));
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["abs_err"] = error;
    state.counters["ref"] = ref;
}

BENCHMARK(BM_Mango_Div_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// QuantLib discrete dividend benchmark
// ============================================================================

static void BM_QuantLib_Div_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    auto base = get_div_base_grid();

    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    double price = 0.0;
    for (auto _ : state) {
        price = price_ql_div(nx, nt);
        benchmark::DoNotOptimize(price);
    }

    double ref = get_div_reference_price();
    double error = std::abs(price - ref);

    state.SetLabel(std::format("QuantLib-div {}x{}", nx, nt));
    state.counters["nx"] = static_cast<double>(nx);
    state.counters["nt"] = static_cast<double>(nt);
    state.counters["price"] = price;
    state.counters["abs_err"] = error;
    state.counters["ref"] = ref;
}

BENCHMARK(BM_QuantLib_Div_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Summary
// ============================================================================

static void BM_Info(benchmark::State& state) {
    auto base = get_base_grid();
    double ref = get_reference_price();

    for (auto _ : state) {}

    state.SetLabel(std::format("base={}x{} ref={:.6f}", base.nx, base.nt, ref));
    state.counters["base_nx"] = static_cast<double>(base.nx);
    state.counters["base_nt"] = static_cast<double>(base.nt);
    state.counters["ref_price"] = ref;
}
BENCHMARK(BM_Info)->Iterations(1);

static void BM_Info_Div(benchmark::State& state) {
    auto base = get_div_base_grid();
    double ref = get_div_reference_price();

    for (auto _ : state) {}

    state.SetLabel(std::format("div base={}x{} ref={:.6f}", base.nx, base.nt, ref));
    state.counters["base_nx"] = static_cast<double>(base.nx);
    state.counters["base_nt"] = static_cast<double>(base.nt);
    state.counters["ref_price"] = ref;
}
BENCHMARK(BM_Info_Div)->Iterations(1);

BENCHMARK_MAIN();
