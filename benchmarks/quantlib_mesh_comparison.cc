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

static constexpr double kStrike = 100.0;
static constexpr double kMaturity = 1.0;
static constexpr double kRate = 0.05;
static constexpr double kDivYield = 0.02;

struct BaseGrid {
    size_t nx;
    size_t nt;
    double x_min;
    double x_max;
    double alpha;
};

// 6 test cases: 3 moneyness × 2 vols.
// Averaging over cases washes out per-case error cancellation artifacts.
struct TestCase {
    double spot;
    double vol;
    const char* label;
};

static constexpr std::array<TestCase, 6> kVanillaCases = {{
    {90.0,  0.20, "ITM/lo"},
    {100.0, 0.20, "ATM/lo"},
    {110.0, 0.20, "OTM/lo"},
    {90.0,  0.30, "ITM/hi"},
    {100.0, 0.30, "ATM/hi"},
    {110.0, 0.30, "OTM/hi"},
}};

// ============================================================================
// QuantLib helper
// ============================================================================

// Fixed evaluation date for reproducible results across calendar days
static const ql::Date kEvalDate(1, ql::January, 2024);

static double price_ql(double spot, double vol, size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(kMaturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, kStrike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(spot));
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
// Solve mango vanilla at given case, scale, and alpha
// ============================================================================

static double solve_mango_vanilla(const TestCase& tc, const BaseGrid& base,
                                  int scale, double alpha) {
    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    PricingParams params(
        OptionSpec{.spot = tc.spot, .strike = kStrike, .maturity = kMaturity,
            .rate = kRate, .dividend_yield = kDivYield,
            .option_type = OptionType::PUT},
        tc.vol);

    auto grid_spec = GridSpec<double>::sinh_spaced(base.x_min, base.x_max, nx, alpha).value();
    PDEGridConfig grid_config{.grid_spec = grid_spec, .n_time = nt};

    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(nx), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, nx).value();

    auto solver = AmericanOptionSolver::create(params, workspace, grid_config).value();
    auto result = solver.solve();
    if (!result) throw std::runtime_error("Vanilla solver error");
    return result->value_at(tc.spot);
}

// ============================================================================
// Per-case base grids and reference prices (computed once)
// ============================================================================

struct VanillaCaseData {
    BaseGrid base;
    double ref_price;  // mango 64x self-convergence reference
};

// Equidistribution-optimal: alpha = 2*arcsinh(n_sigma/sqrt(2)), n_sigma=5
static const double kAlphaOpt = 2.0 * std::asinh(5.0 / std::sqrt(2.0));  // ≈ 3.95

/// Lazy-initialized per-case data: base grid from estimate_pde_grid, reference
/// from mango's own 64x solve (opt-alpha).
static const std::array<VanillaCaseData, 6>& get_vanilla_case_data() {
    static auto data = [] {
        std::array<VanillaCaseData, 6> result{};
        for (size_t i = 0; i < kVanillaCases.size(); ++i) {
            const auto& tc = kVanillaCases[i];
            PricingParams params(
                OptionSpec{.spot = tc.spot, .strike = kStrike, .maturity = kMaturity,
                    .rate = kRate, .dividend_yield = kDivYield,
                    .option_type = OptionType::PUT},
                tc.vol);
            auto [gs, td] = estimate_pde_grid(params);
            result[i].base = BaseGrid{gs.n_points(), td.n_steps(), gs.x_min(), gs.x_max(), kAlphaOpt};
            result[i].ref_price = solve_mango_vanilla(tc, result[i].base, 64, kAlphaOpt);
        }
        return result;
    }();
    return data;
}

// ============================================================================
// Mango benchmark: opt-alpha, 6-case average
// ============================================================================

static void BM_Mango_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    const auto& case_data = get_vanilla_case_data();

    std::array<double, 6> prices{};
    for (auto _ : state) {
        for (size_t i = 0; i < kVanillaCases.size(); ++i) {
            prices[i] = solve_mango_vanilla(kVanillaCases[i], case_data[i].base, scale, kAlphaOpt);
            benchmark::DoNotOptimize(prices[i]);
        }
    }

    double sum_rel_err_sq = 0.0;
    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < kVanillaCases.size(); ++i) {
        double rel_err = std::abs(prices[i] - case_data[i].ref_price) / case_data[i].ref_price;
        sum_rel_err_sq += rel_err * rel_err;
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_abs_err += std::abs(prices[i] - case_data[i].ref_price);
        state.counters[std::format("err_{}", kVanillaCases[i].label)] = rel_err;
    }

    double rms_rel_err = std::sqrt(sum_rel_err_sq / kVanillaCases.size());
    state.SetLabel(std::format("mango {}x (6 cases)", scale));
    state.counters["rms_rel_err"] = rms_rel_err;
    state.counters["max_rel_err"] = max_rel_err;
    state.counters["avg_abs_err"] = sum_abs_err / kVanillaCases.size();
}

BENCHMARK(BM_Mango_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// QuantLib benchmark: 6-case average
// ============================================================================

static void BM_QuantLib_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    const auto& case_data = get_vanilla_case_data();

    std::array<double, 6> prices{};
    for (auto _ : state) {
        for (size_t i = 0; i < kVanillaCases.size(); ++i) {
            const auto& base = case_data[i].base;
            size_t nx = base.nx * scale;
            if (nx % 2 == 0) nx++;
            size_t nt = base.nt * scale;
            prices[i] = price_ql(kVanillaCases[i].spot, kVanillaCases[i].vol, nx, nt);
            benchmark::DoNotOptimize(prices[i]);
        }
    }

    double sum_rel_err_sq = 0.0;
    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < kVanillaCases.size(); ++i) {
        double rel_err = std::abs(prices[i] - case_data[i].ref_price) / case_data[i].ref_price;
        sum_rel_err_sq += rel_err * rel_err;
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_abs_err += std::abs(prices[i] - case_data[i].ref_price);
        state.counters[std::format("err_{}", kVanillaCases[i].label)] = rel_err;
    }

    double rms_rel_err = std::sqrt(sum_rel_err_sq / kVanillaCases.size());
    state.SetLabel(std::format("QuantLib {}x (6 cases)", scale));
    state.counters["rms_rel_err"] = rms_rel_err;
    state.counters["max_rel_err"] = max_rel_err;
    state.counters["avg_abs_err"] = sum_abs_err / kVanillaCases.size();
}

BENCHMARK(BM_QuantLib_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Discrete dividend parameters
// ============================================================================

static constexpr double kDivStrike = 100.0;
static constexpr double kDivMaturity = 1.0;
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

// 6 test cases: 3 moneyness × 2 vols.
// Averaging over cases washes out per-case error cancellation artifacts.
struct DivTestCase {
    double spot;
    double vol;
    const char* label;
};

static constexpr std::array<DivTestCase, 6> kDivCases = {{
    {90.0,  0.20, "ITM/lo"},
    {100.0, 0.20, "ATM/lo"},
    {110.0, 0.20, "OTM/lo"},
    {90.0,  0.30, "ITM/hi"},
    {100.0, 0.30, "ATM/hi"},
    {110.0, 0.30, "OTM/hi"},
}};

// ============================================================================
// QuantLib helper with discrete dividends
// ============================================================================

static double price_ql_div(double spot, double vol, size_t grid_steps, size_t time_steps) {
    ql::Date today = kEvalDate;
    ql::Settings::instance().evaluationDate() = today;

    auto maturity_date = today + ql::Period(static_cast<int>(kDivMaturity * 365), ql::Days);

    auto exercise = ql::ext::make_shared<ql::AmericanExercise>(today, maturity_date);
    auto payoff = ql::ext::make_shared<ql::PlainVanillaPayoff>(ql::Option::Put, kDivStrike);
    ql::VanillaOption option(payoff, exercise);

    auto spot_h = ql::Handle<ql::Quote>(ql::ext::make_shared<ql::SimpleQuote>(spot));
    auto rate_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivRate, ql::Actual365Fixed()));
    auto div_h = ql::Handle<ql::YieldTermStructure>(
        ql::ext::make_shared<ql::FlatForward>(today, kDivContYield, ql::Actual365Fixed()));
    auto vol_h = ql::Handle<ql::BlackVolTermStructure>(
        ql::ext::make_shared<ql::BlackConstantVol>(today, ql::NullCalendar(), vol, ql::Actual365Fixed()));

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
// Solve mango with discrete dividends at given case and scale
// ============================================================================

static double solve_mango_div(const DivTestCase& tc, const BaseGrid& base, int scale) {
    size_t nx = base.nx * scale;
    if (nx % 2 == 0) nx++;
    size_t nt = base.nt * scale;

    std::vector<Dividend> divs;
    for (const auto& d : kDividends) {
        divs.push_back({.calendar_time = d.calendar_time, .amount = d.amount});
    }

    PricingParams params(
        OptionSpec{.spot = tc.spot, .strike = kDivStrike, .maturity = kDivMaturity,
            .rate = kDivRate, .dividend_yield = kDivContYield,
            .option_type = OptionType::PUT},
        tc.vol, divs);

    std::vector<double> mandatory_tau;
    for (const auto& d : kDividends) {
        double tau = kDivMaturity - d.calendar_time;
        if (tau > 0.0 && tau < kDivMaturity) {
            mandatory_tau.push_back(tau);
        }
    }

    auto grid_spec = GridSpec<double>::sinh_spaced(
        base.x_min, base.x_max, nx, base.alpha).value();
    PDEGridConfig grid_config{
        .grid_spec = grid_spec,
        .n_time = nt,
        .mandatory_times = mandatory_tau,
    };

    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(nx), &pool);
    auto workspace = PDEWorkspace::from_buffer(buffer, nx).value();

    auto solver = AmericanOptionSolver::create(params, workspace, grid_config).value();
    auto result = solver.solve();
    if (!result) throw std::runtime_error("Dividend solver error");
    return result->value_at(tc.spot);
}

// ============================================================================
// Per-case base grids and reference prices (computed once)
// ============================================================================

struct DivCaseData {
    BaseGrid base;
    double ref_price;  // mango 64x self-convergence reference
};

/// Lazy-initialized per-case data: base grid from estimate_pde_grid, reference
/// from mango's own 64x solve.  Using mango's own reference avoids conflating
/// the QuantLib–mango method difference with true convergence error.
static const std::array<DivCaseData, 6>& get_div_case_data() {
    static auto data = [] {
        std::array<DivCaseData, 6> result{};
        for (size_t i = 0; i < kDivCases.size(); ++i) {
            const auto& tc = kDivCases[i];
            std::vector<Dividend> divs;
            for (const auto& d : kDividends) {
                divs.push_back({.calendar_time = d.calendar_time, .amount = d.amount});
            }
            PricingParams params(
                OptionSpec{.spot = tc.spot, .strike = kDivStrike, .maturity = kDivMaturity,
                    .rate = kDivRate, .dividend_yield = kDivContYield,
                    .option_type = OptionType::PUT},
                tc.vol, divs);
            auto [gs, td] = estimate_pde_grid(params);
            result[i].base = BaseGrid{gs.n_points(), td.n_steps(), gs.x_min(), gs.x_max(), 2.0};
            result[i].ref_price = solve_mango_div(tc, result[i].base, 64);
        }
        return result;
    }();
    return data;
}

// ============================================================================
// Mango discrete dividend benchmark
// ============================================================================

static void BM_Mango_Div_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    const auto& case_data = get_div_case_data();

    std::array<double, 6> prices{};
    for (auto _ : state) {
        for (size_t i = 0; i < kDivCases.size(); ++i) {
            prices[i] = solve_mango_div(kDivCases[i], case_data[i].base, scale);
            benchmark::DoNotOptimize(prices[i]);
        }
    }

    double sum_rel_err_sq = 0.0;
    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < kDivCases.size(); ++i) {
        double rel_err = std::abs(prices[i] - case_data[i].ref_price) / case_data[i].ref_price;
        sum_rel_err_sq += rel_err * rel_err;
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_abs_err += std::abs(prices[i] - case_data[i].ref_price);
        state.counters[std::format("err_{}", kDivCases[i].label)] = rel_err;
    }

    double rms_rel_err = std::sqrt(sum_rel_err_sq / kDivCases.size());
    state.SetLabel(std::format("mango-div {}x (6 cases)", scale));
    state.counters["rms_rel_err"] = rms_rel_err;
    state.counters["max_rel_err"] = max_rel_err;
    state.counters["avg_abs_err"] = sum_abs_err / kDivCases.size();
}

BENCHMARK(BM_Mango_Div_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// QuantLib discrete dividend benchmark
// ============================================================================

static void BM_QuantLib_Div_MeshScale(benchmark::State& state) {
    int scale = static_cast<int>(state.range(0));
    const auto& case_data = get_div_case_data();

    std::array<double, 6> prices{};
    for (auto _ : state) {
        for (size_t i = 0; i < kDivCases.size(); ++i) {
            const auto& base = case_data[i].base;
            size_t nx = base.nx * scale;
            if (nx % 2 == 0) nx++;
            size_t nt = base.nt * scale;
            prices[i] = price_ql_div(kDivCases[i].spot, kDivCases[i].vol, nx, nt);
            benchmark::DoNotOptimize(prices[i]);
        }
    }

    double sum_rel_err_sq = 0.0;
    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < kDivCases.size(); ++i) {
        double rel_err = std::abs(prices[i] - case_data[i].ref_price) / case_data[i].ref_price;
        sum_rel_err_sq += rel_err * rel_err;
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_abs_err += std::abs(prices[i] - case_data[i].ref_price);
        state.counters[std::format("err_{}", kDivCases[i].label)] = rel_err;
    }

    double rms_rel_err = std::sqrt(sum_rel_err_sq / kDivCases.size());
    state.SetLabel(std::format("QuantLib-div {}x (6 cases)", scale));
    state.counters["rms_rel_err"] = rms_rel_err;
    state.counters["max_rel_err"] = max_rel_err;
    state.counters["avg_abs_err"] = sum_abs_err / kDivCases.size();
}

BENCHMARK(BM_QuantLib_Div_MeshScale)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Summary
// ============================================================================

static void BM_Info(benchmark::State& state) {
    const auto& case_data = get_vanilla_case_data();

    for (auto _ : state) {}

    state.SetLabel(std::format("6 vanilla cases"));
    for (size_t i = 0; i < kVanillaCases.size(); ++i) {
        const auto& tc = kVanillaCases[i];
        const auto& cd = case_data[i];
        state.counters[std::format("ref_{}", tc.label)] = cd.ref_price;
        state.counters[std::format("nx_{}", tc.label)] = static_cast<double>(cd.base.nx);
        state.counters[std::format("nt_{}", tc.label)] = static_cast<double>(cd.base.nt);
    }
}
BENCHMARK(BM_Info)->Iterations(1);

static void BM_Info_Div(benchmark::State& state) {
    const auto& case_data = get_div_case_data();

    for (auto _ : state) {}

    state.SetLabel(std::format("6 div cases"));
    for (size_t i = 0; i < kDivCases.size(); ++i) {
        const auto& tc = kDivCases[i];
        const auto& cd = case_data[i];
        state.counters[std::format("ref_{}", tc.label)] = cd.ref_price;
        state.counters[std::format("nx_{}", tc.label)] = static_cast<double>(cd.base.nx);
        state.counters[std::format("nt_{}", tc.label)] = static_cast<double>(cd.base.nt);
    }
}
BENCHMARK(BM_Info_Div)->Iterations(1);

BENCHMARK_MAIN();
