#include "src/option/american_option.hpp"
#include "src/bspline/bspline_4d.hpp"
#include "src/bspline/bspline_fitter_4d.hpp"
#include "src/option/iv_solver.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/normalized_chain_solver.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_workspace.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mango;

namespace {

constexpr int kWarmupIterations = 5;
constexpr double kMinBenchmarkTimeSec = 2.0;

double analytic_bs_price(double S, double K, double tau, double sigma, double r, OptionType type) {
    if (tau <= 0.0) {
        return (type == OptionType::CALL) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }

    const double sqrt_tau = std::sqrt(tau);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau);
    const double d2 = d1 - sigma * sqrt_tau;

    auto Phi = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    if (type == OptionType::CALL) {
        return S * Phi(d1) - K * std::exp(-r * tau) * Phi(d2);
    }

    return K * std::exp(-r * tau) * Phi(-d2) - S * Phi(-d1);
}

struct AnalyticSurfaceFixture {
    double K_ref;
    std::vector<double> m_grid;
    std::vector<double> tau_grid;
    std::vector<double> sigma_grid;
    std::vector<double> rate_grid;
    std::shared_ptr<const BSpline4D> evaluator;
};

const AnalyticSurfaceFixture& GetAnalyticSurfaceFixture() {
    static AnalyticSurfaceFixture* fixture = [] {
        auto fixture_ptr = std::make_unique<AnalyticSurfaceFixture>();
        fixture_ptr->K_ref = 100.0;
        fixture_ptr->m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
        fixture_ptr->tau_grid = {0.1, 0.5, 1.0, 2.0};
        fixture_ptr->sigma_grid = {0.10, 0.15, 0.20, 0.25, 0.30};
        fixture_ptr->rate_grid = {0.0, 0.025, 0.05, 0.10};

        const size_t Nm = fixture_ptr->m_grid.size();
        const size_t Nt = fixture_ptr->tau_grid.size();
        const size_t Nv = fixture_ptr->sigma_grid.size();
        const size_t Nr = fixture_ptr->rate_grid.size();

        std::vector<double> prices(Nm * Nt * Nv * Nr);
        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                for (size_t k = 0; k < Nv; ++k) {
                    for (size_t l = 0; l < Nr; ++l) {
                        const size_t idx = ((i * Nt + j) * Nv + k) * Nr + l;
                        prices[idx] = analytic_bs_price(
                            fixture_ptr->m_grid[i] * fixture_ptr->K_ref,
                            fixture_ptr->K_ref,
                            fixture_ptr->tau_grid[j],
                            fixture_ptr->sigma_grid[k],
                            fixture_ptr->rate_grid[l],
                            OptionType::PUT);
                    }
                }
            }
        }

        auto fitter_result = BSplineFitter4D::create(
            fixture_ptr->m_grid,
            fixture_ptr->tau_grid,
            fixture_ptr->sigma_grid,
            fixture_ptr->rate_grid);

        if (!fitter_result.has_value()) {
            throw std::runtime_error("Failed to create BSplineFitter4D: " + fitter_result.error());
        }

        BSplineFitter4D& fitter = fitter_result.value();

        auto fit_result = fitter.fit(prices);
        if (!fit_result.success) {
            throw std::runtime_error("Failed to fit analytic BSpline surface: " + fit_result.error_message);
        }

        auto workspace_result = PriceTableWorkspace::create(
            fixture_ptr->m_grid,
            fixture_ptr->tau_grid,
            fixture_ptr->sigma_grid,
            fixture_ptr->rate_grid,
            fit_result.coefficients,
            fixture_ptr->K_ref,
            0.0);  // dividend_yield = 0

        if (!workspace_result.has_value()) {
            throw std::runtime_error("Failed to create workspace: " + workspace_result.error());
        }

        fixture_ptr->evaluator = std::make_shared<BSpline4D>(workspace_result.value());

        return fixture_ptr.release();
    }();

    return *fixture;
}

void RunAnalyticBSplineIVBenchmark(benchmark::State& state, const char* label) {
    const auto& surf = GetAnalyticSurfaceFixture();

    // Create solver using factory method
    auto solver_result = IVSolverInterpolated::create(
        surf.evaluator,
        surf.K_ref,
        {surf.m_grid.front(), surf.m_grid.back()},
        {surf.tau_grid.front(), surf.tau_grid.back()},
        {surf.sigma_grid.front(), surf.sigma_grid.back()},
        {surf.rate_grid.front(), surf.rate_grid.back()});

    if (!solver_result) {
        throw std::runtime_error("Failed to create IV solver: " + solver_result.error());
    }
    const auto& solver = solver_result.value();

    constexpr double spot = 103.5;
    constexpr double strike = 100.0;
    constexpr double maturity = 1.0;
    constexpr double rate = 0.05;
    constexpr double sigma_true = 0.20;

    OptionSpec spec{
        .spot = spot,
        .strike = strike,
        .maturity = maturity,
        .rate = rate,
        .dividend_yield = 0.0,
        .type = OptionType::PUT
    };
    IVQuery query{
        .option = spec,
        .market_price = analytic_bs_price(spot, strike, maturity, sigma_true, rate, OptionType::PUT)
    };

    auto run_once = [&]() {
        auto result = solver.solve(query);
        if (!result.converged) {
            throw std::runtime_error(
                result.failure_reason.value_or("Fast IV solver failed"));
        }
        benchmark::DoNotOptimize(result.implied_vol);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetLabel(label);
}

}  // namespace

// ============================================================================
// README Snapshot Benchmarks
// ============================================================================

static void BM_README_AmericanSingle(benchmark::State& state) {
    const size_t n_space = static_cast<size_t>(state.range(0));
    const size_t n_time = static_cast<size_t>(state.range(1));

    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    constexpr double x_min = -3.0;
    constexpr double x_max = 3.0;

    auto workspace = AmericanSolverWorkspace::create(x_min, x_max, n_space, n_time);
    if (!workspace) {
        throw std::runtime_error("Failed to create workspace: " + workspace.error());
    }

    auto run_once = [&]() {
        auto solver = AmericanOptionSolver::create(params, workspace.value());
        if (!solver) {
            throw std::runtime_error("Failed to create solver: " + solver.error());
        }
        auto result = solver.value().solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.counters["n_space"] = static_cast<double>(n_space);
    state.counters["n_time"] = static_cast<double>(n_time);
    if (n_space == 101) {
        state.SetLabel("American (single, 101x1k grid)");
    } else if (n_space == 501) {
        state.SetLabel("American (single, 501x5k grid)");
    } else {
        state.SetLabel("American (single)");
    }
}
BENCHMARK(BM_README_AmericanSingle)
    ->Args({101, 1000})
    ->Args({501, 5000})
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_AmericanBatch64(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));

    std::vector<AmericanOptionParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double strike = 90.0 + i * 0.5;
        batch.push_back(AmericanOptionParams{
            .strike = strike,
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        });
    }

    constexpr double x_min = -3.0;
    constexpr double x_max = 3.0;
    constexpr size_t n_space = 101;
    constexpr size_t n_time = 1000;

    auto run_once = [&]() {
        auto batch_result = solve_american_options_batch(batch, x_min, x_max, n_space, n_time);
        for (const auto& res : batch_result.results) {
            if (!res) {
                throw std::runtime_error(res.error().message);
            }
        }
        benchmark::DoNotOptimize(batch_result);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.SetLabel("American batch (64 options)");
}
BENCHMARK(BM_README_AmericanBatch64)
    ->Arg(64)
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_IV_FDM(benchmark::State& state) {
    const size_t n_space = static_cast<size_t>(state.range(0));
    const size_t n_time = static_cast<size_t>(state.range(1));

    IVParams params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.08,
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = n_space;
    config.grid_n_time = n_time;

    auto run_once = [&]() {
        IVSolver solver(params, config);
        auto result = solver.solve();
        if (!result.converged) {
            throw std::runtime_error(
                result.failure_reason.value_or("FDM IV solver failed"));
        }
        benchmark::DoNotOptimize(result.implied_vol);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.counters["n_space"] = static_cast<double>(n_space);
    state.counters["n_time"] = static_cast<double>(n_time);
    if (n_space == 101) {
        state.SetLabel("American IV (FDM, 101x1k grid)");
    } else if (n_space == 201) {
        state.SetLabel("American IV (FDM, 201x2k grid)");
    } else {
        state.SetLabel("American IV (FDM)");
    }
}
BENCHMARK(BM_README_IV_FDM)
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_IV_BSpline(benchmark::State& state) {
    RunAnalyticBSplineIVBenchmark(state, "American IV (B-spline)");
}
BENCHMARK(BM_README_IV_BSpline)->MinTime(kMinBenchmarkTimeSec);

static void BM_README_PriceTableInterpolation(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();
    const double spot = 103.5;
    const double moneyness = spot / surf.K_ref;
    const double maturity = 0.75;
    const double sigma = 0.22;
    const double rate = 0.02;

    auto run_once = [&]() {
        double price = surf.evaluator->eval(moneyness, maturity, sigma, rate);
        benchmark::DoNotOptimize(price);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetLabel("Price table interpolation");
}
BENCHMARK(BM_README_PriceTableInterpolation)->MinTime(kMinBenchmarkTimeSec);

static void BM_README_PriceTableGreeks(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();
    const double spot = 103.5;
    const double moneyness = spot / surf.K_ref;
    const double maturity = 1.0;
    const double sigma = 0.20;
    const double rate = 0.05;
    const double sigma_eps = 1e-4;
    const double m_eps = 5e-3;

    auto run_once = [&]() {
        const double base = surf.evaluator->eval(moneyness, maturity, sigma, rate);
        const double price_up_sigma = surf.evaluator->eval(moneyness, maturity, sigma + sigma_eps, rate);
        const double price_dn_sigma = surf.evaluator->eval(moneyness, maturity, sigma - sigma_eps, rate);
        double vega = (price_up_sigma - price_dn_sigma) / (2.0 * sigma_eps);

        const double price_up_m = surf.evaluator->eval(moneyness + m_eps, maturity, sigma, rate);
        const double price_dn_m = surf.evaluator->eval(moneyness - m_eps, maturity, sigma, rate);
        double gamma = (price_up_m - 2.0 * base + price_dn_m) / (m_eps * m_eps);

        benchmark::DoNotOptimize(vega);
        benchmark::DoNotOptimize(gamma);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetLabel("Greeks (vega, gamma)");
}
BENCHMARK(BM_README_PriceTableGreeks)->MinTime(kMinBenchmarkTimeSec);

// Benchmark: NormalizedChainSolver - solve once, price all strikes
static void BM_README_NormalizedChain(benchmark::State& state) {
    // Solve once for 5 strikes across 3 maturities
    std::vector<double> tau_snapshots_vec{0.25, 0.5, 1.0};

    NormalizedSolveRequest request{
        .sigma = 0.20,
        .rate = 0.05,
        .dividend = 0.02,
        .option_type = OptionType::PUT,
        .x_min = -2.5,
        .x_max = 2.5,
        .n_space = 101,
        .n_time = 1000,
        .T_max = 1.0,
        .tau_snapshots = std::span<const double>(tau_snapshots_vec)
    };

    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    double spot = 100.0;

    for (auto _ : state) {
        // Create workspace
        auto workspace_result = NormalizedWorkspace::create(request);
        if (!workspace_result) continue;
        auto workspace = std::move(workspace_result.value());
        auto surface = workspace.surface_view();

        // Solve PDE once
        auto solve_result = NormalizedChainSolver::solve(request, workspace, surface);
        if (!solve_result) continue;

        // Price all 15 options (5 strikes × 3 maturities)
        for (double K : strikes) {
            for (double tau : tau_snapshots_vec) {
                double x = std::log(spot / K);
                double u = surface.interpolate(x, tau);
                double price = K * u;
                benchmark::DoNotOptimize(price);
            }
        }
    }

    state.SetLabel("American option chain (5 strikes × 3 maturities)");
}
BENCHMARK(BM_README_NormalizedChain)->MinTime(kMinBenchmarkTimeSec);

class ReadmeSummaryReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        return ConsoleReporter::ReportContext(context);
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);

        for (const auto& run : reports) {
            if (run.report_big_o || run.report_rms || run.run_type != Run::RT_Iteration) {
                continue;
            }
            const std::string name = run.benchmark_name();
            if (name.find("BM_README_") == std::string::npos) {
                continue;
            }
            const double per_iter_in_unit = run.GetAdjustedRealTime();
            if (run.iterations == 0 || per_iter_in_unit <= 0.0) {
                continue;
            }
            const double seconds = per_iter_in_unit / benchmark::GetTimeUnitMultiplier(run.time_unit);
            const double avg_ns = seconds * 1e9;
            summaries_.push_back(SummaryEntry{
                .name = run.report_label.empty() ? name : run.report_label,
                .avg_ns = avg_ns});
        }
    }

    void Finalize() override {
        ConsoleReporter::Finalize();
        if (summaries_.empty()) {
            return;
        }

        std::size_t name_width = 0;
    for (const auto& entry : summaries_) {
        name_width = std::max(name_width, entry.name.size());
    }
    const int label_width = static_cast<int>(name_width);
        constexpr int number_width = 10;

        auto& out = GetOutputStream();
        out << "\nREADME Summary (avg time per op)\n";
        for (const auto& entry : summaries_) {
            const double avg_us = entry.avg_ns / 1e3;
            const double avg_ms = entry.avg_ns / 1e6;
            out << "  - "
                << std::left << std::setw(label_width) << entry.name
                << std::right << " : "
                << std::fixed << std::setprecision(2)
                << std::setw(number_width) << avg_ms << " ms ("
                << std::setw(number_width) << avg_us << " µs, "
                << std::setw(number_width + 2) << entry.avg_ns << " ns)\n";
        }
        out << std::defaultfloat;
    }

private:
    struct SummaryEntry {
        std::string name;
        double avg_ns;
    };

    std::vector<SummaryEntry> summaries_;
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    ReadmeSummaryReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    benchmark::Shutdown();
    return 0;
}
