#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/bspline_price_table.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include "src/option/price_table_workspace.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mango;

namespace {

constexpr int kWarmupIterations = 5;
constexpr double kMinBenchmarkTimeSec = 2.0;

// ============================================================================
// Golden Values (QuantLib FDM with 200 space × 2000 time grid)
// ============================================================================
// These values provide accuracy validation for README benchmarks
// Tolerance: 0.1% relative error or $0.01 absolute error

// BM_README_AmericanSingle: ATM 1yr put (S=100, K=100, T=1.0, r=0.05, q=0.02, σ=0.20)
constexpr double GOLDEN_AMERICAN_SINGLE = 6.65996306;

// BM_README_AmericanBatch64 / Sequential: First 5 strikes (K = 90.0 + i*0.5, T=1.0)
// Used for validation sanity check (full batch has 64 strikes)
constexpr double GOLDEN_BATCH_SAMPLE[5] = {
    2.82114033,  // K=90.0
    3.62668720,  // K=90.5
    4.47960317,  // K=91.0
    5.37850090,  // K=91.5
    6.32175050   // K=92.0
};

// Validation tolerance
constexpr double GOLDEN_REL_TOL = 0.001;  // 0.1%
constexpr double GOLDEN_ABS_TOL = 0.01;   // $0.01

inline void validate_price(const char* benchmark_name, double computed, double expected) {
    double abs_error = std::abs(computed - expected);
    double rel_error = abs_error / expected;

    if (abs_error > GOLDEN_ABS_TOL && rel_error > GOLDEN_REL_TOL) {
        throw std::runtime_error(std::format(
            "{}: Price validation failed! Expected={:.6f}, Computed={:.6f}, "
            "AbsErr={:.6f} (tol={:.2f}), RelErr={:.3f}% (tol={:.1f}%)",
            benchmark_name, expected, computed, abs_error, GOLDEN_ABS_TOL,
            rel_error * 100.0, GOLDEN_REL_TOL * 100.0));
    }
}

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

        auto fitter_result = BSplineNDSeparable<double, 4>::create(
            std::array<std::vector<double>, 4>{
                fixture_ptr->m_grid,
                fixture_ptr->tau_grid,
                fixture_ptr->sigma_grid,
                fixture_ptr->rate_grid});

        if (!fitter_result.has_value()) {
            throw std::runtime_error("Failed to create BSpline fitter: " + fitter_result.error());
        }

        auto& fitter = fitter_result.value();

        auto fit_result = fitter.fit(prices, BSplineNDSeparableConfig<double>{.tolerance = 1e-6});
        if (!fit_result.has_value()) {
            throw std::runtime_error("Failed to fit analytic BSpline surface: " + fit_result.error());
        }

        auto workspace_result = PriceTableWorkspace::create(
            fixture_ptr->m_grid,
            fixture_ptr->tau_grid,
            fixture_ptr->sigma_grid,
            fixture_ptr->rate_grid,
            fit_result->coefficients,
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
        auto err = solver_result.error();
        throw std::runtime_error("Failed to create IV solver (error code " +
            std::to_string(static_cast<int>(err.code)) + ")");
    }
    const auto& solver = solver_result.value();

    constexpr double spot = 103.5;
    constexpr double strike = 100.0;
    constexpr double maturity = 1.0;
    constexpr double rate = 0.05;
    constexpr double sigma_true = 0.20;

    IVQuery query{
        spot,
        strike,
        maturity,
        rate,
        0.0,  // dividend_yield
        OptionType::PUT,
        analytic_bs_price(spot, strike, maturity, sigma_true, rate, OptionType::PUT)
    };

    auto run_once = [&]() {
        auto result = solver.solve_impl(query);
        if (!result.has_value()) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        benchmark::DoNotOptimize(result->implied_vol);
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
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.20    // volatility
    );

    // Use automatic grid estimation
    auto [grid_spec, n_time] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace) {
        throw std::runtime_error("Failed to create workspace: " + workspace.error());
    }

    auto run_once = [&]() {
        AmericanOptionSolver solver(params, workspace.value());
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
        return price;
    };

    // Warmup iterations
    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    // Validate accuracy against QuantLib golden value
    double validation_price = run_once();
    validate_price("BM_README_AmericanSingle", validation_price, GOLDEN_AMERICAN_SINGLE);

    for (auto _ : state) {
        run_once();
    }

    state.counters["n_space"] = static_cast<double>(n);
    state.counters["n_time"] = static_cast<double>(n_time);
    state.SetLabel(std::format("American (single, {}x{})", n, n_time));
}
BENCHMARK(BM_README_AmericanSingle)
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_AmericanSequential(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));

    std::vector<AmericanOptionParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double strike = 90.0 + i * 0.5;
        batch.push_back(AmericanOptionParams(
            100.0,  // spot
            strike, // strike
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::PUT,
            0.20    // volatility
        ));
    }

    auto run_once = [&]() -> double {
        double first_price = 0.0;
        // Sequential processing - no batch API
        for (size_t idx = 0; idx < batch.size(); ++idx) {
            const auto& params = batch[idx];
            auto [grid_spec, n_time] = estimate_grid_for_option(params);
            size_t n = grid_spec.n_points();
            std::pmr::synchronized_pool_resource pool;
            std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

            auto workspace = PDEWorkspace::from_buffer(buffer, n);
            if (!workspace) {
                throw std::runtime_error("Failed to create workspace");
            }

            AmericanOptionSolver solver(params, workspace.value());
            auto result = solver.solve();
            if (!result) {
                throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
            }
            double price = result->value_at(params.spot);
            benchmark::DoNotOptimize(price);

            // Capture first price for validation
            if (idx == 0) {
                first_price = price;
            }
        }
        return first_price;
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    // Validate first strike as sanity check
    double validation_price = run_once();
    validate_price("BM_README_AmericanSequential[K=90.0]", validation_price, GOLDEN_BATCH_SAMPLE[0]);

    for (auto _ : state) {
        run_once();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.SetLabel("American sequential (64 options)");
}
BENCHMARK(BM_README_AmericanSequential)
    ->Arg(64)
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_AmericanBatch64(benchmark::State& state) {
    const size_t batch_size = static_cast<size_t>(state.range(0));

    std::vector<AmericanOptionParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double strike = 90.0 + i * 0.5;
        batch.push_back(AmericanOptionParams(
            100.0,  // spot
            strike, // strike
            1.0,    // maturity
            0.05,   // rate
            0.02,   // dividend_yield
            OptionType::PUT,
            0.20    // volatility
        ));
    }

    BatchAmericanOptionSolver solver;

    auto run_once = [&]() -> double {
        // Use per-option grids with OpenMP parallelization
        auto batch_result = solver.solve_batch(batch, false);  // use_shared_grid=false (per-option grids)
        double first_price = 0.0;
        for (size_t idx = 0; idx < batch_result.results.size(); ++idx) {
            const auto& res = batch_result.results[idx];
            if (!res) {
                throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(res.error().code)));
            }
            // Capture first price for validation
            if (idx == 0) {
                first_price = res->value();
            }
        }
        benchmark::DoNotOptimize(batch_result);
        return first_price;
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    // Validate first strike as sanity check
    double validation_price = run_once();
    validate_price("BM_README_AmericanBatch64[K=90.0]", validation_price, GOLDEN_BATCH_SAMPLE[0]);

    for (auto _ : state) {
        run_once();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.SetLabel("American parallel batch (64 options)");
}
BENCHMARK(BM_README_AmericanBatch64)
    ->Arg(64)
    ->MinTime(kMinBenchmarkTimeSec);

static void BM_README_IV_FDM(benchmark::State& state) {
    const size_t n_space = static_cast<size_t>(state.range(0));
    const size_t n_time = static_cast<size_t>(state.range(1));

    IVQuery query{
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.0,    // dividend_yield
        OptionType::PUT,
        6.08    // market_price
    };

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    // Enable manual grid mode for benchmarking (bypass auto-estimation)
    config.use_manual_grid = true;
    config.grid_n_space = n_space;
    config.grid_n_time = n_time;
    config.grid_x_min = -3.0;
    config.grid_x_max = 3.0;
    config.grid_alpha = 2.0;

    IVSolverFDM solver(config);
    auto run_once = [&]() {
        auto result = solver.solve_impl(query);
        if (!result.has_value()) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        benchmark::DoNotOptimize(result->implied_vol);
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
    // Batch solver with normalized optimization (automatic when eligible)
    // Solve 5 strikes across 3 maturities using shared grid
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    std::vector<double> maturities = {0.25, 0.5, 1.0};
    double spot = 100.0;

    // Create option parameters (all puts with same vol/rate/maturity per group)
    std::vector<AmericanOptionParams> params;
    for (double tau : maturities) {
        for (double K : strikes) {
            params.emplace_back(
                spot,      // spot price
                K,         // strike
                tau,       // maturity
                0.05,      // rate
                0.02,      // dividend yield
                OptionType::PUT,
                0.20       // volatility
            );
        }
    }

    for (auto _ : state) {
        // Solve batch with shared grid (enables normalized optimization)
        BatchAmericanOptionSolver solver;
        auto result = solver.solve_batch(params, true);  // use_shared_grid=true

        // Extract prices (normalized path used automatically if eligible)
        for (const auto& opt_result : result.results) {
            if (opt_result.has_value()) {
                benchmark::DoNotOptimize(opt_result->value());
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
