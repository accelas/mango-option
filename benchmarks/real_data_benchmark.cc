// SPDX-License-Identifier: MIT
/**
 * @file real_data_benchmark.cc
 * @brief Benchmarks using real SPY option chain data from yfinance
 *
 * Mirrors the structure of readme_benchmarks.cc but uses real market data
 * instead of synthetic test cases.
 *
 * Data source: benchmarks/real_market_data.hpp (auto-generated)
 * Regenerate with: python scripts/download_benchmark_data.py SPY
 */

#include "benchmarks/real_market_data.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/option_grid.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_grid_estimator.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/math/black_scholes_analytics.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/option/table/price_table_surface.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <iomanip>
#include <memory>
#include <memory_resource>
#include <set>
#include <string>
#include <vector>

using namespace mango;
using namespace mango::benchmark_data;

namespace {

constexpr int kWarmupIterations = 5;
constexpr double kMinBenchmarkTimeSec = 2.0;

// ============================================================================
// Accuracy bucketing by moneyness
// ============================================================================

enum class MoneynessBucket : int { DeepOTM = 0, NearOTM, ATM, ITM, COUNT };

struct BucketStats {
    size_t count = 0;
    double sum_iv_err_bps = 0.0;
    double max_iv_err_bps = 0.0;
    double sum_price_err_sq = 0.0;
    double sum_vw_sq = 0.0;  // sum of (vega * iv_err)^2
    double sum_vega = 0.0;
};

// Classify put option: S/K > 1 means OTM for puts.
MoneynessBucket classify_put(double spot, double strike) {
    double m = spot / strike;
    if (m > 1.05) return MoneynessBucket::DeepOTM;
    if (m > 1.00) return MoneynessBucket::NearOTM;
    if (m >= 0.97) return MoneynessBucket::ATM;
    return MoneynessBucket::ITM;
}

const char* bucket_label(MoneynessBucket b) {
    switch (b) {
        case MoneynessBucket::DeepOTM: return "deep_otm";
        case MoneynessBucket::NearOTM: return "near_otm";
        case MoneynessBucket::ATM:     return "atm";
        case MoneynessBucket::ITM:     return "itm";
        default:                       return "unknown";
    }
}

void emit_bucket_counters(benchmark::State& state,
                          const std::array<BucketStats, 4>& buckets,
                          double global_sum_price_err_sq, size_t global_count) {
    for (int i = 0; i < static_cast<int>(MoneynessBucket::COUNT); ++i) {
        const auto& b = buckets[i];
        auto name = bucket_label(static_cast<MoneynessBucket>(i));
        state.counters[std::format("{}_n", name)] = static_cast<double>(b.count);
        if (b.count > 0) {
            state.counters[std::format("{}_avg_bps", name)] =
                b.sum_iv_err_bps / b.count;
            state.counters[std::format("{}_max_bps", name)] = b.max_iv_err_bps;
            state.counters[std::format("{}_price_rmse", name)] =
                std::sqrt(b.sum_price_err_sq / b.count);
            if (b.sum_vega > 0.0) {
                state.counters[std::format("{}_vw_bps", name)] =
                    std::sqrt(b.sum_vw_sq) / b.sum_vega;
            }
        }
    }
    if (global_count > 0) {
        state.counters["price_rmse"] =
            std::sqrt(global_sum_price_err_sq / global_count);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

PricingParams make_params(const RealOptionData& opt, double vol = 0.20) {
    return PricingParams(
        SPOT,
        opt.strike,
        opt.maturity,
        RISK_FREE_RATE,
        DIVIDEND_YIELD,
        opt.is_call ? OptionType::CALL : OptionType::PUT,
        vol
    );
}

// Analytic Black-Scholes for European options (used for B-spline surface)
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

// Fixture for B-spline interpolation surface
struct AnalyticSurfaceFixture {
    double K_ref;
    std::shared_ptr<const PriceTableSurface<4>> surface;
};

const AnalyticSurfaceFixture& GetAnalyticSurfaceFixture() {
    static AnalyticSurfaceFixture* fixture = [] {
        auto fixture_ptr = std::make_unique<AnalyticSurfaceFixture>();
        fixture_ptr->K_ref = SPOT;  // Use real spot as reference

        std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
        std::vector<double> tau_grid = {0.01, 0.05, 0.1, 0.25, 0.5};  // Short maturities like real data
        std::vector<double> vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30};
        std::vector<double> rate_grid = {0.0, 0.02, 0.04, 0.06};

        auto result = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, SPOT,
            GridAccuracyParams{}, OptionType::PUT, DIVIDEND_YIELD);
        if (!result) {
            throw std::runtime_error("Failed to create PriceTableBuilder");
        }
        auto [builder, axes] = std::move(result.value());
        auto table = builder.build(axes);
        if (!table) {
            throw std::runtime_error("Failed to build price table");
        }
        fixture_ptr->surface = table->surface;

        return fixture_ptr.release();
    }();

    return *fixture;
}

}  // namespace

// ============================================================================
// Real Data Benchmarks - Mirrors README benchmarks with real market data
// ============================================================================

// BM_RealData_AmericanSingle: Price ATM put from real market data
static void BM_RealData_AmericanSingle(benchmark::State& state) {
    auto params = make_params(ATM_PUT);

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace) {
        throw std::runtime_error("Failed to create workspace: " + workspace.error());
    }

    auto run_once = [&]() {
        auto solver = AmericanOptionSolver::create(params, workspace.value()).value();
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error");
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
        return price;
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.counters["n_space"] = static_cast<double>(n);
    state.counters["n_time"] = static_cast<double>(time_domain.n_steps());
    state.SetLabel(std::format("Real ATM put (K={:.0f}, T={:.3f})", ATM_PUT.strike, ATM_PUT.maturity));
}
BENCHMARK(BM_RealData_AmericanSingle)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_AmericanSequential: Sequential pricing of 64 puts
static void BM_RealData_AmericanSequential(benchmark::State& state) {
    const size_t batch_size = REAL_PUTS.size();

    std::vector<PricingParams> batch;
    batch.reserve(batch_size);
    for (const auto& opt : REAL_PUTS) {
        batch.push_back(make_params(opt));
    }

    auto run_once = [&]() {
        for (const auto& params : batch) {
            auto [grid_spec, time_domain] = estimate_grid_for_option(params);
            size_t n = grid_spec.n_points();
            std::pmr::synchronized_pool_resource pool;
            std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

            auto workspace = PDEWorkspace::from_buffer(buffer, n);
            if (!workspace) {
                throw std::runtime_error("Failed to create workspace");
            }

            auto solver = AmericanOptionSolver::create(params, workspace.value()).value();
            auto result = solver.solve();
            if (!result) {
                throw std::runtime_error("Solver error");
            }
            double price = result->value_at(params.spot);
            benchmark::DoNotOptimize(price);
        }
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.SetLabel(std::format("Real data sequential ({} options)", batch_size));
}
BENCHMARK(BM_RealData_AmericanSequential)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_AmericanBatch: Parallel batch pricing of 64 puts
static void BM_RealData_AmericanBatch(benchmark::State& state) {
    const size_t batch_size = REAL_PUTS.size();

    std::vector<PricingParams> batch;
    batch.reserve(batch_size);
    for (const auto& opt : REAL_PUTS) {
        batch.push_back(make_params(opt));
    }

    BatchAmericanOptionSolver solver;

    auto run_once = [&]() {
        auto batch_result = solver.solve_batch(batch, false);  // per-option grids
        for (const auto& res : batch_result.results) {
            if (!res) {
                throw std::runtime_error("Solver error");
            }
            benchmark::DoNotOptimize(res->value());
        }
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.counters["batch"] = static_cast<double>(batch_size);
    state.SetLabel(std::format("Real data parallel batch ({} options)", batch_size));
}
BENCHMARK(BM_RealData_AmericanBatch)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_IV_FDM: IV calculation using FDM solver
static void BM_RealData_IV_FDM(benchmark::State& state) {
    const auto& opt = ATM_PUT;

    IVQuery query{
        SPOT,
        opt.strike,
        opt.maturity,
        RISK_FREE_RATE,
        DIVIDEND_YIELD,
        opt.is_call ? OptionType::CALL : OptionType::PUT,
        opt.market_price
    };

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;

    IVSolverFDM solver(config);

    auto run_once = [&]() {
        auto result = solver.solve(query);
        if (!result.has_value()) {
            throw std::runtime_error("IV solver failed");
        }
        benchmark::DoNotOptimize(result->implied_vol);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetLabel(std::format("Real IV (FDM, K={:.0f})", opt.strike));
}
BENCHMARK(BM_RealData_IV_FDM)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_IV_BSpline: IV calculation using B-spline interpolation
static void BM_RealData_IV_BSpline(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();

    auto aps = AmericanPriceSurface::create(surf.surface, OptionType::PUT);
    if (!aps) {
        throw std::runtime_error("Failed to create AmericanPriceSurface");
    }
    auto solver_result = IVSolverInterpolatedStandard::create(std::move(*aps));
    if (!solver_result) {
        throw std::runtime_error("Failed to create IV solver");
    }
    const auto& solver = solver_result.value();

    // Query near ATM
    const double spot = SPOT;
    const double strike = SPOT;  // ATM
    constexpr double maturity = 0.1;
    constexpr double sigma_true = 0.20;

    IVQuery query{
        spot,
        strike,
        maturity,
        RISK_FREE_RATE,
        DIVIDEND_YIELD,
        OptionType::PUT,
        analytic_bs_price(spot, strike, maturity, sigma_true, RISK_FREE_RATE, OptionType::PUT)
    };

    auto run_once = [&]() {
        auto result = solver.solve(query);
        if (!result.has_value()) {
            throw std::runtime_error("Solver error");
        }
        benchmark::DoNotOptimize(result->implied_vol);
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        run_once();
    }

    for (auto _ : state) {
        run_once();
    }

    state.SetLabel("Real IV (B-spline)");
}
BENCHMARK(BM_RealData_IV_BSpline)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_PriceTableInterpolation: Price table lookup
static void BM_RealData_PriceTableInterpolation(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();

    // Query near real spot
    const double moneyness = 1.0;  // ATM
    constexpr double maturity = 0.1;
    constexpr double sigma = 0.20;
    const double rate = RISK_FREE_RATE;

    auto run_once = [&]() {
        double price = surf.surface->value({moneyness, maturity, sigma, rate});
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
BENCHMARK(BM_RealData_PriceTableInterpolation)
    ->MinTime(kMinBenchmarkTimeSec);

// BM_RealData_PriceTableGreeks: Vega and gamma via finite differences
static void BM_RealData_PriceTableGreeks(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();

    const double moneyness = 1.0;
    constexpr double maturity = 0.1;
    constexpr double sigma = 0.20;
    const double rate = RISK_FREE_RATE;
    constexpr double sigma_eps = 1e-4;
    constexpr double m_eps = 5e-3;

    auto run_once = [&]() {
        const double base = surf.surface->value({moneyness, maturity, sigma, rate});
        const double price_up_sigma = surf.surface->value({moneyness, maturity, sigma + sigma_eps, rate});
        const double price_dn_sigma = surf.surface->value({moneyness, maturity, sigma - sigma_eps, rate});
        double vega = (price_up_sigma - price_dn_sigma) / (2.0 * sigma_eps);

        const double price_up_m = surf.surface->value({moneyness + m_eps, maturity, sigma, rate});
        const double price_dn_m = surf.surface->value({moneyness - m_eps, maturity, sigma, rate});
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
BENCHMARK(BM_RealData_PriceTableGreeks)
    ->MinTime(kMinBenchmarkTimeSec);

// Helper to build option chain from real market data for IV smile benchmarks
struct IVSmileFixture {
    OptionGrid chain;
    std::vector<RealOptionData> smile_options;
    double target_maturity;
    GridSpec<double> grid_spec;

    IVSmileFixture()
        : grid_spec(GridSpec<double>::uniform(-3.0, 3.0, 101).value())
    {
        // Find the most common maturity (first maturity with multiple strikes)
        target_maturity = REAL_PUTS[0].maturity;
        for (const auto& opt : REAL_PUTS) {
            if (std::abs(opt.maturity - target_maturity) < 0.001) {
                smile_options.push_back(opt);
            }
        }

        // Build OptionGrid structure
        chain.ticker = SYMBOL;
        chain.spot = SPOT;
        chain.dividend_yield = DIVIDEND_YIELD;

        // Extract unique strikes
        std::set<double> unique_strikes;
        for (const auto& opt : smile_options) {
            unique_strikes.insert(opt.strike);
        }
        chain.strikes.assign(unique_strikes.begin(), unique_strikes.end());

        // Maturity grid needs at least 4 points for B-spline fitting
        chain.maturities = {
            target_maturity * 0.5,
            target_maturity * 0.75,
            target_maturity,
            target_maturity * 1.5,
            target_maturity * 2.0
        };

        // Volatility grid covering typical range (needs at least 4 points)
        chain.implied_vols = {0.10, 0.15, 0.20, 0.25, 0.30, 0.40};

        // Rate grid (needs at least 4 points)
        chain.rates = {0.02, 0.03, RISK_FREE_RATE, 0.05};
    }
};

const IVSmileFixture& GetIVSmileFixture() {
    static IVSmileFixture fixture;
    return fixture;
}

// Stage 1: Benchmark building the price table from real option chain
static void BM_RealData_IVSmile_BuildTable(benchmark::State& state) {
    const auto& fixture = GetIVSmileFixture();

    size_t n_pde_solves = 0;

    for (auto _ : state) {
        auto builder_result = PriceTableBuilder<4>::from_grid(
            fixture.chain,
            ExplicitPDEGrid{fixture.grid_spec, 200},
            OptionType::PUT,
            0.1   // allow 10% failure rate
        );

        if (!builder_result) {
            throw std::runtime_error("Failed to create price table builder");
        }

        auto& [builder, axes] = builder_result.value();

        auto table_result = builder.build(axes);
        if (!table_result) {
            throw std::runtime_error("Failed to build price table");
        }

        benchmark::DoNotOptimize(table_result->surface);
        n_pde_solves = table_result->n_pde_solves;
    }

    state.counters["n_strikes"] = static_cast<double>(fixture.chain.strikes.size());
    state.counters["n_maturities"] = static_cast<double>(fixture.chain.maturities.size());
    state.counters["n_vols"] = static_cast<double>(fixture.chain.implied_vols.size());
    state.counters["pde_solves"] = static_cast<double>(n_pde_solves);
    state.SetLabel(std::format("Build table ({} strikes × {} τ × {} σ)",
        fixture.chain.strikes.size(),
        fixture.chain.maturities.size(),
        fixture.chain.implied_vols.size()));
}
BENCHMARK(BM_RealData_IVSmile_BuildTable)
    ->MinTime(kMinBenchmarkTimeSec);

// Stage 2: Benchmark IV smile query speed (table pre-built)
static void BM_RealData_IVSmile_Query(benchmark::State& state) {
    const auto& fixture = GetIVSmileFixture();

    // Build price table once (not timed)
    auto builder_result = PriceTableBuilder<4>::from_grid(
        fixture.chain,
        ExplicitPDEGrid{fixture.grid_spec, 200},
        OptionType::PUT,
        0.1
    );
    if (!builder_result) {
        throw std::runtime_error("Failed to create price table builder");
    }

    auto& [builder, axes] = builder_result.value();
    auto table_result = builder.build(axes);
    if (!table_result) {
        throw std::runtime_error("Failed to build price table");
    }

    // Create IV solver
    auto aps_query = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
    if (!aps_query) {
        throw std::runtime_error("Failed to create AmericanPriceSurface");
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps_query));
    if (!iv_solver_result) {
        throw std::runtime_error("Failed to create IV solver");
    }
    const auto& iv_solver = iv_solver_result.value();

    // Prepare IV queries for each strike (the IV smile)
    std::vector<IVQuery> queries;
    queries.reserve(fixture.smile_options.size());
    for (const auto& opt : fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }

    // Warmup
    for (int i = 0; i < kWarmupIterations; ++i) {
        for (const auto& query : queries) {
            auto result = iv_solver.solve(query);
            benchmark::DoNotOptimize(result);
        }
    }

    // Benchmark: Calculate IV for all strikes (the smile)
    for (auto _ : state) {
        for (const auto& query : queries) {
            auto result = iv_solver.solve(query);
            if (result.has_value()) {
                benchmark::DoNotOptimize(result->implied_vol);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * queries.size());
    state.counters["n_queries"] = static_cast<double>(queries.size());
    state.SetLabel(std::format("Query IV smile ({} strikes, T={:.3f})",
        queries.size(), fixture.target_maturity));
}
BENCHMARK(BM_RealData_IVSmile_Query)
    ->MinTime(kMinBenchmarkTimeSec);

// Stage 3: Compare IV smile accuracy - Interpolated vs FDM
static void BM_RealData_IVSmile_FDM(benchmark::State& state) {
    const auto& fixture = GetIVSmileFixture();

    // Prepare IV queries for each strike
    std::vector<IVQuery> queries;
    queries.reserve(fixture.smile_options.size());
    for (const auto& opt : fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    IVSolverFDM fdm_solver(config);

    // Benchmark FDM IV calculation for entire smile
    for (auto _ : state) {
        for (const auto& query : queries) {
            auto result = fdm_solver.solve(query);
            if (result.has_value()) {
                benchmark::DoNotOptimize(result->implied_vol);
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * queries.size());
    state.counters["n_queries"] = static_cast<double>(queries.size());
    state.SetLabel(std::format("FDM IV smile ({} strikes)", queries.size()));
}
BENCHMARK(BM_RealData_IVSmile_FDM)
    ->MinTime(kMinBenchmarkTimeSec);

// Stage 4: Compare IV smile accuracy between Interpolated and FDM
static void BM_RealData_IVSmile_Accuracy(benchmark::State& state) {
    const auto& fixture = GetIVSmileFixture();

    // Build price table
    auto builder_result = PriceTableBuilder<4>::from_grid(
        fixture.chain,
        ExplicitPDEGrid{fixture.grid_spec, 200},
        OptionType::PUT,
        0.1
    );
    if (!builder_result) {
        throw std::runtime_error("Failed to create price table builder");
    }

    auto& [builder, axes] = builder_result.value();
    auto table_result = builder.build(axes);
    if (!table_result) {
        throw std::runtime_error("Failed to build price table");
    }

    // Create interpolated IV solver
    auto aps_acc = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
    if (!aps_acc) {
        throw std::runtime_error("Failed to create AmericanPriceSurface");
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps_acc));
    if (!iv_solver_result) {
        throw std::runtime_error("Failed to create IV solver");
    }
    const auto& interp_solver = iv_solver_result.value();

    // Create FDM IV solver
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    IVSolverFDM fdm_solver(config);

    // Prepare IV queries
    std::vector<IVQuery> queries;
    queries.reserve(fixture.smile_options.size());
    for (const auto& opt : fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }

    // Track accuracy metrics
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    size_t valid_count = 0;

    for (auto _ : state) {
        max_abs_error = 0.0;
        sum_abs_error = 0.0;
        sum_rel_error = 0.0;
        valid_count = 0;

        for (const auto& query : queries) {
            auto fdm_result = fdm_solver.solve(query);
            auto interp_result = interp_solver.solve(query);

            if (fdm_result.has_value() && interp_result.has_value()) {
                double fdm_iv = fdm_result->implied_vol;
                double interp_iv = interp_result->implied_vol;
                double abs_error = std::abs(fdm_iv - interp_iv);
                double rel_error = abs_error / fdm_iv;

                max_abs_error = std::max(max_abs_error, abs_error);
                sum_abs_error += abs_error;
                sum_rel_error += rel_error;
                valid_count++;
            }
        }

        benchmark::DoNotOptimize(max_abs_error);
    }

    if (valid_count > 0) {
        state.counters["max_abs_err_bps"] = max_abs_error * 10000.0;  // basis points
        state.counters["avg_abs_err_bps"] = (sum_abs_error / valid_count) * 10000.0;
        state.counters["avg_rel_err_pct"] = (sum_rel_error / valid_count) * 100.0;
        state.counters["valid_count"] = static_cast<double>(valid_count);
    }

    state.SetLabel(std::format("Accuracy: max={:.1f}bps, avg={:.1f}bps ({} options)",
        max_abs_error * 10000.0,
        (valid_count > 0 ? sum_abs_error / valid_count : 0.0) * 10000.0,
        valid_count));
}
BENCHMARK(BM_RealData_IVSmile_Accuracy)
    ->MinTime(kMinBenchmarkTimeSec);

// Helper to create dense grid fixture with configurable resolution
struct DenseGridFixture {
    OptionGrid chain;
    std::vector<RealOptionData> smile_options;
    double target_maturity;
    GridSpec<double> grid_spec;
    size_t n_maturities;
    size_t n_vols;
    size_t n_rates;

    DenseGridFixture(size_t n_mat, size_t n_vol, size_t n_rate)
        : grid_spec(GridSpec<double>::uniform(-3.0, 3.0, 101).value())
        , n_maturities(n_mat)
        , n_vols(n_vol)
        , n_rates(n_rate)
    {
        // Find the most common maturity
        target_maturity = REAL_PUTS[0].maturity;
        for (const auto& opt : REAL_PUTS) {
            if (std::abs(opt.maturity - target_maturity) < 0.001) {
                smile_options.push_back(opt);
            }
        }

        chain.ticker = SYMBOL;
        chain.spot = SPOT;
        chain.dividend_yield = DIVIDEND_YIELD;

        // Extract unique strikes
        std::set<double> unique_strikes;
        for (const auto& opt : smile_options) {
            unique_strikes.insert(opt.strike);
        }
        chain.strikes.assign(unique_strikes.begin(), unique_strikes.end());

        // Build maturity grid centered around target
        chain.maturities.clear();
        double mat_min = target_maturity * 0.5;
        double mat_max = target_maturity * 2.0;
        for (size_t i = 0; i < n_mat; ++i) {
            double t = mat_min + (mat_max - mat_min) * i / (n_mat - 1);
            chain.maturities.push_back(t);
        }

        // Build volatility grid (uniform spacing for better accuracy)
        chain.implied_vols.clear();
        double vol_min = 0.05;
        double vol_max = 0.50;
        for (size_t i = 0; i < n_vol; ++i) {
            double v = vol_min + (vol_max - vol_min) * i / (n_vol - 1);
            chain.implied_vols.push_back(v);
        }

        // Build rate grid
        chain.rates.clear();
        double rate_min = 0.01;
        double rate_max = 0.06;
        for (size_t i = 0; i < n_rate; ++i) {
            double r = rate_min + (rate_max - rate_min) * i / (n_rate - 1);
            chain.rates.push_back(r);
        }
    }
};

// Parameterized benchmark to find optimal grid density for 1 bps accuracy
static void BM_RealData_GridDensity(benchmark::State& state) {
    const size_t n_maturities = state.range(0);
    const size_t n_vols = state.range(1);
    const size_t n_rates = state.range(2);

    DenseGridFixture fixture(n_maturities, n_vols, n_rates);

    // Build price table
    auto builder_result = PriceTableBuilder<4>::from_grid(
        fixture.chain,
        ExplicitPDEGrid{fixture.grid_spec, 200},
        OptionType::PUT,
        0.1
    );
    if (!builder_result) {
        state.SkipWithError("Failed to create price table builder");
        return;
    }

    auto& [builder, axes] = builder_result.value();
    auto table_result = builder.build(axes);
    if (!table_result) {
        state.SkipWithError("Failed to build price table");
        return;
    }

    // Create interpolated IV solver
    auto aps_dense = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
    if (!aps_dense) {
        state.SkipWithError("Failed to create AmericanPriceSurface");
        return;
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps_dense));
    if (!iv_solver_result) {
        state.SkipWithError("Failed to create IV solver");
        return;
    }
    const auto& interp_solver = iv_solver_result.value();

    // Create FDM IV solver (ground truth)
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    IVSolverFDM fdm_solver(config);

    // Prepare IV queries
    std::vector<IVQuery> queries;
    for (const auto& opt : fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }

    // Track accuracy
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    size_t valid_count = 0;

    for (auto _ : state) {
        max_abs_error = 0.0;
        sum_abs_error = 0.0;
        valid_count = 0;

        for (const auto& query : queries) {
            auto fdm_result = fdm_solver.solve(query);
            auto interp_result = interp_solver.solve(query);

            if (fdm_result.has_value() && interp_result.has_value()) {
                double abs_error = std::abs(fdm_result->implied_vol - interp_result->implied_vol);
                max_abs_error = std::max(max_abs_error, abs_error);
                sum_abs_error += abs_error;
                valid_count++;
            }
        }
        benchmark::DoNotOptimize(max_abs_error);
    }

    size_t n_pde_solves = table_result->n_pde_solves;

    state.counters["pde_solves"] = static_cast<double>(n_pde_solves);
    state.counters["max_err_bps"] = max_abs_error * 10000.0;
    state.counters["avg_err_bps"] = (valid_count > 0 ? sum_abs_error / valid_count : 0.0) * 10000.0;
    state.counters["valid"] = static_cast<double>(valid_count);

    state.SetLabel(std::format("{}×{}×{} grid, {} solves, max={:.1f}bps, avg={:.1f}bps",
        n_maturities, n_vols, n_rates,
        n_pde_solves,
        max_abs_error * 10000.0,
        (valid_count > 0 ? sum_abs_error / valid_count : 0.0) * 10000.0));
}

// Test different grid densities: (maturities, vols, rates)
BENCHMARK(BM_RealData_GridDensity)
    ->Args({5, 6, 4})      // Current: sparse
    ->Args({9, 12, 6})     // Dense
    ->Args({13, 18, 8})    // Extra dense
    ->Args({17, 25, 10})   // Very dense - targeting ~1 bps
    ->Args({21, 31, 12})   // Ultra dense
    ->MinTime(kMinBenchmarkTimeSec);

// ============================================================================
// Automatic Grid Estimation Validation
// ============================================================================

// Test from_grid_auto with different target IV errors
static void BM_RealData_GridEstimator(benchmark::State& state) {
    const double target_error_bps = static_cast<double>(state.range(0));
    const double target_iv_error = target_error_bps / 10000.0;

    // Create a fixture with real market data
    const auto& iv_fixture = GetIVSmileFixture();

    // Configure accuracy parameters
    PriceTableGridAccuracyParams<4> accuracy;
    accuracy.target_iv_error = target_iv_error;

    // Build price table with automatic grid estimation
    auto builder_result = PriceTableBuilder<4>::from_grid_auto(
        iv_fixture.chain,
        ExplicitPDEGrid{GridSpec<double>::uniform(-3.0, 3.0, 101).value(), 200},
        OptionType::PUT,
        accuracy
    );
    if (!builder_result) {
        state.SkipWithError("Failed to create price table builder");
        return;
    }

    auto& [builder, axes] = builder_result.value();
    auto table_result = builder.build(axes);
    if (!table_result) {
        state.SkipWithError("Failed to build price table");
        return;
    }

    // Create interpolated IV solver
    auto aps_est = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
    if (!aps_est) {
        state.SkipWithError("Failed to create AmericanPriceSurface");
        return;
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps_est));
    if (!iv_solver_result) {
        state.SkipWithError("Failed to create IV solver");
        return;
    }
    const auto& interp_solver = iv_solver_result.value();

    // Create FDM IV solver (ground truth)
    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    IVSolverFDM fdm_solver(config);

    // Prepare IV queries
    std::vector<IVQuery> queries;
    for (const auto& opt : iv_fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            iv_fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }
    // Track accuracy
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    size_t valid_count = 0;

    for (auto _ : state) {
        max_abs_error = 0.0;
        sum_abs_error = 0.0;
        valid_count = 0;

        for (const auto& query : queries) {
            auto fdm_result = fdm_solver.solve(query);
            auto interp_result = interp_solver.solve(query);

            if (fdm_result.has_value() && interp_result.has_value()) {
                double abs_error = std::abs(fdm_result->implied_vol - interp_result->implied_vol);
                max_abs_error = std::max(max_abs_error, abs_error);
                sum_abs_error += abs_error;
                valid_count++;
            }
        }
        benchmark::DoNotOptimize(max_abs_error);
    }

    // Report grid sizes
    size_t n_m = axes.grids[0].size();
    size_t n_tau = axes.grids[1].size();
    size_t n_sigma = axes.grids[2].size();
    size_t n_rate = axes.grids[3].size();

    state.counters["target_bps"] = target_error_bps;
    state.counters["actual_max_bps"] = max_abs_error * 10000.0;
    state.counters["actual_avg_bps"] = (valid_count > 0 ? sum_abs_error / valid_count : 0.0) * 10000.0;
    state.counters["n_m"] = static_cast<double>(n_m);
    state.counters["n_tau"] = static_cast<double>(n_tau);
    state.counters["n_sigma"] = static_cast<double>(n_sigma);
    state.counters["n_rate"] = static_cast<double>(n_rate);
    state.counters["pde_solves"] = static_cast<double>(table_result->n_pde_solves);

    // Check if achieved target (allowing 2× safety margin)
    bool achieved = (max_abs_error <= 2.0 * target_iv_error);
    state.counters["achieved"] = achieved ? 1.0 : 0.0;

    state.SetLabel(std::format("target={}bps, actual_max={:.1f}bps, grid={}×{}×{}×{}, {} solves {}",
        static_cast<int>(target_error_bps),
        max_abs_error * 10000.0,
        n_m, n_tau, n_sigma, n_rate,
        table_result->n_pde_solves,
        achieved ? "✓" : "✗"));
}

// Test different target accuracy levels
BENCHMARK(BM_RealData_GridEstimator)
    ->Arg(100)   // 100 bps = 1% (very loose)
    ->Arg(50)    // 50 bps = 0.5%
    ->Arg(10)    // 10 bps = 0.1% (default)
    ->Arg(5)     // 5 bps = 0.05%
    ->Arg(1)     // 1 bps = 0.01% (high accuracy)
    ->MinTime(kMinBenchmarkTimeSec);

// ============================================================================
// Automatic Grid Profiles (end-to-end)
// ============================================================================

static void BM_RealData_GridProfiles(benchmark::State& state) {
    const int profile = static_cast<int>(state.range(0));
    const auto grid_profile = (profile == 0)
        ? PriceTableGridProfile::Low
        : (profile == 1 ? PriceTableGridProfile::Medium
                        : (profile == 2 ? PriceTableGridProfile::High : PriceTableGridProfile::Ultra));
    const auto pde_profile = (profile == 0)
        ? GridAccuracyProfile::Low
        : (profile == 1 ? GridAccuracyProfile::Medium
                        : (profile == 2 ? GridAccuracyProfile::High : GridAccuracyProfile::Ultra));

    const auto& iv_fixture = GetIVSmileFixture();

    auto builder_result = PriceTableBuilder<4>::from_grid_auto_profile(
        iv_fixture.chain,
        grid_profile,
        pde_profile,
        OptionType::PUT);
    if (!builder_result) {
        state.SkipWithError("Failed to create price table builder");
        return;
    }

    auto& [builder, axes] = builder_result.value();
    auto table_result = builder.build(axes);
    if (!table_result) {
        state.SkipWithError("Failed to build price table");
        return;
    }

    auto aps_prof = AmericanPriceSurface::create(table_result->surface, OptionType::PUT);
    if (!aps_prof) {
        state.SkipWithError("Failed to create AmericanPriceSurface");
        return;
    }
    auto iv_solver_result = IVSolverInterpolatedStandard::create(std::move(*aps_prof));
    if (!iv_solver_result) {
        state.SkipWithError("Failed to create IV solver");
        return;
    }
    const auto& interp_solver = iv_solver_result.value();

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    IVSolverFDM fdm_solver(config);

    std::vector<IVQuery> queries;
    for (const auto& opt : iv_fixture.smile_options) {
        queries.push_back(IVQuery(
            SPOT,
            opt.strike,
            iv_fixture.target_maturity,
            RISK_FREE_RATE,
            DIVIDEND_YIELD,
            OptionType::PUT,
            opt.market_price
        ));
    }

    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    double sum_price_err_sq = 0.0;
    size_t valid_count = 0;
    double last_iter_seconds = 0.0;
    double last_iter_iv_us = 0.0;
    double fdm_iter_seconds = 0.0;
    double fdm_iv_us = 0.0;
    std::array<BucketStats, static_cast<int>(MoneynessBucket::COUNT)> buckets{};

    // Compute accuracy once (outside timed loop)
    for (const auto& query : queries) {
        auto fdm_result = fdm_solver.solve(query);
        auto interp_result = interp_solver.solve(query);

        if (fdm_result.has_value() && interp_result.has_value()) {
            double abs_error = std::abs(fdm_result->implied_vol - interp_result->implied_vol);
            max_abs_error = std::max(max_abs_error, abs_error);
            sum_abs_error += abs_error;
            valid_count++;

            // Bucketed metrics
            double rate = get_zero_rate(query.rate, query.maturity);
            double vega = bs_vega(SPOT, query.strike, query.maturity,
                                  fdm_result->implied_vol, rate, DIVIDEND_YIELD);
            double iv_err_bps = abs_error * 10000.0;
            double price_err = vega * abs_error;
            sum_price_err_sq += price_err * price_err;

            auto bucket = classify_put(SPOT, query.strike);
            auto& b = buckets[static_cast<int>(bucket)];
            b.count++;
            b.sum_iv_err_bps += iv_err_bps;
            b.max_iv_err_bps = std::max(b.max_iv_err_bps, iv_err_bps);
            b.sum_price_err_sq += price_err * price_err;
            b.sum_vw_sq += (vega * iv_err_bps) * (vega * iv_err_bps);
            b.sum_vega += vega;
        }
    }

    // Measure FDM-only time once (auto-grid per query)
    if (!queries.empty()) {
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& query : queries) {
            auto fdm_result = fdm_solver.solve(query);
            benchmark::DoNotOptimize(fdm_result);
        }
        auto t1 = std::chrono::steady_clock::now();
        fdm_iter_seconds = std::chrono::duration<double>(t1 - t0).count();
        fdm_iv_us = (fdm_iter_seconds / static_cast<double>(queries.size())) * 1e6;
    }

    // Timed loop: interpolation only
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& query : queries) {
            auto interp_result = interp_solver.solve(query);
            benchmark::DoNotOptimize(interp_result);
        }
        auto t1 = std::chrono::steady_clock::now();
        last_iter_seconds = std::chrono::duration<double>(t1 - t0).count();
        if (!queries.empty()) {
            last_iter_iv_us = (last_iter_seconds / static_cast<double>(queries.size())) * 1e6;
        }
    }

    size_t n_m = axes.grids[0].size();
    size_t n_tau = axes.grids[1].size();
    size_t n_sigma = axes.grids[2].size();
    size_t n_rate = axes.grids[3].size();

    const double target_bps = (profile == 0) ? 200.0 : (profile == 1 ? 150.0 : (profile == 2 ? 100.0 : 75.0));
    const double max_err_bps = max_abs_error * 10000.0;
    const double avg_err_bps = (valid_count > 0 ? sum_abs_error / valid_count : 0.0) * 10000.0;
    const bool achieved = (max_err_bps <= target_bps);

    state.counters["profile"] = static_cast<double>(profile);
    state.counters["target_bps"] = target_bps;
    state.counters["max_err_bps"] = max_err_bps;
    state.counters["avg_err_bps"] = avg_err_bps;
    state.counters["achieved"] = achieved ? 1.0 : 0.0;
    state.counters["n_m"] = static_cast<double>(n_m);
    state.counters["n_tau"] = static_cast<double>(n_tau);
    state.counters["n_sigma"] = static_cast<double>(n_sigma);
    state.counters["n_rate"] = static_cast<double>(n_rate);
    state.counters["pde_solves"] = static_cast<double>(table_result->n_pde_solves);
    if (last_iter_seconds > 0.0) {
        state.counters["interp_ivs_per_sec"] =
            static_cast<double>(queries.size()) / last_iter_seconds;
    }
    if (last_iter_iv_us > 0.0) {
        state.counters["interp_iv_us"] = last_iter_iv_us;
    }
    if (fdm_iter_seconds > 0.0) {
        state.counters["fdm_ivs_per_sec"] =
            static_cast<double>(queries.size()) / fdm_iter_seconds;
    }
    if (fdm_iv_us > 0.0) {
        state.counters["fdm_iv_us"] = fdm_iv_us;
    }

    // Bucketed accuracy counters
    emit_bucket_counters(state, buckets, sum_price_err_sq, valid_count);

    const char* label = (profile == 0) ? "low" : (profile == 1 ? "medium" : (profile == 2 ? "high" : "ultra"));
    // ATM bucket stats for label
    const auto& atm = buckets[static_cast<int>(MoneynessBucket::ATM)];
    double atm_avg = atm.count > 0 ? atm.sum_iv_err_bps / atm.count : 0.0;
    double price_rmse = valid_count > 0 ? std::sqrt(sum_price_err_sq / valid_count) : 0.0;
    state.SetLabel(std::format("{} profile, grid={}×{}×{}×{}, {} solves, interp_iv={:.2f}us, atm={:.1f}bps, price_rmse=${:.4f}, max={:.1f}bps, avg={:.1f}bps {}",
        label,
        n_m, n_tau, n_sigma, n_rate,
        table_result->n_pde_solves,
        last_iter_iv_us,
        atm_avg,
        price_rmse,
        max_err_bps,
        avg_err_bps,
        achieved ? "✓" : "✗"));


    if (!achieved) {
        state.SkipWithError("accuracy target not met");
    }
}

BENCHMARK(BM_RealData_GridProfiles)
    ->Arg(0)  // low
    ->Arg(1)  // medium
    ->Arg(2)  // high
    ->Arg(3)  // ultra
    ->MinTime(kMinBenchmarkTimeSec);

// ============================================================================
// Summary Reporter
// ============================================================================

class RealDataSummaryReporter : public benchmark::ConsoleReporter {
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
            if (name.find("BM_RealData_") == std::string::npos) {
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
        out << "\nReal Data Benchmark Summary (avg time per op)\n";
        out << "Symbol: " << SYMBOL << ", Spot: $" << std::fixed << std::setprecision(2) << SPOT << "\n";
        out << "Rate: " << std::fixed << std::setprecision(2) << (RISK_FREE_RATE * 100.0) << "%, ";
        out << "Div Yield: " << std::fixed << std::setprecision(2) << (DIVIDEND_YIELD * 100.0) << "%\n\n";

        for (const auto& entry : summaries_) {
            const double avg_us = entry.avg_ns / 1e3;
            const double avg_ms = entry.avg_ns / 1e6;
            out << "  - "
                << std::left << std::setw(label_width) << entry.name
                << std::right << " : "
                << std::fixed << std::setprecision(2)
                << std::setw(number_width) << avg_ms << " ms ("
                << std::setw(number_width) << avg_us << " us, "
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
    RealDataSummaryReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    benchmark::Shutdown();
    return 0;
}
