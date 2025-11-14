/**
 * @file component_performance.cc
 * @brief Component-level performance benchmarks for C++20 implementation
 *
 * Benchmarks individual components to understand performance characteristics:
 * - American option pricing (single calculation)
 * - Implied volatility solving (single calculation)
 * - Batch operations
 *
 * Run with: bazel run //benchmarks:component_performance
 */

#include "src/option/american_option.hpp"
#include "src/option/slice_solver_workspace.hpp"
#include "src/interpolation/bspline_4d.hpp"
#include "src/interpolation/bspline_fitter_4d.hpp"
#include "src/option/iv_solver.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/price_table_4d_builder.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace mango;

namespace {

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
    std::unique_ptr<BSpline4D> evaluator;
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

        if (!fitter_result) {
            throw std::runtime_error("Failed to create BSpline fitter: " + fitter_result.error());
        }

        auto fit_result = fitter_result->fit(prices);
        if (!fit_result.success) {
            throw std::runtime_error("Failed to fit analytic BSpline surface: " + fit_result.error_message);
        }

        fixture_ptr->evaluator = std::make_unique<BSpline4D>(
            fixture_ptr->m_grid,
            fixture_ptr->tau_grid,
            fixture_ptr->sigma_grid,
            fixture_ptr->rate_grid,
            fit_result.coefficients);

        return fixture_ptr.release();
    }();

    return *fixture;
}

}  // namespace

// ============================================================================
// American Option Pricing Benchmarks
// ============================================================================

static void BM_AmericanPut_ATM_1Y(benchmark::State& state) {
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

    AmericanOptionGrid grid;
    grid.n_space = state.range(0);
    grid.n_time = 1000;
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid.x_min, grid.x_max, grid.n_space);

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("ATM Put, T=1Y, σ=0.20");
}
BENCHMARK(BM_AmericanPut_ATM_1Y)->Arg(101)->Arg(201)->Arg(501);

static void BM_AmericanPut_OTM_3M(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 110.0,  // OTM put
        .maturity = 0.25,
        .volatility = 0.30,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = state.range(0);
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid.x_min, grid.x_max, grid.n_space);

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("OTM Put, T=3M, σ=0.30");
}
BENCHMARK(BM_AmericanPut_OTM_3M)->Arg(500)->Arg(1000)->Arg(2000);

static void BM_AmericanPut_ITM_2Y(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 90.0,  // ITM put
        .maturity = 2.0,
        .volatility = 0.25,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid.x_min, grid.x_max, grid.n_space);

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("ITM Put, T=2Y, σ=0.25");
}
BENCHMARK(BM_AmericanPut_ITM_2Y);

static void BM_AmericanCall_WithDividends(benchmark::State& state) {
    AmericanOptionParams params{
        .strike = 100.0,
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::CALL,
        .discrete_dividends = {{0.25, 2.0}, {0.5, 2.0}, {0.75, 2.0}}
    };

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;

    for (auto _ : state) {
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
    }

    state.SetLabel("Call with 3 discrete dividends");
}
BENCHMARK(BM_AmericanCall_WithDividends);

// ============================================================================
// Implied Volatility Benchmarks
// ============================================================================

static void BM_ImpliedVol_ATM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 100.0,
        .strike = 100.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .market_price = 6.0,  // Approximate for σ=0.20
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ATM Put, T=1Y");
}
BENCHMARK(BM_ImpliedVol_ATM_Put);

static void BM_ImpliedVol_OTM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 110.0,
        .strike = 100.0,
        .time_to_maturity = 0.25,
        .risk_free_rate = 0.05,
        .market_price = 0.80,  // OTM put
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("OTM Put, T=3M");
}
BENCHMARK(BM_ImpliedVol_OTM_Put);

static void BM_ImpliedVol_ITM_Put(benchmark::State& state) {
    IVParams params{
        .spot_price = 90.0,
        .strike = 100.0,
        .time_to_maturity = 2.0,
        .risk_free_rate = 0.05,
        .market_price = 15.0,  // ITM put
        .is_call = false
    };

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        IVSolver solver(params, config);
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ITM Put, T=2Y");
}
BENCHMARK(BM_ImpliedVol_ITM_Put);

static void BM_ImpliedVol_BSplineSurface(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();

    // Wrap fixture in PriceTableSurface for clean API
    PriceTableGrid grid{
        .moneyness = surf.m_grid,
        .maturity = surf.tau_grid,
        .volatility = surf.sigma_grid,
        .rate = surf.rate_grid,
        .K_ref = surf.K_ref
    };

    PriceTableSurface surface(
        std::make_shared<BSpline4D>(*surf.evaluator),  // Share ownership
        std::move(grid),
        0.0  // dividend yield
    );

    IVSolverInterpolated solver(surface);

    constexpr double spot = 103.5;
    constexpr double strike = 100.0;
    constexpr double maturity = 1.0;
    constexpr double rate = 0.05;
    constexpr double sigma_true = 0.20;

    IVQuery query{
        .market_price = analytic_bs_price(spot, strike, maturity, sigma_true, rate, OptionType::PUT),
        .spot = spot,
        .strike = strike,
        .maturity = maturity,
        .rate = rate,
        .option_type = OptionType::PUT};

    for (auto _ : state) {
        auto result = solver.solve(query);
        if (!result.converged) {
            throw std::runtime_error(result.failure_reason.value_or("Fast IV solver failed"));
        }
        benchmark::DoNotOptimize(result.implied_vol);
    }

    state.SetLabel("B-spline IV (table-based)");
}
BENCHMARK(BM_ImpliedVol_BSplineSurface);

// ============================================================================
// Grid Resolution Impact
// ============================================================================

static void BM_AmericanPut_GridResolution(benchmark::State& state) {
    size_t n_space = state.range(0);
    size_t n_time = state.range(1);

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

    AmericanOptionGrid grid;
    grid.n_space = n_space;
    grid.n_time = n_time;

    double total_time_ns = 0.0;
    size_t iterations = 0;

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        AmericanOptionSolver solver(params, grid);
        auto result = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();

        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        benchmark::DoNotOptimize(*result);
        total_time_ns += std::chrono::duration<double, std::nano>(end - start).count();
        iterations++;
    }

    // Add time in milliseconds for easier reading
    if (iterations > 0) {
        double avg_time_ms = (total_time_ns / iterations) / 1e6;
        state.counters["time_ms"] = avg_time_ms;
    }

    state.SetLabel("Grid: " + std::to_string(n_space) + "x" + std::to_string(n_time));
}
BENCHMARK(BM_AmericanPut_GridResolution)
    ->Args({51, 500})
    ->Args({101, 1000})
    ->Args({201, 2000})
    ->Args({501, 5000});

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

static void BM_AmericanPut_Batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    // Generate batch of strike prices around ATM
    std::vector<AmericanOptionParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double strike = 90.0 + i * 0.5;  // Strikes from 90 to 90 + batch_size*0.5
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

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;

    for (auto _ : state) {
        auto results = solve_american_options_batch(batch, grid);
        for (const auto& res : results) {
            if (!res) {
                throw std::runtime_error(res.error().message);
            }
        }
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Parallel batch: " + std::to_string(batch_size) + " options");
}
BENCHMARK(BM_AmericanPut_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

static void BM_ImpliedVol_Batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    // Generate batch of market prices
    std::vector<IVParams> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double market_price = 5.0 + i * 0.1;  // Different prices
        batch.push_back(IVParams{
            .spot_price = 100.0,
            .strike = 100.0,
            .time_to_maturity = 1.0,
            .risk_free_rate = 0.05,
            .market_price = market_price,
            .is_call = false
        });
    }

    IVConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    for (auto _ : state) {
        auto results = solve_implied_vol_batch(batch, config);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Parallel batch: " + std::to_string(batch_size) + " IVs");
}
BENCHMARK(BM_ImpliedVol_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

BENCHMARK_MAIN();
