// SPDX-License-Identifier: MIT
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
#include "src/pde/core/pde_workspace.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/american_price_surface.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <memory>
#include <memory_resource>
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
    std::shared_ptr<const PriceTableSurface<4>> surface;
};

const AnalyticSurfaceFixture& GetAnalyticSurfaceFixture() {
    static AnalyticSurfaceFixture* fixture = [] {
        auto fixture_ptr = std::make_unique<AnalyticSurfaceFixture>();
        fixture_ptr->K_ref = 100.0;

        std::vector<double> m_grid = {0.8, 0.9, 1.0, 1.1, 1.2};
        std::vector<double> tau_grid = {0.1, 0.5, 1.0, 2.0};
        std::vector<double> vol_grid = {0.10, 0.15, 0.20, 0.25, 0.30};
        std::vector<double> rate_grid = {0.0, 0.025, 0.05, 0.10};

        auto result = PriceTableBuilder<4>::from_vectors(
            m_grid, tau_grid, vol_grid, rate_grid, 100.0,
            GridAccuracyParams{}, OptionType::PUT, 0.0);
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
// American Option Pricing Benchmarks
// ============================================================================

static void BM_AmericanPut_ATM_1Y(benchmark::State& state) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.20    // volatility
    );

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }
    auto workspace = workspace_result.value();

    for (auto _ : state) {
        AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("ATM Put, T=1Y, σ=0.20");
}
BENCHMARK(BM_AmericanPut_ATM_1Y)->Arg(101)->Arg(201)->Arg(501);

static void BM_AmericanPut_OTM_3M(benchmark::State& state) {
    AmericanOptionParams params(
        110.0,  // spot (OTM put)
        100.0,  // strike
        0.25,   // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.30    // volatility
    );

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }
    auto workspace = workspace_result.value();

    for (auto _ : state) {
        AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("OTM Put, T=3M, σ=0.30");
}
BENCHMARK(BM_AmericanPut_OTM_3M)->Arg(500)->Arg(1000)->Arg(2000);

static void BM_AmericanPut_ITM_2Y(benchmark::State& state) {
    AmericanOptionParams params(
        90.0,   // spot (ITM put)
        100.0,  // strike
        2.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.25    // volatility
    );

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }
    auto workspace = workspace_result.value();

    for (auto _ : state) {
        AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("ITM Put, T=2Y, σ=0.25");
}
BENCHMARK(BM_AmericanPut_ITM_2Y);

static void BM_AmericanCall_WithDividends(benchmark::State& state) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::CALL,
        0.20,   // volatility
        {
            {0.25, 2.0},
            {0.5, 2.0},
            {0.75, 2.0}
        }
    );

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }
    auto workspace = workspace_result.value();

    for (auto _ : state) {
        AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();
        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
    }

    state.SetLabel("Call with 3 discrete dividends");
}
BENCHMARK(BM_AmericanCall_WithDividends);

// ============================================================================
// Implied Volatility Benchmarks
// ============================================================================

static void BM_ImpliedVol_ATM_Put(benchmark::State& state) {
    IVQuery query(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, 6.0);

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    IVSolverFDM solver(config);

    for (auto _ : state) {
        auto result = solver.solve_impl(query);
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ATM Put, T=1Y");
}
BENCHMARK(BM_ImpliedVol_ATM_Put);

static void BM_ImpliedVol_OTM_Put(benchmark::State& state) {
    IVQuery query(110.0, 100.0, 0.25, 0.05, 0.0, OptionType::PUT, 0.80);

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    IVSolverFDM solver(config);

    for (auto _ : state) {
        auto result = solver.solve_impl(query);
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("OTM Put, T=3M");
}
BENCHMARK(BM_ImpliedVol_OTM_Put);

static void BM_ImpliedVol_ITM_Put(benchmark::State& state) {
    IVQuery query(90.0, 100.0, 2.0, 0.05, 0.0, OptionType::PUT, 15.0);

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    IVSolverFDM solver(config);

    for (auto _ : state) {
        auto result = solver.solve_impl(query);
        benchmark::DoNotOptimize(result);
    }

    state.SetLabel("ITM Put, T=2Y");
}
BENCHMARK(BM_ImpliedVol_ITM_Put);

static void BM_ImpliedVol_BSplineSurface(benchmark::State& state) {
    const auto& surf = GetAnalyticSurfaceFixture();

    // Create AmericanPriceSurface wrapper and IV solver
    auto aps = AmericanPriceSurface::create(surf.surface, OptionType::PUT);
    if (!aps) {
        throw std::runtime_error("Failed to create AmericanPriceSurface");
    }
    auto solver_result = IVSolverInterpolated::create(std::move(*aps));

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

    // Use a representative ATM put price for the IV query
    IVQuery query(spot, strike, maturity, rate, 0.0, OptionType::PUT,
                  analytic_bs_price(spot, strike, maturity, sigma_true, rate, OptionType::PUT));

    for (auto _ : state) {
        auto result = solver.solve_impl(query);
        if (!result.has_value()) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        benchmark::DoNotOptimize(result->implied_vol);
    }

    state.SetLabel("B-spline IV (table-based)");
}
BENCHMARK(BM_ImpliedVol_BSplineSurface);

// ============================================================================
// Grid Resolution Impact
// ============================================================================

static void BM_AmericanPut_GridResolution(benchmark::State& state) {
    AmericanOptionParams params(
        100.0,  // spot
        100.0,  // strike
        1.0,    // maturity
        0.05,   // rate
        0.02,   // dividend_yield
        OptionType::PUT,
        0.20    // volatility
    );

    auto [grid_spec, time_domain] = estimate_grid_for_option(params);

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }
    auto workspace = workspace_result.value();

    double total_time_ns = 0.0;
    size_t iterations = 0;

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        AmericanOptionSolver solver(params, workspace);
        auto result = solver.solve();
        auto end = std::chrono::high_resolution_clock::now();

        if (!result) {
            throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(result.error().code)));
        }
        double price = result->value_at(params.spot);
        benchmark::DoNotOptimize(price);
        total_time_ns += std::chrono::duration<double, std::nano>(end - start).count();
        iterations++;
    }

    // Add time in milliseconds for easier reading
    if (iterations > 0) {
        double avg_time_ms = (total_time_ns / iterations) / 1e6;
        state.counters["time_ms"] = avg_time_ms;
    }

    state.SetLabel("Grid: " + std::to_string(grid_spec.n_points()) + "x" + std::to_string(time_domain.n_steps()));
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

    auto [grid_spec, time_domain] = compute_global_grid_for_batch(batch);
    (void)time_domain;  // Not used in this benchmark

    // Allocate buffer for workspace
    size_t n = grid_spec.n_points();
    std::pmr::synchronized_pool_resource pool;
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result) {
        state.SkipWithError(workspace_result.error().c_str());
        return;
    }

    for (auto _ : state) {
        std::vector<std::expected<AmericanOptionResult, SolverError>> results;
        results.reserve(batch_size);

        // Sequential processing for now (batch API may not exist)
        for (const auto& params : batch) {
            AmericanOptionSolver solver(params, workspace_result.value());
            results.push_back(solver.solve());
        }

        for (const auto& res : results) {
            if (!res) {
                throw std::runtime_error("Solver error code " + std::to_string(static_cast<int>(res.error().code)));
            }
        }
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Sequential batch: " + std::to_string(batch_size) + " options");
}
BENCHMARK(BM_AmericanPut_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

static void BM_ImpliedVol_Batch(benchmark::State& state) {
    size_t batch_size = state.range(0);

    // Generate batch of market prices
    std::vector<IVQuery> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        double market_price = 5.0 + i * 0.1;  // Different prices
        batch.push_back(IVQuery(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::PUT, market_price));
    }

    IVSolverFDMConfig config;
    config.root_config.max_iter = 100;
    config.root_config.tolerance = 1e-6;
    config.grid_n_space = 101;
    config.grid_n_time = 1000;

    IVSolverFDM solver(config);
    for (auto _ : state) {
        std::vector<std::expected<IVSuccess, IVError>> results;
        results.reserve(batch_size);

        // Sequential processing for now (batch API may not exist)
        for (const auto& params : batch) {
            results.push_back(solver.solve_impl(params));
        }

        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("Sequential batch: " + std::to_string(batch_size) + " IVs");
}
BENCHMARK(BM_ImpliedVol_Batch)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

BENCHMARK_MAIN();
