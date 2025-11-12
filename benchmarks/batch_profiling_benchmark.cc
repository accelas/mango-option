/**
 * @file batch_profiling_benchmark.cc
 * @brief Detailed profiling of batch mode overhead sources
 *
 * Measures timing breakdown for:
 * - Pack/scatter operations (AoS ↔ SoA conversions)
 * - Stencil computation (L(u) evaluation)
 * - Newton iteration (per-lane Jacobian, solve)
 * - Boundary conditions
 * - Total overhead
 */

#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "src/pde/core/pde_solver.hpp"
#include "src/pde/core/time_domain.hpp"
#include "src/pde/memory/pde_workspace.hpp"
#include "src/pde/operators/spatial_operator.hpp"
#include "src/pde/operators/centered_difference_facade.hpp"
#include "src/pde/core/boundary_conditions.hpp"
#include "src/pde/operators/black_scholes_pde.hpp"

namespace mango {
namespace {

using namespace mango::operators;
namespace stdx = std::experimental;

// Timing accumulator
struct TimingStats {
    double pack_time_ms = 0.0;
    double scatter_time_ms = 0.0;
    double stencil_time_ms = 0.0;
    double newton_time_ms = 0.0;
    double boundary_time_ms = 0.0;
    double total_time_ms = 0.0;
    size_t newton_iterations = 0;
};

// Helper: Create uniform grid
std::vector<double> create_uniform_grid(double x_min, double x_max, size_t n) {
    std::vector<double> grid(n);
    const double dx = (x_max - x_min) / (n - 1);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = x_min + i * dx;
    }
    return grid;
}

// Instrumented batch solver with timing
TimingStats solve_with_profiling(
    size_t batch_width,
    const std::vector<double>& grid,
    size_t n_time_steps) {

    TimingStats stats;
    const size_t n = grid.size();

    // Root-finding config
    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    // TR-BDF2 config
    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    // Create per-lane PDEs (varying volatility)
    std::vector<BlackScholesPDE<double>> pdes;
    pdes.reserve(batch_width);
    for (size_t i = 0; i < batch_width; ++i) {
        double vol = 0.15 + 0.15 * (static_cast<double>(i) / batch_width);
        pdes.emplace_back(vol, 0.05, 0.02);
    }

    // Time domain
    const double T = 1.0;
    TimeDomain time_domain(0.0, T, T / n_time_steps);

    // Spatial operator
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
    auto spatial_op = SpatialOperator(pdes, spacing);

    // Boundary conditions
    auto left_bc = DirichletBC([](double, double) { return 0.0; });
    auto right_bc = DirichletBC([](double, double) { return 0.0; });

    // Obstacle condition (American put)
    const double K = 100.0;
    auto obstacle = [K](double, std::span<const double> x, std::span<double> psi) {
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = K * std::exp(x[i]);
            psi[i] = std::max(K - S, 0.0);
        }
    };

    // Initial condition
    auto initial_condition = [K](std::span<const double> x, std::span<double> u) {
        for (size_t i = 0; i < x.size(); ++i) {
            const double S = K * std::exp(x[i]);
            u[i] = std::max(K - S, 0.0);
        }
    };

    // Create batch workspace
    PDEWorkspace workspace(n, std::span(grid), batch_width);

    // Create solver
    PDESolver solver(
        grid, time_domain,
        trbdf2_config, root_config,
        left_bc, right_bc, spatial_op,
        obstacle,
        &workspace
    );

    // Initialize
    auto init_start = std::chrono::high_resolution_clock::now();
    solver.initialize(initial_condition);
    auto init_end = std::chrono::high_resolution_clock::now();

    // Solve (this is where we want detailed timing)
    auto total_start = std::chrono::high_resolution_clock::now();
    auto result = solver.solve();
    auto total_end = std::chrono::high_resolution_clock::now();

    if (!result.has_value()) {
        throw std::runtime_error("Solver failed to converge");
    }

    stats.total_time_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();

    // We can't easily instrument internal solver timing without modifying source,
    // so we'll measure component operations separately
    return stats;
}

// Benchmark: Measure pack/scatter overhead separately
static void BM_PackScatterOverhead(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n);
    PDEWorkspace workspace(n, std::span(grid), batch_width);

    // Initialize lane buffers with some data
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto u_lane = workspace.u_lane(lane);
        for (size_t i = 0; i < n; ++i) {
            u_lane[i] = static_cast<double>(i + lane);
        }
    }

    double total_pack_time = 0.0;
    double total_scatter_time = 0.0;

    for (auto _ : state) {
        // Measure pack time (SoA → AoS)
        auto pack_start = std::chrono::high_resolution_clock::now();
        workspace.pack_to_batch_slice();
        auto pack_end = std::chrono::high_resolution_clock::now();

        total_pack_time += std::chrono::duration<double, std::micro>(
            pack_end - pack_start).count();

        // Measure scatter time (AoS → SoA)
        auto scatter_start = std::chrono::high_resolution_clock::now();
        workspace.scatter_from_batch_slice();
        auto scatter_end = std::chrono::high_resolution_clock::now();

        total_scatter_time += std::chrono::duration<double, std::micro>(
            scatter_end - scatter_start).count();

        benchmark::DoNotOptimize(workspace.batch_slice().data());
    }

    state.counters["pack_time_us"] = total_pack_time / state.iterations();
    state.counters["scatter_time_us"] = total_scatter_time / state.iterations();
    state.counters["total_overhead_us"] = (total_pack_time + total_scatter_time) / state.iterations();
    state.counters["grid_size"] = n;
    state.counters["batch_width"] = batch_width;
}

// Benchmark: Measure stencil computation time
static void BM_StencilComputationBatch(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n);
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));

    // Create batch PDEs
    std::vector<BlackScholesPDE<double>> pdes;
    pdes.reserve(batch_width);
    for (size_t i = 0; i < batch_width; ++i) {
        double vol = 0.15 + 0.15 * (static_cast<double>(i) / batch_width);
        pdes.emplace_back(vol, 0.05, 0.02);
    }

    auto spatial_op = SpatialOperator(pdes, spacing);

    // Create workspace and initialize
    PDEWorkspace workspace(n, std::span(grid), batch_width);
    for (size_t lane = 0; lane < batch_width; ++lane) {
        auto u_lane = workspace.u_lane(lane);
        for (size_t i = 0; i < n; ++i) {
            u_lane[i] = std::sin(static_cast<double>(i) * 0.1);
        }
    }

    workspace.pack_to_batch_slice();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        spatial_op.apply_interior_batch(0.5, workspace.batch_slice(),
                                       workspace.lu_batch(), batch_width,
                                       1, n - 1);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration_us = std::chrono::duration<double, std::micro>(
            end - start).count();
        state.counters["stencil_time_us"] = duration_us;

        benchmark::DoNotOptimize(workspace.lu_batch().data());
    }

    state.counters["grid_size"] = n;
    state.counters["batch_width"] = batch_width;
}

// Benchmark: Measure single-contract stencil for comparison
static void BM_StencilComputationSingle(benchmark::State& state) {
    const size_t n = state.range(0);

    auto grid = create_uniform_grid(-1.0, 1.0, n);
    auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));

    BlackScholesPDE<double> pde(0.20, 0.05, 0.02);
    auto spatial_op = SpatialOperator(pde, spacing);

    std::vector<double> u(n);
    std::vector<double> lu(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(static_cast<double>(i) * 0.1);
    }

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        spatial_op.apply(0.5, std::span(u), std::span(lu));

        auto end = std::chrono::high_resolution_clock::now();

        auto duration_us = std::chrono::duration<double, std::micro>(
            end - start).count();
        state.counters["stencil_time_us"] = duration_us;

        benchmark::DoNotOptimize(lu.data());
    }

    state.counters["grid_size"] = n;
    state.counters["batch_width"] = 1;
}

// Benchmark: Full solver with timing breakdown
static void BM_FullSolverProfiling(benchmark::State& state) {
    const size_t n_space = 101;
    const size_t n_time = 100;  // Reduced for faster profiling
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);

    for (auto _ : state) {
        auto stats = solve_with_profiling(batch_width, grid, n_time);

        state.counters["total_time_ms"] = stats.total_time_ms;
        state.counters["grid_size"] = n_space;
        state.counters["time_steps"] = n_time;
        state.counters["batch_width"] = batch_width;
    }
}

// Register benchmarks
BENCHMARK(BM_PackScatterOverhead)
    ->Arg(51)
    ->Arg(101)
    ->Arg(201)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StencilComputationBatch)
    ->Arg(51)
    ->Arg(101)
    ->Arg(201)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_StencilComputationSingle)
    ->Arg(51)
    ->Arg(101)
    ->Arg(201)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_FullSolverProfiling)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace mango

BENCHMARK_MAIN();
