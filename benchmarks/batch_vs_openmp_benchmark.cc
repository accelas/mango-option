/**
 * @file batch_vs_openmp_benchmark.cc
 * @brief Compare batch mode vs OpenMP parallel single-contract mode
 *
 * Three approaches tested:
 * 1. Single-threaded batch mode (horizontal SIMD)
 * 2. OpenMP parallel single-contract mode (thread-level parallelism)
 * 3. OpenMP parallel batch mode (thread + SIMD parallelism)
 *
 * Goal: Determine which approach delivers best performance for solving
 * multiple option contracts simultaneously.
 */

#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <cmath>
#include <omp.h>

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

// Contract parameters
struct ContractParams {
    double strike;
    double volatility;
    double rate;
    double dividend;
    double maturity;
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

// Generate contract chain (homogeneous maturity and strike for fair comparison)
std::vector<ContractParams> generate_homogeneous_contracts(size_t n_contracts) {
    std::vector<ContractParams> contracts;
    contracts.reserve(n_contracts);

    const double K = 100.0;
    const double T = 1.0;  // Same maturity for all
    const double r = 0.05;
    const double q = 0.02;

    for (size_t i = 0; i < n_contracts; ++i) {
        // Vary volatility: 15%-30%
        double vol = 0.15 + 0.15 * (static_cast<double>(i) / n_contracts);
        contracts.push_back({K, vol, r, q, T});
    }

    return contracts;
}

// Approach 1: Single-threaded batch mode (current implementation)
void solve_batch_single_threaded(
    const std::vector<ContractParams>& contracts,
    const std::vector<double>& grid,
    size_t n_time_steps,
    size_t batch_width) {

    const size_t n = grid.size();
    const size_t n_contracts = contracts.size();

    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };
    auto left_bc = DirichletBC(left_bc_func);
    auto right_bc = DirichletBC(right_bc_func);

    // Process contracts in batches
    for (size_t batch_start = 0; batch_start < n_contracts; batch_start += batch_width) {
        size_t current_batch_width = std::min(batch_width, n_contracts - batch_start);

        const auto& first_contract = contracts[batch_start];
        TimeDomain time_domain(0.0, first_contract.maturity,
                              first_contract.maturity / n_time_steps);

        // Create batch workspace
        PDEWorkspace workspace_batch(n, std::span(grid), current_batch_width);

        // Create per-contract PDEs
        std::vector<BlackScholesPDE<double>> pdes;
        pdes.reserve(current_batch_width);
        for (size_t i = 0; i < current_batch_width; ++i) {
            const auto& contract = contracts[batch_start + i];
            pdes.emplace_back(contract.volatility, contract.rate, contract.dividend);
        }

        auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
        auto spatial_op = SpatialOperator<BlackScholesPDE<double>, double>(pdes, spacing);

        // Initial condition (American put payoff)
        auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
            const auto& contract = first_contract;
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                u[i] = std::max(contract.strike - S, 0.0);
            }
        };

        // Obstacle condition
        auto obstacle = [&](double, std::span<const double> x, std::span<double> psi) {
            const auto& contract = first_contract;
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                psi[i] = std::max(contract.strike - S, 0.0);
            }
        };

        PDESolver solver_batch(
            grid, time_domain,
            trbdf2_config, root_config,
            left_bc, right_bc, spatial_op,
            obstacle,
            &workspace_batch
        );

        solver_batch.initialize(initial_condition);
        auto result = solver_batch.solve();

        if (!result.has_value()) {
            throw std::runtime_error("Batch solver failed to converge");
        }
    }
}

// Approach 2: OpenMP parallel single-contract mode
void solve_openmp_single_contract(
    const std::vector<ContractParams>& contracts,
    const std::vector<double>& grid,
    size_t n_time_steps) {

    const size_t n_contracts = contracts.size();

    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };

    // Parallel loop over contracts
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < n_contracts; ++idx) {
        const auto& contract = contracts[idx];

        TimeDomain time_domain(0.0, contract.maturity,
                              contract.maturity / n_time_steps);

        auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                u[i] = std::max(contract.strike - S, 0.0);
            }
        };

        auto obstacle = [&](double, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                psi[i] = std::max(contract.strike - S, 0.0);
            }
        };

        BlackScholesPDE pde(contract.volatility, contract.rate, contract.dividend);
        auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
        auto spatial_op = SpatialOperator(pde, spacing);

        auto left_bc = DirichletBC(left_bc_func);
        auto right_bc = DirichletBC(right_bc_func);

        PDESolver solver(
            grid, time_domain,
            trbdf2_config, root_config,
            left_bc, right_bc, spatial_op,
            obstacle
        );

        solver.initialize(initial_condition);
        auto result = solver.solve();

        if (!result.has_value()) {
            #pragma omp critical
            {
                throw std::runtime_error("Single-contract solver failed to converge");
            }
        }
    }
}

// Approach 3: OpenMP parallel batch mode
void solve_openmp_batch(
    const std::vector<ContractParams>& contracts,
    const std::vector<double>& grid,
    size_t n_time_steps,
    size_t batch_width) {

    const size_t n = grid.size();
    const size_t n_contracts = contracts.size();

    RootFindingConfig root_config{
        .max_iter = 100,
        .tolerance = 1e-6,
        .jacobian_fd_epsilon = 1e-7,
        .brent_tol_abs = 1e-6
    };

    TRBDF2Config trbdf2_config{
        .max_iter = 100,
        .tolerance = 1e-6
    };

    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };

    // Parallel loop over batches
    #pragma omp parallel for schedule(dynamic)
    for (size_t batch_start = 0; batch_start < n_contracts; batch_start += batch_width) {
        size_t current_batch_width = std::min(batch_width, n_contracts - batch_start);

        const auto& first_contract = contracts[batch_start];
        TimeDomain time_domain(0.0, first_contract.maturity,
                              first_contract.maturity / n_time_steps);

        PDEWorkspace workspace_batch(n, std::span(grid), current_batch_width);

        std::vector<BlackScholesPDE<double>> pdes;
        pdes.reserve(current_batch_width);
        for (size_t i = 0; i < current_batch_width; ++i) {
            const auto& contract = contracts[batch_start + i];
            pdes.emplace_back(contract.volatility, contract.rate, contract.dividend);
        }

        auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
        auto spatial_op = SpatialOperator<BlackScholesPDE<double>, double>(pdes, spacing);

        auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
            const auto& contract = first_contract;
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                u[i] = std::max(contract.strike - S, 0.0);
            }
        };

        auto obstacle = [&](double, std::span<const double> x, std::span<double> psi) {
            const auto& contract = first_contract;
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                psi[i] = std::max(contract.strike - S, 0.0);
            }
        };

        auto left_bc = DirichletBC(left_bc_func);
        auto right_bc = DirichletBC(right_bc_func);

        PDESolver solver_batch(
            grid, time_domain,
            trbdf2_config, root_config,
            left_bc, right_bc, spatial_op,
            obstacle,
            &workspace_batch
        );

        solver_batch.initialize(initial_condition);
        auto result = solver_batch.solve();

        if (!result.has_value()) {
            #pragma omp critical
            {
                throw std::runtime_error("Batch solver failed to converge");
            }
        }
    }
}

// Benchmark: Single-threaded batch mode (baseline)
static void BM_BatchMode_SingleThreaded(benchmark::State& state) {
    const size_t n_contracts = state.range(0);
    const size_t n_space = 101;
    const size_t n_time = 1000;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);
    auto contracts = generate_homogeneous_contracts(n_contracts);

    for (auto _ : state) {
        solve_batch_single_threaded(contracts, grid, n_time, batch_width);
    }

    state.counters["contracts"] = n_contracts;
    state.counters["batch_width"] = batch_width;
    state.counters["threads"] = 1;
    state.counters["throughput_contracts_per_sec"] = benchmark::Counter(
        n_contracts * state.iterations(), benchmark::Counter::kIsRate);
}

// Benchmark: OpenMP parallel single-contract mode
static void BM_OpenMP_SingleContract(benchmark::State& state) {
    const size_t n_contracts = state.range(0);
    const size_t n_space = 101;
    const size_t n_time = 1000;

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);
    auto contracts = generate_homogeneous_contracts(n_contracts);

    // Set OpenMP thread count
    const int n_threads = state.range(1);
    omp_set_num_threads(n_threads);

    for (auto _ : state) {
        solve_openmp_single_contract(contracts, grid, n_time);
    }

    state.counters["contracts"] = n_contracts;
    state.counters["batch_width"] = 1;
    state.counters["threads"] = n_threads;
    state.counters["throughput_contracts_per_sec"] = benchmark::Counter(
        n_contracts * state.iterations(), benchmark::Counter::kIsRate);
}

// Benchmark: OpenMP parallel batch mode
static void BM_OpenMP_BatchMode(benchmark::State& state) {
    const size_t n_contracts = state.range(0);
    const size_t n_space = 101;
    const size_t n_time = 1000;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);
    auto contracts = generate_homogeneous_contracts(n_contracts);

    // Set OpenMP thread count
    const int n_threads = state.range(1);
    omp_set_num_threads(n_threads);

    for (auto _ : state) {
        solve_openmp_batch(contracts, grid, n_time, batch_width);
    }

    state.counters["contracts"] = n_contracts;
    state.counters["batch_width"] = batch_width;
    state.counters["threads"] = n_threads;
    state.counters["throughput_contracts_per_sec"] = benchmark::Counter(
        n_contracts * state.iterations(), benchmark::Counter::kIsRate);
}

// Register benchmarks with different contract counts and thread counts
// Test with 16, 32, 64 contracts and 1, 4, 8, 16 threads

// Baseline: Single-threaded batch mode
BENCHMARK(BM_BatchMode_SingleThreaded)
    ->Args({16, 1})
    ->Args({32, 1})
    ->Args({64, 1})
    ->Unit(benchmark::kMillisecond)
    ->MinTime(3.0);

// OpenMP single-contract mode with varying thread counts
BENCHMARK(BM_OpenMP_SingleContract)
    ->Args({16, 4})
    ->Args({16, 8})
    ->Args({16, 16})
    ->Args({32, 4})
    ->Args({32, 8})
    ->Args({32, 16})
    ->Args({64, 4})
    ->Args({64, 8})
    ->Args({64, 16})
    ->Unit(benchmark::kMillisecond)
    ->MinTime(3.0);

// OpenMP batch mode with varying thread counts
BENCHMARK(BM_OpenMP_BatchMode)
    ->Args({16, 4})
    ->Args({16, 8})
    ->Args({16, 16})
    ->Args({32, 4})
    ->Args({32, 8})
    ->Args({32, 16})
    ->Args({64, 4})
    ->Args({64, 8})
    ->Args({64, 16})
    ->Unit(benchmark::kMillisecond)
    ->MinTime(3.0);

}  // namespace
}  // namespace mango

BENCHMARK_MAIN();
