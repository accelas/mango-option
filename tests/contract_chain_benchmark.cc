/**
 * @file contract_chain_benchmark.cc
 * @brief Benchmark batch mode vs single-contract mode for solving contract chains
 *
 * Contract chains are sequences of option contracts with varying parameters
 * (strikes, volatilities, maturities). Batch mode uses cross-contract SIMD
 * vectorization to solve multiple contracts simultaneously, while single-contract
 * mode solves each contract individually.
 *
 * Expected speedup with SIMD vectorization:
 * - AVX2 (SIMD width = 4): ~3-3.5x speedup
 * - AVX-512 (SIMD width = 8): ~6-7x speedup
 *
 * Benchmarks measure:
 * - Total wall-clock time
 * - Throughput (contracts/second)
 * - Per-contract average time
 * - Speedup ratio (single-contract time / batch time)
 */

#include <benchmark/benchmark.h>
#include <experimental/simd>
#include <vector>
#include <cmath>
#include <iostream>

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

// Contract parameters for benchmarking
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

// Generate realistic contract chain (varying strikes, volatilities, maturities)
std::vector<ContractParams> generate_contract_chain(size_t n_contracts) {
    std::vector<ContractParams> contracts;
    contracts.reserve(n_contracts);

    // Realistic parameter ranges for American options
    const double base_strike = 100.0;
    const double base_vol = 0.20;  // 20% base volatility
    const double base_rate = 0.05;  // 5% risk-free rate
    const double dividend = 0.02;  // 2% dividend yield
    const double base_maturity = 1.0;  // 1 year

    for (size_t i = 0; i < n_contracts; ++i) {
        // Vary strikes: 80-120 (±20% from ATM)
        double strike_ratio = 0.8 + 0.4 * (static_cast<double>(i) / n_contracts);
        double strike = base_strike * strike_ratio;

        // Vary volatility: 15%-30%
        double vol = 0.15 + 0.15 * (static_cast<double>(i) / n_contracts);

        // Vary maturity: 0.25-2.0 years
        double maturity = 0.25 + 1.75 * (static_cast<double>(i) / n_contracts);

        contracts.push_back({strike, vol, base_rate, dividend, maturity});
    }

    return contracts;
}

// Solve contract chain in single-contract mode (one at a time)
void solve_chain_single_contract(
    const std::vector<ContractParams>& contracts,
    const std::vector<double>& grid,
    size_t n_time_steps) {

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

    // Boundary condition functions (Dirichlet, zero at boundaries)
    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };

    // Solve each contract individually
    for (const auto& contract : contracts) {
        // Time domain
        TimeDomain time_domain(0.0, contract.maturity, contract.maturity / n_time_steps);

        // Initial condition: American put payoff max(K - S, 0)
        auto initial_condition = [&](std::span<const double> x, std::span<double> u) {
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                u[i] = std::max(contract.strike - S, 0.0);
            }
        };

        // Obstacle condition: American put payoff
        auto obstacle = [&](double, std::span<const double> x, std::span<double> psi) {
            for (size_t i = 0; i < x.size(); ++i) {
                const double S = contract.strike * std::exp(x[i]);
                psi[i] = std::max(contract.strike - S, 0.0);
            }
        };

        // Black-Scholes PDE
        BlackScholesPDE pde(contract.volatility, contract.rate, contract.dividend);
        auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
        auto spatial_op = SpatialOperator(pde, spacing);

        // Boundary conditions
        auto left_bc = DirichletBC(left_bc_func);
        auto right_bc = DirichletBC(right_bc_func);

        // Create solver (single-contract mode)
        PDESolver solver(
            grid, time_domain,
            trbdf2_config, root_config,
            left_bc, right_bc, spatial_op,
            obstacle
        );

        solver.initialize(initial_condition);
        auto result = solver.solve();

        if (!result.has_value()) {
            throw std::runtime_error("Single-contract solver failed to converge");
        }
    }
}

// Solve contract chain in batch mode (multiple contracts at once)
void solve_chain_batch_mode(
    const std::vector<ContractParams>& contracts,
    const std::vector<double>& grid,
    size_t n_time_steps,
    size_t batch_width) {

    const size_t n = grid.size();
    const size_t n_contracts = contracts.size();

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

    // Boundary condition functions
    auto left_bc_func = [](double, double) { return 0.0; };
    auto right_bc_func = [](double, double) { return 0.0; };
    auto left_bc = DirichletBC(left_bc_func);
    auto right_bc = DirichletBC(right_bc_func);

    // Process contracts in batches
    for (size_t batch_start = 0; batch_start < n_contracts; batch_start += batch_width) {
        size_t current_batch_width = std::min(batch_width, n_contracts - batch_start);

        // For simplicity, assume all contracts in batch have same maturity
        // (real implementation would handle varying time domains)
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

        // Create spatial operator with batch PDEs
        auto spacing = std::make_shared<GridSpacing<double>>(GridView<double>(grid));
        auto spatial_op = SpatialOperator<BlackScholesPDE<double>, double>(pdes, spacing);

        // Initial condition (same for all lanes in this batch)
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

        // Create batch solver
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

// Benchmark: Single-contract mode (baseline)
static void BM_ContractChain_SingleContract(benchmark::State& state) {
    const size_t n_contracts = state.range(0);
    const size_t n_space = 101;  // Realistic grid size
    const size_t n_time = 1000;  // Realistic time steps

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);
    auto contracts = generate_contract_chain(n_contracts);

    for (auto _ : state) {
        solve_chain_single_contract(contracts, grid, n_time);
    }

    // Report metrics
    const size_t simd_width = stdx::native_simd<double>::size();
    state.counters["contracts"] = n_contracts;
    state.counters["simd_width"] = simd_width;
    state.counters["throughput_contracts_per_sec"] = benchmark::Counter(
        n_contracts * state.iterations(), benchmark::Counter::kIsRate);
}

// Benchmark: Batch mode
static void BM_ContractChain_BatchMode(benchmark::State& state) {
    const size_t n_contracts = state.range(0);
    const size_t n_space = 101;
    const size_t n_time = 1000;
    const size_t batch_width = stdx::native_simd<double>::size();

    auto grid = create_uniform_grid(-1.0, 1.0, n_space);
    auto contracts = generate_contract_chain(n_contracts);

    for (auto _ : state) {
        solve_chain_batch_mode(contracts, grid, n_time, batch_width);
    }

    // Report metrics
    state.counters["contracts"] = n_contracts;
    state.counters["simd_width"] = batch_width;
    state.counters["throughput_contracts_per_sec"] = benchmark::Counter(
        n_contracts * state.iterations(), benchmark::Counter::kIsRate);
}

// Register benchmarks for chain sizes: 10, 20, 30, 50 contracts
// Use smaller iteration count to get results faster
BENCHMARK(BM_ContractChain_SingleContract)
    ->Arg(10)
    ->Arg(20)
    ->Arg(30)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(5.0);  // Run for at least 5 seconds

BENCHMARK(BM_ContractChain_BatchMode)
    ->Arg(10)
    ->Arg(20)
    ->Arg(30)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(5.0);  // Run for at least 5 seconds

}  // namespace
}  // namespace mango

BENCHMARK_MAIN();
