#include "src/option/option_chain_solver.hpp"
#include "src/option/american_option.hpp"
#include <benchmark/benchmark.h>
#include <vector>

namespace mango {
namespace {

// Benchmark: Old batch API (no workspace reuse)
static void BM_OldBatchAPI_10Strikes(benchmark::State& state) {
    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    // Create 10 separate option params (old API style)
    std::vector<AmericanOptionParams> params;
    for (int i = 0; i < 10; ++i) {
        AmericanOptionParams p;
        p.strike = 90.0 + i * 2.0;  // 90, 92, 94, ..., 108
        p.spot = 100.0;
        p.maturity = 1.0;
        p.volatility = 0.20;
        p.rate = 0.05;
        p.continuous_dividend_yield = 0.02;
        p.option_type = OptionType::PUT;
        params.push_back(p);
    }

    for (auto _ : state) {
        auto results = BatchAmericanOptionSolver::solve_batch(params, grid);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * 10);  // 10 options per iteration
}

// Benchmark: New chain API (with workspace reuse)
static void BM_NewChainAPI_10Strikes(benchmark::State& state) {
    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    // Create chain with 10 strikes
    AmericanOptionChain chain;
    chain.spot = 100.0;
    chain.maturity = 1.0;
    chain.volatility = 0.20;
    chain.rate = 0.05;
    chain.continuous_dividend_yield = 0.02;
    chain.option_type = OptionType::PUT;
    chain.strikes = {90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0};

    for (auto _ : state) {
        auto results = OptionChainSolver::solve_chain(chain, grid);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * 10);  // 10 options per iteration
}

// Benchmark: Multiple chains in parallel
static void BM_MultipleChains_Parallel(benchmark::State& state) {
    const size_t n_chains = state.range(0);

    AmericanOptionGrid grid;
    grid.n_space = 101;
    grid.n_time = 1000;
    grid.x_min = -3.0;
    grid.x_max = 3.0;

    // Create multiple chains with slight variations
    std::vector<AmericanOptionChain> chains;
    for (size_t i = 0; i < n_chains; ++i) {
        AmericanOptionChain chain;
        chain.spot = 100.0;
        chain.maturity = 1.0;
        chain.volatility = 0.20 + i * 0.01;  // Vary volatility slightly
        chain.rate = 0.05;
        chain.continuous_dividend_yield = 0.02;
        chain.option_type = OptionType::PUT;
        chain.strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
        chains.push_back(chain);
    }

    size_t total_options = n_chains * 5;  // 5 strikes per chain

    for (auto _ : state) {
        auto results = OptionChainSolver::solve_chains(chains, grid);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * total_options);
}

BENCHMARK(BM_OldBatchAPI_10Strikes)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_NewChainAPI_10Strikes)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MultipleChains_Parallel)->Arg(1)->Arg(4)->Arg(8)->Arg(16)->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace mango

BENCHMARK_MAIN();
