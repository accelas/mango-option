# Option Chain Batch Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add proper batch API for option chains that exploits workspace reuse and OpenMP parallelization for 1.1-1.25x speedup and 10x less memory allocation.

**Architecture:** Three-mode API: (1) Sequential within chain with workspace reuse, (2) Parallel across chains (default), (3) Advanced thread pool with dynamic scheduling. Key insight: parallelize where workspace reuse doesn't matter (across chains), sequential where it matters (within chain).

**Tech Stack:** C++23, OpenMP, GoogleTest, Google Benchmark, Bazel

---

## Background

### Current Batch API Problems

The existing `BatchAmericanOptionSolver` (`src/option/american_option.hpp:323-352`) is just a convenience wrapper that solves each option independently with `#pragma omp parallel for`. It doesn't exploit option chain structure:
- Same underlying, expiration, volatility, rate across all strikes
- Only strike K differs

This results in:
- No workspace reuse (~10 KB allocated per option)
- Cold cache for each thread
- No exploitation of shared parameters

### Option Chain Structure

An option chain has:
- **Shared parameters**: spot S, maturity T, volatility σ, rate r, dividend q, option_type
- **Variable parameter**: strikes K₁, K₂, ..., Kₙ (typically 10-50)

Example: SPY Dec-19-2025 Puts with strikes [400, 420, 440, 460, ...]

### Performance Opportunity

**Current**: 10 strikes × 10 KB/strike = 100 KB allocations, cold cache
**Target**: 1 workspace × 10 KB = 10 KB allocations, warm cache
**Expected**: 1.1-1.25x speedup + 10x less memory

## Phase 1: Core Data Structures

### Task 1.1: Create Chain Configuration Struct

**Files:**
- Create: `src/option/option_chain_solver.hpp`

**Step 1: Create header with option chain struct**

Create `src/option/option_chain_solver.hpp`:

```cpp
/**
 * @file option_chain_solver.hpp
 * @brief Batch solver optimized for option chains (same S, T, σ, r, q; different K)
 *
 * An option chain is a set of options sharing all parameters except strike.
 * This solver exploits that structure for better performance:
 * - Workspace reuse: one SliceSolverWorkspace per chain
 * - Cache-friendly: sequential solving keeps workspace hot
 * - 10x less allocation: ~10 KB per chain vs ~10 KB per option
 */

#pragma once

#include "src/option/american_option.hpp"
#include "src/option/slice_solver_workspace.hpp"
#include "src/support/expected.hpp"
#include <vector>
#include <span>

namespace mango {

/**
 * Option chain configuration.
 *
 * Represents multiple options sharing all parameters except strike.
 * Typical use: All puts (or calls) for same underlying and expiration.
 */
struct AmericanOptionChain {
    double spot;                        ///< Current spot price (shared)
    double maturity;                    ///< Time to maturity in years (shared)
    double volatility;                  ///< Implied volatility (shared)
    double rate;                        ///< Risk-free rate (shared)
    double continuous_dividend_yield;   ///< Continuous dividend yield (shared)
    OptionType option_type;             ///< CALL or PUT (shared)
    std::vector<double> strikes;        ///< Strike prices [K₁, K₂, ..., Kₙ] (variable)

    /// Optional: discrete dividends (shared across all strikes)
    std::vector<std::pair<double, double>> discrete_dividends;

    /// Validate chain parameters
    expected<void, std::string> validate() const {
        if (spot <= 0.0) {
            return unexpected("Spot price must be positive");
        }
        if (maturity <= 0.0) {
            return unexpected("Maturity must be positive");
        }
        if (volatility <= 0.0) {
            return unexpected("Volatility must be positive");
        }
        if (continuous_dividend_yield < 0.0) {
            return unexpected("Continuous dividend yield must be non-negative");
        }
        if (strikes.empty()) {
            return unexpected("Chain must have at least one strike");
        }
        for (double k : strikes) {
            if (k <= 0.0) {
                return unexpected("All strikes must be positive");
            }
        }
        // Validate discrete dividends
        for (const auto& [time, amount] : discrete_dividends) {
            if (time < 0.0 || time > maturity) {
                return unexpected("Discrete dividend time must be in [0, maturity]");
            }
            if (amount < 0.0) {
                return unexpected("Discrete dividend amount must be non-negative");
            }
        }
        return {};
    }
};

/**
 * Result for one strike in a chain.
 */
struct ChainStrikeResult {
    double strike;                                          ///< Strike price
    expected<AmericanOptionResult, SolverError> result;     ///< Price + Greeks or error
};

}  // namespace mango
```

**Step 2: Add BUILD.bazel target**

Modify `src/option/BUILD.bazel`, add to `hdrs` list of existing `option` target:

```python
cc_library(
    name = "option",
    hdrs = [
        "american_option.hpp",
        "iv_solver.hpp",
        "option_chain_solver.hpp",  # ADD THIS LINE
        "price_table_4d_builder.hpp",
        "price_table_snapshot_collector.hpp",
        "slice_solver_workspace.hpp",
    ],
    # ... rest unchanged
)
```

**Step 3: Verify it compiles**

Run:
```bash
bazel build //src/option:option
```

Expected: Build succeeds

**Step 4: Commit header**

```bash
git add src/option/option_chain_solver.hpp src/option/BUILD.bazel
git commit -m "feat(option): add option chain configuration struct"
```

---

### Task 1.2: Add Validation Test

**Files:**
- Create: `tests/option_chain_solver_test.cc`
- Modify: `tests/BUILD.bazel`

**Step 1: Write failing validation test**

Create `tests/option_chain_solver_test.cc`:

```cpp
#include "src/option/option_chain_solver.hpp"
#include <gtest/gtest.h>

namespace mango {
namespace {

TEST(AmericanOptionChainTest, ValidChainPassesValidation) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0},
        .discrete_dividends = {}
    };

    auto result = chain.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(AmericanOptionChainTest, NegativeSpotFails) {
    AmericanOptionChain chain{
        .spot = -100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("Spot"), std::string::npos);
}

TEST(AmericanOptionChainTest, EmptyStrikesFails) {
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {},  // Empty!
        .discrete_dividends = {}
    };

    auto result = chain.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("at least one strike"), std::string::npos);
}

}  // namespace
}  // namespace mango
```

**Step 2: Add test target to BUILD.bazel**

Modify `tests/BUILD.bazel`, add:

```python
cc_test(
    name = "option_chain_solver_test",
    size = "small",
    srcs = ["option_chain_solver_test.cc"],
    deps = [
        "//src/option:option",
        "@googletest//:gtest_main",
    ],
)
```

**Step 3: Run test to verify it passes**

Run:
```bash
bazel test //tests:option_chain_solver_test --test_output=all
```

Expected: All 3 tests PASS (validation is in header, already implemented)

**Step 4: Commit validation test**

```bash
git add tests/option_chain_solver_test.cc tests/BUILD.bazel
git commit -m "test(option): add option chain validation tests"
```

---

## Phase 2: Sequential Chain Solver (Mode 1)

### Task 2.1: Implement Sequential Solver

**Files:**
- Modify: `src/option/option_chain_solver.hpp`

**Step 1: Add OptionChainSolver class declaration**

Add to `src/option/option_chain_solver.hpp` after `ChainStrikeResult`:

```cpp
/**
 * Batch solver optimized for option chains.
 *
 * Provides three modes:
 * 1. solve_chain() - Sequential within chain (workspace reuse)
 * 2. solve_chains() - Parallel across chains (default)
 * 3. solve_chains_advanced() - Thread pool with dynamic scheduling (future)
 */
class OptionChainSolver {
public:
    /**
     * Solve option chain sequentially with workspace reuse.
     *
     * Creates one SliceSolverWorkspace for entire chain and solves
     * all strikes sequentially. This keeps the workspace "hot" and
     * minimizes allocation overhead.
     *
     * Use when: Single chain, or when called from parallel context.
     *
     * Performance: ~10x less allocation, cache-friendly.
     *
     * @param chain Chain configuration (shared params, different strikes)
     * @param grid PDE grid configuration
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration
     * @return Results for each strike (same order as chain.strikes)
     */
    static std::vector<ChainStrikeResult> solve_chain(
        const AmericanOptionChain& chain,
        const AmericanOptionGrid& grid,
        const TRBDF2Config& trbdf2_config = {},
        const RootFindingConfig& root_config = {});
};
```

**Step 2: Create implementation file**

Create `src/option/option_chain_solver.cpp`:

```cpp
#include "src/option/option_chain_solver.hpp"
#include "src/option/american_option.hpp"
#include "src/option/slice_solver_workspace.hpp"

namespace mango {

std::vector<ChainStrikeResult> OptionChainSolver::solve_chain(
    const AmericanOptionChain& chain,
    const AmericanOptionGrid& grid,
    const TRBDF2Config& trbdf2_config,
    const RootFindingConfig& root_config)
{
    // Validate chain
    auto validation = chain.validate();
    if (!validation.has_value()) {
        // Return error for all strikes
        std::vector<ChainStrikeResult> results;
        results.reserve(chain.strikes.size());
        for (double strike : chain.strikes) {
            results.push_back({
                strike,
                unexpected(SolverError{
                    .type = SolverErrorType::VALIDATION_ERROR,
                    .message = validation.error()
                })
            });
        }
        return results;
    }

    // Validate grid
    auto grid_validation = AmericanOptionGrid::validate_expected(grid);
    if (!grid_validation.has_value()) {
        // Return error for all strikes
        std::vector<ChainStrikeResult> results;
        results.reserve(chain.strikes.size());
        for (double strike : chain.strikes) {
            results.push_back({
                strike,
                unexpected(SolverError{
                    .type = SolverErrorType::VALIDATION_ERROR,
                    .message = grid_validation.error()
                })
            });
        }
        return results;
    }

    std::vector<ChainStrikeResult> results;
    results.reserve(chain.strikes.size());

    // Create workspace ONCE for entire chain
    auto workspace = std::make_shared<SliceSolverWorkspace>(
        grid.x_min, grid.x_max, grid.n_space);

    // Solve each strike SEQUENTIALLY (workspace stays hot)
    for (double strike : chain.strikes) {
        AmericanOptionParams params{
            .strike = strike,
            .spot = chain.spot,
            .maturity = chain.maturity,
            .volatility = chain.volatility,
            .rate = chain.rate,
            .continuous_dividend_yield = chain.continuous_dividend_yield,
            .option_type = chain.option_type,
            .discrete_dividends = chain.discrete_dividends
        };

        // Reuse workspace for this strike
        AmericanOptionSolver solver(params, grid, workspace, trbdf2_config, root_config);
        auto result = solver.solve();

        results.push_back({strike, std::move(result)});
    }

    return results;
}

}  // namespace mango
```

**Step 3: Update BUILD.bazel**

Modify `src/option/BUILD.bazel`, add to `srcs` list:

```python
cc_library(
    name = "option",
    hdrs = [
        # ... existing hdrs
        "option_chain_solver.hpp",
    ],
    srcs = [
        # ... existing srcs
        "option_chain_solver.cpp",  # ADD THIS
    ],
    # ... rest unchanged
)
```

**Step 4: Build and verify**

Run:
```bash
bazel build //src/option:option
```

Expected: Build succeeds

**Step 5: Commit implementation**

```bash
git add src/option/option_chain_solver.hpp src/option/option_chain_solver.cpp src/option/BUILD.bazel
git commit -m "feat(option): implement sequential option chain solver"
```

---

### Task 2.2: Test Sequential Solver

**Files:**
- Modify: `tests/option_chain_solver_test.cc`

**Step 1: Write test for sequential solver**

Add to `tests/option_chain_solver_test.cc`:

```cpp
TEST(OptionChainSolverTest, SolvesSimpleChainSequentially) {
    // Create chain: 5 put strikes around ATM
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0},
        .discrete_dividends = {}
    };

    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    auto results = OptionChainSolver::solve_chain(chain, grid);

    ASSERT_EQ(results.size(), 5);

    // All should converge
    for (const auto& [strike, result] : results) {
        ASSERT_TRUE(result.has_value()) << "Strike " << strike << " failed to converge";
        EXPECT_GT(result->value, 0.0) << "Strike " << strike << " has non-positive value";
    }

    // Put values should increase as strike increases (intrinsic value)
    EXPECT_LT(results[0].result->value, results[4].result->value)
        << "Deep OTM put should be cheaper than deep ITM put";
}

TEST(OptionChainSolverTest, HandlesInvalidChainGracefully) {
    // Invalid chain: negative volatility
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = -0.20,  // Invalid!
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {100.0}
    };

    AmericanOptionGrid grid;

    auto results = OptionChainSolver::solve_chain(chain, grid);

    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].result.has_value());
    EXPECT_EQ(results[0].result.error().type, SolverErrorType::VALIDATION_ERROR);
}
```

**Step 2: Run tests**

Run:
```bash
bazel test //tests:option_chain_solver_test --test_output=all
```

Expected: All tests PASS

**Step 3: Commit test**

```bash
git add tests/option_chain_solver_test.cc
git commit -m "test(option): add sequential chain solver tests"
```

---

## Phase 3: Parallel Chain Solver (Mode 2)

### Task 3.1: Implement Parallel Solver

**Files:**
- Modify: `src/option/option_chain_solver.hpp`
- Modify: `src/option/option_chain_solver.cpp`

**Step 1: Add solve_chains declaration**

Add to `OptionChainSolver` class in `src/option/option_chain_solver.hpp`:

```cpp
    /**
     * Solve multiple option chains in parallel.
     *
     * Each chain is solved sequentially (workspace reuse), but chains
     * are processed in parallel using OpenMP. This is the recommended
     * mode for typical use cases.
     *
     * Parallelization strategy:
     * - Parallelize ACROSS chains (not within)
     * - Each thread gets one chain at a time
     * - Sequential solve within chain keeps workspace hot
     *
     * Use when: Multiple chains (typical case).
     *
     * Performance: Same per-chain benefit as solve_chain(), scaled across cores.
     *
     * @param chains Vector of option chains
     * @param grid PDE grid configuration (shared across all chains)
     * @param trbdf2_config TR-BDF2 solver configuration
     * @param root_config Root finding configuration
     * @return Results for each chain (same order as input)
     */
    static std::vector<std::vector<ChainStrikeResult>> solve_chains(
        std::span<const AmericanOptionChain> chains,
        const AmericanOptionGrid& grid,
        const TRBDF2Config& trbdf2_config = {},
        const RootFindingConfig& root_config = {});
```

**Step 2: Implement solve_chains**

Add to `src/option/option_chain_solver.cpp`:

```cpp
std::vector<std::vector<ChainStrikeResult>> OptionChainSolver::solve_chains(
    std::span<const AmericanOptionChain> chains,
    const AmericanOptionGrid& grid,
    const TRBDF2Config& trbdf2_config,
    const RootFindingConfig& root_config)
{
    std::vector<std::vector<ChainStrikeResult>> all_results(chains.size());

    // Parallelize ACROSS chains (not within)
    // Each thread solves one chain sequentially with workspace reuse
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t i = 0; i < chains.size(); ++i) {
        all_results[i] = solve_chain(chains[i], grid, trbdf2_config, root_config);
    }

    return all_results;
}
```

**Step 3: Build and verify**

Run:
```bash
bazel build //src/option:option
```

Expected: Build succeeds

**Step 4: Commit implementation**

```bash
git add src/option/option_chain_solver.hpp src/option/option_chain_solver.cpp
git commit -m "feat(option): implement parallel option chain solver"
```

---

### Task 3.2: Test Parallel Solver

**Files:**
- Modify: `tests/option_chain_solver_test.cc`

**Step 1: Write test for parallel solver**

Add to `tests/option_chain_solver_test.cc`:

```cpp
TEST(OptionChainSolverTest, SolvesMultipleChainsInParallel) {
    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    // Create 3 chains with different parameters
    std::vector<AmericanOptionChain> chains = {
        // Chain 1: ATM puts
        {
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .strikes = {95.0, 100.0, 105.0}
        },
        // Chain 2: OTM calls
        {
            .spot = 100.0,
            .maturity = 0.5,
            .volatility = 0.30,
            .rate = 0.03,
            .continuous_dividend_yield = 0.01,
            .option_type = OptionType::CALL,
            .strikes = {105.0, 110.0, 115.0}
        },
        // Chain 3: Deep ITM puts
        {
            .spot = 100.0,
            .maturity = 2.0,
            .volatility = 0.25,
            .rate = 0.04,
            .continuous_dividend_yield = 0.015,
            .option_type = OptionType::PUT,
            .strikes = {120.0, 125.0, 130.0}
        }
    };

    auto all_results = OptionChainSolver::solve_chains(chains, grid);

    ASSERT_EQ(all_results.size(), 3);

    // Check each chain
    for (size_t i = 0; i < chains.size(); ++i) {
        ASSERT_EQ(all_results[i].size(), chains[i].strikes.size())
            << "Chain " << i << " has wrong number of results";

        for (const auto& [strike, result] : all_results[i]) {
            ASSERT_TRUE(result.has_value())
                << "Chain " << i << ", strike " << strike << " failed";
            EXPECT_GT(result->value, 0.0)
                << "Chain " << i << ", strike " << strike << " has non-positive value";
        }
    }
}

TEST(OptionChainSolverTest, ParallelMatchesSequential) {
    // Test that parallel execution gives same results as sequential
    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 95.0, 100.0, 105.0, 110.0}
    };

    // Solve sequentially
    auto sequential_results = OptionChainSolver::solve_chain(chain, grid);

    // Solve in parallel (single chain, but tests parallel path)
    std::vector<AmericanOptionChain> chains = {chain};
    auto parallel_results = OptionChainSolver::solve_chains(chains, grid);

    ASSERT_EQ(parallel_results.size(), 1);
    ASSERT_EQ(parallel_results[0].size(), sequential_results.size());

    // Compare results
    for (size_t i = 0; i < sequential_results.size(); ++i) {
        ASSERT_TRUE(sequential_results[i].result.has_value());
        ASSERT_TRUE(parallel_results[0][i].result.has_value());

        EXPECT_NEAR(sequential_results[i].result->value,
                   parallel_results[0][i].result->value,
                   1e-10)
            << "Mismatch at strike " << sequential_results[i].strike;
    }
}
```

**Step 2: Run tests**

Run:
```bash
bazel test //tests:option_chain_solver_test --test_output=all
```

Expected: All tests PASS

**Step 3: Commit test**

```bash
git add tests/option_chain_solver_test.cc
git commit -m "test(option): add parallel chain solver tests"
```

---

## Phase 4: Performance Validation

### Task 4.1: Create Performance Benchmark

**Files:**
- Create: `benchmarks/option_chain_benchmark.cc`
- Modify: `benchmarks/BUILD.bazel`

**Step 1: Write benchmark comparing old vs new API**

Create `benchmarks/option_chain_benchmark.cc`:

```cpp
#include "src/option/option_chain_solver.hpp"
#include "src/option/american_option.hpp"
#include <benchmark/benchmark.h>
#include <vector>

namespace mango {
namespace {

// Benchmark: Old batch API (no workspace reuse)
static void BM_OldBatchAPI_10Strikes(benchmark::State& state) {
    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    // Create 10 separate option params (old API style)
    std::vector<AmericanOptionParams> params;
    for (int i = 0; i < 10; ++i) {
        params.push_back({
            .strike = 90.0 + i * 2.0,  // 90, 92, 94, ..., 108
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20,
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .discrete_dividends = {}
        });
    }

    for (auto _ : state) {
        auto results = BatchAmericanOptionSolver::solve_batch(params, grid);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * 10);  // 10 options per iteration
}

// Benchmark: New chain API (with workspace reuse)
static void BM_NewChainAPI_10Strikes(benchmark::State& state) {
    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    // Create chain with 10 strikes
    AmericanOptionChain chain{
        .spot = 100.0,
        .maturity = 1.0,
        .volatility = 0.20,
        .rate = 0.05,
        .continuous_dividend_yield = 0.02,
        .option_type = OptionType::PUT,
        .strikes = {90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0}
    };

    for (auto _ : state) {
        auto results = OptionChainSolver::solve_chain(chain, grid);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() * 10);  // 10 options per iteration
}

// Benchmark: Multiple chains in parallel
static void BM_MultipleChains_Parallel(benchmark::State& state) {
    const size_t n_chains = state.range(0);

    AmericanOptionGrid grid{
        .n_space = 101,
        .n_time = 1000,
        .x_min = -3.0,
        .x_max = 3.0
    };

    // Create multiple chains with slight variations
    std::vector<AmericanOptionChain> chains;
    for (size_t i = 0; i < n_chains; ++i) {
        chains.push_back({
            .spot = 100.0,
            .maturity = 1.0,
            .volatility = 0.20 + i * 0.01,  // Vary volatility slightly
            .rate = 0.05,
            .continuous_dividend_yield = 0.02,
            .option_type = OptionType::PUT,
            .strikes = {90.0, 95.0, 100.0, 105.0, 110.0}
        });
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
```

**Step 2: Add benchmark target**

Modify `benchmarks/BUILD.bazel`, add:

```python
cc_test(
    name = "option_chain_benchmark",
    srcs = ["option_chain_benchmark.cc"],
    deps = [
        "//src/option:option",
        "@com_google_benchmark//:benchmark_main",
    ],
)
```

**Step 3: Build benchmark**

Run:
```bash
bazel build //benchmarks:option_chain_benchmark
```

Expected: Build succeeds

**Step 4: Run benchmark and capture results**

Run:
```bash
bazel run //benchmarks:option_chain_benchmark --config=opt
```

Expected output (approximate):
```
BM_OldBatchAPI_10Strikes          XXX ms          XXX ms
BM_NewChainAPI_10Strikes          YYY ms          YYY ms    (YYY < XXX, ~10-20% faster)
BM_MultipleChains_Parallel/1      ZZZ ms
BM_MultipleChains_Parallel/4      AAA ms          (scales with cores)
BM_MultipleChains_Parallel/8      BBB ms
BM_MultipleChains_Parallel/16     CCC ms
```

**Step 5: Commit benchmark**

```bash
git add benchmarks/option_chain_benchmark.cc benchmarks/BUILD.bazel
git commit -m "perf(option): add option chain solver benchmarks"
```

---

## Phase 5: Documentation

### Task 5.1: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add option chain section to CLAUDE.md**

Add after the "Unified Root-Finding API" section (around line 269):

```markdown
## Option Chain Batch Solver

The library provides an optimized batch solver for option chains that exploits workspace reuse and cache locality.

### What is an Option Chain?

An **option chain** is a set of options sharing all parameters except strike:
- Same underlying (SPY, AAPL, etc.)
- Same expiration date
- Same volatility, rate, dividend
- **Different strikes** (typically 10-50)

Example: SPY Dec-19-2025 Puts with strikes [400, 420, 440, 460, ...]

### Why Use Option Chain Solver?

**Performance benefits over naive batch solving:**
- **10x less memory allocation**: One workspace per chain (~10 KB) vs per option (~10 KB each)
- **Cache-friendly**: Sequential solving keeps workspace hot
- **1.1-1.25x speedup**: From workspace reuse and cache warmth

### Basic Usage

```cpp
#include "src/option/option_chain_solver.hpp"

// Define chain: 5 put strikes, shared parameters
mango::AmericanOptionChain chain{
    .spot = 100.0,
    .maturity = 1.0,
    .volatility = 0.20,
    .rate = 0.05,
    .continuous_dividend_yield = 0.02,
    .option_type = mango::OptionType::PUT,
    .strikes = {90.0, 95.0, 100.0, 105.0, 110.0}
};

mango::AmericanOptionGrid grid{
    .n_space = 101,
    .n_time = 1000
};

// Solve entire chain with workspace reuse
auto results = mango::OptionChainSolver::solve_chain(chain, grid);

// Access results
for (const auto& [strike, result] : results) {
    if (result.has_value()) {
        std::cout << "Strike " << strike << ": $" << result->value << "\n";
    }
}
```

### Multiple Chains (Parallel)

```cpp
// Create multiple chains (e.g., different expirations or underlyings)
std::vector<mango::AmericanOptionChain> chains = {
    { /* chain 1: Dec-2025 */ },
    { /* chain 2: Jan-2026 */ },
    { /* chain 3: Feb-2026 */ }
};

// Solve in parallel (chains parallelized, strikes sequential within chain)
auto all_results = mango::OptionChainSolver::solve_chains(chains, grid);

// all_results[i][j] = result for chains[i].strikes[j]
```

### When to Use

✅ **Use OptionChainSolver when:**
- Pricing multiple strikes with same expiration/parameters
- Processing option chains from market data
- Building IV surfaces (multiple chains across expirations)

❌ **Use BatchAmericanOptionSolver when:**
- Options have heterogeneous parameters (different T, σ, r)
- No shared structure to exploit
- Convenience wrapper for embarrassingly parallel work

### Performance Characteristics

**Single chain (10 strikes)**:
- Old batch API: 10 × 10 KB = 100 KB allocated, cold cache
- New chain API: 1 × 10 KB = 10 KB allocated, warm cache
- Speedup: ~1.1-1.25x

**Multiple chains (5 chains × 10 strikes = 50 options)**:
- Parallelization: Across chains (not within)
- Each thread: Sequential solve with workspace reuse
- Scales well up to number of chains
```

**Step 2: Commit documentation**

```bash
git add CLAUDE.md
git commit -m "docs: add option chain batch solver documentation"
```

---

## Phase 6: Python Bindings (Optional)

### Task 6.1: Add Python Bindings for Chain API

**Files:**
- Modify: `python/mango_bindings.cpp`

**Step 1: Add chain struct and function bindings**

Add to `python/mango_bindings.cpp` in the appropriate section:

```cpp
// Add after AmericanOptionParams binding
py::class_<AmericanOptionChain>(m, "AmericanOptionChain")
    .def(py::init<>())
    .def_readwrite("spot", &AmericanOptionChain::spot)
    .def_readwrite("maturity", &AmericanOptionChain::maturity)
    .def_readwrite("volatility", &AmericanOptionChain::volatility)
    .def_readwrite("rate", &AmericanOptionChain::rate)
    .def_readwrite("continuous_dividend_yield", &AmericanOptionChain::continuous_dividend_yield)
    .def_readwrite("option_type", &AmericanOptionChain::option_type)
    .def_readwrite("strikes", &AmericanOptionChain::strikes)
    .def_readwrite("discrete_dividends", &AmericanOptionChain::discrete_dividends)
    .def("validate", &AmericanOptionChain::validate);

py::class_<ChainStrikeResult>(m, "ChainStrikeResult")
    .def_readonly("strike", &ChainStrikeResult::strike)
    .def_readonly("result", &ChainStrikeResult::result);

// Add chain solver functions
m.def("solve_option_chain",
    [](const AmericanOptionChain& chain, const AmericanOptionGrid& grid) {
        return OptionChainSolver::solve_chain(chain, grid);
    },
    py::arg("chain"),
    py::arg("grid"),
    "Solve option chain with workspace reuse");

m.def("solve_option_chains",
    [](const std::vector<AmericanOptionChain>& chains, const AmericanOptionGrid& grid) {
        return OptionChainSolver::solve_chains(chains, grid);
    },
    py::arg("chains"),
    py::arg("grid"),
    "Solve multiple option chains in parallel");
```

**Step 2: Build Python module**

Run:
```bash
bazel build //python:mango_option
```

Expected: Build succeeds

**Step 3: Test from Python**

Run:
```bash
python3 -c "
import mango_option

chain = mango_option.AmericanOptionChain()
chain.spot = 100.0
chain.maturity = 1.0
chain.volatility = 0.20
chain.rate = 0.05
chain.continuous_dividend_yield = 0.02
chain.option_type = mango_option.OptionType.PUT
chain.strikes = [90.0, 95.0, 100.0, 105.0, 110.0]

grid = mango_option.AmericanOptionGrid()
results = mango_option.solve_option_chain(chain, grid)

print(f'Solved {len(results)} strikes')
for r in results:
    print(f'Strike {r.strike}: ${r.result.value:.2f}')
"
```

Expected: Prints 5 strike prices

**Step 4: Commit Python bindings**

```bash
git add python/mango_bindings.cpp
git commit -m "feat(python): add option chain solver bindings"
```

---

## Completion Checklist

- [ ] Phase 1: Core data structures (AmericanOptionChain, ChainStrikeResult)
- [ ] Phase 2: Sequential chain solver (solve_chain with workspace reuse)
- [ ] Phase 3: Parallel chain solver (solve_chains across chains)
- [ ] Phase 4: Performance benchmarks (validate 1.1-1.25x speedup)
- [ ] Phase 5: Documentation (update CLAUDE.md)
- [ ] Phase 6: Python bindings (optional, for Python users)

## Testing Strategy

**Unit tests** (`tests/option_chain_solver_test.cc`):
- Validation (positive and negative cases)
- Sequential solver correctness
- Parallel solver correctness
- Sequential vs parallel consistency

**Performance tests** (`benchmarks/option_chain_benchmark.cc`):
- Old batch API vs new chain API
- Single chain performance
- Multi-chain scaling with thread count

## Performance Expectations

**Expected improvements over BatchAmericanOptionSolver**:
- Memory: 10x reduction (10 KB per chain vs 10 KB per option)
- Speed: 1.1-1.25x faster (from cache warmth + reduced allocation)
- Scalability: Good up to number of chains (parallelism at chain level)

## Future Enhancements

**Phase 7 (Future): Advanced Thread Pool Mode**
- Flatten all strikes into single work queue
- Thread-local workspaces with dynamic scheduling
- Better load balancing for heterogeneous chains
- Target: Additional 1.1-1.2x speedup for 100+ total strikes

Implementation deferred until benchmarks show need.
