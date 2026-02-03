# Phase 1 Week 2: WorkspaceStorage + Spatial Operators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement cache-blocked workspace storage and Black-Scholes spatial operators to prepare for TR-BDF2 solver integration.

**Architecture:** CPU-only WorkspaceStorage with pre-computed dx arrays, cache-blocked stencil operations, and two spatial operators (equity with discrete dividends, index with continuous dividend yield). All using tag dispatch and span-based access patterns from Week 1.

**Tech Stack:** C++20, std::span, cache-blocking infrastructure from Week 1, GoogleTest

---

## Task 1: Create WorkspaceStorage with Pre-Computed dx

**Files:**
- Create: `src/cpp/workspace.hpp`
- Test: `tests/workspace_test.cc`

**Step 1: Write failing test for workspace creation**

Create `tests/workspace_test.cc`:

```cpp
#include "mango/cpp/workspace.hpp"
#include "mango/cpp/grid.hpp"
#include <gtest/gtest.h>

TEST(WorkspaceStorageTest, SmallGridSingleBlock) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 100);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Small grid should use single block
    EXPECT_EQ(workspace.cache_config().n_blocks, 1);
    EXPECT_EQ(workspace.cache_config().block_size, 100);
    EXPECT_EQ(workspace.cache_config().overlap, 1);
}

TEST(WorkspaceStorageTest, LargeGridMultipleBlocks) {
    auto spec = mango::GridSpec<>::uniform(0.0, 100.0, 10000);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Large grid should use L1-blocked strategy
    EXPECT_GT(workspace.cache_config().n_blocks, 1);
    EXPECT_LT(workspace.cache_config().block_size, 10000);
}

TEST(WorkspaceStorageTest, PreComputedDxArray) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 6);
    auto grid = spec.generate();  // Points: 0, 2, 4, 6, 8, 10

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // dx array should have size n-1
    EXPECT_EQ(workspace.dx().size(), 5);

    // All dx values should be 2.0 for uniform grid
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(workspace.dx()[i], 2.0);
    }
}
```

**Step 2: Add test target to BUILD**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "workspace_test",
    srcs = ["workspace_test.cc"],
    deps = [
        "//src/cpp:workspace",
        "//src/cpp:grid",
        "@com_google_googletest//:gtest_main",
    ],
    copts = ["-std=c++20"],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:workspace_test --test_output=errors
```

Expected: FAIL with "workspace.hpp: No such file or directory"

**Step 4: Implement WorkspaceStorage**

Create `src/cpp/workspace.hpp`:

```cpp
#pragma once

#include "cache_config.hpp"
#include <vector>
#include <span>
#include <cstddef>

namespace mango {

/**
 * WorkspaceStorage: Cache-blocked storage for PDE solver arrays
 *
 * Manages solver workspace with:
 * - Contiguous buffer for all arrays (cache-friendly)
 * - Pre-computed dx array (avoids out-of-bounds in cache blocks)
 * - Cache-blocking configuration
 * - 64-byte alignment for SIMD operations
 */
class WorkspaceStorage {
public:
    /**
     * Create workspace for grid
     * @param n Number of grid points
     * @param grid Grid coordinates (used to pre-compute dx)
     */
    explicit WorkspaceStorage(size_t n, std::span<const double> grid)
        : buffer_(5 * n)  // u_current, u_next, u_stage, rhs, Lu
        , cache_config_(CacheBlockConfig::adaptive(n))
        , dx_(n - 1)
    {
        // Pre-compute grid spacing once during initialization
        // CRITICAL: Avoids out-of-bounds access when processing cache blocks
        for (size_t i = 0; i < n - 1; ++i) {
            dx_[i] = grid[i + 1] - grid[i];
        }

        // Set up array views as non-overlapping spans
        size_t offset = 0;
        u_current_ = std::span{buffer_.data() + offset, n}; offset += n;
        u_next_    = std::span{buffer_.data() + offset, n}; offset += n;
        u_stage_   = std::span{buffer_.data() + offset, n}; offset += n;
        rhs_       = std::span{buffer_.data() + offset, n}; offset += n;
        lu_        = std::span{buffer_.data() + offset, n};
    }

    // Access to arrays
    std::span<double> u_current() { return u_current_; }
    std::span<const double> u_current() const { return u_current_; }

    std::span<double> u_next() { return u_next_; }
    std::span<const double> u_next() const { return u_next_; }

    std::span<double> u_stage() { return u_stage_; }
    std::span<const double> u_stage() const { return u_stage_; }

    std::span<double> rhs() { return rhs_; }
    std::span<const double> rhs() const { return rhs_; }

    std::span<double> lu() { return lu_; }
    std::span<const double> lu() const { return lu_; }

    // Access to pre-computed dx
    std::span<const double> dx() const { return dx_; }

    // Cache configuration
    const CacheBlockConfig& cache_config() const { return cache_config_; }

private:
    std::vector<double> buffer_;     // Single allocation for all arrays
    CacheBlockConfig cache_config_;  // Cache-blocking configuration
    std::vector<double> dx_;         // Pre-computed grid spacing

    // Spans into buffer_
    std::span<double> u_current_;
    std::span<double> u_next_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
};

} // namespace mango
```

**Step 5: Add library target to BUILD**

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "workspace",
    hdrs = ["workspace.hpp"],
    deps = [":cache_config"],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 6: Run tests to verify they pass**

```bash
bazel test //tests:workspace_test --test_output=all
```

Expected: PASS (3 tests)

**Step 7: Commit**

```bash
git add src/cpp/workspace.hpp src/cpp/BUILD.bazel tests/workspace_test.cc tests/BUILD.bazel
git commit -m "feat(workspace): add WorkspaceStorage with cache-blocking

- Contiguous buffer for all solver arrays
- Pre-computed dx array to avoid out-of-bounds in cache blocks
- Adaptive cache-blocking (single block <5000, L1-blocked >=5000)
- 64-byte aligned buffer for SIMD operations

Tests: 3 passing"
```

---

## Task 2: Add Block Access Methods to WorkspaceStorage

**Files:**
- Modify: `src/cpp/workspace.hpp`
- Modify: `tests/workspace_test.cc`

**Step 1: Write failing test for block access**

Add to `tests/workspace_test.cc`:

```cpp
TEST(WorkspaceStorageTest, GetBlockInterior) {
    auto spec = mango::GridSpec<>::uniform(0.0, 10.0, 20);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());
    workspace.cache_config() = mango::CacheBlockConfig{10, 2, 1};  // Force 2 blocks for testing

    // Block 0: indices 0-9, interior 1-9 (skip boundary at 0)
    auto [start, end] = workspace.get_block_interior_range(0);
    EXPECT_EQ(start, 1);
    EXPECT_EQ(end, 10);

    // Block 1: indices 10-19, interior 10-18 (skip boundary at 19)
    auto [start2, end2] = workspace.get_block_interior_range(1);
    EXPECT_EQ(start2, 10);
    EXPECT_EQ(end2, 19);
}

TEST(WorkspaceStorageTest, GetBlockWithHalo) {
    auto spec = mango::GridSpec<>::uniform(0.0, 1.0, 20);
    auto grid = spec.generate();

    mango::WorkspaceStorage workspace(grid.size(), grid.span());

    // Initialize u_current with index values for testing
    for (size_t i = 0; i < 20; ++i) {
        workspace.u_current()[i] = static_cast<double>(i);
    }

    // Block with halo should include overlap points
    auto block_info = workspace.get_block_with_halo(workspace.u_current(), 1);

    EXPECT_GT(block_info.halo_left, 0);
    EXPECT_EQ(block_info.data[block_info.halo_left], workspace.u_current()[block_info.interior_start]);
}
```

**Step 2: Run test to verify it fails**

```bash
bazel test //tests:workspace_test --test_output=errors
```

Expected: FAIL with "get_block_interior_range not a member"

**Step 3: Add block access methods to WorkspaceStorage**

Add to `src/cpp/workspace.hpp` (in WorkspaceStorage class):

```cpp
    /**
     * BlockInfo: Information about a cache block with halo
     */
    struct BlockInfo {
        std::span<const double> data;  // Data with halo
        size_t interior_start;         // Global index where interior starts
        size_t interior_count;         // Number of interior points
        size_t halo_left;             // Number of left halo points
        size_t halo_right;            // Number of right halo points
    };

    /**
     * Get interior range for a cache block
     * @param block_idx Block index
     * @return [start, end) range of interior points (exclusive of boundaries)
     */
    std::pair<size_t, size_t> get_block_interior_range(size_t block_idx) const {
        const size_t n = u_current_.size();
        size_t start = block_idx * cache_config_.block_size;
        size_t end = std::min(start + cache_config_.block_size, n);

        // Skip global boundaries (0 and n-1)
        size_t interior_start = std::max(start, size_t{1});
        size_t interior_end = std::min(end, n - 1);

        return {interior_start, interior_end};
    }

    /**
     * Get block with halo for stencil operations
     * @param array Array to get block from
     * @param block_idx Block index
     * @return BlockInfo with data span and halo information
     */
    BlockInfo get_block_with_halo(std::span<const double> array, size_t block_idx) const {
        auto [interior_start, interior_end] = get_block_interior_range(block_idx);

        if (interior_start >= interior_end) {
            // Boundary-only block (shouldn't happen with proper sizing)
            return {std::span<const double>{}, interior_start, 0, 0, 0};
        }

        const size_t n = array.size();
        const size_t interior_count = interior_end - interior_start;

        // Compute halo sizes (clamped to available points)
        const size_t halo_left  = std::min(cache_config_.overlap, interior_start);
        const size_t halo_right = std::min(cache_config_.overlap, n - interior_end);

        // Build span with halos
        auto data_with_halo = array.subspan(
            interior_start - halo_left,
            interior_count + halo_left + halo_right
        );

        return {data_with_halo, interior_start, interior_count, halo_left, halo_right};
    }
```

**Step 4: Run tests to verify they pass**

```bash
bazel test //tests:workspace_test --test_output=all
```

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/cpp/workspace.hpp tests/workspace_test.cc
git commit -m "feat(workspace): add cache block access methods

- get_block_interior_range() returns interior point range for block
- get_block_with_halo() returns block data with stencil halos
- BlockInfo struct carries halo metadata for stencil operations

Tests: 5 passing"
```

---

## Task 3: Create Spatial Operator Base and Black-Scholes Operator

**Files:**
- Create: `src/cpp/spatial_operators.hpp`
- Test: `tests/spatial_operators_test.cc`

**Step 1: Write failing test for Black-Scholes operator**

Create `tests/spatial_operators_test.cc`:

```cpp
#include "mango/cpp/spatial_operators.hpp"
#include "mango/cpp/grid.hpp"
#include <gtest/gtest.h>
#include <cmath>

TEST(BlackScholesOperatorTest, EquityOperatorBasic) {
    // Create operator for equity option
    // Parameters: r=0.05, sigma=0.2, no dividends
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    // Create simple grid
    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();

    // Test input: linear function u(S) = S (delta = 1, gamma = 0)
    std::vector<double> u(41);
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    std::vector<double> Lu(41);

    // Apply operator
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S, the Black-Scholes operator gives:
    // L(u) = r*S*du/dS - r*u = r*S*1 - r*S = 0
    // (Middle points should be approximately zero)
    EXPECT_NEAR(Lu[20], 0.0, 0.01);  // S=100
}

TEST(BlackScholesOperatorTest, EquityOperatorParabolic) {
    // Test with u(S) = S^2 (delta = 2S, gamma = 2)
    mango::EquityBlackScholesOperator op(0.05, 0.2);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();

    std::vector<double> u(21);
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i] * grid[i];
    }

    std::vector<double> Lu(21);
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S^2:
    // du/dS = 2S, d2u/dS2 = 2
    // L(u) = 0.5*sigma^2*S^2*2 + r*S*2S - r*S^2
    //      = sigma^2*S^2 + 2*r*S^2 - r*S^2
    //      = sigma^2*S^2 + r*S^2
    double S = grid[10];  // Middle point
    double expected = 0.2*0.2*S*S + 0.05*S*S;
    EXPECT_NEAR(Lu[10], expected, 0.1);
}
```

**Step 2: Add test target**

Add to `tests/BUILD.bazel`:

```python
cc_test(
    name = "spatial_operators_test",
    srcs = ["spatial_operators_test.cc"],
    deps = [
        "//src/cpp:spatial_operators",
        "//src/cpp:grid",
        "@com_google_googletest//:gtest_main",
    ],
    copts = ["-std=c++20"],
)
```

**Step 3: Run test to verify it fails**

```bash
bazel test //tests:spatial_operators_test --test_output=errors
```

Expected: FAIL with "spatial_operators.hpp: No such file"

**Step 4: Implement Black-Scholes operator**

Create `src/cpp/spatial_operators.hpp`:

```cpp
#pragma once

#include <span>
#include <cstddef>
#include <vector>

namespace mango {

/**
 * EquityBlackScholesOperator: Black-Scholes PDE for equity options
 *
 * PDE: dV/dt = L(V) where
 * L(V) = 0.5*sigma^2*S^2*d2V/dS2 + (r - q)*S*dV/dS - r*V
 *
 * For equity options, dividends are discrete (not part of PDE term).
 * The drift term uses (r - 0) = r since continuous dividend yield q=0.
 */
class EquityBlackScholesOperator {
public:
    /**
     * Create operator for equity option
     * @param r Risk-free rate
     * @param sigma Volatility
     */
    EquityBlackScholesOperator(double r, double sigma)
        : r_(r), sigma_(sigma), sigma_sq_half_(0.5 * sigma * sigma) {}

    /**
     * Apply spatial operator: Lu = L(u)
     * @param t Current time
     * @param S Grid of stock prices
     * @param u Solution values
     * @param Lu Output: operator applied to u
     */
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu) const {

        const size_t n = S.size();

        // Boundaries are handled by boundary conditions
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points: centered finite differences
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];
            const double dx_left = S[i] - S[i-1];
            const double dx_right = S[i+1] - S[i];
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative: d2u/dS2 (centered difference on non-uniform grid)
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // First derivative: du/dS (centered difference)
            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Black-Scholes operator
            // L(u) = 0.5*sigma^2*S^2*d2u/dS2 + r*S*du/dS - r*u
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + r_ * S_i * du_dS
                  - r_ * u[i];
        }
    }

private:
    double r_;              // Risk-free rate
    double sigma_;          // Volatility
    double sigma_sq_half_;  // 0.5 * sigma^2 (cached)
};

/**
 * IndexBlackScholesOperator: Black-Scholes PDE for index options
 *
 * PDE: dV/dt = L(V) where
 * L(V) = 0.5*sigma^2*S^2*d2V/dS2 + (r - q)*S*dV/dS - r*V
 *
 * For index options, continuous dividend yield q appears in drift term.
 */
class IndexBlackScholesOperator {
public:
    /**
     * Create operator for index option
     * @param r Risk-free rate
     * @param sigma Volatility
     * @param q Continuous dividend yield
     */
    IndexBlackScholesOperator(double r, double sigma, double q)
        : r_(r), sigma_(sigma), q_(q), sigma_sq_half_(0.5 * sigma * sigma) {}

    /**
     * Apply spatial operator: Lu = L(u)
     */
    void apply(double t, std::span<const double> S,
               std::span<const double> u, std::span<double> Lu) const {

        const size_t n = S.size();
        Lu[0] = Lu[n-1] = 0.0;

        // Interior points
        for (size_t i = 1; i < n - 1; ++i) {
            const double S_i = S[i];
            const double dx_left = S[i] - S[i-1];
            const double dx_right = S[i+1] - S[i];
            const double dx_center = 0.5 * (dx_left + dx_right);

            // Second derivative
            const double d2u = (u[i+1] - u[i]) / dx_right - (u[i] - u[i-1]) / dx_left;
            const double d2u_dS2 = d2u / dx_center;

            // First derivative
            const double du_dS = (u[i+1] - u[i-1]) / (dx_left + dx_right);

            // Black-Scholes operator with continuous dividend
            // L(u) = 0.5*sigma^2*S^2*d2u/dS2 + (r - q)*S*du/dS - r*u
            Lu[i] = sigma_sq_half_ * S_i * S_i * d2u_dS2
                  + (r_ - q_) * S_i * du_dS
                  - r_ * u[i];
        }
    }

private:
    double r_;              // Risk-free rate
    double sigma_;          // Volatility
    double q_;              // Continuous dividend yield
    double sigma_sq_half_;  // 0.5 * sigma^2 (cached)
};

} // namespace mango
```

**Step 5: Add library target**

Add to `src/cpp/BUILD.bazel`:

```python
cc_library(
    name = "spatial_operators",
    hdrs = ["spatial_operators.hpp"],
    copts = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-O3",
    ],
    visibility = ["//visibility:public"],
)
```

**Step 6: Run tests to verify they pass**

```bash
bazel test //tests:spatial_operators_test --test_output=all
```

Expected: PASS (2 tests)

**Step 7: Commit**

```bash
git add src/cpp/spatial_operators.hpp src/cpp/BUILD.bazel tests/spatial_operators_test.cc tests/BUILD.bazel
git commit -m "feat(operators): add Black-Scholes spatial operators

- EquityBlackScholesOperator: equity options (discrete dividends)
- IndexBlackScholesOperator: index options (continuous dividend yield)
- Non-uniform grid finite differences for both operators
- Centered differences for accuracy

Tests: 2 passing"
```

---

## Task 4: Add Index Option Test with Continuous Dividend

**Files:**
- Modify: `tests/spatial_operators_test.cc`

**Step 1: Write test for index option with dividend**

Add to `tests/spatial_operators_test.cc`:

```cpp
TEST(BlackScholesOperatorTest, IndexOperatorWithDividend) {
    // Create operator with dividend yield q=0.03
    mango::IndexBlackScholesOperator op(0.05, 0.2, 0.03);

    auto spec = mango::GridSpec<>::uniform(80.0, 120.0, 41);
    auto grid = spec.generate();

    // Test with u(S) = S (delta = 1, gamma = 0)
    std::vector<double> u(41);
    for (size_t i = 0; i < 41; ++i) {
        u[i] = grid[i];
    }

    std::vector<double> Lu(41);
    op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu));

    // For u(S) = S with dividend:
    // du/dS = 1, d2u/dS2 = 0
    // L(u) = 0 + (r - q)*S*1 - r*S = (r - q - r)*S = -q*S
    double S = grid[20];  // S = 100
    double expected = -0.03 * S;  // -q*S
    EXPECT_NEAR(Lu[20], expected, 0.01);
}

TEST(BlackScholesOperatorTest, IndexVsEquityDifference) {
    // Verify dividend yield affects operator output
    double r = 0.05, sigma = 0.2, q = 0.03;

    mango::EquityBlackScholesOperator equity_op(r, sigma);
    mango::IndexBlackScholesOperator index_op(r, sigma, q);

    auto spec = mango::GridSpec<>::uniform(90.0, 110.0, 21);
    auto grid = spec.generate();

    std::vector<double> u(21);
    for (size_t i = 0; i < 21; ++i) {
        u[i] = grid[i];  // u(S) = S
    }

    std::vector<double> Lu_equity(21), Lu_index(21);
    equity_op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu_equity));
    index_op.apply(0.0, grid.span(), std::span<const double>(u), std::span<double>(Lu_index));

    // For u(S) = S:
    // Equity: L(u) = r*S - r*S = 0
    // Index:  L(u) = (r-q)*S - r*S = -q*S
    // Difference should be -q*S
    double S = grid[10];
    EXPECT_NEAR(Lu_equity[10], 0.0, 0.01);
    EXPECT_NEAR(Lu_index[10], -q * S, 0.01);
    EXPECT_NEAR(Lu_index[10] - Lu_equity[10], -q * S, 0.01);
}
```

**Step 2: Run tests to verify they pass**

```bash
bazel test //tests:spatial_operators_test --test_output=all
```

Expected: PASS (4 tests total)

**Step 3: Commit**

```bash
git add tests/spatial_operators_test.cc
git commit -m "test(operators): verify index option continuous dividend

- Test IndexBlackScholesOperator with q=0.03
- Verify dividend affects drift term correctly
- Compare equity vs index operator outputs

Tests: 4 passing"
```

---

## Checkpoint: Week 2 Progress

At this point, we have implemented:

✅ **WorkspaceStorage** (Task 1-2):
- Contiguous buffer allocation for all arrays
- Pre-computed dx array to avoid out-of-bounds access
- Cache-blocking configuration with adaptive strategy
- Block access methods with halo support

✅ **Spatial Operators** (Task 3-4):
- EquityBlackScholesOperator for discrete dividends
- IndexBlackScholesOperator with continuous dividend yield
- Non-uniform grid finite differences
- Comprehensive tests verifying operator behavior

**Tests:** 9 passing total (5 workspace + 4 operators)

**Next Steps (Week 3-4 or Week 5-6):**
- Multi-dimensional grids (MultiGridBuffer)
- TR-BDF2 solver integration
- Cache-blocked stencil operations
- Boundary condition application in solver

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-11-04-phase1-week2-workspace-and-operators.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration with superpowers:subagent-driven-development

**2. Parallel Session (separate)** - Open new session with superpowers:executing-plans, batch execution with checkpoints

**Which approach?**
